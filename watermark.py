import torch
import random
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.utils import weight_norm
from resnet import ResNet293

LRELU_SLOPE = 0.1

# generate a random wm of size (32, 25), dim = 106 (all in-use ascii char), 25@106
# return: (32, 25)
def random_watermark(batch_size, h):
    sign = torch.randint(low=0, high=h.Watermark["digit_size"], size=(batch_size, h.Watermark["capacity"]))
    return sign

# cast a random wm to match with audio feature space
class WatermarkEncoder(torch.nn.Module):
    def __init__(self, h):
        super(WatermarkEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=h.Watermark['digit_size'], embedding_dim=h.Watermark['embed_dim'])
        self.final_dim = h.Watermark["encoder_final_dim"]
        assert self.final_dim * h.Watermark["capacity"] == 512 * 50
        print("Watermark feature vector repeat:", 50/h.Watermark["capacity"], " times.")

        upcast_dim = [h.Watermark["embed_dim"]] + h.Watermark["encoder_mlp_dims"] + [self.final_dim]
        layers = []
        for i in range(len(upcast_dim) - 2):
            layers.append(weight_norm(nn.Linear(upcast_dim[i], upcast_dim[i + 1])))
            layers.append(nn.LeakyReLU(negative_slope=LRELU_SLOPE))
        # last layer (no activation)
        layers.append(weight_norm(nn.Linear(upcast_dim[-2], upcast_dim[-1])))
        self.up_cast = nn.Sequential(*layers)

    def forward(self, x):
        # (32, 25, 32) -> (32, 25, 1024)
        x = self.embedding(x)  # (batch, capacity, embed_dim): (32, 25, 32)
        # (32, 25, 32) -> (32, 25, 1024)
        x = self.up_cast(x)  # (batch, capacity, self.final_dim): (32, 25, 1024)
        x = x.view(x.shape[0], 50, 512)  # (batch, 50, 512): (32, 50, 512)
        return x

# Discarded
class WatermarkDecoder(nn.Module):
    def __init__(self, h):
        super(WatermarkDecoder, self).__init__()
        # resnet_embed_dim determines the output dimension of resnet.
        self.recover = ResNet293(
            feat_dim=h.Watermark['resnet_feat_dim'],
            embed_dim=h.Watermark['resnet_embed_dim'],
            pooling_func='MQMHASTP'
        )
        self.final_dim = h.Watermark["capacity"]

        dims = [h.Watermark['resnet_embed_dim']] + h.Watermark['decoder_mlp_dims'] + [self.final_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(weight_norm(nn.Linear(dims[i], dims[i + 1])))
            layers.append(nn.LeakyReLU(negative_slope=LRELU_SLOPE))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1])))
        self.to_wm = nn.Sequential(*layers)

    # input (32, 80, 50)
    def forward(self, x):
        # (batch_size, num_frames, num_mel_filters (80))
        x = x.transpose(1, 2)  # (batch, 80, 50) -> (batch, 50, 80)
        x = self.recover(x)[-1]  # (batch, 1024)
        x = self.to_wm(x)  # (batch, 25)
        return x

class ImprovedWatermarkDecoder(nn.Module):
    def __init__(self, h):
        super(ImprovedWatermarkDecoder, self).__init__()

        self.capacity = h.Watermark["capacity"]  # 25 digits
        self.num_classes = h.Watermark["digit_size"]  # 106 possibilities per digit
        resnet_feat_dim = h.Watermark["resnet_feat_dim"]  # 80
        resnet_embed_dim = h.Watermark["resnet_embed_dim"]  # 1024

        self.recover = ResNet293(
            feat_dim=resnet_feat_dim,
            embed_dim=resnet_embed_dim,
            pooling_func='MQMHASTP'
        )

        # This shared MLP processes the ResNet embedding before per-digit classification.
        mlp_dims = [resnet_embed_dim] + h.Watermark['decoder_mlp_dims']
        layers = []
        dropout_rate = h.Watermark.get('dropout_rate', 0.3)
        for i in range(len(mlp_dims) - 1):
            layers.append(weight_norm(nn.Linear(mlp_dims[i], mlp_dims[i + 1])))
            layers.append(nn.LeakyReLU(negative_slope=LRELU_SLOPE))
            layers.append(nn.Dropout(p=dropout_rate))
        self.mlp_body = nn.Sequential(*layers)

        # Create one per-digit classifier.
        # Each classifier is a simple linear layer that outputs a vector of logits over the possible classes.
        self.classifiers = nn.ModuleList([
            weight_norm(nn.Linear(mlp_dims[-1], self.num_classes))
            for _ in range(self.capacity)
        ])

    # input (32, 80, 50)
    def forward(self, x):
        # (batch_size, num_frames, num_mel_filters (80))
        x = x.transpose(1, 2)  # (batch, 80, 50) -> (batch, 50, 80)
        x = self.recover(x)[-1]  # (batch, 50, 80) -> (batch, 1024)
        x = self.mlp_body(x)  # (batch, 1024) -> (batch, 512) -> (batch, 128)

        # For each digit position, get the classifier’s logits.
        logits_list = []
        for clf in self.classifiers:
            digit_logits = clf(x)  # (batch, 128) -> (batch, digit_size): (batch, 128) -> (batch, 106)
            logits_list.append(digit_logits.unsqueeze(1))  # Expand to (batch, 1, 106)
        # Concatenate all digit logits to form the final output.
        logits = torch.cat(logits_list, dim=1)  # (batch, capacity, num_classes)

        rec_watermark = torch.argmax(logits, dim=-1)

        return logits, rec_watermark

def watermark_loss(reconstruct_sign_logit, sign):
    """
    Args:
        reconstruct_sign_logit: Tensor of shape (batch, capacity, num_classes)
        sign: Tensor of shape (batch, capacity) with integer labels in [0, num_classes-1]
    Returns:
        Scalar loss (averaged cross entropy over all digits in the batch)
    """
    batch_size, capacity, num_classes = reconstruct_sign_logit.shape

    # Reshape for CrossEntropyLoss: (batch * capacity, num_classes)
    logits_flat = reconstruct_sign_logit.view(-1, num_classes)
    sign_flat = sign.view(-1)

    loss = func.cross_entropy(logits_flat, sign_flat)
    return loss

def count_common_digit(watermarkA, watermarkB):
    cnt = 0
    for i in range(watermarkA.size(1)):
        if watermarkA[0][i] == watermarkB[0][i]:
            cnt += 1
    return cnt

def attack(y_g_hat, sr, order_list=None):
    # attack is used for whole batch
    # order is tuple，[(CLP, 0.4), (RSP-16, 0.3), (Noise-W20, 0.3)]
    """
    Close loop: no effect: pass √
    Re sampling: Uniformly resampled to 70%: RSP-70 √
    Time stretching: stretched to 0.9 (TS-90) or 1.1 (TS-110) or original
    Lossy compression: MP3 64 kbps (MP3-64) × not differentiable
    Random noise: Noise type is uniformly sampled from White Noise yielding 55 SNR (Noise-W55) √
    Gain: Gain multiplies a random amplitude to reduce or increase the volume, 0.5 amplitude scaling (APS-50) √ 2.5 amplitude scaling (APS-250) √
    Pass filter: HPF500 √, LPF1000 √
    Masking: randomly mask out samples: SS-01 √
    Echo augmentation: EA-0301 √
    Smooth-out: median filter: MF-6 √
    """
    # Select an operation based on its probability
    operations, probabilities = zip(*order_list)
    operation = random.choices(operations, weights=probabilities, k=1)[0]

    # No effect
    if operation == "Pass":
        y_g_hat_att = y_g_hat
        return y_g_hat_att, operation

    # resample, but no speed up/down
    if operation == "RSP-70":
        resample1 = torchaudio.transforms.Resample(sr, sr*0.7).to(dtype=y_g_hat.dtype, device=y_g_hat.device)
        resample2 = torchaudio.transforms.Resample(sr*0.7, sr).to(dtype=y_g_hat.dtype, device=y_g_hat.device)
        y_g_hat_att = resample1(y_g_hat)
        y_g_hat_att = resample2(y_g_hat_att)
        return y_g_hat_att, operation

    # white noise
    if operation == "Noise-W55":
        def generate_white_noise(X, N, snr):
            noise = torch.randn(N)
            noise = noise.to(X.device)
            snr = 10 ** (snr / 10)
            power = torch.mean(torch.square(X))
            npower = power / snr
            noise = noise * torch.sqrt(npower)
            X = X + noise
            return X, noise

        y_g_hat_att, noise = generate_white_noise(y_g_hat, y_g_hat.shape[2], 55)
        return y_g_hat_att, operation

    # random masking
    if operation == "SS-01":
        def generate_random_tensor(N, rate):
            num_zeros = int(N * rate)
            num_ones = N - num_zeros
            tensor_data = np.concatenate((np.zeros(num_zeros), np.ones(num_ones)))
            np.random.shuffle(tensor_data)
            mask = torch.tensor(tensor_data).float()
            return mask

        mask = generate_random_tensor(y_g_hat.shape[2], 0.1)
        mask = mask.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * mask

        return y_g_hat_att, operation

    # Gain
    if operation == "AS-50":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.5)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * rate_para

        return y_g_hat_att, operation

    # Gain
    if operation == "AS-250":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 2.5)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * rate_para

        return y_g_hat_att, operation

    # speed up (time stretch)
    if operation == "TS-90":
        speed_factor = 0.9
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=int(sr * speed_factor)
        ).to(dtype=y_g_hat.dtype, device=y_g_hat.device)

        y_g_hat_att = resampler(y_g_hat)

        return y_g_hat_att, operation

    # speed down (time stretch)
    if operation == "TS-110":
        speed_factor = 1.1
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=int(sr * speed_factor)
        ).to(dtype=y_g_hat.dtype, device=y_g_hat.device)

        y_g_hat_att = resampler(y_g_hat)

        return y_g_hat_att, operation

    # Echo Augmentation:
    # The operation introduces a delayed and quieter copy of the audio, simulating an echo effect.
    if operation == "EA-0301":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.3)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_echo = y_g_hat * rate_para
        shift_amount = int(y_g_hat.size(2) * 0.1)
        y_g_hat_truncated = y_g_hat.clone()[:, :, :shift_amount]
        y_g_hat_truncated[:, :, :shift_amount] = 0
        padded_tensor = torch.cat((y_g_hat_truncated, y_g_hat_echo), dim=2)
        y_g_hat_att = padded_tensor[:, :, :y_g_hat.size(2)] + y_g_hat

        return y_g_hat_att, operation

    # lowpass filter
    if operation == "LP1000":
        y_g_hat_att = torchaudio.functional.lowpass_biquad(y_g_hat, sr, cutoff_freq=1000, Q=0.707)

        return y_g_hat_att, operation

    # highpass filter
    if operation == "HP500":
        # Apply a high-pass filter with a cutoff frequency of 100 Hz
        y_g_hat_att = torchaudio.functional.highpass_biquad(y_g_hat, sr, cutoff_freq=500, Q=0.707)
        return y_g_hat_att, operation

    # The median filter smooths the audio by replacing each sample with the median of a small window of neighboring samples.
    if operation == "MF-6":
        def median_filter(y, kernel_size=6):
            pad = kernel_size // 2
            y_padded = func.pad(y, (pad, pad), mode="reflect")
            # y: (batch, channels, time)
            y_unfolded = y_padded.unfold(dimension=2, size=kernel_size, step=1)  # shape: (B, C, T, K)
            med_filtered, _ = torch.median(y_unfolded, dim=-1)  # take median across window
            return med_filtered

        y_g_hat_att = median_filter(y_g_hat, kernel_size=6)
        return y_g_hat_att, operation

# Keep audio length after time stretching attack
def restore_audio(attacked_audio, original_length = 12000):
    new_length = attacked_audio.shape[-1]
    if original_length < new_length:
        attacked_audio = attacked_audio[..., :original_length]
    elif original_length > new_length:
        pad_amount = original_length - attacked_audio.shape[-1]
        left_pad = pad_amount // 2
        right_pad = pad_amount - left_pad
        attacked_audio = torch.nn.functional.pad(attacked_audio, (left_pad, right_pad), mode='reflect')

    return attacked_audio