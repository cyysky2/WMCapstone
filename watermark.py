import torch
import random
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
from torch.nn import Linear
from torch.nn.utils import weight_norm
from resnet import ResNet293

DIGIT_SIZE = 106
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


def watermark_loss(reconstruct_sign, sign):
    loss = func.cross_entropy(reconstruct_sign, sign)
    return loss

def count_common_digit(watermarkA, watermarkB):
    cnt = 0
    for i in range(watermarkA.size(1)):
        if watermarkA[0][i] == watermarkB[0][i]:
            cnt += 1
    return cnt

def attack(y_g_hat, order_list=None):
    # attack is used for whole batch
    # order is tuple，[(CLP, 0.4), (RSP-16, 0.3), (Noise-W20, 0.3)]
    """
    Close loop: no effect: CLP √
    Re sampling: Uniformly resampled to 90%: RSP-90 √
    Time stretching: stretched to 0.9 (TS-90) or 1.1 (TS-110) or original
    Lossy compression: MP3 64 kbps (MP3-64) × not differentiable
    Random noise: Noise type is uniformly sampled from White Noise 35dB (Noise-W35) √
    Gain: Gain multiplies a random amplitude to reduce or increase the volume, 0.9 amplitude scaling (APS-90) √ 1.5 amplitude scaling (APS-150) √
    Pass filter: HPF100 √, LPF5000 √
    Masking: randomly mask out samples: SS-01 √
    Echo augmentation: EA-0301 √
    Smooth-out: median filter: MF-3 √
    """
    # Select an operation based on its probability
    operations, probabilities = zip(*order_list)
    operation = random.choices(operations, weights=probabilities, k=1)[0]

    # No effect
    if operation == "CLP":
        y_g_hat_att = y_g_hat
        return y_g_hat_att, operation

    # resample, but no speed up/down
    if operation == "RSP-90":
        resample1 = torchaudio.transforms.Resample(24000, 21600).to(y_g_hat.device)
        resample2 = torchaudio.transforms.Resample(21600, 24000).to(y_g_hat.device)
        y_g_hat_att = resample1(y_g_hat)
        y_g_hat_att = resample2(y_g_hat_att)
        return y_g_hat_att, operation

    # white noise
    if operation == "Noise-W35":
        def generate_white_noise(X, N, snr):
            noise = torch.randn(N)
            noise = noise.to(X.device)
            snr = 10 ** (snr / 10)
            power = torch.mean(torch.square(X))
            npower = power / snr
            noise = noise * torch.sqrt(npower)
            X = X + noise
            return X, noise

        y_g_hat_att, noise = generate_white_noise(y_g_hat, y_g_hat.shape[2], 35)
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

        mask = generate_random_tensor(y_g_hat.shape[2], 0.001)
        mask = mask.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * mask

        return y_g_hat_att, operation

    # Gain
    if operation == "AS-90":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.9)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * rate_para

        return y_g_hat_att, operation

    # Gain
    if operation == "AS-150":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 1.5)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * rate_para

        return y_g_hat_att, operation

    # speed up (time stretch)
    if operation == "TS-90":
        speed_factor = 0.9
        resampler = torchaudio.transforms.Resample(
            orig_freq=24000,
            new_freq=int(24000 * speed_factor)
        ).to(y_g_hat.device)

        y_g_hat_att = resampler(y_g_hat)

        return y_g_hat_att, operation

    # speed down (time stretch)
    if operation == "TS-110":
        speed_factor = 110
        resampler = torchaudio.transforms.Resample(
            orig_freq=24000,
            new_freq=int(24000 * speed_factor)
        ).to(y_g_hat.device)

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
        shift_amount = int(y_g_hat.size(2) * 0.15)
        y_g_hat_truncated = y_g_hat.clone()[:, :, :shift_amount]
        y_g_hat_truncated[:, :, :shift_amount] = 0
        padded_tensor = torch.cat((y_g_hat_truncated, y_g_hat_echo), dim=2)
        y_g_hat_att = padded_tensor[:, :, :y_g_hat.size(2)] + y_g_hat

        return y_g_hat_att, operation

    # lowpass filter
    if operation == "LP5000":
        y_g_hat_att = torchaudio.functional.lowpass_biquad(y_g_hat, 24000, cutoff_freq=5000, Q=0.707)

        return y_g_hat_att, operation

    # highpass filter
    if operation == "HP100":
        # Apply a high-pass filter with a cutoff frequency of 100 Hz
        y_g_hat_att = torchaudio.functional.highpass_biquad(y_g_hat, 24000, cutoff_freq=100, Q=0.707)
        return y_g_hat_att, operation

    # The median filter smooths the audio by replacing each sample with the median of a small window of neighboring samples.
    if operation == "MF-3":
        window_size = 3
        filtered_signal = torch.zeros_like(y_g_hat)
        for i in range(y_g_hat.size(2)):
            start = max(0, i - window_size // 2)
            end = min(y_g_hat.size(2), i + window_size // 2 + 1)
            window = y_g_hat[:, :, start:end]
            filtered_signal[:, :, start:end] = torch.median(window)

        y_g_hat_att = filtered_signal
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