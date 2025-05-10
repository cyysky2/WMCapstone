import torch
import torchaudio
import random
import numpy as np
import torch.nn.functional as func


attack_methods = ["Pass", 
                  "RSP-30", "RSP-50", "RSP-70", "RSP-90", 
                  "Noise-W10", "Noise-W15", "Noise-W20", "Noise-W25",
                  "SS-015", "SS-01", "SS-005", "SS-003",
                  "AS-10", "AS-15", "AS-20", "AS-30",
                  "AS-450", "AS-400", "AS-350", "AS-300",
                  "TS-85", "TS-90", "TS-95", "TS-97",
                  "TS-103", "TS-105", "TS-110", "TS-115",
                  "EA-03005", "EA-0301", "EA-03015", "EA-0302",
                  "EA-02005", "EA-0201", "EA-02015", "EA-0202", 
                  "LP500", "LP700", "LP1000", "LP1500",
                  "HP6000", "HP3000", "HP2000", "HP1000",
                  "MF-12", "MF-9", "MF-6", "MF-3"]

def all_attacks():
    return [(method, 1) for method in attack_methods]

def attack(y_g_hat, sr, order_list=None):
    # attack is used for whole batch
    # order is tuple，[(CLP, 0.4), (RSP-16, 0.3), (Noise-W20, 0.3)]
    """
    Close loop: no effect: pass √
    Re sampling: Uniformly resampled to 50%: RSP-50 √
    Time stretching: stretched to 0.9 (TS-90) or 1.1 (TS-110) or original
    Lossy compression: MP3 64 kbps (MP3-64) × not differentiable
    Random noise: Noise type is uniformly sampled from White Noise 85dB (Noise-W85) √
    Gain: Gain multiplies a random amplitude to reduce or increase the volume, 0.2 amplitude scaling (APS-20) √ 3.5 amplitude scaling (APS-350) √
    Pass filter: HPF5000 √, LPF1000 √
    Masking: randomly mask out samples: SS-01 √
    Echo augmentation: EA-0301 √
    Smooth-out: median filter: MF-8 √
    """
    # Select an operation based on its probability
    operations, probabilities = zip(*order_list)
    operation = random.choices(operations, weights=probabilities, k=1)[0]

    # No effect
    if operation == "Pass":
        y_g_hat_att = y_g_hat
        return y_g_hat_att, operation

    # resample, but no speed up/down
    if operation == "RSP-30":
        resample1 = torchaudio.transforms.Resample(sr, sr*0.3).to(dtype=y_g_hat.dtype, device=y_g_hat.device)
        resample2 = torchaudio.transforms.Resample(sr*0.3, sr).to(dtype=y_g_hat.dtype, device=y_g_hat.device)
        y_g_hat_att = resample1(y_g_hat)
        y_g_hat_att = resample2(y_g_hat_att)
        return y_g_hat_att, operation
    
    # resample, but no speed up/down
    if operation == "RSP-50":
        resample1 = torchaudio.transforms.Resample(sr, sr*0.5).to(dtype=y_g_hat.dtype, device=y_g_hat.device)
        resample2 = torchaudio.transforms.Resample(sr*0.5, sr).to(dtype=y_g_hat.dtype, device=y_g_hat.device)
        y_g_hat_att = resample1(y_g_hat)
        y_g_hat_att = resample2(y_g_hat_att)
        return y_g_hat_att, operation
    
    # resample, but no speed up/down
    if operation == "RSP-70":
        resample1 = torchaudio.transforms.Resample(sr, sr*0.7).to(dtype=y_g_hat.dtype, device=y_g_hat.device)
        resample2 = torchaudio.transforms.Resample(sr*0.7, sr).to(dtype=y_g_hat.dtype, device=y_g_hat.device)
        y_g_hat_att = resample1(y_g_hat)
        y_g_hat_att = resample2(y_g_hat_att)
        return y_g_hat_att, operation
    
    # resample, but no speed up/down
    if operation == "RSP-90":
        resample1 = torchaudio.transforms.Resample(sr, sr*0.9).to(dtype=y_g_hat.dtype, device=y_g_hat.device)
        resample2 = torchaudio.transforms.Resample(sr*0.9, sr).to(dtype=y_g_hat.dtype, device=y_g_hat.device)
        y_g_hat_att = resample1(y_g_hat)
        y_g_hat_att = resample2(y_g_hat_att)
        return y_g_hat_att, operation

    # white noise: signal: noise = 10
    if operation == "Noise-W10":
        def generate_white_noise(X, N, snr):
            noise = torch.randn(N)
            noise = noise.to(X.device)
            snr = 10 ** (snr / 10)
            power = torch.mean(torch.square(X))
            npower = power / snr
            noise = noise * torch.sqrt(npower)
            X = X + noise
            return X, noise

        y_g_hat_att, noise = generate_white_noise(y_g_hat, y_g_hat.shape[2], 10)
        return y_g_hat_att, operation
    
    # white noise: signal: noise = 31.6
    if operation == "Noise-W15":
        def generate_white_noise(X, N, snr):
            noise = torch.randn(N)
            noise = noise.to(X.device)
            snr = 10 ** (snr / 10)
            power = torch.mean(torch.square(X))
            npower = power / snr
            noise = noise * torch.sqrt(npower)
            X = X + noise
            return X, noise

        y_g_hat_att, noise = generate_white_noise(y_g_hat, y_g_hat.shape[2], 15)
        return y_g_hat_att, operation
    
    # white noise: signal: noise = 100
    if operation == "Noise-W20":
        def generate_white_noise(X, N, snr):
            noise = torch.randn(N)
            noise = noise.to(X.device)
            snr = 10 ** (snr / 10)
            power = torch.mean(torch.square(X))
            npower = power / snr
            noise = noise * torch.sqrt(npower)
            X = X + noise
            return X, noise

        y_g_hat_att, noise = generate_white_noise(y_g_hat, y_g_hat.shape[2], 20)
        return y_g_hat_att, operation
    
    # white noise:  signal: noise = 316
    if operation == "Noise-W25":
        def generate_white_noise(X, N, snr):
            noise = torch.randn(N)
            noise = noise.to(X.device)
            snr = 10 ** (snr / 10)
            power = torch.mean(torch.square(X))
            npower = power / snr
            noise = noise * torch.sqrt(npower)
            X = X + noise
            return X, noise

        y_g_hat_att, noise = generate_white_noise(y_g_hat, y_g_hat.shape[2], 25)
        return y_g_hat_att, operation
    
    # random masking
    if operation == "SS-015":
        def generate_random_tensor(N, rate):
            num_zeros = int(N * rate)
            num_ones = N - num_zeros
            tensor_data = np.concatenate((np.zeros(num_zeros), np.ones(num_ones)))
            np.random.shuffle(tensor_data)
            mask = torch.tensor(tensor_data).float()
            return mask

        mask = generate_random_tensor(y_g_hat.shape[2], 0.15)
        mask = mask.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * mask

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
    
    # random masking
    if operation == "SS-005":
        def generate_random_tensor(N, rate):
            num_zeros = int(N * rate)
            num_ones = N - num_zeros
            tensor_data = np.concatenate((np.zeros(num_zeros), np.ones(num_ones)))
            np.random.shuffle(tensor_data)
            mask = torch.tensor(tensor_data).float()
            return mask

        mask = generate_random_tensor(y_g_hat.shape[2], 0.05)
        mask = mask.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * mask

        return y_g_hat_att, operation
    
    # random masking
    if operation == "SS-003":
        def generate_random_tensor(N, rate):
            num_zeros = int(N * rate)
            num_ones = N - num_zeros
            tensor_data = np.concatenate((np.zeros(num_zeros), np.ones(num_ones)))
            np.random.shuffle(tensor_data)
            mask = torch.tensor(tensor_data).float()
            return mask
        
        mask = generate_random_tensor(y_g_hat.shape[2], 0.03)
        mask = mask.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * mask

        return y_g_hat_att, operation

    # Gain
    if operation == "AS-10":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.1)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * rate_para

        return y_g_hat_att, operation
    
    # Gain
    if operation == "AS-15":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.15)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * rate_para

        return y_g_hat_att, operation
    
    # Gain
    if operation == "AS-20":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.2)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * rate_para

        return y_g_hat_att, operation
    
    # Gain
    if operation == "AS-30":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.3)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * rate_para

        return y_g_hat_att, operation

    # Gain
    if operation == "AS-450":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 4.5)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * rate_para

        return y_g_hat_att, operation
    
    if operation == "AS-400":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 4)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * rate_para

        return y_g_hat_att, operation
    
    # Gain
    if operation == "AS-350":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 3.5)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * rate_para

        return y_g_hat_att, operation
    
    # Gain
    if operation == "AS-300":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 3)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_att = y_g_hat * rate_para

        return y_g_hat_att, operation
    
    
    # speed up (time stretch)
    if operation == "TS-85":
        speed_factor = 0.85
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=int(sr * speed_factor)
        ).to(dtype=y_g_hat.dtype, device=y_g_hat.device)

        y_g_hat_att = resampler(y_g_hat)

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
    
    # speed up (time stretch)
    if operation == "TS-95":
        speed_factor = 0.95
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=int(sr * speed_factor)
        ).to(dtype=y_g_hat.dtype, device=y_g_hat.device)

        y_g_hat_att = resampler(y_g_hat)

        return y_g_hat_att, operation
    
    # speed up (time stretch)
    if operation == "TS-97":
        speed_factor = 0.97
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=int(sr * speed_factor)
        ).to(dtype=y_g_hat.dtype, device=y_g_hat.device)

        y_g_hat_att = resampler(y_g_hat)

        return y_g_hat_att, operation

    # speed down (time stretch)
    if operation == "TS-103":
        speed_factor = 1.03
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=int(sr * speed_factor)
        ).to(dtype=y_g_hat.dtype, device=y_g_hat.device)

        y_g_hat_att = resampler(y_g_hat)

        return y_g_hat_att, operation
    
    # speed down (time stretch)
    if operation == "TS-105":
        speed_factor = 1.05
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
    
    # speed down (time stretch)
    if operation == "TS-115":
        speed_factor = 1.15
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=int(sr * speed_factor)
        ).to(dtype=y_g_hat.dtype, device=y_g_hat.device)

        y_g_hat_att = resampler(y_g_hat)

        return y_g_hat_att, operation

    
    # Echo Augmentation:
    # The operation introduces a delayed and quieter copy of the audio, simulating an echo effect.
    if operation == "EA-03005":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.3)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_echo = y_g_hat * rate_para
        shift_amount = int(y_g_hat.size(2) * 0.05)
        y_g_hat_truncated = y_g_hat.clone()[:, :, :shift_amount]
        y_g_hat_truncated[:, :, :shift_amount] = 0
        padded_tensor = torch.cat((y_g_hat_truncated, y_g_hat_echo), dim=2)
        y_g_hat_att = padded_tensor[:, :, :y_g_hat.size(2)] + y_g_hat

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
    
    # Echo Augmentation:
    # The operation introduces a delayed and quieter copy of the audio, simulating an echo effect.
    if operation == "EA-03015":
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
    
    # Echo Augmentation:
    # The operation introduces a delayed and quieter copy of the audio, simulating an echo effect.
    if operation == "EA-0302":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.3)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_echo = y_g_hat * rate_para
        shift_amount = int(y_g_hat.size(2) * 0.2)
        y_g_hat_truncated = y_g_hat.clone()[:, :, :shift_amount]
        y_g_hat_truncated[:, :, :shift_amount] = 0
        padded_tensor = torch.cat((y_g_hat_truncated, y_g_hat_echo), dim=2)
        y_g_hat_att = padded_tensor[:, :, :y_g_hat.size(2)] + y_g_hat

        return y_g_hat_att, operation
    
    
    # Echo Augmentation:
    # The operation introduces a delayed and quieter copy of the audio, simulating an echo effect.
    if operation == "EA-02005":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.2)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_echo = y_g_hat * rate_para
        shift_amount = int(y_g_hat.size(2) * 0.05)
        y_g_hat_truncated = y_g_hat.clone()[:, :, :shift_amount]
        y_g_hat_truncated[:, :, :shift_amount] = 0
        padded_tensor = torch.cat((y_g_hat_truncated, y_g_hat_echo), dim=2)
        y_g_hat_att = padded_tensor[:, :, :y_g_hat.size(2)] + y_g_hat

        return y_g_hat_att, operation
    
    # Echo Augmentation:
    # The operation introduces a delayed and quieter copy of the audio, simulating an echo effect.
    if operation == "EA-0201":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.2)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_echo = y_g_hat * rate_para
        shift_amount = int(y_g_hat.size(2) * 0.1)
        y_g_hat_truncated = y_g_hat.clone()[:, :, :shift_amount]
        y_g_hat_truncated[:, :, :shift_amount] = 0
        padded_tensor = torch.cat((y_g_hat_truncated, y_g_hat_echo), dim=2)
        y_g_hat_att = padded_tensor[:, :, :y_g_hat.size(2)] + y_g_hat

        return y_g_hat_att, operation
    
    # Echo Augmentation:
    # The operation introduces a delayed and quieter copy of the audio, simulating an echo effect.
    if operation == "EA-02015":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.2)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_echo = y_g_hat * rate_para
        shift_amount = int(y_g_hat.size(2) * 0.15)
        y_g_hat_truncated = y_g_hat.clone()[:, :, :shift_amount]
        y_g_hat_truncated[:, :, :shift_amount] = 0
        padded_tensor = torch.cat((y_g_hat_truncated, y_g_hat_echo), dim=2)
        y_g_hat_att = padded_tensor[:, :, :y_g_hat.size(2)] + y_g_hat

        return y_g_hat_att, operation
    
    # Echo Augmentation:
    # The operation introduces a delayed and quieter copy of the audio, simulating an echo effect.
    if operation == "EA-0202":
        def generate_rate_tensor(N, rate):
            tensor = torch.full((N,), rate)
            return tensor

        rate_para = generate_rate_tensor(y_g_hat.shape[2], 0.2)
        rate_para = rate_para.to(y_g_hat.device)
        y_g_hat_echo = y_g_hat * rate_para
        shift_amount = int(y_g_hat.size(2) * 0.2)
        y_g_hat_truncated = y_g_hat.clone()[:, :, :shift_amount]
        y_g_hat_truncated[:, :, :shift_amount] = 0
        padded_tensor = torch.cat((y_g_hat_truncated, y_g_hat_echo), dim=2)
        y_g_hat_att = padded_tensor[:, :, :y_g_hat.size(2)] + y_g_hat

        return y_g_hat_att, operation

    # lowpass filter
    if operation == "LP500":
        y_g_hat_att = torchaudio.functional.lowpass_biquad(y_g_hat, sr, cutoff_freq=500, Q=0.707)

        return y_g_hat_att, operation
    
    # lowpass filter
    if operation == "LP700":
        y_g_hat_att = torchaudio.functional.lowpass_biquad(y_g_hat, sr, cutoff_freq=700, Q=0.707)

        return y_g_hat_att, operation
    
    # lowpass filter
    if operation == "LP1000":
        y_g_hat_att = torchaudio.functional.lowpass_biquad(y_g_hat, sr, cutoff_freq=1000, Q=0.707)

        return y_g_hat_att, operation
    
    # lowpass filter
    if operation == "LP1500":
        y_g_hat_att = torchaudio.functional.lowpass_biquad(y_g_hat, sr, cutoff_freq=1500, Q=0.707)

        return y_g_hat_att, operation

    # highpass filter
    if operation == "HP6000":
        # Apply a high-pass filter with a cutoff frequency of 6000 Hz
        y_g_hat_att = torchaudio.functional.highpass_biquad(y_g_hat, sr, cutoff_freq=6000, Q=0.707)
        return y_g_hat_att, operation
    
    # highpass filter
    if operation == "HP3000":
        # Apply a high-pass filter with a cutoff frequency of 3000 Hz
        y_g_hat_att = torchaudio.functional.highpass_biquad(y_g_hat, sr, cutoff_freq=3000, Q=0.707)
        return y_g_hat_att, operation
    
    # highpass filter
    if operation == "HP2000":
        # Apply a high-pass filter with a cutoff frequency of 2000 Hz
        y_g_hat_att = torchaudio.functional.highpass_biquad(y_g_hat, sr, cutoff_freq=2000, Q=0.707)
        return y_g_hat_att, operation
    
    # highpass filter
    if operation == "HP1000":
        # Apply a high-pass filter with a cutoff frequency of 1000 Hz
        y_g_hat_att = torchaudio.functional.highpass_biquad(y_g_hat, sr, cutoff_freq=1000, Q=0.707)
        return y_g_hat_att, operation

    
    # The median filter smooths the audio by replacing each sample with the median of a small window of neighboring samples.
    if operation == "MF-12":
        def median_filter(y, kernel_size=12):
            if y.dim() == 2:
                y = y.unsqueeze(0)  # add batch dimension
            if y.dim() == 4:
                y = y.squeeze(1)    # remove extra batch if needed
        
            pad = kernel_size // 2
            # Padding only the time dimension (last dimension)
            y_padded = torch.nn.functional.pad(y, (pad, pad), mode="reflect")  # pad last dim
            y_unfolded = y_padded.unfold(dimension=2, size=kernel_size, step=1)  # unfold time axis
            med_filtered, _ = torch.median(y_unfolded, dim=-1)  # median over window
            return med_filtered

        y_g_hat_att = median_filter(y_g_hat, 12)
        return y_g_hat_att, operation
    
    # The median filter smooths the audio by replacing each sample with the median of a small window of neighboring samples.
    if operation == "MF-9":
        def median_filter(y, kernel_size=12):
            if y.dim() == 2:
                y = y.unsqueeze(0)  # add batch dimension
            if y.dim() == 4:
                y = y.squeeze(1)    # remove extra batch if needed
        
            pad = kernel_size // 2
            # Padding only the time dimension (last dimension)
            y_padded = torch.nn.functional.pad(y, (pad, pad), mode="reflect")  # pad last dim
            y_unfolded = y_padded.unfold(dimension=2, size=kernel_size, step=1)  # unfold time axis
            med_filtered, _ = torch.median(y_unfolded, dim=-1)  # median over window
            return med_filtered

        y_g_hat_att = median_filter(y_g_hat, 9)
        return y_g_hat_att, operation
    
    # The median filter smooths the audio by replacing each sample with the median of a small window of neighboring samples.
    if operation == "MF-6":
        def median_filter(y, kernel_size=12):
            if y.dim() == 2:
                y = y.unsqueeze(0)  # add batch dimension
            if y.dim() == 4:
                y = y.squeeze(1)    # remove extra batch if needed
        
            pad = kernel_size // 2
            # Padding only the time dimension (last dimension)
            y_padded = torch.nn.functional.pad(y, (pad, pad), mode="reflect")  # pad last dim
            y_unfolded = y_padded.unfold(dimension=2, size=kernel_size, step=1)  # unfold time axis
            med_filtered, _ = torch.median(y_unfolded, dim=-1)  # median over window
            return med_filtered

        y_g_hat_att = median_filter(y_g_hat, 6)
        return y_g_hat_att, operation
    
    # The median filter smooths the audio by replacing each sample with the median of a small window of neighboring samples.
    if operation == "MF-3":
        def median_filter(y, kernel_size=12):
            if y.dim() == 2:
                y = y.unsqueeze(0)  # add batch dimension
            if y.dim() == 4:
                y = y.squeeze(1)    # remove extra batch if needed
        
            pad = kernel_size // 2
            # Padding only the time dimension (last dimension)
            y_padded = torch.nn.functional.pad(y, (pad, pad), mode="reflect")  # pad last dim
            y_unfolded = y_padded.unfold(dimension=2, size=kernel_size, step=1)  # unfold time axis
            med_filtered, _ = torch.median(y_unfolded, dim=-1)  # median over window
            return med_filtered

        y_g_hat_att = median_filter(y_g_hat, 3)
        return y_g_hat_att, operation
    
    
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