import torch
import torch.utils.data
import os
import math
import random
import numpy as np
from scipy.io.wavfile import read
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression_torch(x, channel=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * channel)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

# return the file path to training and validation files as two lists
def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = []
        for x in fi.read().split('\n'):
            training_files.append(x)

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = []
        for x in fi.read().split('\n'):
            validation_files.append(x)

    return training_files, validation_files

mel_basis = {}
hann_window = {}
def mel_spectrogram(audio,
                    n_fft,
                    num_mels,
                    sampling_rate,
                    hop_size,
                    win_size,
                    freq_min,
                    freq_max,
                    center=False):
    # check normalization
    if torch.min(audio) < -1.:
        print('min value is ', torch.min(audio))
    if torch.max(audio) > 1.:
        print('max value is ', torch.max(audio))

    global mel_basis, hann_window
    # NOTE: if condition Modified
    if str(freq_max) + '_' + str(audio.device)  not in mel_basis:
        # Generates the Mel filter bank matrix.
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, freq_min, freq_max)
        mel_basis[str(freq_max) + '_' + str(audio.device)] = torch.from_numpy(mel).float().to(audio.device)
        # A 1-D tensor of size (window_length,), containing the window (weight)
        hann_window[str(audio.device)] = torch.hann_window(win_size).to(audio.device)

    # Padding for FFT to capture all info.
    audio = torch.nn.functional.pad(
        audio.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode='reflect')
    audio = audio.squeeze(1)

    # center: The FFT window is centered at the middle of each frame.
    # This means that some samples before and after the frame's boundaries are included (hence, the need for padding).
    # Centering ensures that the window's peak (maximum weight) aligns with the center of the frame, improving the analysis' stability.
    # spec: (N_fft / 2 + 1,  # of frame), # of frame = 12000/240(hop size) = 50
    # (512, 50)
    spec = torch.stft(
        audio,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(audio.device)],
        center=center,
        pad_mode='reflect',
        normalized=False,
        onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    spec = torch.matmul(mel_basis[str(freq_max) + '_' + str(audio.device)], spec)
    spec = spectral_normalize_torch(spec)

    # mel_spec: (N_mel, #of frames)
    # (80,50)-- add batch dim --> (32, 80, 50)
    return spec

class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        # The n_cache_reuse determines how many times the cached data can be reused before reloading.
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            try:
                audio, sampling_rate = load_wav(filename)
                audio = audio / MAX_WAV_VALUE
                if not self.fine_tuning:
                    audio = normalize(audio) * 0.95
            except:
                print (f"Error on audio: {filename}")
                audio = np.random.normal(size=(160000,)) * 0.05
                sampling_rate = self.sampling_rate
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        # adds a batch dimension
        # (# of audio in a batch, # of audio sample points)
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            # A segment of fixed size (self.segment_size) is extracted randomly from the audio.
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    # pad zero if the audio is not long enough
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
            # (N_mel, #of frames)
            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        # Fine-tuning mode: the mel spectrogram of the audio is preloaded.
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                # (batch dim, N_mel, #frame)
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze()

    def __len__(self):
        return len(self.audio_files)
