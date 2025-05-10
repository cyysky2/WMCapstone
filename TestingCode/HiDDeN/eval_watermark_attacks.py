#!/usr/bin/env python3
"""
Batch evaluation: attack -> decode -> metric -> csv

Using restore_audio() after attacks to keep time axis fixed.
"""

import os, csv, math, random, pathlib
import numpy as np
import soundfile as sf
import librosa
import torch, torchvision
from PIL import Image
from tqdm import tqdm

# ---- PATH CONFIG ----
ENCODED_DIR = os.path.expanduser("~/autodl-tmp/HiDDeN_dataset/LibriSpeech_dev-clean_mel/val/val_class_encoded")
OUT_AUDIO   = os.path.join(ENCODED_DIR, "attacked_wav")
OUT_MEL     = os.path.join(ENCODED_DIR, "attacked_mel")
CSV_PATH    = os.path.join(ENCODED_DIR, "attack_results.csv")
WATERMARK_BITS = os.path.expanduser("~/autodl-tmp/HiDDeN/runs/mel128_emnist784 2025.04.24--20-05-22/watermark_bits.npy")
RUN_FOLDER     = os.path.expanduser("~/autodl-tmp/HiDDeN/runs/mel128_emnist784 2025.04.24--20-05-22")
CKPT_FILE      = "checkpoints/mel128_emnist784--epoch-98.pyt"

batch_size = 16
SR         = 24_000
N_MELS     = 80
N_FFT      = 1024
HOP        = 240
GL_NITER   = 64
DYN_RANGE  = 80.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- IMPORT PROJECT MODULES ----
import sys
sys.path.append(os.getcwd())
from options import HiDDenConfiguration
from model.hidden import Hidden
import utils
from noise_layers.noiser import Noiser
from attack import all_attacks, attack, restore_audio  

# ---- LOAD MODEL ----
opts_file = os.path.join(RUN_FOLDER, "options-and-config.pickle")
train_opts, hidden_cfg, noise_cfg = utils.load_options(opts_file)
noiser = Noiser(noise_cfg, device= DEVICE)
model  = Hidden(hidden_cfg, DEVICE, noiser, tb_logger=None)
ckpt   = torch.load(os.path.join(RUN_FOLDER, CKPT_FILE), map_location=DEVICE)
utils.model_from_checkpoint(model, ckpt)

# ---- ORIGINAL WATERMARK ----
orig_bits = np.load(WATERMARK_BITS)
orig_img = orig_bits.reshape(28, 28).astype(np.float32)

# ---- UTILITIES ----
to_tensor = torchvision.transforms.ToTensor()

def png_to_mel(png_path):
    img = Image.open(png_path).convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr[:N_MELS, :]
    m_db = arr * DYN_RANGE - DYN_RANGE
    m_power = librosa.db_to_power(m_db)
    return m_power

def mel_to_audio(m_power):
    return librosa.feature.inverse.mel_to_audio(
        M=m_power, sr=SR, n_fft=N_FFT, hop_length=HOP,
        power=2.0, n_iter=GL_NITER)

def audio_to_mel_png(y, out_png):
    m = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, power=2.0)
    m_db = librosa.power_to_db(m, ref=np.max, top_db=DYN_RANGE)
    img = (m_db + DYN_RANGE) / DYN_RANGE
    img = (img * 255).clip(0, 255).astype(np.uint8)

    h, w = img.shape
    if w > 128:
        img = img[:, :128]
    elif w < 128:
        pad_w = 128 - w
        img = np.pad(img, ((0,0), (0,pad_w)), mode='constant')

    canvas = np.zeros((128,128), np.uint8)
    canvas[:80, :128] = img
    Image.fromarray(canvas).save(out_png)

def mse_psnr(a, b):
    mse = np.mean((a - b) ** 2)
    psnr = 99.0 if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))
    return mse, psnr

os.makedirs(OUT_AUDIO, exist_ok=True)
os.makedirs(OUT_MEL, exist_ok=True)

# ---- MAIN BATCH LOOP ----
png_files = sorted(p for p in pathlib.Path(ENCODED_DIR).glob("*.png"))
assert png_files, "No PNGs found to encode."

results = {}

batch_audio_paths = []
batch_audios = []
batch_png_names = []

for idx, png_path in enumerate(tqdm(png_files, desc="Preparing batches")):
    mel_power = png_to_mel(png_path)
    audio = mel_to_audio(mel_power)
    batch_audios.append(audio)
    batch_png_names.append(png_path.stem)

    if len(batch_audios) == batch_size or idx == len(png_files) - 1:
        batch_tensor = torch.stack([
            torch.from_numpy(a).unsqueeze(0).unsqueeze(0) for a in batch_audios
        ]).float().to('cpu')  # (B,1,1,T)

        for atk_name, param in all_attacks():
            atk_batch, _ = attack(batch_tensor, SR, order_list=[(atk_name, param)])
            atk_batch = [restore_audio(w, batch_tensor.size(-1)) for w in atk_batch]

            # For each attacked audio
            for bidx, atk_wave in enumerate(atk_batch):
                atk_wave_np = atk_wave.squeeze(0).squeeze(0).numpy()

                wav_out = os.path.join(OUT_AUDIO, f"{batch_png_names[bidx]}_{atk_name}.wav")
                sf.write(wav_out, atk_wave_np, SR)

                mel_png_out = os.path.join(OUT_MEL, f"{batch_png_names[bidx]}_{atk_name}.png")
                audio_to_mel_png(atk_wave_np, mel_png_out)

        batch_audio_paths.extend(batch_png_names)
        batch_audios.clear()
        batch_png_names.clear()

# ---- DECODING + METRICS ----
tqdm.write("Starting decoding and evaluation...")

attack_types = [atk_name for atk_name, _ in all_attacks()]

for atk_name in attack_types:
    decode_tensors = []
    decode_names = []

    for mel_path in pathlib.Path(OUT_MEL).glob(f"*_{" + atk_name + "}.png"):
        tensor = to_tensor(Image.open(mel_path).convert("RGB")) * 2 - 1
        decode_tensors.append(tensor)
        decode_names.append(mel_path.stem)

        if len(decode_tensors) == batch_size or mel_path == list(pathlib.Path(OUT_MEL).glob(f"*_{" + atk_name + "}.png"))[-1]:
            batch = torch.stack(decode_tensors).to(DEVICE)
            dummy_msg = torch.zeros((batch.size(0), hidden_cfg.message_length), device=DEVICE)
            with torch.no_grad():
                _, (_, _, decoded) = model.validate_on_batch([batch, dummy_msg])

            for i in range(batch.size(0)):
                decoded_bits = (decoded[i] > 0).float().cpu().numpy()
                dec_img = decoded_bits.reshape(28,28)

                mse, psnr = mse_psnr(orig_img, dec_img)
                results.setdefault(atk_name, [[], []])
                results[atk_name][0].append(mse)
                results[atk_name][1].append(psnr)

            decode_tensors.clear()
            decode_names.clear()

# ---- SAVE CSV ----
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["attack", "avg_mse", "std_mse", "avg_psnr", "std_psnr", "N"])
    for atk_name, (mse_list, psnr_list) in results.items():
        writer.writerow([
            atk_name,
            np.mean(mse_list), np.std(mse_list),
            np.mean(psnr_list), np.std(psnr_list),
            len(mse_list)
        ])

print(f"\nSaved attack evaluation CSV to {CSV_PATH}")
