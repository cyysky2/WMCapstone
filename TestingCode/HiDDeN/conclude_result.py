#!/usr/bin/env python3
"""
Standalone evaluation summary script:
- Decodes all mel spectrogram PNGs from attack_mel/
- Compares each decoded watermark to the original EMNIST letter
- Computes MSE and PSNR
- Aggregates results per attack type
- Saves CSV

Usage: python conclude_watermark_eval.py
"""

import os, re, csv, math
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch, torchvision

# --- CONFIG ---
MEL_DIR       = os.path.expanduser("~/autodl-tmp/HiDDeN_dataset/LibriSpeech_dev-clean_mel/val/val_class_encoded/attacked_mel")
WATERMARK_NPY = os.path.expanduser("~/autodl-tmp/HiDDeN/runs/mel128_emnist784 2025.04.24--20-05-22/watermark_bits.npy")
RUN_FOLDER    = os.path.expanduser("~/autodl-tmp/HiDDeN/runs/mel128_emnist784 2025.04.24--20-05-22")
CKPT_FILE     = "checkpoints/mel128_emnist784--epoch-77.pyt"
OUT_CSV       = os.path.expanduser("~/autodl-tmp/HiDDeN_dataset/LibriSpeech_dev-clean_mel/val/attack_results_conclude.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD MODEL ---
from options import HiDDenConfiguration
from model.hidden import Hidden
import utils
from noise_layers.noiser import Noiser

opts_file = os.path.join(RUN_FOLDER, "options-and-config.pickle")
train_opts, hidden_cfg, noise_cfg = utils.load_options(opts_file)
noiser = Noiser(noise_cfg, device = DEVICE)
model = Hidden(hidden_cfg, DEVICE, noiser, tb_logger=None)
ckpt = torch.load(os.path.join(RUN_FOLDER, CKPT_FILE), map_location=DEVICE)
utils.model_from_checkpoint(model, ckpt)

# --- LOAD GROUND TRUTH WATERMARK ---
gt_bits = np.load(WATERMARK_NPY)
gt_img = gt_bits.reshape(28, 28).astype(np.float32)

# --- UTILS ---
to_tensor = torchvision.transforms.ToTensor()

def decode_watermark_from_png(png_path):
    img = Image.open(png_path).convert("RGB")
    x = to_tensor(img).unsqueeze(0).to(DEVICE) * 2 - 1
    dummy_msg = torch.zeros((1, hidden_cfg.message_length), device=DEVICE)
    with torch.no_grad():
        _, (_, _, decoded) = model.validate_on_batch([x, dummy_msg])
    decoded_bits = (decoded.squeeze(0) > 0).float().cpu().numpy()
    return decoded_bits

def compute_metrics(decoded_img, gt_img, scale_to_255=False):
    if scale_to_255:
        decoded_img = decoded_img * 255.0
        gt_img = gt_img * 255.0
    mse = np.mean((decoded_img - gt_img) ** 2)
    psnr = 99.0 if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse)) if scale_to_255 else 20 * math.log10(1.0 / math.sqrt(mse))
    return mse, psnr


# --- SCAN FILES ---
png_files = [f for f in os.listdir(MEL_DIR) if f.lower().endswith(".png")]
assert png_files, f"No PNG files found in {MEL_DIR}"

# --- EVAL LOOP ---
results = {}

for fname in tqdm(png_files, desc="Evaluating"):
    attack_match = re.search(r"_([A-Za-z0-9\-]+)\.png$", fname)
    if not attack_match:
        continue  # skip files without attack suffix
    atk_name = attack_match.group(1)

    fpath = os.path.join(MEL_DIR, fname)
    decoded_bits = decode_watermark_from_png(fpath)
    decoded_img = decoded_bits.reshape(28, 28)
    mse, psnr = compute_metrics(decoded_img, gt_img, True)
    results.setdefault(atk_name, [[], []])
    results[atk_name][0].append(mse)
    results[atk_name][1].append(psnr)

# --- SAVE CSV ---
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["attack", "avg_mse", "std_mse", "avg_psnr", "std_psnr", "N"])
    for atk_name, (mse_list, psnr_list) in sorted(results.items()):
        writer.writerow([
            atk_name,
            np.mean(mse_list), np.std(mse_list),
            np.mean(psnr_list), np.std(psnr_list),
            len(mse_list)
        ])

print(f"\n✅ Concluded result written to: {OUT_CSV}")
