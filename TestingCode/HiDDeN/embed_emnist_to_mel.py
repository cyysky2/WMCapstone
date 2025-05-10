#!/usr/bin/env python3
"""
Embed a random EMNIST-letters glyph (28×28 → 784 bits) into every
mel-spectrogram PNG in the validation set using a trained HiDDeN model.

Required paths – edit if your layout differs.
"""
import os, random, pathlib, argparse
import numpy as np
import torch, torchvision
from PIL import Image
import idx2numpy                                    # pip install idx2numpy

# ──────────── USER-EDITABLE PATHS ──────────────────────────────────────────
EMNIST_IMG_FILE = "~/autodl-tmp/EMNIST_letters/emnist-letters-test-images-idx3-ubyte"
RUN_FOLDER      = "~/autodl-tmp/HiDDeN/runs/mel128_emnist784 2025.04.24--20-05-22"
CKPT_FILE       = "checkpoints/mel128_emnist784--epoch-98.pyt"
VAL_DIR         = "~/autodl-tmp/HiDDeN_dataset/LibriSpeech_dev-clean_mel/val/val_class"
# ───────────────────────────────────────────────────────────────────────────

# ---------- resolve & sanity-check file system -----------------------------
EMNIST_IMG_FILE = os.path.expanduser(EMNIST_IMG_FILE)
RUN_FOLDER      = os.path.expanduser(RUN_FOLDER)
CKPT_PATH       = os.path.join(RUN_FOLDER, CKPT_FILE)
OPTIONS_FILE    = os.path.join(RUN_FOLDER, "options-and-config.pickle")

assert os.path.isfile(EMNIST_IMG_FILE), f"Missing EMNIST file: {EMNIST_IMG_FILE}"
assert os.path.isfile(CKPT_PATH),       f"Missing checkpoint: {CKPT_PATH}"
assert os.path.isfile(OPTIONS_FILE),    f"Missing options/config pickle in run folder."
assert os.path.isdir(VAL_DIR := os.path.expanduser(VAL_DIR)), "Validation PNG folder not found"
# --------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- load repo objects -----------------------------
from options import HiDDenConfiguration
from model.hidden import Hidden
import utils
from noise_layers.noiser import Noiser
# --------------------------------------------------------------------------

# --------------------------- 1. load config -------------------------------
train_opts, hidden_cfg, noise_cfg = utils.load_options(OPTIONS_FILE)
noiser = Noiser(noise_cfg, device = DEVICE)
model  = Hidden(hidden_cfg, DEVICE, noiser, tb_logger=None)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
utils.model_from_checkpoint(model, ckpt)
print(f"Loaded checkpoint «{os.path.basename(CKPT_PATH)}» "
      f"(epoch {ckpt.get('epoch', '?')})")
# --------------------------------------------------------------------------

# --------------------------- 2. EMNIST glyph ------------------------------
emnist_imgs = idx2numpy.convert_from_file(EMNIST_IMG_FILE)  # (N, 28, 28, uint8)
glyph = emnist_imgs[random.randrange(len(emnist_imgs))]
glyph = np.rot90(glyph, -1)[:, ::-1]                        # orient upright
bits  = (glyph > 0).astype(np.float32).flatten()            # → 784 bits
assert len(bits) == hidden_cfg.message_length, \
       f"Message length mismatch ({len(bits)} vs {hidden_cfg.message_length})"

# NEW ­– save a copy once for later evaluations
np.save(os.path.join(RUN_FOLDER, "watermark_bits.npy"), bits)
print("Saved watermark_bits.npy to run folder")

message = torch.from_numpy(bits).unsqueeze(0).to(DEVICE)    # (1, 784)
# --------------------------------------------------------------------------

# --------------------------- 3. I/O helpers -------------------------------
to_tensor = torchvision.transforms.ToTensor()
def load_png(path):
    return to_tensor(Image.open(path).convert("RGB"))        # (3,128,128) in [0,1]

def save_png(tensor, path):
    tensor = tensor.clamp(0, 1).cpu()
    torchvision.utils.save_image(tensor, path)
# --------------------------------------------------------------------------

OUT_DIR = pathlib.Path(VAL_DIR + "_encoded")
OUT_DIR.mkdir(parents=True, exist_ok=True)

png_files = sorted(p for p in pathlib.Path(VAL_DIR).glob("*.png"))
assert png_files, "No PNGs found to encode."

with torch.no_grad():
    for idx, png_path in enumerate(png_files, 1):
        cover = load_png(png_path).unsqueeze(0).to(DEVICE) * 2 - 1  # → [-1,1]
        _, (encoded, _, decoded) = model.validate_on_batch([cover, message])

        save_png((encoded.squeeze(0) + 1) / 2, OUT_DIR / png_path.name)

        if idx == 1:   # quick sanity
            dec = (decoded.squeeze(0) > 0).float().cpu().numpy()
            ber = np.mean(dec != bits)
            print(f"[{png_path.name}] BER = {ber:.4f}")

        if idx % 100 == 0 or idx == len(png_files):
            print(f"Processed {idx}/{len(png_files)}")

print(f"\nAll encoded images written to: {OUT_DIR}")
