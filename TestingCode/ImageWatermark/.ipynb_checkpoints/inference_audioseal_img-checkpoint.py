#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed 28×28 EMNIST-letters images as 16-bit chunks across VoxPopuli clips
using AudioSeal (16 bps).  One glyph → 49 clips → 784 bits  → 28×28 bitmap.

Imperceptibility is still measured per-clip exactly as before.  
Watermark robustness will be computed per-image later from the
    <image_tag>_orig.png     (ground-truth glyph)
    <image_tag>_recon_<atk>.png
files produced here.
"""
import os, struct, random, traceback, math
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch, torchaudio
import soundfile as sf
from PIL import Image
from tqdm import tqdm
import argparse

from audioseal import AudioSeal
from attack import all_attacks, attack, restore_audio

# ---------------------------- CONFIG -------------------------------- #
EMNIST_IDX_IMG = "/root/autodl-tmp/EMNIST_letters/emnist-letters-test-images-idx3-ubyte"

SAMPLE_RATE    = 16_000
CAPACITY_BITS  = 16                # AudioSeal 16 bps
BITS_PER_IMG   = 28*28             # 784
CLIPS_PER_IMG  = BITS_PER_IMG // CAPACITY_BITS   # 49
SEED           = 42
# -------------------------------------------------------------------- #

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detector   = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
generator  = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
msg_bits   = CAPACITY_BITS

##############################################################################
# Helper: read EMNIST idx-3 file in raw form (uint8 0-255)
##############################################################################
def load_emnist_images(idx_path: str) -> np.ndarray:
    idx_path = os.path.expanduser(idx_path)
    with open(idx_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "Not a valid IDX image file"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)

emnist_imgs = load_emnist_images(EMNIST_IDX_IMG)
print(f"Loaded {emnist_imgs.shape[0]} EMNIST glyphs.")

##############################################################################
# Misc. I/O helpers
##############################################################################
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def image_to_bits(img: np.ndarray) -> np.ndarray:
    """Binarise (threshold 128) and flatten to length-784 0/1 array."""
    return (img > 127).astype(np.uint8).flatten()

def bits_to_image(bits: List[int]) -> np.ndarray:
    """Opposite of image_to_bits."""
    return np.array(bits, dtype=np.uint8).reshape(28, 28) * 255

def save_png(img_arr: np.ndarray, path: str):
    Image.fromarray(img_arr, mode="L").save(path)

##############################################################################
# Check-pointing helpers
##############################################################################
def is_clip_completed(root: str, rel_base: str) -> bool:
    """True if *all* required outputs for this clip already exist."""
    subdir = os.path.join(root, os.path.dirname(rel_base))
    base   = os.path.basename(rel_base)

    req = [f"{base}_{tag}.txt" for tag in ["message"]]
    req += [f"{base}_{s}.wav"  for s in ["original", "watermarked"]]
    req += [f"{base}_detect_{atk[0].replace('-', '')}.txt"
            for atk in all_attacks()]
    # audio for attacks is optional – comment out if you need wavs
    return all(os.path.exists(os.path.join(subdir, f)) for f in req)

##############################################################################
# Main
##############################################################################
@torch.no_grad()
def main(VAL_FILELIST, DATASET_BASE_DIR, OUTPUT_DIR):
    filelist_path = os.path.expanduser(VAL_FILELIST)
    output_dir    = os.path.expanduser(OUTPUT_DIR)
    ensure_dir(output_dir)

    with open(filelist_path, "r") as f:
        all_clips = [ln.strip() for ln in f if ln.strip()]

    # How many full images can we embed?
    n_images = len(all_clips) // CLIPS_PER_IMG
    if n_images == 0:
        print("Not enough clips for even one image – abort.")
        return

    chosen_idx = random.sample(range(emnist_imgs.shape[0]), n_images)

    # Drive the big loop -----------------------------------------------------
    for img_no in tqdm(range(n_images), desc="Images processed"):
        img        = emnist_imgs[chosen_idx[img_no]]
        img_bits   = image_to_bits(img)                     # (784,)
        clip_batch = all_clips[img_no*CLIPS_PER_IMG : (img_no+1)*CLIPS_PER_IMG]

        # Accumulators per-attack-type (including 'clean')
        recon_bits: Dict[str, List[int]] = {"clean": []}
        for atk_name, _ in all_attacks():
            recon_bits[atk_name] = []

        image_tag = f"img_{img_no:05d}"
        # Optionally keep some sample PNGs
        save_png(img, os.path.join(output_dir, f"{image_tag}_orig.png"))

        # ------------------------------------------------------------------ #
        for seg, wav_path in enumerate(clip_batch):
            seg_bits = img_bits[seg*CAPACITY_BITS : (seg+1)*CAPACITY_BITS]
            assert len(seg_bits) == CAPACITY_BITS

            wav_path = os.path.expanduser(wav_path)
            filename = os.path.splitext(os.path.basename(wav_path))[0]
            # relative base for exactly the same tree as before
            rel_path = os.path.relpath(
                wav_path,
                os.path.expanduser(DATASET_BASE_DIR)
            )
            base_name = os.path.splitext(rel_path)[0]

            if is_clip_completed(output_dir, base_name):
                # Still need its bits for reconstruction ─ decode again quickly
                recon_clean, _ = quick_decode(wav_path, seg_bits)
                recon_bits["clean"].extend(recon_clean)
                for atk_name, _ in all_attacks():
                    recon_bits[atk_name].extend(
                        quick_decode(os.path.join(
                            output_dir,
                            os.path.dirname(base_name),
                            f"{os.path.basename(base_name)}_attacked_{atk_name.replace('-', '')}.wav"
                        ), seg_bits)[0]
                    )
                continue

            try:
                # ===== 1. LOAD + PRE-PROCESS AUDIO ======================== #
                waveform, sr = torchaudio.load(wav_path)
                if sr != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                    waveform  = resampler(waveform)
                if waveform.shape[0] > 1:                          # mono-fy
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = waveform.to(device)                     # (1, N)
                if waveform.dim() == 2:
                    waveform = waveform.unsqueeze(1)               # (1,1,N)

                # ===== 2. PREPARE MESSAGE (shape: [B, nbits]) ============
                msg_tensor = torch.tensor(seg_bits, dtype=torch.int,
                                          device=device).unsqueeze(0)

                # ===== 3. EMBED WATERMARK ================================ #
                wm = generator.get_watermark(waveform, SAMPLE_RATE, message=msg_tensor)
                wav_wm = waveform + wm

                # ===== 4. SAVE ORIGINAL / WM ============================ #
                out_subdir = os.path.join(output_dir, os.path.dirname(base_name))
                ensure_dir(out_subdir)
                base_file  = os.path.basename(base_name)

                sf.write(os.path.join(out_subdir, f"{base_file}_original.wav"),
                         waveform.squeeze().cpu().numpy(), SAMPLE_RATE)
                sf.write(os.path.join(out_subdir, f"{base_file}_watermarked.wav"),
                         wav_wm.squeeze().cpu().numpy(), SAMPLE_RATE)

                with open(os.path.join(out_subdir, f"{base_file}_message.txt"), "w") as fmsg:
                    fmsg.write(" ".join(map(str, seg_bits)))

                # ===== 5. DECODE FROM CLEAN ============================= #
                res_clean, reco_bits = detector.detect_watermark(wav_wm, SAMPLE_RATE)
                reco_bits = reco_bits.squeeze().cpu().numpy().astype(int).tolist()
                recon_bits["clean"].extend(reco_bits)

                # ===== 6. RUN AND SAVE ALL ATTACKS ====================== #
                for atk_name, _ in all_attacks():
                    atk_wav, _ = attack(wav_wm, SAMPLE_RATE, [(atk_name, 1.0)])
                    atk_wav = restore_audio(atk_wav, wav_wm.size(-1))

                    # decode
                    _, atk_bits = detector.detect_watermark(atk_wav, SAMPLE_RATE)
                    atk_bits = atk_bits.squeeze().cpu().numpy().astype(int).tolist()
                    recon_bits[atk_name].extend(atk_bits)

                    atk_tag = atk_name.replace("-", "")
                    # Optional: uncomment next line if you *need* the attacked audio
                    # sf.write(os.path.join(out_subdir, f"{base_file}_attacked_{atk_tag}.wav"),
                    #          atk_wav.squeeze().cpu().numpy(), SAMPLE_RATE)

                    with open(os.path.join(out_subdir, f"{base_file}_detect_{atk_tag}.txt"), "w") as fdet:
                        ber = 100.0 * (np.array(atk_bits) != np.array(seg_bits)).mean()
                        fdet.write(" ".join(map(str, atk_bits)) + "\n")
                        fdet.write(f"Bit Error Rate: {ber:.2f}%\n")

                # -------- tidy up GPU memory --------------------------------
                del waveform, wm, wav_wm
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"[ERROR] {wav_path}: {e}")
                traceback.print_exc()
                continue
        # -------- End-for (49 clips) --------------------------------------

        # ===== 7. RECONSTRUCT IMAGES & SAVE SOME PNGs ====================== #
        for key, bits in recon_bits.items():
            if len(bits) != BITS_PER_IMG:               # sanity
                print(f"[WARN] {image_tag}: {key} -> got {len(bits)} bits, skip.")
                continue
            img_arr = bits_to_image(bits)
            png_name = f"{image_tag}_recon_{key}.png" if key != "clean" \
                       else f"{image_tag}_recon.png"
            save_png(img_arr, os.path.join(output_dir, png_name))

    # ---------------------------- DONE ---------------------------------- #
    print(f"Finished embedding {n_images} EMNIST glyphs across "
          f"{n_images*CLIPS_PER_IMG} audio clips.")

##############################################################################
def quick_decode(wav_path: str, gt_bits: List[int]):
    """
    Helper to quickly decode a *clean* clip that was already processed.
    Only used when we skip completed clips at script restart.
    """
    if not os.path.isfile(wav_path):
        return [], []
    wav, _ = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(0).to(device)
    _, bits = detector.detect_watermark(wav, SAMPLE_RATE)
    bits = bits.squeeze().cpu().numpy().astype(int).tolist()
    return bits, gt_bits

##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filelist_path") 
    parser.add_argument("-d", "--data_path") 
    parser.add_argument("-o", "--output_dir")
    args = parser.parse_args()

    # Paths
    VAL_FILELIST = os.path.expanduser(args.filelist_path)
    DATASET_BASE_DIR = os.path.expanduser(args.data_path)
    OUTPUT_DIR = os.path.expanduser(args.output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok = True)
        
    main(VAL_FILELIST, DATASET_BASE_DIR, OUTPUT_DIR)
