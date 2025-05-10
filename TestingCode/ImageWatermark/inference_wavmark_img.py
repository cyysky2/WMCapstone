#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed EMNIST glyphs in VoxPopuli clips with WavMark (32-bps watermark).

≈ 25 clips (1 s @ 16 kHz each)   →   1 glyph (784 bits)

Directory layout, file names and .txt detection files are kept 100 %
compatible with the AudioSeal pipeline so your evaluation scripts
(`evaluate_imperceptibility`, the new image-level `evaluate_robustness`)
work unchanged – just point them to the WavMark result directory.

"""
# ───────────────────────── Imports & config ──────────────────────────────
import os, struct, random, math, traceback, re
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch, torchaudio, soundfile as sf
from PIL import Image
from tqdm import tqdm

import wavmark
from wavmark.utils import file_reader
from attack import all_attacks, attack, restore_audio
import argparse

# --------------------------- USER CONFIG ---------------------------------
EMNIST_IDX_IMG = "~/autodl-tmp/EMNIST_letters/emnist-letters-test-images-idx3-ubyte"

SAMPLE_RATE    = 16_000

CAPACITY_BITS  = 32                 # WavMark payload length
BITS_PER_IMG   = 28 * 28            # 784
CLIPS_PER_IMG  = math.ceil(BITS_PER_IMG / CAPACITY_BITS)   # 25
PAD_VALUE      = 0                  # pad final 16 bits
SEED           = 42
# ------------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = wavmark.load_model().to(device).eval()

# ─────────────────────────── Helper utils ───────────────────────────────
def load_emnist_idx3(path: str) -> np.ndarray:
    path = os.path.expanduser(path)
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "Not an IDX3 image file"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def binarise(img: np.ndarray) -> np.ndarray:      # (28,28) → 0/1
    return (img > 127).astype(np.uint8)

def image_to_bits(img: np.ndarray) -> np.ndarray: # → (784,) 0/1
    return binarise(img).flatten()

def bits_to_image(bits: List[int]) -> np.ndarray: # (784,) → (28,28) uint8
    return (np.array(bits[:BITS_PER_IMG]).reshape(28, 28) * 255).astype(np.uint8)

def save_png(arr: np.ndarray, path: str):
    Image.fromarray(arr, mode="L").save(path)

def is_clip_done(out_root: str, rel_base: str) -> bool:
    sub = os.path.join(out_root, os.path.dirname(rel_base))
    base = os.path.basename(rel_base)
    req  = [f"{base}_original.wav",
            f"{base}_watermarked.wav",
            f"{base}_payload.txt"] \
         + [f"{base}_detect_{atk[0].replace('-', '')}.txt"
            for atk in all_attacks()]
    return all(os.path.exists(os.path.join(sub, r)) for r in req)

# quick helper so we can re-decode if script restarts --------------------
@torch.no_grad()
def decode_bits(audio_np: np.ndarray) -> np.ndarray:
    """Return int(0/1) ndarray length=32 (or None on failure)."""
    ten = torch.tensor(audio_np, dtype=torch.float32, device=device
                       ).unsqueeze(0)
    out = model.decode(ten)
    if out is None:
        return None
    return (out >= 0.5).int().cpu().numpy().squeeze()

# ─────────────────────────── Main routine ───────────────────────────────
@torch.no_grad()
def main(VAL_FILELIST, DATASET_BASE_DIR, OUTPUT_DIR):
    ensure_dir(os.path.expanduser(OUTPUT_DIR))
    emnist_imgs = load_emnist_idx3(EMNIST_IDX_IMG)
    print(f"Loaded {emnist_imgs.shape[0]} EMNIST glyphs.")

    with open(os.path.expanduser(VAL_FILELIST)) as f:
        clips_all = [ln.strip() for ln in f if ln.strip()]

    n_images = len(clips_all) // CLIPS_PER_IMG
    if n_images == 0:
        print("Not enough 1-second clips for even a single glyph.")
        return

    chosen = random.sample(range(emnist_imgs.shape[0]), n_images)

    # --------------------------- big loop ---------------------------------
    for img_no in tqdm(range(n_images), desc="Embedding images"):
        img      = emnist_imgs[chosen[img_no]]
        bits_gt  = image_to_bits(img)
        # pad to exactly 25 × 32
        pad_len  = CLIPS_PER_IMG * CAPACITY_BITS - BITS_PER_IMG
        bits_padded = np.concatenate([bits_gt,
                                      np.full(pad_len, PAD_VALUE, dtype=np.uint8)])

        clip_batch = clips_all[img_no*CLIPS_PER_IMG : (img_no+1)*CLIPS_PER_IMG]

        image_tag   = f"img_{img_no:05d}"
        recon_bits: Dict[str, List[int]] = {"clean": []}
        for atk,_ in all_attacks():
            recon_bits[atk] = []

        
        save_png(bits_to_image(bits_gt),
                     os.path.join(os.path.expanduser(OUTPUT_DIR),
                                  f"{image_tag}_orig.png"))

        # ───────────────── per-clip work ────────────────────────────────
        for seg, wav_path in enumerate(clip_batch):
            seg_bits = bits_padded[seg*CAPACITY_BITS : (seg+1)*CAPACITY_BITS]
            assert len(seg_bits) == CAPACITY_BITS

            wav_path  = os.path.expanduser(wav_path)
            rel_path  = os.path.relpath(wav_path,
                                        os.path.expanduser(DATASET_BASE_DIR))
            base_name = os.path.splitext(rel_path)[0]
            out_sub   = os.path.join(os.path.expanduser(OUTPUT_DIR),
                                     os.path.dirname(base_name))
            ensure_dir(out_sub)
            base_file = os.path.basename(base_name)

            if is_clip_done(os.path.expanduser(OUTPUT_DIR), base_name):
                # already processed – read detection txt to grab bits --------
                for atk,_ in all_attacks():
                    tag = atk[0].replace("-", "")
                    txt = os.path.join(out_sub,
                                       f"{base_file}_detect_{tag}.txt")
                    if not os.path.isfile(txt):
                        continue
                    with open(txt) as fh:
                        line_bits = re.search(r"\[(.*)\]", fh.readline())
                        if line_bits:
                            bits = np.array(eval(line_bits.group(0)),
                                            dtype=np.uint8)
                            if len(bits) == CAPACITY_BITS:
                                recon_bits[atk[0]].extend(bits)
                # clean reconstruction ─ read watermarked wav and decode
                wm_wav = os.path.join(out_sub, f"{base_file}_watermarked.wav")
                if os.path.isfile(wm_wav):
                    sig, _ = torchaudio.load(wm_wav)
                    bits   = decode_bits(sig.squeeze().cpu().numpy())
                    if bits is not None:
                        recon_bits["clean"].extend(bits.astype(int))
                continue   # proceed to next clip

            # ----------------------- load & trim / pad ---------------------
            sig_np = file_reader.read_as_single_channel(wav_path,
                                                        aim_sr=SAMPLE_RATE)
            if len(sig_np) > SAMPLE_RATE:
                sig_np = sig_np[:SAMPLE_RATE]
            elif len(sig_np) < SAMPLE_RATE:
                sig_np = np.pad(sig_np, (0, SAMPLE_RATE - len(sig_np)),
                                mode='constant')

            # ------------------------- embed -------------------------------
            sig_t     = torch.tensor(sig_np, dtype=torch.float32,
                                     device=device).unsqueeze(0)
            payload_t = torch.tensor(seg_bits, dtype=torch.float32,
                                     device=device).unsqueeze(0)
            wm_t      = model.encode(sig_t, payload_t)       # (1, T)
            wm_np     = wm_t.cpu().squeeze().numpy()

            # save originals / watermarked ---------------------------------
            sf.write(os.path.join(out_sub, f"{base_file}_original.wav"),
                     sig_np, SAMPLE_RATE)
            sf.write(os.path.join(out_sub, f"{base_file}_watermarked.wav"),
                     wm_np, SAMPLE_RATE)
            with open(os.path.join(out_sub, f"{base_file}_payload.txt"), "w") as fh:
                fh.write(" ".join(map(str, seg_bits.tolist())) + "\n")

            # decode from clean --------------------------------------------
            dec_clean = decode_bits(wm_np)
            if dec_clean is None:
                dec_clean = np.zeros(CAPACITY_BITS, dtype=np.uint8)
            recon_bits["clean"].extend(dec_clean.tolist())

            # ---------------------- iterate attacks -----------------------
            for atk_name, _ in all_attacks():
                atk_tag = atk_name.replace("-", "")
                atk_t, _ = attack(torch.tensor(wm_np).unsqueeze(0
                             ).unsqueeze(0).to(torch.float32),
                                  SAMPLE_RATE, [(atk_name, 1.0)])
                atk_t   = restore_audio(atk_t, wm_np.shape[-1])
                atk_np  = atk_t.squeeze().cpu().numpy()

                dec = decode_bits(atk_np)
                if dec is None:
                    dec = np.zeros(CAPACITY_BITS, dtype=np.uint8)
                    ber = 100.0
                else:
                    ber = (dec != seg_bits).mean() * 100

                recon_bits[atk_name].extend(dec.tolist())

                # (attacked wav is huge – skip saving to save disk)
                with open(os.path.join(out_sub,
                            f"{base_file}_detect_{atk_tag}.txt"), "w") as fh:
                    fh.write(f"Decoded Payload: {dec.tolist()}\n")
                    fh.write(f"Bit Error Rate: {ber:.1f}%\n")

            # tidy GPU
            del sig_t, wm_t, payload_t
            torch.cuda.empty_cache()

        # ─────────────── finished 25 clips → rebuild images ───────────────
        for atk, bit_list in recon_bits.items():
            if len(bit_list) < BITS_PER_IMG:
                print(f"[WARN] {image_tag}/{atk}: only {len(bit_list)} bits")
                continue
            img_arr = bits_to_image(bit_list)
            
            name = (f"{image_tag}_recon.png"
                    if atk == "clean"
                    else f"{image_tag}_recon_{atk.replace('-', '')}.png")
            save_png(img_arr,
                     os.path.join(os.path.expanduser(OUTPUT_DIR), name))

    print(f"Completed: {n_images} images embedded "
          f"({n_images*CLIPS_PER_IMG} clips).")

# ─────────────────────────────────────────────────────────────────────────
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
