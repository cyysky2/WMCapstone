#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed EMNIST glyphs into VoxPopuli clips with the traditional
DCT-b1 watermark (Kosta-PMF 2021).

Each 1-second, 16 kHz clip carries 10 frames × 24 bits = 240 bits.
Therefore one 784-bit glyph needs 4 clips (last clip is zero-padded).

File naming and directory layout are identical to the AudioSeal /
WavMark pipelines so your evaluation scripts work unchanged.
"""
from __future__ import annotations
import os, struct, random, math, re, traceback
from pathlib import Path
from typing import List, Dict

import numpy as np
import soundfile as sf
from scipy.fftpack import dct, idct
from PIL import Image
from tqdm import tqdm
import torch
import argparse

from attack import attack, restore_audio, all_attacks

# ─────────────────────────── Global constants ──────────────────────────
SR                 = 16_000                            # sample-rate
LT, LW             = 23, 1486                          # DCT-b1 params
LF                 = LT + LW                           # frame length
BAND_SZ, LG1, LG2  = 30, 24, 6
CAPACITY_BITS      = 10 * LG1                          # 240
BITS_PER_IMG       = 28 * 28                           # 784
CLIPS_PER_IMG      = math.ceil(BITS_PER_IMG / CAPACITY_BITS)  # 4
PAD_VAL            = 0                                 # pad last clip
SEED               = 42                                # reproducibility
# paths ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­
EMNIST_IDX3  = "~/autodl-tmp/EMNIST_letters/emnist-letters-test-images-idx3-ubyte"
random.seed(SEED)
np.random.seed(SEED)

# ───────────────────────── DCT-b1 low-level code ───────────────────────
def _band_repr_freq(band_idx, band_size, n_coeff):
    a = band_idx * band_size * SR / (2 * n_coeff)
    b = (band_idx + 1) * band_size * SR / (2 * n_coeff)
    return (a + b) / 2

def _band_masking_energy(C, band_idx, band_size, n_coeff):
    freq = _band_repr_freq(band_idx, band_size, n_coeff)
    bark = 13 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq / 7500) ** 2)
    a_tmn = -0.275 * bark - 15.025
    return 10 ** (a_tmn / 10) * np.sum(C ** 2)

def _divide_groups(C):
    g1 = np.random.choice(range(C.shape[0]), size=LG1, replace=False)
    g2 = np.array([i for i in range(C.shape[0]) if i not in g1])
    return np.sort(g1), np.sort(g2)

def _energy_compensation(C, G2, niT):
    C_hat = C.copy()
    if niT < 0:
        for ind in G2:
            C_hat[ind] = np.sign(C[ind]) * math.sqrt(C[ind]**2 - niT / LG2)
    elif niT > 0:
        ni = niT
        for k, ind in enumerate(sorted(G2, key=lambda i: abs(C[i]))):
            C_hat[ind] = np.sign(C[ind]) * math.sqrt(
                max(0, C[ind]**2 - ni / (LG2 - (k + 1))))
            ni -= (C[ind]**2 - C_hat[ind]**2)
    return C_hat

def _embed_bits_band(C, bits):
    delta = math.sqrt(_band_masking_energy(C, 0, BAND_SZ, LW))
    G1, G2 = _divide_groups(C)
    C_hat = C.copy()
    for bit, idx in zip(bits, G1):
        if bit == 0:
            C_hat[idx] = math.floor(C[idx] / delta + 0.5) * delta
        else:
            C_hat[idx] = math.floor(C[idx] / delta) * delta + delta / 2
    niT = sum(C_hat[G1]) - sum(C[G1])
    return _energy_compensation(C_hat, G2, niT), G1

def dctb1_embed(sig: np.ndarray, payload_bits: np.ndarray):
    """Return watermarked signal and frame-key list (len = n_frames)."""
    n_frames = len(sig) // LF
    if len(payload_bits) != n_frames * LG1:
        raise ValueError("payload length mismatch")
    wm = sig.copy()
    key = []
    for f in range(n_frames):
        frame = sig[f*LF : f*LF+LF]
        C = dct(frame[LT:], norm='ortho')
        C_hat, G1 = _embed_bits_band(C[:BAND_SZ],
                                     payload_bits[f*LG1:(f+1)*LG1])
        C_full = C.copy()
        C_full[:BAND_SZ] = C_hat
        wm_frame = np.zeros_like(frame)
        wm_frame[:LT] = frame[:LT]
        wm_frame[LT:] = idct(C_full, norm='ortho')
        wm[f*LF : f*LF+LF] = wm_frame
        key.append(G1)
    return wm, key

def dctb1_detect(sig_wm: np.ndarray, key):
    n_frames = len(sig_wm) // LF
    bits = []
    for f in range(n_frames):
        frame = sig_wm[f*LF : f*LF+LF]
        C = dct(frame[LT:], norm='ortho')
        delta = math.sqrt(_band_masking_energy(C[:BAND_SZ], 0, BAND_SZ, LW))
        for idx in key[f]:
            bits.append(1 if abs(C[idx] / delta - np.floor(C[idx] / delta) - 0.5) < .25
                        else 0)
    return np.array(bits, dtype=int)

# ───────────────────────── Helper utilities ────────────────────────────
def ensure_dir(p: str):  os.makedirs(p, exist_ok=True)

def load_idx3(path: str) -> np.ndarray:
    path = os.path.expanduser(path)
    with open(path, "rb") as f:
        magic, num, r, c = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, r, c)

def bits_from_img(img: np.ndarray) -> np.ndarray:  # (28,28) -> (784,) 0/1
    return (img > 127).astype(np.uint8).flatten()

def img_from_bits(bits: List[int]) -> np.ndarray:  # -> (28,28) uint8
    return (np.array(bits[:BITS_PER_IMG]).reshape(28, 28) * 255).astype(np.uint8)

def save_png(arr: np.ndarray, path: str):
    Image.fromarray(arr, mode="L").save(path)

def parse_bits(txt_path: str) -> np.ndarray | None:
    if not os.path.isfile(txt_path): return None
    with open(txt_path) as fh:
        m = re.search(r"\[(.*)\]", fh.read())
        if m:
            return np.array(eval(m.group(0)), dtype=np.uint8)
    return None

def clip_done(out_root: str, rel_base: str) -> bool:
    req = [f"{rel_base}_original.wav",
           f"{rel_base}_watermarked.wav",
           f"{rel_base}_payload.txt",
           f"{rel_base}_detect_clean.txt"] + \
          [f"{rel_base}_detect_{atk[0].replace('-','')}.txt"
           for atk in all_attacks()]
    return all(os.path.isfile(os.path.join(out_root, p)) for p in req)

# ─────────────────────────── Main routine ──────────────────────────────
@torch.no_grad()
def main(VAL_FILELIST, VOX_BASE_DIR, OUT_DIR):
    ensure_dir(os.path.expanduser(OUT_DIR))
    emnist = load_idx3(EMNIST_IDX3)
    print(f"Loaded {emnist.shape[0]} EMNIST glyphs.")

    with open(os.path.expanduser(VAL_FILELIST)) as f:
        clips = [ln.strip() for ln in f if ln.strip()]

    n_imgs = len(clips) // CLIPS_PER_IMG
    if n_imgs == 0:
        print("Not enough clips for one glyph.")
        return
    chosen = random.sample(range(emnist.shape[0]), n_imgs)

    for img_no in tqdm(range(n_imgs), desc="DCT-b1 embedding"):
        glyph     = emnist[chosen[img_no]]
        bits_gt   = bits_from_img(glyph)
        pad_len   = CLIPS_PER_IMG * CAPACITY_BITS - BITS_PER_IMG
        bits_pad  = np.concatenate([bits_gt,
                                    np.full(pad_len, PAD_VAL, dtype=np.uint8)])
        recon: Dict[str, List[int]] = {"clean": []}
        for atk,_ in all_attacks():
            recon[atk] = []

        tag = f"img_{img_no:05d}"
        
        save_png(img_from_bits(bits_gt),
                     os.path.join(os.path.expanduser(OUT_DIR),
                                  f"{tag}_orig.png"))

        # iterate over 4 clips ------------------------------------------------
        for seg, wav_path in enumerate(
                clips[img_no*CLIPS_PER_IMG : (img_no+1)*CLIPS_PER_IMG]):
            seg_bits = bits_pad[seg*CAPACITY_BITS:(seg+1)*CAPACITY_BITS]
            wav_path = os.path.expanduser(wav_path)
            rel_path = os.path.relpath(wav_path,
                                       os.path.expanduser(VOX_BASE_DIR))
            base_no_ext = os.path.splitext(rel_path)[0]
            out_sub = os.path.join(os.path.expanduser(OUT_DIR),
                                   os.path.dirname(base_no_ext))
            ensure_dir(out_sub)
            base = os.path.basename(base_no_ext)

            # skip if already done -----------------------------------------
            if clip_done(os.path.expanduser(OUT_DIR),
                         os.path.join(os.path.dirname(base_no_ext), base)):
                # reuse existing decoded payloads
                for atk_name, _ in [("clean", None)] + all_attacks():
                    tag_atk = atk_name.replace('-', '')
                    bits = parse_bits(os.path.join(
                        out_sub,
                        f"{base}_detect_{tag_atk}.txt"))
                    if bits is not None:
                        recon[atk_name if atk_name!="clean" else "clean"
                             ].extend(bits.tolist())
                continue

            # load & force to exactly 1 s -----------------------------------
            sig, sr = sf.read(wav_path)
            if sr != SR:
                sig = sf.resample(sig, sr, SR)  # rarely needed
            sig = sig[:SR] if len(sig) >= SR else \
                  np.pad(sig, (0, SR-len(sig)))

            # embed ---------------------------------------------------------
            wm_sig, key = dctb1_embed(sig, seg_bits)

            sf.write(os.path.join(out_sub, f"{base}_original.wav"), sig, SR)
            sf.write(os.path.join(out_sub, f"{base}_watermarked.wav"), wm_sig, SR)
            with open(os.path.join(out_sub, f"{base}_payload.txt"), "w") as f:
                f.write(" ".join(map(str, seg_bits.tolist())) + "\n")

            # decode clean immediately & save ------------------------------
            det_clean = dctb1_detect(wm_sig, key)
            ber_clean = (det_clean[:len(seg_bits)] != seg_bits).mean() * 100
            recon["clean"].extend(det_clean.tolist())
            with open(os.path.join(out_sub, f"{base}_detect_clean.txt"), "w") as f:
                f.write("Decoded Payload: " + str(det_clean.tolist()) + "\n")
                f.write(f"Bit Error Rate: {ber_clean:.1f}%\n")

            # attacks -------------------------------------------------------
            for atk_name, _ in all_attacks():
                atk_tag = atk_name.replace('-', '')
                atk_t, _ = attack(torch.tensor(wm_sig).unsqueeze(0
                             ).unsqueeze(0).float(),
                                  SR, [(atk_name, 1.0)])
                atk_np = restore_audio(atk_t, wm_sig.shape[-1]
                                       ).squeeze().numpy()

                det_bits = dctb1_detect(atk_np, key)
                ber = (det_bits[:len(seg_bits)] != seg_bits).mean() * 100
                recon[atk_name].extend(det_bits.tolist())

                with open(os.path.join(out_sub,
                             f"{base}_detect_{atk_tag}.txt"), "w") as fh:
                    fh.write("Decoded Payload: " + str(det_bits.tolist()) + "\n")
                    fh.write(f"Bit Error Rate: {ber:.1f}%\n")

            del sig, wm_sig, key  # free mem

        # finished 4 clips → rebuild images ---------------------------------
        for atk, bit_list in recon.items():
            if len(bit_list) < BITS_PER_IMG:
                continue
            img_arr = img_from_bits(bit_list)
            name = (f"{tag}_recon.png" if atk == "clean"
                    else f"{tag}_recon_{atk.replace('-', '')}.png")
            
            save_png(img_arr, os.path.join(os.path.expanduser(OUT_DIR), name))

    print(f"Done: {n_imgs} glyphs → {n_imgs*CLIPS_PER_IMG} clips.")

# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filelist_path") 
    parser.add_argument("-d", "--data_path") 
    parser.add_argument("-o", "--output_dir")
    args = parser.parse_args()

    VAL_FILELIST = os.path.expanduser(args.filelist_path)
    VOX_BASE_DIR = os.path.expanduser(args.data_path)
    OUTPUT_DIR = os.path.expanduser(args.output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    
    main(VAL_FILELIST, VOX_BASE_DIR, OUTPUT_DIR)
