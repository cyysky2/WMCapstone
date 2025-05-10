#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed EMNIST glyphs in VoxPopuli validation clips with the
Patchwork blind watermark (40 bps per 1-s clip).

Output tree and file names follow the _exact_ convention used in the
other pipelines, so the same imperceptibility and PNG-based robustness
evaluators work without modification.

"""
# ───────────────────────── Imports & paths ─────────────────────────────
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
# ----------------------------------------------------------------------
EMNIST_IDX3  = "~/autodl-tmp/EMNIST_letters/emnist-letters-test-images-idx3-ubyte"

SR                 = 16_000
CAPACITY_BITS      = 40                       # payload per clip
BITS_PER_IMG       = 28 * 28                  # 784
CLIPS_PER_IMG      = math.ceil(BITS_PER_IMG / CAPACITY_BITS)   # 20
PAD_VAL            = 0
SEED               = 42

random.seed(SEED)
np.random.seed(SEED)

# ───────────────────── Patchwork embed / detect ───────────────────────
FS_WM, FE_WM   = 3000, 7000         # band (Hz)
K1, K2         = 0.195, 0.08        # spread factors

def _idx_bounds(sr, L):
    return int(FS_WM / (sr/L)), int(FE_WM / (sr/L))

def patchwork_embed(sig: np.ndarray, payload: np.ndarray, sr: int = SR):
    L = len(sig)
    si, ei = _idx_bounds(sr, L)
    X = dct(sig, norm='ortho')
    Xs = X[si:ei+1]

    Ls = len(Xs)
    if Ls % (payload.size * 2) != 0:
        Ls -= Ls % (payload.size * 2)
    Xs = Xs[:Ls]

    Xsp = np.dstack((Xs[:Ls//2], Xs[:(Ls//2-1):-1])).flatten()
    segments = np.array_split(Xsp, payload.size * 2)

    wm_segments = []
    eps = 1e-12
    for i in range(0, len(segments), 2):
        j = i // 2 + 1
        rj = K1 * math.exp(-K2 * j)

        seg1, seg2 = segments[i], segments[i+1]
        m1, m2 = np.mean(np.abs(seg1)) + eps, np.mean(np.abs(seg2)) + eps
        mj, mmj = (m1 + m2) / 2, min(m1, m2)

        if payload[j-1] == 0 and (m1 - m2) < rj * mmj:
            m1p, m2p = mj + rj * mmj / 2, mj - rj * mmj / 2
        elif payload[j-1] == 1 and (m2 - m1) < rj * mmj:
            m1p, m2p = mj - rj * mmj / 2, mj + rj * mmj / 2
        else:
            m1p, m2p = m1, m2

        wm_segments.append(seg1 * m1p / m1)
        wm_segments.append(seg2 * m2p / m2)

    Ysp = np.hstack(wm_segments)
    Ys  = np.hstack([Ysp[::2], Ysp[-1::-2]])
    Y   = X.copy()
    Y[si:si+Ls] = Ys
    return idct(Y, norm='ortho')

def patchwork_detect(sig: np.ndarray, watermark_len: int, sr: int = SR):
    L = len(sig)
    si, ei = _idx_bounds(sr, L)
    X  = dct(sig, norm='ortho')
    Xs = X[si:ei+1]
    Ls = len(Xs)
    if Ls % (watermark_len*2) != 0:
        Ls -= Ls % (watermark_len*2)
    Xs  = Xs[:Ls]

    Xsp = np.dstack((Xs[:Ls//2], Xs[:(Ls//2-1):-1])).flatten()
    segs = np.array_split(Xsp, watermark_len*2)

    bits = []
    for i in range(0, len(segs), 2):
        m1 = np.mean(np.abs(segs[i]))
        m2 = np.mean(np.abs(segs[i+1]))
        bits.append(0 if (m1 - m2) >= 0 else 1)
    return np.array(bits, dtype=int)

# ───────────────────────── Helper utilities ───────────────────────────
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_idx3(path: str) -> np.ndarray:
    with open(os.path.expanduser(path), 'rb') as f:
        magic, n, r, c = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)

def bits_from_img(img: np.ndarray) -> np.ndarray:
    return (img > 127).astype(np.uint8).flatten()

def img_from_bits(bits: List[int]) -> np.ndarray:
    return (np.array(bits[:BITS_PER_IMG]).reshape(28, 28) * 255).astype(np.uint8)

def save_png(arr: np.ndarray, path: str):
    Image.fromarray(arr, mode="L").save(path)

def parse_bits(txt_path: str):
    if not os.path.isfile(txt_path): return None
    with open(txt_path) as fh:
        m = re.search(r"\[(.*)\]", fh.read())
        if m:
            return np.array(eval(m.group(0)), dtype=np.uint8)
    return None

def clip_done(out_root, rel_base):
    req = [f"{rel_base}_original.wav",
           f"{rel_base}_watermarked.wav",
           f"{rel_base}_payload.txt",
           f"{rel_base}_detect_clean.txt"] + \
          [f"{rel_base}_detect_{atk[0].replace('-','')}.txt"
           for atk in all_attacks()]
    return all(os.path.isfile(os.path.join(out_root, p)) for p in req)

# ───────────────────────────── Main loop ──────────────────────────────
@torch.no_grad()
def main(VAL_FILELIST, VOX_BASE_DIR, OUT_DIR):
    ensure_dir(os.path.expanduser(OUT_DIR))
    emnist = load_idx3(EMNIST_IDX3)
    print(f"Loaded {emnist.shape[0]} EMNIST glyphs.")

    with open(os.path.expanduser(VAL_FILELIST)) as f:
        clips_all = [ln.strip() for ln in f if ln.strip()]

    n_imgs = len(clips_all) // CLIPS_PER_IMG
    if n_imgs == 0:
        print("Not enough clips for even one glyph."); return
    chosen = random.sample(range(emnist.shape[0]), n_imgs)

    for img_no in tqdm(range(n_imgs), desc="Patchwork embedding"):
        glyph   = emnist[chosen[img_no]]
        bits_gt = bits_from_img(glyph)
        pad_len = CLIPS_PER_IMG * CAPACITY_BITS - BITS_PER_IMG
        bits_pad = np.concatenate([bits_gt,
                                   np.full(pad_len, PAD_VAL, dtype=np.uint8)])

        tag = f"img_{img_no:05d}"
        
        save_png(img_from_bits(bits_gt),
                     os.path.join(os.path.expanduser(OUT_DIR),
                                  f"{tag}_orig.png"))

        recon: Dict[str, List[int]] = {"clean": []}
        for atk,_ in all_attacks(): recon[atk] = []

        # -------- iterate over 20 clips ----------------------------------
        for seg, wav_path in enumerate(
                clips_all[img_no*CLIPS_PER_IMG:(img_no+1)*CLIPS_PER_IMG]):
            seg_bits = bits_pad[seg*CAPACITY_BITS:(seg+1)*CAPACITY_BITS]
            wav_path = os.path.expanduser(wav_path)
            rel_path = os.path.relpath(wav_path, os.path.expanduser(VOX_BASE_DIR))
            base_no_ext = os.path.splitext(rel_path)[0]
            out_sub = os.path.join(os.path.expanduser(OUT_DIR),
                                   os.path.dirname(base_no_ext))
            ensure_dir(out_sub)
            base = os.path.basename(base_no_ext)

            if clip_done(os.path.expanduser(OUT_DIR),
                         os.path.join(os.path.dirname(base_no_ext), base)):
                # reuse existing decoded bits
                for atk_name in ["clean"] + [a[0] for a in all_attacks()]:
                    tag_atk = atk_name.replace('-', '')
                    bits = parse_bits(os.path.join(
                        out_sub, f"{base}_detect_{tag_atk}.txt"))
                    if bits is not None:
                        recon["clean" if atk_name == "clean" else atk_name].extend(bits.tolist())
                continue

            # ---- load & fix length --------------------------------------
            sig, sr = sf.read(wav_path)
            if sr != SR: raise RuntimeError(f"{wav_path} must be 16 kHz")
            sig = sig[:SR] if len(sig) >= SR else np.pad(sig, (0, SR-len(sig)))

            # ---- embed ---------------------------------------------------
            wm_sig = patchwork_embed(sig, seg_bits)

            sf.write(os.path.join(out_sub, f"{base}_original.wav"), sig, SR)
            sf.write(os.path.join(out_sub, f"{base}_watermarked.wav"), wm_sig, SR)
            with open(os.path.join(out_sub, f"{base}_payload.txt"), "w") as fp:
                fp.write(" ".join(map(str, seg_bits.tolist())) + "\n")

            det_clean = patchwork_detect(wm_sig, CAPACITY_BITS)
            ber_clean = (det_clean != seg_bits).mean() * 100
            recon["clean"].extend(det_clean.tolist())
            with open(os.path.join(out_sub, f"{base}_detect_clean.txt"), "w") as fp:
                fp.write("Decoded Payload: " + str(det_clean.tolist()) + "\n")
                fp.write(f"Bit Error Rate: {ber_clean:.1f}%\n")

            # ---- attacks -------------------------------------------------
            for atk_name, _ in all_attacks():
                atk_tag = atk_name.replace('-', '')
                wm_tensor = torch.tensor(wm_sig, dtype=torch.float32
                             ).unsqueeze(0).unsqueeze(0)
                atk_tensor, _ = attack(wm_tensor, SR, [(atk_name, 1.0)])
                atk_np = restore_audio(atk_tensor, wm_sig.size).squeeze().cpu().numpy()

                det_bits = patchwork_detect(atk_np, CAPACITY_BITS)
                ber = (det_bits != seg_bits).mean() * 100
                recon[atk_name].extend(det_bits.tolist())

                with open(os.path.join(out_sub,
                         f"{base}_detect_{atk_tag}.txt"), "w") as fp:
                    fp.write("Decoded Payload: " + str(det_bits.tolist()) + "\n")
                    fp.write(f"Bit Error Rate: {ber:.1f}%\n")

        # -------- reconstruct 28×28 images ------------------------------
        for atk, bit_list in recon.items():
            if len(bit_list) < BITS_PER_IMG: continue
            img_arr = img_from_bits(bit_list)
            
            name = (f"{tag}_recon.png"
                        if atk == "clean"
                        else f"{tag}_recon_{atk.replace('-', '')}.png")
            save_png(img_arr, os.path.join(os.path.expanduser(OUT_DIR), name))

    print(f"Finished {n_imgs} glyphs → {n_imgs*CLIPS_PER_IMG} clips.")

# ───────────────────────────────────────────────────────────────────────
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
