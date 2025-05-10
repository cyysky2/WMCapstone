#!/usr/bin/env python3
"""
Run the Patchwork-multilayer audio watermark on LibriSpeech dev-clean
(1-second, 16-kHz snippets) and dump all artefacts needed by the
summary pipeline.

Outputs go to   ~/autodl-tmp/Patchwork/OutputResult/
"""

# ---------------------------------------------------------------------
# 0.  Imports
# ---------------------------------------------------------------------
import os, traceback, math, random
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from scipy.fftpack import dct, idct
import argparse

# ---- attacks utils --------------------------------------------------
from attack import attack, restore_audio, all_attacks         # same helpers as before

# ---------------------------------------------------------------------
# 1.  Patchwork implementation   (adapted & bug-fixed)
#     Source: kosta-pmf/audio-watermarking,
#     functions lines 12-17 are reproduced verbatim :contentReference[oaicite:0]{index=0}
# ---------------------------------------------------------------------
FS_WM   = 3000          # embedding band start (Hz)
FE_WM   = 7000          # embedding band end   (Hz)
K1, K2  = 0.195, 0.08   # energy spread factors

def _idx_bounds(sr, L):
    """Return start & end indices in the DCT spectrum for fs..fe (inclusive)."""
    si = int(FS_WM/(sr/L))
    ei = int(FE_WM/(sr/L))
    return si, ei

def patchwork_embed(sig: np.ndarray, payload: np.ndarray, sr:int=16_000):
    """Return watermarked signal (same length as input)."""
    L   = len(sig)
    si, ei = _idx_bounds(sr, L)
    X   = dct(sig, type=2, norm='ortho')
    Xs  = X[si:ei+1]

    Ls = len(Xs)
    if Ls % (len(payload)*2) != 0:
        Ls -= Ls % (len(payload)*2)
    Xs  = Xs[:Ls]

    # mirror pairing (paper’s Fig. 2)
    Xsp = np.dstack((Xs[:Ls//2], Xs[:(Ls//2-1):-1])).flatten()
    segments = np.array_split(Xsp, len(payload)*2)

    watermarked = []
    for i in range(0, len(segments), 2):
        j   = i//2 + 1                           # 1-based layer index
        rj  = K1 * math.exp(-K2*j)

        seg1, seg2 = segments[i], segments[i+1]
        m1, m2     = np.mean(np.abs(seg1)), np.mean(np.abs(seg2))
        mj, mmj    = (m1+m2)/2, min(m1, m2)

        # adjust magnitudes to create sign difference
        if payload[j-1] == 0 and (m1 - m2) < rj*mmj:
            m1p, m2p = mj + rj*mmj/2, mj - rj*mmj/2
        elif payload[j-1] == 1 and (m2 - m1) < rj*mmj:
            m1p, m2p = mj - rj*mmj/2, mj + rj*mmj/2
        else:
            m1p, m2p = m1, m2

        watermarked.append(seg1 * m1p/m1)
        watermarked.append(seg2 * m2p/m2)

    Ysp = np.hstack(watermarked)
    Ys  = np.hstack([Ysp[::2], Ysp[-1::-2]])
    Y   = X.copy()
    Y[si:si+Ls] = Ys

    return idct(Y, type=2, norm='ortho')

def patchwork_detect(wm_sig: np.ndarray, watermark_len:int,
                     sr:int=16_000) -> np.ndarray:
    """Return detected bit vector (length == watermark_len)."""
    L   = len(wm_sig)
    si, ei = _idx_bounds(sr, L)
    X   = dct(wm_sig, type=2, norm='ortho')
    Xs  = X[si:ei+1]

    Ls = len(Xs)
    if Ls % (watermark_len*2) != 0:
        Ls -= Ls % (watermark_len*2)
    Xs  = Xs[:Ls]

    Xsp = np.dstack((Xs[:Ls//2], Xs[:(Ls//2-1):-1])).flatten()
    segments = np.array_split(Xsp, watermark_len*2)

    bits = []
    for i in range(0, len(segments), 2):
        m1 = np.mean(np.abs(segments[i]))
        m2 = np.mean(np.abs(segments[i+1]))
        bits.append(0 if (m1 - m2) >= 0 else 1)
    return np.array(bits, dtype=int)

# ---------------------------------------------------------------------
# 2.  Boiler-plate: dataset & resume logic
# ---------------------------------------------------------------------
SR        = 16_000
FRAME_SEC = 1.0                       # cut / pad to exactly 1 s (16 000 sampl.)
PAYLOAD_L = 40                        # bits per clip (first layer uses all)

def _done(out_root, base):
    files = [f"{base}_original.wav", f"{base}_watermarked.wav",
             f"{base}_payload.txt"] + \
            [f"{base}_detect_{a[0].replace('-','')}.txt" for a in all_attacks()]
    return all(os.path.exists(os.path.join(out_root, os.path.dirname(base), f))
               for f in files)

# ---------------------------------------------------------------------
# 3.  Main loop
# ---------------------------------------------------------------------
def main(val_list, wav_root, out_root):   

    with open(val_list) as fp:
        wav_files = [ln.strip() for ln in fp if ln.strip()]

    for wav in tqdm(wav_files, desc="Patchwork"):
        try:
            rel   = os.path.relpath(wav, wav_root)
            base  = os.path.splitext(rel)[0]
            sub   = os.path.join(out_root, os.path.dirname(base))
            os.makedirs(sub, exist_ok=True)
            if _done(out_root, base):
                continue

            sig, sr = sf.read(wav)
            if sr != SR:
                raise RuntimeError(f"{wav} is not 16 kHz")

            sig = sig[:SR] if len(sig) >= SR else np.pad(sig, (0, SR-len(sig)))
            payload = np.random.randint(0, 2, size=PAYLOAD_L)

            wm_sig = patchwork_embed(sig, payload, SR)

            # --- save originals & payload
            sf.write(os.path.join(sub, f"{os.path.basename(base)}_original.wav"), sig, SR)
            sf.write(os.path.join(sub, f"{os.path.basename(base)}_watermarked.wav"), wm_sig, SR)
            with open(os.path.join(sub, f"{os.path.basename(base)}_payload.txt"), "w") as fp:
                fp.write("Original Payload:\n")
                fp.write(" ".join(map(str, payload.tolist())))

            # --- per-attack loop
            for atk_name, _ in all_attacks():
                tag = atk_name.replace('-', '')

                # ATTACK INPUT MUST BE A TENSOR
                wm_tensor = torch.tensor(wm_sig, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                atk_tensor, _ = attack(wm_tensor, SR, [(atk_name, 1.0)])

                # restore & convert back to numpy
                atk_tensor = restore_audio(atk_tensor, wm_sig.shape[-1])
                atk_np     = atk_tensor.squeeze().cpu().numpy()

                det_bits = patchwork_detect(atk_np, PAYLOAD_L, SR)
                ber      = (det_bits != payload).mean() * 100.0

                with open(os.path.join(sub, f"{os.path.basename(base)}_detect_{tag}.txt"), "w") as fp:
                    fp.write("Decoded Payload:\n")
                    fp.write(" ".join(map(str, det_bits.tolist())) + "\n")
                    fp.write(f"Bit Error Rate: {ber:.1f}%\n")

                # optionally save attacked audio
                # sf.write(os.path.join(sub,f"{os.path.basename(base)}_attacked_{tag}.wav"),
                #          atk_np, SR)

        except Exception as e:
            print("ERROR:", wav, e)
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filelist_path") 
    parser.add_argument("-d", "--data_path") 
    parser.add_argument("-o", "--output_dir")
    args = parser.parse_args()
    # Paths
    val_list = os.path.expanduser(args.filelist_path)
    wav_root = os.path.expanduser(args.data_path)
    out_root = os.path.expanduser(args.output_dir)
    
    os.makedirs(out_root, exist_ok=True)
    main(val_list, wav_root, out_root)
