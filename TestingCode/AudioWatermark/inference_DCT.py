"""
Run the traditional DCT-b1 audio-watermark (Kosta-PMF, 2021) on
LibriSpeech dev-clean (16 kHz WAV) and save:

  ▸ *_original.wav          – 16 kHz, 1-s (16 000 sample) excerpt
  ▸ *_watermarked.wav       – signal after embedding
  ▸ *_payload.txt           – original payload bits
  ▸ *_attacked_<tag>.wav    – (optional – keep commented if disk is tight)
  ▸ *_detect_<tag>.txt      – decoded bits + BER for each attack

The script is **resume-safe**: if every expected file is already present
for a sample it is skipped.
"""
from __future__ import annotations
import os, traceback, random, math
import numpy as np
from scipy.fftpack import dct, idct
import soundfile as sf
import torch
from tqdm import tqdm
import argparse

# ---------------------------------------------------------------------
# 1.  ---  low-level DCT-b1 functions  --------------------------------
#         (adapted / patched from
#          https://github.com/kosta-pmf/audio-watermarking)  :contentReference[oaicite:0]{index=0}
# ---------------------------------------------------------------------
SR = 16_000
LT, LW = 23, 1486           # transition / embedding samples
LF  = LT + LW               # frame length
BAND_SZ, LG1, LG2 = 30, 24, 6      # DCT band + group sizes

def _band_repr_freq(band_idx, band_size, n_coeff):
    start = band_idx * band_size * SR / (2 * n_coeff)
    end   = (band_idx + 1) * band_size * SR / (2 * n_coeff)
    return (start + end) / 2

def _band_masking_energy(C, band_idx, band_size, n_coeff):
    freq = _band_repr_freq(band_idx, band_size, n_coeff)
    bark = 13 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq / 7500) ** 2)
    a_tmn = -0.275 * bark - 15.025
    return 10 ** (a_tmn / 10) * np.sum(C ** 2)

def _divide_groups(C):
    choice = np.random.choice(range(C.shape[0]), size=(LG1,), replace=False)
    rest   = np.array([i for i in range(C.shape[0]) if i not in choice])
    return np.sort(choice), np.sort(rest)

def _energy_compensation(C, G2_idx, niT):
    C_hat = C.copy()
    if niT < 0:
        for ind in G2_idx:
            C_hat[ind] = np.sign(C[ind]) * math.sqrt(C[ind] ** 2 - niT / LG2)
    elif niT > 0:
        ni = niT
        for k, ind in enumerate(sorted(G2_idx, key=lambda i: abs(C[i]))):
            C_hat[ind] = np.sign(C[ind]) * math.sqrt(max(0, C[ind] ** 2 - ni / (LG2 - (k + 1))))
            ni -= (C[ind] ** 2 - C_hat[ind] ** 2)
    return C_hat

def _embed_bits_in_band(C, bits):
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
    """Return watermarked signal and G1-index key list (per frame)."""
    n_frames = len(sig) // LF
    exp_len  = n_frames * LG1
    if len(payload_bits) != exp_len:
        raise ValueError(f"Payload must be {exp_len} bits for {n_frames} frames")

    wm = sig.copy()
    key = []
    for f in range(n_frames):
        frame = sig[f*LF : f*LF + LF]
        C = dct(frame[LT:], norm='ortho')
        C_hat, G1 = _embed_bits_in_band(C[:BAND_SZ], payload_bits[f*LG1:(f+1)*LG1])
        C_full = C.copy()
        C_full[:BAND_SZ] = C_hat
        wm_frame = np.zeros_like(frame)
        wm_frame[:LT]  = frame[:LT]
        wm_frame[LT:]  = idct(C_full, norm='ortho')
        wm[f*LF : f*LF + LF] = wm_frame
        key.append(G1)
    return wm, key

def dctb1_detect(sig_wm: np.ndarray, key: list[np.ndarray]):
    """Return detected bit array (len = len(key)*LG1)."""
    n_frames = len(sig_wm) // LF
    bits = []
    for f in range(n_frames):
        frame = sig_wm[f*LF : f*LF + LF]
        C = dct(frame[LT:], norm='ortho')
        delta = math.sqrt(_band_masking_energy(C[:BAND_SZ], 0, BAND_SZ, LW))
        for idx in key[f]:
            bits.append(1 if abs(C[idx] / delta - np.floor(C[idx] / delta) - 0.5) < 0.25
                        else 0)
    return np.array(bits, dtype=int)

# ---------------------------------------------------------------------
# 2.  ---  utility functions copied from your WavMark pipeline ---------
# ---------------------------------------------------------------------
from attack import attack, all_attacks, restore_audio   

def _done(out_dir, base):
    need = [f"{base}_original.wav", f"{base}_watermarked.wav", f"{base}_payload.txt"] + \
           [f"{base}_detect_{atk[0].replace('-','')}.txt" for atk in all_attacks()]
    return all(os.path.exists(os.path.join(out_dir, os.path.dirname(base), f))
               for f in need)

# ---------------------------------------------------------------------
# 3.  ---  main loop ---------------------------------------------------
# ---------------------------------------------------------------------
def main(val_list, root_dir, out_dir):   

    with open(val_list) as fp:
        files = [ln.strip() for ln in fp if ln.strip()]

    for wav_path in tqdm(files, desc="DCT-b1"):
        try:
            rel   = os.path.relpath(wav_path, root_dir)
            base  = os.path.splitext(rel)[0]
            sub   = os.path.join(out_dir, os.path.dirname(base))
            os.makedirs(sub, exist_ok=True)
            if _done(out_dir, base):
                continue

            # 3-a)  load & make exactly 1-s snippet (16 000 samples)
            sig, sr = sf.read(wav_path)
            if sr != SR:
                raise RuntimeError(f"{wav_path} is not 16 kHz")
            sig = sig[:SR] if len(sig) >= SR else np.pad(sig, (0, SR-len(sig)))

            # 3-b)  embed
            n_frames = (len(sig)//LF)
            payload  = np.random.randint(0, 2, size=n_frames*LG1)
            wm_sig, key = dctb1_embed(sig, payload)

            # 3-c)  save originals
            sf.write(os.path.join(sub, f"{os.path.basename(base)}_original.wav"), sig, SR)
            sf.write(os.path.join(sub, f"{os.path.basename(base)}_watermarked.wav"), wm_sig, SR)
            with open(os.path.join(sub, f"{os.path.basename(base)}_payload.txt"), "w") as fp:
                fp.write("Original Payload:\n")
                fp.write(" ".join(map(str, payload.tolist())))

            # 3-d)  every attack
            for atk_name, _ in all_attacks():
                tag = atk_name.replace('-','')
                atk_tensor, _ = attack(torch.tensor(wm_sig).unsqueeze(0).unsqueeze(0).float(),
                                       SR, [(atk_name, 1.0)])
                atk_tensor = restore_audio(atk_tensor, wm_sig.shape[-1]).squeeze().numpy()
                det_bits   = dctb1_detect(atk_tensor, key)
                ber        = (det_bits != payload[:len(det_bits)]).mean() * 100
                with open(os.path.join(sub, f"{os.path.basename(base)}_detect_{tag}.txt"),"w") as fp:
                    fp.write("Decoded Payload:\n")
                    fp.write(" ".join(map(str, det_bits.tolist())) + "\n")
                    fp.write(f"Bit Error Rate: {ber:.1f}%\n")
                # Optional: save attacked audio
                # sf.write(os.path.join(sub,f"{os.path.basename(base)}_attacked_{tag}.wav"),
                #          atk_tensor, SR)

        except Exception as e:
            print("ERROR:", wav_path, e)
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filelist_path") 
    parser.add_argument("-d", "--data_path") 
    parser.add_argument("-o", "--output_dir")
    args = parser.parse_args()
    # Paths
    val_list = os.path.expanduser(args.filelist_path)
    root_dir = os.path.expanduser(args.data_path)
    out_dir = os.path.expanduser(args.output_dir)
    
    os.makedirs(out_dir, exist_ok=True)
    main(val_list, root_dir, out_dir)
