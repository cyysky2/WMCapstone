#!/usr/bin/env python3
# ================================================================
# prepare_detector_dataset.py
#
# Create partially-watermarked audio + per-sample labels according
# to AudioSeal augmentation (k = 5 by default).
#
#  ┌ wm_dir
#  │   └─ …/1234/5678.wav
#  └ orig_dir
#      └─ …/1234/5678.wav
#            │                 └ labels saved as .npy
#  output_dir
#      └─ …/1234/5678_det.wav  ┘ (same rel-path)
#
# Each .npy contains uint8 {0,1} with   1 → watermark present  
#                                       0 → replaced / erased
# ================================================================
import argparse, json, os, random
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


MAX_WAV_VALUE  = 32768.0
P_ORIG, P_ZEROS, P_OTHER, P_KEEP = 0.4, 0.2, 0.2, 0.2


# -----------------------------------------------------------------
def pick_non_overlapping_starts(total_len, seg_len, k):
    """Return *k* random start indices whose segments don't overlap."""
    available = list(range(0, total_len - seg_len, seg_len))
    random.shuffle(available)
    starts = sorted(available[:k])
    return starts


def grab_random_other_segment(all_files, needed_len, sr):
    """Get a random audio snippet (np.float32) of needed_len samples."""
    while True:
        path = random.choice(all_files)
        audio, sr2 = sf.read(path, always_2d=False, dtype="float32")
        if sr2 != sr:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) >= needed_len:
            start = random.randint(0, len(audio) - needed_len)
            return audio[start : start + needed_len]


# -----------------------------------------------------------------
def process_one(
    wm_path, orig_root, out_root, k, all_orig_files, rng
):
    rel             = wm_path.relative_to(args.wm_dir)
    orig_path       = orig_root / rel
    out_wav_path = out_root / rel.parent / (rel.stem + "_det.wav")
    out_lbl_path    = out_wav_path.with_suffix(".npy")

    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- load ----------
    wm_audio, sr  = sf.read(wm_path, dtype="float32", always_2d=False)
    orig_audio, _ = sf.read(orig_path, dtype="float32", always_2d=False)

    if wm_audio.ndim  > 1: wm_audio  = wm_audio.mean(axis=1)
    if orig_audio.ndim> 1: orig_audio= orig_audio.mean(axis=1)

    wm_audio = wm_audio[:len(orig_audio)]             # trims only if longer

    n_samples = len(wm_audio)

    # ---------- initialise outputs ----------
    out_audio      = wm_audio.copy()
    labels         = np.ones(n_samples, dtype=np.uint8)

    seg_len        = n_samples // (2 * k)
    starts         = pick_non_overlapping_starts(n_samples, seg_len, k)

    for s in starts:
        choice = rng.random()
        seg_slice = slice(s, s + seg_len)

        # 1) revert to original
        if choice < P_ORIG:
            out_audio[seg_slice] = orig_audio[seg_slice]

        # 2) replace with zeros
        elif choice < P_ORIG + P_ZEROS:
            out_audio[seg_slice] = 0.0

        # 3) replace with other audio
        elif choice < P_ORIG + P_ZEROS + P_OTHER:
            other = grab_random_other_segment(all_orig_files, seg_len, sr)
            out_audio[seg_slice] = other

        # 4) keep watermarked → do nothing

        # In all first three options watermark is *absent*
        if choice < P_ORIG + P_ZEROS + P_OTHER:
            labels[seg_slice] = 0

    # ---------- save ----------
    int16 = np.clip(out_audio * MAX_WAV_VALUE, -MAX_WAV_VALUE, MAX_WAV_VALUE - 1).astype(
        np.int16
    )
    sf.write(out_wav_path, int16, sr, subtype="PCM_16")
    np.save(out_lbl_path, labels)


# -----------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Generate detector-training data à la AudioSeal")
    ap.add_argument("--wm_dir",   required=True, help="Root of fully watermarked WAVs")
    ap.add_argument("--orig_dir", required=True, help="Root of original LibriSpeech WAVs")
    ap.add_argument("--output_dir", required=True, help="Where to write partial data")
    ap.add_argument("--k", type=int, default=5, help="number of masked segments per file")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    wm_files    = sorted(Path(args.wm_dir).rglob("*.wav"))
    orig_files  = sorted(Path(args.orig_dir).rglob("*.wav"))  # list once for 'other'

    print(f"Found {len(wm_files)} watermarked wavs – generating detector set …")
    for wm in tqdm(wm_files):
        process_one(
            wm_path=wm,
            orig_root=Path(args.orig_dir),
            out_root=Path(args.output_dir),
            k=args.k,
            all_orig_files=orig_files,
            rng=rng,
        )
    print("✓ detector dataset written to", args.output_dir)
