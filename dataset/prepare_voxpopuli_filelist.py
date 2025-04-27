#!/usr/bin/env python3
"""
prepare_voxpopuli.py  –  Resumable pre-processor *with* progress bar

  • 95 % / 5 % split (persisted to voxpopuli_split_manifest.tsv)
  • Train  : convert OGG → 24 k Hz WAV, delete OGG
  • Val    : convert OGG → 24 k Hz WAV + 16 k Hz copy, delete OGG
  • Safe to re-run; it skips work already finished.
  • Shows a live progress bar (tqdm).
"""

import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List

# ---------- NEW: progress bar --------------------------------------------- #
try:
    from tqdm.auto import tqdm           # pretty in notebooks & terminals
except ModuleNotFoundError:              # graceful fallback
    def tqdm(x, **kwargs):               # type: ignore
        return x                         # no-op iterator
# --------------------------------------------------------------------------- #

RAW_ROOT      = Path("/root/autodl-tmp/voxpopuli_dataset/raw_audios/en")
PROC_ROOT     = Path("/root/autodl-tmp/voxpopuli_dataset")

MANIFEST      = PROC_ROOT / "voxpopuli_split_manifest.tsv"

TRAIN_DIR_24  = PROC_ROOT / "train_voxpopuli_24k_wav"
VAL_DIR_24    = PROC_ROOT / "val_voxpopuli_24k_wav"
VAL_DIR_16    = PROC_ROOT / "val_voxpopuli_16k_wav"

TRAIN_LIST    = PROC_ROOT / "voxpopuli_train_filelist.txt"
VAL_LIST_24   = PROC_ROOT / "voxpopuli_val_filelist_24k.txt"
VAL_LIST_16   = PROC_ROOT / "voxpopuli_val_filelist_16k.txt"

SEED          = 42
TRAIN_RATIO   = 0.9                     

# ────────────────────────────────────────────────────────────────────────────
def discover_ogg_files(root: Path) -> List[Path]:
    return [Path(dp) / f
            for dp, _, fs in os.walk(root)
            for f in fs if f.lower().endswith(".ogg")]

def save_manifest(mapping: Dict[str, str], file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", encoding="utf-8") as f:
        for ogg_path, split in mapping.items():
            f.write(f"{ogg_path}\t{split}\n")

def load_manifest(file: Path) -> Dict[str, str]:
    mapping = {}
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            ogg_path, split = line.rstrip("\n").split("\t")
            mapping[ogg_path] = split
    return mapping

def ffmpeg_convert(inp: Path, out: Path, sr: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(inp),
        "-ar", str(sr),
        str(out)
    ]
    subprocess.run(cmd, check=True)

def build_or_load_manifest() -> Dict[str, str]:
    if MANIFEST.exists():
        print(f"↻  Manifest found – resuming from {MANIFEST}")
        return load_manifest(MANIFEST)

    print("✱ First run – building split manifest …")
    ogg_files = discover_ogg_files(RAW_ROOT)
    if not ogg_files:
        raise RuntimeError(f"No .ogg files found under {RAW_ROOT}")

    random.seed(SEED)
    random.shuffle(ogg_files)

    split_idx = int(len(ogg_files) * TRAIN_RATIO)
    manifest  = {str(p): ("train" if i < split_idx else "val")
                 for i, p in enumerate(ogg_files)}

    save_manifest(manifest, MANIFEST)
    print(f"  Saved manifest with {len(manifest)} entries.")
    return manifest

# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    manifest = build_or_load_manifest()

    train_paths_24 : List[str] = []
    val_paths_24   : List[str] = []
    val_paths_16   : List[str] = []

    items = list(manifest.items())
    pbar  = tqdm(items, desc="Processing clips", unit="file", ncols=100)

    for ogg_str, split in pbar:
        ogg = Path(ogg_str)
        rel_wav = ogg.relative_to(RAW_ROOT).with_suffix(".wav")

        # update bar label occasionally
        pbar.set_postfix(split=split, refresh=False)

        # -------------- TRAIN ------------------------------------------------
        if split == "train":
            wav24 = TRAIN_DIR_24 / rel_wav

            if not wav24.exists():
                if ogg.exists():
                    ffmpeg_convert(ogg, wav24, 24_000)
                    ogg.unlink(missing_ok=True)
                else:
                    pbar.write(f"⚠️  Missing train file: {ogg}")

            train_paths_24.append(str(wav24))

        # -------------- VALIDATION ------------------------------------------
        else:
            wav24 = VAL_DIR_24 / rel_wav
            wav16 = VAL_DIR_16 / rel_wav

            if not wav24.exists():
                if ogg.exists():
                    ffmpeg_convert(ogg, wav24, 24_000)
                    ogg.unlink(missing_ok=True)
                else:
                    pbar.write(f"⚠️  Missing val  file: {ogg}")
                    continue

            if not wav16.exists():
                ffmpeg_convert(wav24, wav16, 16_000)

            val_paths_24.append(str(wav24))
            val_paths_16.append(str(wav16))

    # ────────────────────────────────────────────────────────────────────────
    def dump(paths: List[str], file: Path) -> None:
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w", encoding="utf-8") as f:
            for p in sorted(set(paths)):
                f.write(f"{p}\n")

    dump(train_paths_24, TRAIN_LIST)
    dump(val_paths_24,  VAL_LIST_24)
    dump(val_paths_16,  VAL_LIST_16)

    pbar.close()
    print("\n✓ Finished – all file-lists refreshed.")
    print(f"  Train wavs (24 kHz): {len(train_paths_24)}")
    print(f"  Val   wavs (24 kHz): {len(val_paths_24)}")
    print(f"  Val   wavs (16 kHz): {len(val_paths_16)}")

if __name__ == "__main__":
    main()
