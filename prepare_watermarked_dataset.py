#!/usr/bin/env python3
# ================================================================
# prepare_watemarked_dataset.py
#
# Embed a *random* watermark in every 0.5-s segment of every WAV
# file in a LibriSpeech directory tree (24 kHz).  Outputs the fully
# watermarked waveform with the same relative path under
# --output_dir and writes a JSON side-car with the random keys
# actually embedded (one per 0.5 s chunk).
#
# Requirements: the same model classes/utilities used in
# inference_sample.py, PyTorch, soundfile, tqdm.
'''
python prepare_watermarked_dataset.py \
--input_dir  /root/autodl-tmp/LibriSpeech/LibriSpeech_24k_wav/train-clean-360 \
  --output_dir /root/autodl-tmp/LibriSpeech/LibriSpeech_24k_wav_wm/train-clean-360 \
  --checkpoint /root/autodl-tmp/WMCapstone/ckpt \
  --device cuda:0          # or "cpu"
'''
# ================================================================

import argparse, json, os, sys
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import soundfile as sf

from models import Generator, Encoder, Quantizer
from utils import load_checkpoint, scan_checkpoint, AttrDict
from watermark import (
    WatermarkEncoder,
    random_watermark,
)

# ---------- helper ------------------------------------------------------------
MAX_WAV_VALUE = 32768.0           # same constant used in dataset.py
SEG_DUR_SEC   = 0.5               # model expects 0.5 s @ 24 kHz
SEG_SAMPLES   = int(24000 * SEG_DUR_SEC)


@torch.no_grad()
def load_modules(cfg, ckpt_path, device):
    """Instantiate and load all network modules (eval mode, no W-norm)."""
    # Model skeletons
    generator         = Generator(cfg).to(device)
    encoder           = Encoder(cfg).to(device)
    quantizer         = Quantizer(cfg).to(device)
    wm_encoder        = WatermarkEncoder(cfg).to(device)

    # Checkpoint
    ckpt_codec_path  = scan_checkpoint(ckpt_path, "generator_")
    state_dict       = load_checkpoint(ckpt_codec_path, device)
    generator.load_state_dict        (state_dict["generator"])
    encoder.load_state_dict          (state_dict["encoder"])
    quantizer.load_state_dict        (state_dict["quantizer"])
    wm_encoder.load_state_dict       (state_dict["watermark_encoder"])

    # inference / weight-norm clean-up
    generator.eval().remove_weight_norm()
    encoder  .eval().remove_weight_norm()
    quantizer.eval()
    wm_encoder.eval()

    return generator, encoder, quantizer, wm_encoder


@torch.no_grad()
def embed_full_watermark(
    wav_tensor, *, sr, generator, encoder, quantizer, wm_encoder,
    cfg, device, batch_size
):
    """Vectorised over (B, 1, 1, 12 000)."""
    if sr != 24000:
        raise ValueError(f"Input must be 24 kHz; got {sr}")

    samples = wav_tensor.squeeze(0).to(device)          # (T,)
    num_chunks = (samples.numel() + SEG_SAMPLES - 1) // SEG_SAMPLES
    pad_len    = num_chunks * SEG_SAMPLES - samples.numel()
    if pad_len:
        samples = F.pad(samples, (0, pad_len))

    chunks     = samples.view(num_chunks, SEG_SAMPLES)   # (N, L)

    wm_chunks, wm_keys = [], []

    for i in range(0, num_chunks, batch_size):
        batch = chunks[i : i + batch_size]              # (B, L)
        batch = (
            batch.unsqueeze(1)                          # (B, 1, L)                    
                  .to(device)
        )

        # random bits             → latent watermark feature
        watermark = random_watermark(batch_size=len(batch), h=cfg).to(device)
        wm_feat   = wm_encoder(watermark)               # (B, F, T′)

        # host signal + watermark → encoded → quantised → regenerated audio
        enc_feat  = encoder(batch, wm_feat)             # (B, T′, F)
        q_feat, *_ = quantizer(enc_feat)
        batch_wm  = generator(q_feat).squeeze(1).squeeze(1)  # (B, L)

        wm_chunks.append(batch_wm.cpu())
        wm_keys  .extend(watermark.cpu())

    wm_audio = torch.cat(wm_chunks).flatten()[: samples.numel()]  # (T,)
    return wm_audio.unsqueeze(0), wm_keys

# ============================================================================
# Slightly modified process_file to pass chunk_batch
# ============================================================================
def process_file(
    in_path, out_root, modules, cfg, device,
    chunk_batch=64, skip_existing=True
):
    rel   = in_path.relative_to(args.input_dir)
    out_w = out_root / rel
    out_j = out_w.with_suffix(".json")
    if skip_existing and out_w.exists():
        return

    out_w.parent.mkdir(parents=True, exist_ok=True)
    wav, sr = sf.read(in_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav).unsqueeze(0)

    g, e, q, wm_enc = modules
    wm_wav, wm_bits = embed_full_watermark(
        wav, sr=sr, generator=g, encoder=e, quantizer=q,
        wm_encoder=wm_enc, cfg=cfg, device=device, batch_size=chunk_batch
    )

    int16 = (wm_wav.squeeze() * MAX_WAV_VALUE).clamp(
        -MAX_WAV_VALUE, MAX_WAV_VALUE - 1
    ).short().cpu().numpy()
    sf.write(out_w, int16, sr, subtype="PCM_16")

    with open(out_j, "w") as fp:
        json.dump({
            "source_file": str(in_path),
            "sampling_rate": sr,
            "segment_samples": SEG_SAMPLES,
            "watermarks": [w.tolist() for w in wm_bits],
        }, fp, indent=2)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Bulk embed random watermarks in LibriSpeech 24 kHz audio."
    )
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Root of LibriSpeech_24k_wav/train-clean-360",
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Where to write watermarked WAVs (mirrors input tree)",
    )
    ap.add_argument("--checkpoint", required=True, help="Path to model ckpt")
    ap.add_argument("--device", default="cuda:0", help="cpu | cuda:0 …")
    ap.add_argument("--workers", type=int, default=1, help="I/O worker threads")
    ap.add_argument("--no-skip", action="store_true", help="Overwrite outputs")
    ap.add_argument("--chunk_batch", type=int, default=64,
                help="how many 0.5-s segments to embed in one GPU batch")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Load model-specific hyper-parameters
    cfg_path = Path(args.checkpoint).with_name("config.json")
    with open(cfg_path) as f:
        h = AttrDict(json.load(f))

    modules = load_modules(h, args.checkpoint, device)

    in_files = sorted(Path(args.input_dir).rglob("*.wav"))

    print(f"Found {len(in_files)} WAV files – starting watermark pass…")
    for wav_file in tqdm(in_files):
        process_file(
            wav_file,
            Path(args.output_dir),
            modules,
            h,
            device,
            chunk_batch = args.chunk_batch,
            skip_existing=not args.no_skip,
        )

    print("✓ All done – watermarked dataset written to", args.output_dir)
