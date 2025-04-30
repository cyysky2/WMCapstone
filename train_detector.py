#!/usr/bin/env python3
# ================================================================
# train_detector.py
#
# Simple, self-contained trainer for the AudioSealDetector.
#   • sample-level BCE loss (watermark present = 1, absent = 0)
#   • automatic padding / masking for variable-length wavs
#   • single-GPU, single-process   (can be extended to DDP easily)
# ================================================================

'''
python train_detector.py \
  --data_dir /root/autodl-tmp/LibriSpeech/detector_data_k5 \
  --epochs   10 \
  --batch    8 \
  --lr       1e-4 \
  --device   cuda:0 \
  --save_ckpt detector_audioseal_k5.pt

'''

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from utils import time_str
from dataset import DetectorDataset
from models import AudioSealDetector


# -----------------------  Collate  --------------------------------
def collate(batch):
    """Pad to max length in batch; return mask for valid samples."""
    audios, labels, srs = zip(*batch)
    assert len(set(srs)) == 1, "mixed sample rates in batch"
    sr = srs[0]

    lens = [a.size(0) for a in audios]
    max_len = max(lens)

    A = torch.zeros(len(audios), max_len)
    L = torch.zeros(len(audios), max_len)
    M = torch.zeros(len(audios), max_len, dtype=torch.bool)   # mask
    for i, (a, l) in enumerate(zip(audios, labels)):
        A[i, : lens[i]] = a
        L[i, : lens[i]] = l
        M[i, : lens[i]] = True
    return A, L, M, sr

# -----------------------  Train / Eval  ---------------------------
def run_epoch(model, loader, optimiser, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss, n_frames = 0.0, 0
    torch.set_grad_enabled(train)
    for audio, gt, mask, sr in loader:
        audio, gt, mask = audio.to(device), gt.to(device), mask.to(device)
        # detector(x) already does internal resampling if sr != 16k
        logits = model.detector(audio)               # (B, 2, T) since nbits=0
        wm_prob = torch.softmax(logits[:, :2, :], dim=1)[:, 1, :]   # (B, T)

        loss = F.binary_cross_entropy(wm_prob[mask], gt[mask])

        if train:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        total_loss += loss.item() * int(mask.sum())
        n_frames   += int(mask.sum())

    return total_loss / n_frames

def main():
    parser = argparse.ArgumentParser("Train AudioSeal detector only")
    parser.add_argument("--data_dir", required=True,
                        help="root folder that holds *_det.wav & *_det.npy")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch",  type=int, default=4)
    parser.add_argument("--lr",     type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_ckpt", default="detector_final.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = DetectorDataset(Path(args.data_dir), "train")
    val_ds   = DetectorDataset(Path(args.data_dir), "val")
    train_ld = DataLoader(train_ds, args.batch, shuffle=True,
                          collate_fn=collate, num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds, args.batch, shuffle=False,
                          collate_fn=collate, num_workers=4, pin_memory=True)

    model = AudioSealDetector(nbits=0).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"{time_str()}  |  start training on {len(train_ds)} files "
          f"(val {len(val_ds)})")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_ld, optimiser, device, train=True)
        vl_loss = run_epoch(model, val_ld, optimiser, device, train=False)

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), args.save_ckpt)

        print(f"{time_str()}  |  epoch {epoch:02d}  "
              f"train BCE {tr_loss:.4f}   val BCE {vl_loss:.4f}")

    print("✓ done, best model saved to", args.save_ckpt)


if __name__ == "__main__":
    main()