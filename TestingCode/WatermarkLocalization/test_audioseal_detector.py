import os, argparse, random
from pathlib import Path

import torch, torchaudio, soundfile as sf
import numpy as np, pandas as pd
from tqdm import tqdm
from audioseal import AudioSeal
from attack import all_attacks, attack, restore_audio

MAX_WAV_VALUE = 32768.0
P_ORIG, P_ZEROS, P_OTHER = 0.4, 0.2, 0.2

# ---------------- models ----------------
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device).eval()
generator= AudioSeal.load_generator("audioseal_wm_16bits").to(device).eval()

@torch.no_grad()    
def localise_mask(audio, sr, thr=0.5):
    wm_probs = detector(audio, sr)[0][:, 1, :]  # softmax already applied in detector
    return (wm_probs > thr)    


def iou(pred, gt):
    inter = (pred & gt).sum(-1).float()
    union = (pred | gt).sum(-1).float().clamp_min(1)
    return (inter / union).mean().item()


info_cache = {}        # tiny dict – path ➞ (frames, sr)
def grab_random_other_segment(all_files, needed_len, sr, max_tries=25):
    for _ in range(max_tries):
        path = random.choice(all_files)
        if path not in info_cache:
            i = sf.info(path)
            info_cache[path] = (i.frames, i.samplerate)
        frames, sr2 = info_cache[path]
        if sr2 != sr or frames < needed_len:
            continue

        start = random.randint(0, frames - needed_len)
        with sf.SoundFile(path) as f:
            f.seek(start)
            seg = f.read(needed_len, dtype="float32")   # read just the slice
        if seg.ndim > 1: seg = seg.mean(axis=1)
        return seg
    # fallback – silence
    return np.zeros(needed_len, dtype="float32")


def augment_mask_mix(wm_audio, clean_audio, all_clean_files, sr, k=2, rng=None):
    if rng is None: rng = random.Random()
    n = len(wm_audio)
    seg_len = max(1, n // (2 * k))
    pool    = list(range(0, n - seg_len, seg_len))
    starts  = sorted(rng.sample(pool, min(k, len(pool))))

    out    = wm_audio.clone()
    labels = torch.ones(n, dtype=torch.bool)

    for s in starts:
        seg = slice(s, s + seg_len)
        r   = rng.random()
        if r < P_ORIG:
            out[seg] = clean_audio[seg]
        elif r < P_ORIG + P_ZEROS:
            out[seg] = 0.0
        elif r < P_ORIG + P_ZEROS + P_OTHER:
            other = grab_random_other_segment(all_clean_files, seg_len, sr)
            out[seg] = torch.from_numpy(other)
        # else keep watermark

        if r < P_ORIG + P_ZEROS + P_OTHER:
            labels[seg] = False
    return out, labels

# --------------- main loop ---------------
def main(args):
    wavs = [os.path.join(dp,f) for dp,_,fs in os.walk(args.data_root)
                                 for f in fs if f.endswith(".wav")]
    all_clean_files = [str(p) for p in Path(args.data_root).rglob("*.wav")]

    stats = {atk[0]: {"iou": [], "acc": []} for atk in all_attacks()}


    for wav_path in tqdm(wavs, desc="files"):
        wav, sr = torchaudio.load(wav_path)
        if sr != 16_000:
            wav = torchaudio.functional.resample(wav, sr, 16_000); sr = 16_000
        if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
        wav = wav.to(device).unsqueeze(1)                       # (1,1,T)

        # 1) watermark
        msg      = torch.randint(0, 2, (1, generator.msg_processor.nbits), device=wav.device)
        wm_wave  = generator.get_watermark(wav, sr, message=msg)
        wm_audio = wav + wm_wave

        # 2) mask‑mix augmentation
        aug_audio, gt_mask = augment_mask_mix(
            wm_audio.squeeze(), wav.squeeze(), all_clean_files, sr, k=5)
        aug_audio = aug_audio.unsqueeze(0).unsqueeze(0).to(device)
        gt_mask   = gt_mask.cpu()

        # 3) attacks & IoU
        for atk_name, _ in all_attacks():
            attacked,_ = attack(aug_audio, sr, [(atk_name, 1.0)])
            attacked   = restore_audio(attacked, aug_audio.size(-1))

            pred_mask  = localise_mask(attacked, sr)[0].cpu()
            iou_score  = iou(pred_mask, gt_mask)
            acc_score  = (pred_mask == gt_mask).float().mean().item()  # accuracy ∈ [0,1]
            
            stats[atk_name]["iou"].append(iou_score)
            stats[atk_name]["acc"].append(acc_score)

    # 4) CSV
    rows = [
    (atk,
     sum(v["iou"])/len(v["iou"]),
     100 * sum(v["acc"])/len(v["acc"]))         # accuracy in %
    for atk, v in stats.items()
    ]
    df = pd.DataFrame(rows, columns=["attack", "mean_IoU", "accuracy_pct"])
    df.to_csv(args.out_csv, index=False)
    print("Saved", args.out_csv)

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_root",
                    default="/root/autodl-tmp/LibriSpeech/LibriSpeech_24k_wav/dev-clean")
    pa.add_argument("--out_csv", default="audioseal_iou_devclean.csv")
    main(pa.parse_args())
