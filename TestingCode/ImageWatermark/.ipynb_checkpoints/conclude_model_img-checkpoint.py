import os
import numpy as np
import re
import csv
from tqdm import tqdm
import soundfile as sf
import torch
import jiwer
from scipy.spatial.distance import euclidean
from pystoi import stoi
from pesq import pesq
from PIL import Image
import argparse
import math
# Evaluation modes
SAMPLE_RATE = 16000

# Compute DTW (normalized distance)
def compute_dtw(x, xw):
    """
    Computes DTW as SNR between original and watermarked signal.
    x: original waveform (numpy array)
    xw: watermarked waveform (numpy array)
    Returns: DTW score in dB
    """
    x = np.asarray(x)
    xw = np.asarray(xw)
    power = np.sum(x ** 2)
    noise = np.sum((xw - x) ** 2)
    if noise == 0:
        return float('inf')
    return 10 * np.log10(power / noise)

# Compute STOI
def compute_stoi(x, y, sr=SAMPLE_RATE):
    return stoi(x, y, sr, extended=False)

# Compute PESQ
def compute_pesq(x, y, sr=SAMPLE_RATE):
    try:
        return pesq(sr, x, y, 'wb')
    except:
        return np.nan

# Load WAV as float
def load_wav(path):
    audio, sr = sf.read(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Sample rate mismatch: {sr}")
    return audio

def _write_csv(out_path: str, header: list[str], rows: list[tuple]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

# Analyze imperceptibility
def evaluate_imperceptibility(result_dir):
    scores = {
        "DTW": [], "PESQ": [], "STOI": []
    }
    print("\nEvaluating Imperceptibility...")
    for root, _, files in os.walk(result_dir):
        for file in files:
            if file.endswith("_original.wav"):
                base = file.replace("_original.wav", "")
                orig_path = os.path.join(root, f"{base}_original.wav")
                wm_path = os.path.join(root, f"{base}_watermarked.wav")
                if not os.path.exists(wm_path):
                    continue
                x = load_wav(orig_path)
                y = load_wav(wm_path)
                scores["DTW"].append(compute_dtw(x, y))
                scores["PESQ"].append(compute_pesq(x, y))
                scores["STOI"].append(compute_stoi(x, y))

    print("Imperceptibility Summary:")
    imp_rows = []                                    # for the csv
    for metric, vals in scores.items():
        arr   = np.array(vals, dtype=float)
        mean  = np.nanmean(arr)
        std   = np.nanstd(arr)
        count = np.sum(~np.isnan(arr))
        print(f"{metric}:  Mean={mean:.4f},  Std={std:.4f},  Count={count}")
        imp_rows.append((metric, f"{mean:.6f}", f"{std:.6f}", count))
    
    _write_csv(
        os.path.join(result_dir, "imperceptibility_summary.csv"),
        header=["Metric", "Mean", "Std", "Count"],
        rows=imp_rows,
    )
    return scores

# -----------------------------------------------------------
#  Image–level robustness: compare *_orig.png with *_recon*.png
# -----------------------------------------------------------
def evaluate_robustness(result_dir):
    """
    Scan `result_dir` for files that follow the naming pattern
        img_00042_orig.png
        img_00042_recon.png           (clean / no-attack)
        img_00042_recon_MF3.png       (example attack name)
        img_00042_recon_MP3Cutter.png ( … )
    and compute the Mean-Squared-Error (MSE) and Peak-SNR (PSNR)
    between every original / reconstructed image pair.

    Returns
    -------
    atk_scores : dict  { attack_name : [mse_1, mse_2, …] }
    """
    # helper lambdas --------------------------------------------------------
    to_np = lambda p: np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    psnr  = lambda mse: 10.0 * math.log10(255.0**2 / mse) if mse > 0 else float("inf")

    orig_pat  = re.compile(r"(img_\d{5})_orig\.png$")
    recon_pat = re.compile(r"(img_\d{5})_recon_([\w\-]+)\.png$")  # group(2) = attack

    atk_scores = {}   # e.g. {"clean": [mse, …], "MF3": […], …}

    # first pass – collect every original → its reconstructions -------------
    for root, _, files in os.walk(result_dir):
        files = [f for f in files if f.endswith(".png")]
        tag2orig = {orig_pat.match(f).group(1): os.path.join(root, f)
                    for f in files if orig_pat.match(f)}
        for f in files:
            m = recon_pat.match(f)
            if not m:
                continue
            tag, atk = m.groups()
            atk = atk or "clean"                #  *_recon.png  → clean
            if tag not in tag2orig:             # no ground-truth image found
                continue

            # load & score ---------------------------------------------------
            orig_img  = to_np(tag2orig[tag])
            recon_img = to_np(os.path.join(root, f))

            mse  = float(np.mean((orig_img - recon_img) ** 2))
            atk_scores.setdefault(atk, []).append(mse)

    # print summary ---------------------------------------------------------
    print("\nRobustness Summary (per attack, image level):")
    rob_rows = []
    for atk in sorted(atk_scores.keys()):
        vals      = np.array(atk_scores[atk], dtype=float)
        mean_mse  = float(vals.mean())
        mean_psnr = psnr(mean_mse)
        samples   = len(vals)
        print(f"{atk:>15}:  MSE = {mean_mse:8.4f}   PSNR = {mean_psnr:6.2f} dB   Samples = {samples}")
        rob_rows.append((atk, f"{mean_mse:.6f}", f"{mean_psnr:.6f}", samples))
    _write_csv(
        os.path.join(result_dir, "robustness_summary.csv"),
        header=["Attack", "Mean_MSE", "Mean_PSNR_dB", "Samples"],
        rows=rob_rows,
    )

    return atk_scores

# Run analysis
def summarize_results(result_dir):
    # evaluate_imperceptibility(result_dir)
    evaluate_robustness(result_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_path")
    args = parser.parse_args()
    summarize_results(args.result_path)

if __name__ == "__main__":
    main()

