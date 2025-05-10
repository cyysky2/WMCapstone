import os
import numpy as np
import re
from tqdm import tqdm
import soundfile as sf
import torch
import jiwer
from scipy.spatial.distance import euclidean
from pystoi import stoi
from pesq import pesq
import argparse
import csv

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
    imp_rows = []
    for key in scores:
        arr = np.array(scores[key], dtype=float)
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        count = np.sum(~np.isnan(arr))
        print(f"{key}: Mean={mean:.4f}, Std={std:.4f}, Count={count}")
        imp_rows.append((key, f"{mean:.6f}", f"{std:.6f}", int(count)))
    
    _write_csv(
        os.path.join(result_dir, "imperceptibility_summary.csv"),
        header=["Metric", "Mean", "Std", "Count"],
        rows=imp_rows,
    )
    return scores

# Analyze robustness
def evaluate_robustness(result_dir):
    attack_scores = {}
    print("\nEvaluating Robustness...")
    for root, _, files in os.walk(result_dir):
        for file in files:
            if file.startswith("recovery") or file.endswith(".txt"):
                match = re.match(r"(.+)_detect_([A-Za-z0-9]+)\.txt", file)
                if not match:
                    continue
                base_name, attack_name = match.groups()
                detect_path = os.path.join(root, file)

                with open(detect_path, "r") as f:
                    lines = f.readlines()
                    try:
                        decoded = eval(re.search(r"Decoded Payload: (\[.*\])", lines[0]).group(1))
                        ber = float(re.search(r"Bit Error Rate: ([\d.]+)", lines[1]).group(1))
                    except:
                        continue

                if attack_name not in attack_scores:
                    attack_scores[attack_name] = {"ber": [], "acc": []}

                attack_scores[attack_name]["ber"].append(ber)
                acc = 100.0 - ber
                attack_scores[attack_name]["acc"].append(acc)

    print("Robustness Summary (per attack):")
    rob_rows = []
    for atk in sorted(attack_scores.keys()):
        ber_vals = np.array(attack_scores[atk]["ber"], dtype=float)
        acc_vals = np.array(attack_scores[atk]["acc"], dtype=float)
        mean_ber = np.mean(ber_vals)
        mean_acc = np.mean(acc_vals)
        print(f"{atk}: Avg BER = {mean_ber:.2f}%, Accuracy = {mean_acc:.2f}%")
        rob_rows.append((atk, f"{mean_ber:.6f}", f"{mean_acc:.6f}"))
    
    _write_csv(
        os.path.join(result_dir, "robustness_summary.csv"),
        header=["Attack", "Mean_BER_percent", "Mean_Accuracy_percent"],
        rows=rob_rows,
    )

    return attack_scores

# Run analysis
def summarize_results(result_dir):
    evaluate_imperceptibility(result_dir)
    evaluate_robustness(result_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_path")
    args = parser.parse_args()
    summarize_results(args.result_path)

if __name__ == "__main__":
    main()
