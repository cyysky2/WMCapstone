"""
Summarize imperceptibility (DTW/SNR, PESQ-wb, STOI) and robustness
(BER / accuracy per attack) for DCT-b1 watermarking results.

Run:  
For DCT result:
python conclude_DCT_and_patchwork.py --result_dir ~/autodl-tmp/DCTb1/OutputResult
For Patchwork result:
python conclude_DCT_and_patchwork.py --result_dir ~/autodl-tmp/Patchwork/OutputResult
"""
import os, re, argparse, numpy as np, soundfile as sf
from tqdm import tqdm
from pystoi import stoi
from pesq import pesq                              # pip install pesq
import csv
SAMPLE_RATE = 16_000

# ---------- helpers ---------------------------------------------------
def load_wav(path):
    audio, sr = sf.read(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} has {sr} Hz (expected {SAMPLE_RATE})")
    return audio.astype(np.float32)

def compute_dtw(x, y):
    power  = np.sum(x**2)
    noise  = np.sum((y - x)**2)
    return np.inf if noise == 0 else 10*np.log10(power / noise)

def compute_pesq_wb(x, y):
    try:
        return pesq(SAMPLE_RATE, x, y, 'wb')
    except Exception:
        return np.nan

def _write_csv(out_path: str, header: list[str], rows: list[tuple]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def imperceptibility(root):
    metrics = {"DTW": [], "PESQ": [], "STOI": []}
    print("\nEvaluating imperceptibility …")
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.endswith("_original.wav"):
                continue
            base = f[:-13]                     # strip "_original.wav"
            try:
                x  = load_wav(os.path.join(dirpath, f))
                y  = load_wav(os.path.join(dirpath, base + "_watermarked.wav"))
                metrics["DTW"].append(compute_dtw(x, y))
                metrics["PESQ"].append(compute_pesq_wb(x, y))
                metrics["STOI"].append(stoi(x, y, SAMPLE_RATE, False))
            except Exception as e:
                print("  !!", os.path.join(dirpath, base), e)

    print("Imperceptibility summary:")
    imp_rows = []
    for k, v in metrics.items():
        arr = np.asarray(v, dtype=float)
        mean = np.nanmean(arr)
        std = np.nanstd(arr)
        count = np.sum(~np.isnan(arr))
        print(f"  {k:<4}: μ = {mean:6.3f}   σ = {std:6.3f}   N = {count}")
        imp_rows.append((k, f"{mean:.6f}", f"{std:.6f}", int(count)))
    
    _write_csv(
        os.path.join(root, "imperceptibility_summary.csv"),
        header=["Metric", "Mean", "Std", "Count"],
        rows=imp_rows,
    )
    return metrics

_detect_re = re.compile(r"(.+)_detect_([A-Za-z0-9]+)\.txt")

def robustness(root):
    _detect_re = re.compile(r"^(.*)_detect_([A-Za-z0-9_-]+)\.txt$", re.IGNORECASE)
    atk_stats = {}
    print("\nEvaluating robustness …")

    for dirpath, _, files in os.walk(root):
        for fname in files:
            m = _detect_re.match(fname)
            if not m:
                continue                            # not a *_detect_*.txt file
            _, attack = m.groups()

            with open(os.path.join(dirpath, fname)) as fh:
                lines = fh.readlines()[:3]

            try:
                ber = float(re.search(r"Bit Error Rate:\s*([\d.]+)", lines[2]).group(1))
            except Exception:
                continue                            # malformed file; skip

            atk_stats.setdefault(attack, {"ber": [], "acc": []})
            atk_stats[attack]["ber"].append(ber)
            atk_stats[attack]["acc"].append(100.0 - ber)

    if not atk_stats:
        print("  No *_detect_*.txt files matched — check the filenames.")
        return atk_stats

    print("Robustness summary (per attack):")
    rob_rows = []
    for atk in sorted(atk_stats):
        ber_arr = np.asarray(atk_stats[atk]["ber"], dtype=float)
        acc_arr = np.asarray(atk_stats[atk]["acc"], dtype=float)
        mean_ber = np.mean(ber_arr)
        mean_acc = np.mean(acc_arr)
        count = len(ber_arr)
        print(f"  {atk:<10}: Avg BER = {mean_ber:6.2f}%   Accuracy = {mean_acc:6.2f}%   N={count}")
        rob_rows.append((atk, f"{mean_ber:.6f}", f"{mean_acc:.6f}", int(count)))
    
    _write_csv(
        os.path.join(root, "robustness_summary.csv"),
        header=["Attack", "Mean_BER_percent", "Mean_Accuracy_percent", "Count"],
        rows=rob_rows,
    )
    return atk_stats

# ---------- CLI -------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", "-r",
                    default=os.path.expanduser("~/autodl-tmp/DCTb1/OutputResult"),
                    help="root folder that contains *_original.wav etc.")
    args = ap.parse_args()

    imperceptibility(args.result_dir)
    robustness(args.result_dir)

if __name__ == "__main__":
    main()
