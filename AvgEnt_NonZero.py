import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm

# === Input folders ===
single_csv_folder = "/home/user/Desktop/DIP/histograms_54_csv"
double_csv_folder = "/home/user/Desktop/DIP/histograms_01_csv"

# === Output folders ===
output_root = "/home/user/Desktop/DIP/entropy_nonzero_per_image"
output_summary_csv = "entropy_sparsity_summary.csv"
os.makedirs(f"{output_root}/single", exist_ok=True)
os.makedirs(f"{output_root}/double", exist_ok=True)

def compute_entropy_nonzero(folder, label):
    results = []
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    print(f"🔍 Processing {len(files)} files from: {label}")
    for f in tqdm(files, desc=f"Computing for {label}"):
        path = os.path.join(folder, f)
        try:
            df = pd.read_csv(path)
            h = df.to_numpy().flatten()
            h_sum = np.sum(h)
            if h_sum > 0:
                p = h / h_sum
                ent = entropy(p, base=2)
            else:
                ent = 0
            nonzero = np.count_nonzero(h)

            # Save per-image CSV
            out_df = pd.DataFrame({
                "filename": [f],
                "entropy": [ent],
                "non_zero_bins": [nonzero],
                "class": [label]
            })
            out_path = os.path.join(output_root, label, f"{os.path.splitext(f)[0]}.csv")
            out_df.to_csv(out_path, index=False)

            # Store summary row
            results.append({
                "filename": f,
                "entropy": ent,
                "non_zero_bins": nonzero,
                "class": label
            })
        except Exception as e:
            print(f"⚠️ Failed to process {f}: {e}")
    return results

# === Run computation for both classes
single_stats = compute_entropy_nonzero(single_csv_folder, "single")
double_stats = compute_entropy_nonzero(double_csv_folder, "double")

# === Combine and save full results
all_stats = pd.DataFrame(single_stats + double_stats)
all_stats.to_csv("entropy_sparsity_per_image.csv", index=False)
print("📁 Full per-image stats saved to: entropy_sparsity_per_image.csv")

# === Summary statistics
summary = all_stats.groupby("class")[["entropy", "non_zero_bins"]].mean().round(4)
summary.to_csv(output_summary_csv)
print(f"📊 Summary saved to: {output_summary_csv}")
print(summary)
