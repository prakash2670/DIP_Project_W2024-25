import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Update these paths ===
single_csv_folder = "/home/user/Desktop/DIP/histograms_54_csv"
double_csv_folder = "/home/user/Desktop/DIP/histograms_01_csv"

b = 100
bins = np.arange(-b, b + 1)

def load_histograms(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    all_H = []
    for f in files:
        path = os.path.join(folder, f)
        try:
            df = pd.read_csv(path)
            all_H.append(df.to_numpy())
        except Exception as e:
            print(f"Failed loading {f}: {e}")
    return np.array(all_H)

# Load
H_single = load_histograms(single_csv_folder)
H_double = load_histograms(double_csv_folder)

# Average histograms
mean_single = np.mean(H_single, axis=0)
mean_double = np.mean(H_double, axis=0)

# Plot representative frequencies
freq_indices = [0, 18, 63]  # DC, mid, high freq
freq_labels = ["DC (0,0)", "Mid (2,2)", "High (7,7)"]

plt.figure(figsize=(10, 6))
for idx, label in zip(freq_indices, freq_labels):
    plt.plot(bins, mean_single[idx], label=f"{label} - Single", color='blue', linestyle='--')
    plt.plot(bins, mean_double[idx], label=f"{label} - Double", color='red')

plt.title("Mean Histogram Comparison for Selected DCT Frequencies")
plt.xlabel("DCT Coefficient Value")
plt.ylabel("Count")
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig("lineplot_dct_comparison.jpg")
plt.show()

