import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import dct
from multiprocessing import Pool, cpu_count
import pandas as pd

# === CONFIGURATION ===
input_folder = "/home/user/Desktop/DIP/extracted_01_images"      
visual_folder = "/home/user/Desktop/DIP/histograms_01_visualizations"      
csv_folder = "/home/user/Desktop/DIP/histograms_01_csv"              

b = 100
bins = np.arange(-b, b + 1)
quant_matrix = np.ones((8, 8)) * 10  # Simplified Q-matrix

os.makedirs(visual_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)

# === DCT Histogram Processing Functions ===

def rgb_to_y_channel(img):
    img = np.asarray(img.convert('RGB'), dtype=np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b

def block_dct(block):
    return np.round(dct(dct(block.T, norm='ortho').T, norm='ortho') / quant_matrix)

def compute_dct_histogram(image):
    y = rgb_to_y_channel(image)
    h, w = y.shape
    y = y[:h - h % 8, :w - w % 8]
    H = np.zeros((64, len(bins)), dtype=np.int32)

    for i in range(0, y.shape[0], 8):
        for j in range(0, y.shape[1], 8):
            block = y[i:i+8, j:j+8] - 128
            q_dct = block_dct(block)
            for u in range(8):
                for v in range(8):
                    idx = u * 8 + v
                    val = int(q_dct[u, v])
                    if -b <= val <= b:
                        H[idx, val + b] += 1
    return H

def process_image(file):
    try:
        path = os.path.join(input_folder, file)
        key = os.path.splitext(file)[0]
        img = Image.open(path)
        H = compute_dct_histogram(img)

        # Save visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(H, cmap='viridis', aspect='auto')
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(visual_folder, f"{key}.jpg"), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Save histogram CSV
        df = pd.DataFrame(H, columns=[str(x) for x in bins])
        df.to_csv(os.path.join(csv_folder, f"{key}.csv"), index=False)

        return f"✅ Processed {file}"
    except Exception as e:
        return f"❌ Error processing {file}: {e}"

# === Run in Parallel ===

def main():
    all_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    print(f"\n🔍 Found {len(all_files)} image files.")
    print(f"⚡ Using {cpu_count()} CPU cores for parallel processing.\n")

    with Pool(cpu_count()) as pool:
        for msg in pool.imap_unordered(process_image, all_files):
            print(msg)

    print(f"\n✅ All histograms saved to:\n - Heatmaps: {visual_folder}\n - CSVs: {csv_folder}")

if __name__ == "__main__":
    main()

