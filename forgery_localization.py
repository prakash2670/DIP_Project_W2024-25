import os
import numpy as np
from PIL import Image, ImageDraw
from scipy.fftpack import dct
from scipy.stats import entropy
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# === CONFIGURATION ===
input_folder = "/home/user/Desktop/DIP/extracted_54_images"
output_folder = "/home/user/Desktop/DIP/Image_Forgery_54"
os.makedirs(output_folder, exist_ok=True)

block_size = 64
dct_block = 8
quant_matrix = np.ones((8, 8)) * 10
entropy_threshold = 0.25  # adjust for sensitivity

def rgb_to_y(image):
    image = np.asarray(image.convert("RGB"), dtype=np.float32)
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b

def block_dct_entropy(block):
    h, w = block.shape
    entropies = []
    for i in range(0, h, dct_block):
        for j in range(0, w, dct_block):
            sub_block = block[i:i+dct_block, j:j+dct_block] - 128
            dct_coeff = dct(dct(sub_block.T, norm='ortho').T, norm='ortho')
            q_dct = np.round(dct_coeff / quant_matrix).flatten()
            hist, _ = np.histogram(q_dct, bins=201, range=(-100, 100), density=True)
            ent = entropy(hist + 1e-8, base=2)
            entropies.append(ent)
    return np.mean(entropies)

def process_image(filename):
    try:
        path = os.path.join(input_folder, filename)
        img = Image.open(path).convert("L")
        y = rgb_to_y(img)
        h, w = y.shape
        y = y[:h - h % block_size, :w - w % block_size]

        entropies = []
        coords = []

        for i in range(0, y.shape[0], block_size):
            for j in range(0, y.shape[1], block_size):
                block = y[i:i+block_size, j:j+block_size]
                ent = block_dct_entropy(block)
                entropies.append(ent)
                coords.append((i, j))

        entropies = np.array(entropies)
        median = np.median(entropies)
        suspicious = [coords[k] for k, e in enumerate(entropies) if e < (median - entropy_threshold)]

        img_rgb = Image.open(path).convert("RGB").resize((y.shape[1], y.shape[0]))
        draw = ImageDraw.Draw(img_rgb)
        if suspicious:
            for (i, j) in suspicious:
                draw.rectangle([j, i, j + block_size, i + block_size], outline="red", width=2)
        else:
            draw.rectangle([5, 5, 20, 20], outline="green", width=2)

        out_path = os.path.join(output_folder, filename)
        img_rgb.save(out_path)
        return True
    except Exception as e:
        print(f"❌ Failed: {filename} — {e}")
        return False

# === MAIN: Multiprocessing ===
if __name__ == "__main__":
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"🚀 Processing {len(files)} images using {cpu_count()} CPU cores...")
    with Pool(processes=8) as pool:  # or use cpu_count()
        list(tqdm(pool.imap_unordered(process_image, files), total=len(files)))
    print(f"✅ All outputs saved in: {output_folder}")

