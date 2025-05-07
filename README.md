# Block-Level JPEG Compression Analysis Using DCT Histogram Features

## 📌 Overview

This project detects and analyzes **single vs. double JPEG compression** in images using **block-level DCT (Discrete Cosine Transform) histogram analysis**. The approach is based on Section 2 of the paper _“Block-level double JPEG compression detection for image forgery localization”_. It relies purely on the statistical behavior of quantized DCT coefficient histograms, without using CNNs or machine learning, for a lightweight and interpretable image forensics solution.

---

## 🧩 Project Workflow

### 1. **Dataset Handling**
- Extracted over **20,000 single** and **20,000 double** compressed JPEG images from a HuggingFace dataset in **Apache Arrow** format.
- Parsed and saved images separately into two directories:
  - `extracted_54_images/` for **single compressed**
  - `extracted_01_images/` for **double compressed**

### 2. **DCT Histogram Generation**
- Converted images to the **Y (luminance)** channel.
- Applied **2D DCT** to each non-overlapping **8x8** block of the image.
- Quantized the DCT coefficients using a fixed matrix.
- Computed **64 histograms per image** (one for each DCT frequency index).
- Saved:
  - Histogram matrices as `.csv` files
  - Visualizations as `.jpg` heatmaps

### 3. **Feature Extraction**
For each image:
- **Shannon Entropy** of the full histogram vector:
  > \( H = -\sum p(x_i) \log_2 p(x_i) \)
- **Non-Zero Bin Count**: The number of histogram bins with values greater than 0.
  
These metrics were saved in the following files:
- `entropy_sparsity_per_image.csv`
- Per-image stats stored in `entropy_nonzero_per_image/` folder

### 4. **Comparative Analysis**
- Plotted log-scale line graphs for DCT frequencies (e.g., DC, mid, high).
- Created heatmaps showing the **difference of histograms** between compression types.
- Compared distributions of entropy and non-zero bins.

---

## 📈 Results Summary

| Compression Type | Avg. Entropy | Avg. Non-Zero Bins |
|------------------|--------------|---------------------|
| Single JPEG      | 6.5401       | 349.10              |
| Double JPEG      | 6.5150       | 338.73              |

Double-compressed images show:
- Slightly **lower entropy**
- **Fewer non-zero bins**
  
These are consistent with the **increased regularity and sparsity** caused by double quantization.

---

## 🔧 Tools Used

- Python 3.12.3
- `numpy`, `pandas`, `scipy`, `matplotlib`
- `PIL`, `pyarrow`, `tqdm`
- Ubuntu 22.04, 8-core CPU

---

## 📂 Folder Structure

```text
DIP_Project/
├── extracted_01_images/              # Double compressed images
├── extracted_54_images/              # Single compressed images
├── histograms_01_csv/               # DCT histograms (double)
├── histograms_54_csv/               # DCT histograms (single)
├── histograms_01_visualizations/    # Heatmap JPGs (double)
├── histograms_54_visualizations/    # Heatmap JPGs (single)
├── entropy_nonzero_per_image/       # Per-image entropy/sparsity files
├── entropy_sparsity_per_image.csv   # Summary CSV
├── entropy_sparsity_summary.csv     # Class-wise averages
├── plots/                           # Comparison plots
├── *.py                             # Code scripts



```

## 📌 Authors

- Guguloth Bhanu Prakash  
- Katta Mukesh  
- Pothuri Nikhil Babu  
*Department of Computer Science, IIT Bhilai*

---

## 📚 Reference

Z. Wang and X. Zhang,  
“Block-level double JPEG compression detection for image forgery localization,”  
_IEEE Transactions on Information Forensics and Security_
