import os
import numpy as np
from PIL import Image

# === CONFIGURATION ===
INPUT_DIR = "images_in"
OUTPUT_DIR = "images_out"
RESIZE_TO = (64, 64)
NOISE_STD = 0.01   # standard deviation of Gaussian noise (0–1 scale)
REPORT = True

# === FUNCTIONS ===

def lp_norm(delta, p=2):
    delta = delta.reshape(-1)
    if p == np.inf:
        return np.max(np.abs(delta))
    return (np.sum(np.abs(delta) ** p)) ** (1.0 / p)

def lp_per_pixel(delta, p=2):
    n = delta.size
    if p == np.inf:
        return np.max(np.abs(delta))
    return (np.mean(np.abs(delta) ** p)) ** (1.0 / p)

def to_float(img):
    return np.asarray(img, dtype=np.float32) / 255.0

def to_uint8(img):
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)

def add_noise(img, std=0.01):
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    out = np.clip(img + noise, 0.0, 1.0)
    return out

def process_image(path):
    name = os.path.basename(path)
    img = Image.open(path).convert("L").resize(RESIZE_TO)
    x = to_float(img)

    # apply perturbation
    x_pert = add_noise(x, NOISE_STD)

    # compute metrics
    delta = x_pert - x
    metrics = {
        "L1": lp_norm(delta, 1),
        "L2": lp_norm(delta, 2),
        "L∞": lp_norm(delta, np.inf),
        "L1_pixel": lp_per_pixel(delta, 1),
        "L2_pixel": lp_per_pixel(delta, 2),
    }

    # stack original and perturbed for visual comparison
    stacked = np.hstack([x, x_pert])
    out_img = Image.fromarray(to_uint8(stacked))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"pert_{name}")
    out_img.save(out_path)

    if REPORT:
        print(f"{name:20s} | "
              f"L1={metrics['L1']:.4f}  L2={metrics['L2']:.4f}  "
              f"L∞={metrics['L∞']:.4f}  "
              f"L2/pix={metrics['L2_pixel']:.5f}")

def main():
    imgs = [f for f in os.listdir(INPUT_DIR)
            if f.lower().endswith(('.jpg', '.jpeg'))]
    if not imgs:
        print("No JPEG files found in", INPUT_DIR)
        return

    for fname in imgs:
        process_image(os.path.join(INPUT_DIR, fname))

if __name__ == "__main__":
    main()
