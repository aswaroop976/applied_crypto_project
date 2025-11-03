import os
import numpy as np
from PIL import Image
import pdqhash

# === CONFIGURATION ===
INPUT_DIR = "images_in"
OUTPUT_DIR = "images_out"
RESIZE_TO = (64, 64)
NOISE_STD = 0.03   # standard deviation of Gaussian noise (0–1 scale)
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

def _ensure_rgb_uint8(arr):
    """
    PDQ expects an 8-bit array; it works well with 3-channel input.
    arr can be HxW (grayscale) or HxWxC; returns HxWx3 uint8.
    """
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)  # gray -> RGB
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr.astype(np.uint8)

def pdq_dihedral_hashes(arr_uint8_rgb):
    """
    Returns (hashes8, quality) using PDQ dihedral variant for rot/flip robustness.
    hashes8: list/array of 8 binary vectors of length 256 (dtype uint8 or bool)
    """
    hashes8, quality = pdqhash.compute_dihedral(arr_uint8_rgb)
    # Ensure binary arrays are numpy arrays of 0/1 for distance calc
    hashes8 = [np.array(h, dtype=np.uint8) for h in hashes8]
    return hashes8, quality

def pdq_best_distance(hashes_a, hashes_b):
    """
    Brute-force best Hamming distance across dihedral variants.
    """
    best = 256
    for ha in hashes_a:
        for hb in hashes_b:
            d = int(np.count_nonzero(ha != hb))
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best

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

    # === NEW: PDQ similarity (distance + similarity score) ===
    x_u8_rgb = _ensure_rgb_uint8(to_uint8(x))          # HxWx3 uint8
    xpert_u8_rgb = _ensure_rgb_uint8(to_uint8(x_pert)) # HxWx3 uint8

    hv1, q1 = pdq_dihedral_hashes(x_u8_rgb)
    hv2, q2 = pdq_dihedral_hashes(xpert_u8_rgb)

    pdq_dist = pdq_best_distance(hv1, hv2)        # 0..256 (lower is more similar)
    pdq_sim = (256 - pdq_dist) / 256.0            # 0..1 (higher is more similar)

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
              f"L2/pix={metrics['L2_pixel']:.5f}  | "
              f"PDQdist={pdq_dist:3d}  PDQsim={pdq_sim:.3f}  "
              f"Qorig={q1:.1f}  Qpert={q2:.1f}")

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
