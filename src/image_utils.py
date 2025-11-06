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

def pdq_hamming_distance(hashes_a, hashes_b):
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

def pdq_distance_gray(x_float, delta_float):
    """
    Compute PDQ distance (and similarity) between a grayscale base image x and its
    perturbed version x + delta. Both are float arrays in [0,1], shape HxW.

    Returns: (distance_0_256, similarity_0_1, q_x, q_xpert)
    """
    # 1) form perturbed grayscale, clamp to valid range
    x_pert = np.clip(x_float + delta_float, 0.0, 1.0)

    # 2) PDQ expects 8-bit RGB; convert HxW -> HxWx3 uint8
    x_rgb_u8     = _ensure_rgb_uint8(to_uint8(x_float))
    xpert_rgb_u8 = _ensure_rgb_uint8(to_uint8(x_pert))

    # 3) dihedral PDQ hashes (rot/flip robustness)
    hv1, q1 = pdq_dihedral_hashes(x_rgb_u8)
    hv2, q2 = pdq_dihedral_hashes(xpert_rgb_u8)

    # 4) best Hamming distance across 8x8 pairs + similarity
    dist = pdq_hamming_distance(hv1, hv2)       # 0..256
    sim = (256 - dist) / 256.0
    return dist, sim, q1, q2

# === IMPLEMENTATION OF ALGORITHM 3: InverseDelta ===

def resize_float_delta(delta_small, out_size):
    """
    Resize a float single-channel delta (values may be negative) from small -> out_size (W,H).
    We encode into uint8 for resizing via PIL then decode back to float:
      encode: u8 = round((delta + 1) / 2 * 255)
      decode: delta = (u8/255)*2 - 1
    This preserves sign and relative magnitudes within [-1,1].
    """
    # delta_small : Hs x Ws floats (likely small values, e.g. [-0.05, 0.05])
    # map to u8
    u8 = to_uint8((delta_small + 1.0) / 2.0)  # now 0..255
    img = Image.fromarray(u8, mode="L")
    img_resized = img.resize(out_size, resample=Image.BILINEAR)
    u8_resized = np.asarray(img_resized, dtype=np.uint8)
    delta_resized = (u8_resized.astype(np.float32) / 255.0) * 2.0 - 1.0
    return delta_resized.astype(np.float32)

def inverse_delta_map(orig_rgb_float, delta_gray_resized):
    """
    Map a single-channel resized grayscale perturbation (delta_gray_resized: HxW)
    to a 3-channel perturbation delta_rgb (HxWx3) following Algorithm 3.
    Steps:
      - Compute per-pixel mean grayscale intensity mean(X_c) = (R+G+B)/3
      - For each pixel i and each channel c:
          if delta_gray_i <= 0:
              delta_c_i = delta_gray_i * (X_c_i / mean_i)        [if mean>eps]
          else:
              delta_c_i = delta_gray_i * ((1 - X_c_i) / (1 - mean_i)) [if 1-mean>eps]
      - Clamp X_c + delta_c to [0,1] and return the offset (clamped - X_c).
    """
    eps = 1e-8
    # orig_rgb_float: HxWx3 in [0,1]
    X = orig_rgb_float
    if X.ndim != 3 or X.shape[2] != 3:
        raise ValueError("orig_rgb_float must be HxWx3 float in [0,1]")

    H, W, _ = X.shape
    if delta_gray_resized.shape != (H, W):
        raise ValueError("delta_gray_resized must be same spatial size as orig_rgb_float")

    # mean grayscale per pixel
    mean = np.mean(X, axis=2)  # HxW

    # allocate output delta per channel
    delta_rgb = np.zeros_like(X, dtype=np.float32)  # HxWx3

    # Broadcast arrays for vectorized ops
    mean_b = mean[..., None]  # HxWx1
    delta_b = delta_gray_resized[..., None]  # HxWx1

    # Two masks: where grayscale delta is negative or non-positive, and where positive
    neg_mask = (delta_b <= 0.0)
    pos_mask = ~neg_mask

    # SAFE division denominators
    denom_neg = np.where(mean_b > eps, mean_b, np.inf)         # avoid divide-by-zero
    denom_pos = np.where((1.0 - mean_b) > eps, (1.0 - mean_b), np.inf)

    # For negative/zero delta: scale by (X_c / mean)
    delta_neg = delta_b * (X / denom_neg)    # HxWx3, denom_neg inf => zero contribution

    # For positive delta: scale by ((1-X_c) / (1-mean))
    delta_pos = delta_b * ((1.0 - X) / denom_pos)

    # Combine using masks (broadcasted)
    delta_rgb = np.where(neg_mask, delta_neg, delta_pos)  # HxWx3

    # Now clamp: ensure X + delta_rgb in [0,1] and convert delta_rgb to the clamped offset
    X_plus = np.clip(X + delta_rgb, 0.0, 1.0)
    delta_rgb_clamped = X_plus - X

    return delta_rgb_clamped.astype(np.float32)

def process_image(path):
    name = os.path.basename(path)

    # 1) Load original full-resolution RGB image and convert to float [0,1]
    orig = Image.open(path).convert("RGB")
    orig_size = orig.size  # (W, H)
    orig_rgb = to_float(orig)  # HxWx3 floats

    # 2) Build the small grayscale image (this mirrors the attack space)
    small_gray = orig.convert("L").resize(RESIZE_TO)
    x = to_float(small_gray)  # Hs x Ws, floats in [0,1]

    # 3) Apply perturbation in the small grayscale domain (here: Gaussian noise as before)
    x_pert = add_noise(x, NOISE_STD)  # Hs x Ws

    # 4) Small-plane delta (the optimized \bar{δ})
    bar_delta = x_pert - x  # Hs x Ws, can be negative/positive

    # 5) Resize bar_delta to original image spatial size -> δ_gray (HxW)
    #    This corresponds to step: δ_gray = resize(\bar{δ}, n_gray)
    W_orig, H_orig = orig_size
    delta_gray_resized = resize_float_delta(bar_delta, (W_orig, H_orig))  # H x W floats

    # 6) Map grayscale perturbation to RGB perturbation using Algorithm 3
    delta_rgb = inverse_delta_map(orig_rgb, delta_gray_resized)  # HxWx3

    # 7) Apply final perturbation to original image and clamp, create output image
    perturbed_rgb = np.clip(orig_rgb + delta_rgb, 0.0, 1.0)
    perturbed_u8 = to_uint8(perturbed_rgb)
    perturbed_img = Image.fromarray(perturbed_u8, mode="RGB")

    # 8) For visualization: create a small stacked comparison (orig small vs pert small)
    stacked_small = np.hstack([x, x_pert])   # Hs x (2*Ws) single channel
    stacked_small_u8 = to_uint8(stacked_small)
    stacked_img_small = Image.fromarray(stacked_small_u8, mode="L")

    # 9) Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path_full = os.path.join(OUTPUT_DIR, f"pert_resized_{name}")
    perturbed_img.save(out_path_full)

    out_path_small = os.path.join(OUTPUT_DIR, f"pert_small_{name}")
    stacked_img_small.save(out_path_small)

    # 10) PDQ reporting (optional)
    x_for_pdq = _ensure_rgb_uint8(to_uint8(x))           # small original -> RGB for pdq
    xpert_for_pdq = _ensure_rgb_uint8(to_uint8(x_pert))  # small pert -> RGB for pdq
    hv1, q1 = pdq_dihedral_hashes(x_for_pdq)
    hv2, q2 = pdq_dihedral_hashes(xpert_for_pdq)
    pdq_dist = pdq_hamming_distance(hv1, hv2)
    pdq_sim = (256 - pdq_dist) / 256.0

    if REPORT:
        # L1/L2/Linf metrics on small plane (same as you already had)
        delta = bar_delta
        L1 = np.sum(np.abs(delta))
        L2 = np.sqrt(np.sum(delta * delta))
        Linf = np.max(np.abs(delta))
        L2pix = np.sqrt(np.mean(delta * delta))
        print(f"{name:25s} | L1={L1:.4f} L2={L2:.4f} L∞={Linf:.4f} L2/pix={L2pix:.5f} | "
              f"PDQdist={pdq_dist:3d} PDQsim={pdq_sim:.3f} Qorig={q1:.1f} Qpert={q2:.1f} "
              f"SavedFull={out_path_full} SavedSmall={out_path_small}")

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
