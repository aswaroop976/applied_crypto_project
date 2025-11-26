import os
import numpy as np
from PIL import Image
import pdqhash
from image_utils import *

PDQ_THRESHOLD = 35 # Represents hamming distance between PDQ hashes, change later
NUM_ITERATIONS = 200 # Number of iterations our iterative algorithm works for
EPS_MAX = 0.26 # Max L_inf allowed


def project_linf(delta, eps):
    """Project delta (HxW) onto L∞ ball with radius eps (elementwise clamp)."""
    return np.clip(delta, -eps, eps)

def project_l2(delta, eps):
    """Project delta onto L2 ball of radius eps (global)."""
    norm = np.linalg.norm(delta.ravel())
    if norm <= eps or norm == 0:
        return delta
    return (delta / norm) * eps

def best_candidate_from_random(x, delta_cum, num_candidates=200, per_pixel_eps=0.01, candidate_type='gauss'):
    """
    Generate candidates (small HxW perturbations), test pdq distance increase when added to current,
    and return the best candidate (largest resulting PDQ distance).
    - x: base small-grayscale image (HxW floats in [0,1])
    - delta_cum: current cumulative perturbation (HxW floats, can be nonzero)
    - returns (best_candidate, best_dist, best_sim, best_qorig, best_qpert)
    """
    H, W = x.shape
    best = None
    best_dist = -1
    best_sim = None
    best_qorig = None
    best_qpert = None

    base_perturbed = np.clip(x + delta_cum, 0.0, 1.0)  # current perturbed small image

    for i in range(num_candidates):
        if candidate_type == 'gauss':
            cand = np.random.normal(0, per_pixel_eps, size=(H, W)).astype(np.float32)
        elif candidate_type == 'signed':
            cand = (np.random.choice([-1.0, 1.0], size=(H, W)) *
                    np.random.rand(H, W) * per_pixel_eps).astype(np.float32)
        elif candidate_type == 'sparse':
            # small fraction of pixels changed
            cand = np.zeros((H, W), dtype=np.float32)
            mask = np.random.rand(H, W) < 0.01  # 1% pixels
            cand[mask] = np.random.normal(0, per_pixel_eps, size=np.count_nonzero(mask))
        else:
            raise ValueError("unknown candidate_type")

        # create trial cumulative perturbation and limit it (so candidates don't exceed global budget by themselves)
        trial_delta = delta_cum + cand
        trial_delta = project_linf(trial_delta, per_pixel_eps*5)  # optionally cap intermediate values
        trial_delta = project_l2(trial_delta, per_pixel_eps*5)  # optionally cap intermediate values

        # compute pdq distance between base_perturbed and base_perturbed + cand
        # note: pdq_distance_gray expects delta that when added to x produces perturbed image;
        # here we want distance between (x + delta_cum) and (x + delta_cum + cand),
        # so pass base_perturbed as reference by computing delta_rel = (delta_cum + cand) - delta_cum = cand
        dist, sim, q1, q2 = pdq_distance_gray(base_perturbed, cand)

        # choose the candidate that produced the largest PDQ distance
        if dist > best_dist:
            best_dist = dist
            best = cand
            best_sim = sim
            best_qorig = q1
            best_qpert = q2

    return best, best_dist, best_sim, best_qorig, best_qpert
def iterative_greedy_attack(
    orig_rgb,
    x_small,
    its=50,
    M=200,
    eps_step=0.005,
    eps_max=0.05,
    candidate_type='gauss',
    threshold=PDQ_THRESHOLD,
    bar_delta_init=None,
    history_init=None,
    restart_idx=0,
    max_restarts=3,
):
    """
    Iteratively find and accumulate greedy perturbations in small grayscale space.

    - orig_rgb: original full-res RGB float HxWx3
    - x_small: small grayscale image Hs x Ws (float), this is the *original* base
    - its: number of iterations for THIS run
    - M: number of candidates per iteration
    - eps_step: per-pixel step size / candidate scale
    - eps_max: max L_inf allowed for cumulative bar_delta for THIS run
    - threshold: target PDQ distance between x_small and x_small + bar_delta
    - bar_delta_init: optional initial cumulative perturbation to continue from
    - history_init: optional history from previous runs (list of (iter_idx, dist, sim))
    - restart_idx: how many times we've already bumped eps_max
    - max_restarts: maximum number of times we'll increase eps_max and continue

    Returns: bar_delta, delta_rgb, perturbed_rgb, history
    """
    Hs, Ws = x_small.shape

    # Start from previous perturbation if provided, else from zero
    if bar_delta_init is None:
        bar_delta = np.zeros_like(x_small, dtype=np.float32)
    else:
        bar_delta = bar_delta_init.astype(np.float32).copy()

    # Continue history if given, otherwise start fresh
    if history_init is None or len(history_init) == 0:
        # compute initial PDQ distance between x_small and x_small + bar_delta (likely 0)
        dist_cum, sim_cum, q0, q1 = pdq_distance_gray(x_small, bar_delta)
        history = [(0, dist_cum, sim_cum)]
        start_iter = 1
    else:
        history = list(history_init)
        last_iter_idx, dist_cum, sim_cum = history[-1]
        start_iter = last_iter_idx + 1

    print(f"[restart {restart_idx}] eps_max being utilized: {eps_max}")

    # Main iterative loop for THIS run
    for step in range(its):
        iter_idx = start_iter + step

        # generate many small candidates around the *current* perturbed small image
        best_cand, best_dist, best_sim, qorig, qpert = best_candidate_from_random(
            x_small,
            bar_delta,
            num_candidates=M,
            per_pixel_eps=eps_step,
            candidate_type=candidate_type,
        )

        # accumulate perturbation
        bar_delta = bar_delta + best_cand

        # enforce current L_inf budget
        bar_delta = project_linf(bar_delta, eps_max)
        # optionally also L2:
        # bar_delta = project_l2(bar_delta, l2_max)

        # measure PDQ distance vs original small image x_small
        dist_cum, sim_cum, qorig, qpert = pdq_distance_gray(x_small, bar_delta)
        history.append((iter_idx, dist_cum, sim_cum))

        print(
            f"iter {iter_idx:3d}: best_candidate_dist={best_dist:3d} "
            f"=> cum_dist={dist_cum:3d} PDQsim={sim_cum:.3f} "
            f"q={qorig:.1f}/{qpert:.1f}"
        )

        if dist_cum >= threshold:
            print("Reached PDQ distance target; stopping early.")
            break

    # If threshold not met, bump eps_max and CONTINUE from current bar_delta
    if dist_cum < threshold and restart_idx < max_restarts:
        new_eps_max = eps_max + 0.05
        print(
            f"Threshold not reached (dist_cum={dist_cum} < {threshold}). "
            f"Increasing eps_max to {new_eps_max} and continuing attack "
            f"(restart {restart_idx + 1})."
        )
        return iterative_greedy_attack(
            orig_rgb,
            x_small,                  # keep original base image
            its=its,
            M=M,
            eps_step=eps_step,
            eps_max=new_eps_max,
            candidate_type=candidate_type,
            threshold=threshold,
            bar_delta_init=bar_delta,  # <-- continue from current cumulative delta
            history_init=history,      # <-- keep the full history / logs
            restart_idx=restart_idx + 1,
            max_restarts=max_restarts,
        )

    if dist_cum < threshold:
        print(
            f"WARNING: Max restarts ({max_restarts}) reached, "
            f"final PDQdist={dist_cum} still < threshold={threshold}."
        )

    # Produce final mapped RGB perturbation to apply to the original full-res image
    W_orig, H_orig = orig_rgb.shape[1], orig_rgb.shape[0]
    delta_gray_resized = resize_float_delta(bar_delta, (W_orig, H_orig))
    delta_rgb = inverse_delta_map(orig_rgb, delta_gray_resized)
    perturbed_rgb = np.clip(orig_rgb + delta_rgb, 0.0, 1.0)

    return bar_delta, delta_rgb, perturbed_rgb, history

def process_image(path):
    """
    Modified process_image that runs the iterative greedy attack on the image at `path`.
    Saves:
      - images_out/{name}__orig_fullRGB.png          : original full-res RGB (for reference)
      - images_out/{name}__pert_resized_RGB.png      : final perturbed full-res RGB (mapped from small attack)
      - images_out/{name}__pert_smallstack.png       : side-by-side small grayscale original vs small perturbed
      - images_out/{name}__bar_delta_small.png       : visualization of cumulative bar_delta (scaled for viewing)
    Prints PDQ and metrics to console.
    """
    name = os.path.basename(path)
    base, ext = os.path.splitext(name)
    orig = Image.open(path).convert("RGB")
    orig_size = orig.size  # (W, H)
    orig_rgb = to_float(orig)  # HxWx3 floats

    # small grayscale attack space
    small_gray = orig.convert("L").resize(RESIZE_TO)
    x = to_float(small_gray)  # Hs x Ws

    # Attack configuration (tune these)
    # Run iterative greedy attack
    bar_delta, delta_rgb, perturbed_rgb, history = iterative_greedy_attack(
        orig_rgb=orig_rgb,
        x_small=x,
        its=NUM_ITERATIONS,
        M=200,
        eps_step=0.005,
        eps_max=EPS_MAX,
        candidate_type='gauss',
        threshold=PDQ_THRESHOLD,
        bar_delta_init=None,   # <-- start fresh
        history_init=None,     # <-- start fresh
        restart_idx=0,         # <-- first run
        max_restarts=3         # <-- you can tune this
    )

    # Save outputs with clear filenames
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # original full-res (for quick reference)
    orig_out = os.path.join(OUTPUT_DIR, f"{base}__orig_fullRGB.png")
    Image.fromarray(to_uint8(orig_rgb)).save(orig_out)

    # perturbed full-res RGB (mapped from bar_delta)
    pert_resized_out = os.path.join(OUTPUT_DIR, f"{base}__pert_resized_RGB.png")
    Image.fromarray(to_uint8(perturbed_rgb)).save(pert_resized_out)

    # small stacked: original small grayscale vs (x + bar_delta) small grayscale
    x_pert_small = np.clip(x + bar_delta, 0.0, 1.0)
    stacked_small = np.hstack([x, x_pert_small])
    stacked_small_u8 = to_uint8(stacked_small)
    stacked_small_img = Image.fromarray(stacked_small_u8, mode="L")
    stacked_small_out = os.path.join(OUTPUT_DIR, f"{base}__pert_smallstack.png")
    stacked_small_img.save(stacked_small_out)

    # save a visualization of bar_delta (scale to [0,1] for viewing)
    # We'll map bar_delta (which can be negative) to a viewable grayscale: (bar_delta - min) / (max - min)
    bd = bar_delta
    bd_min, bd_max = float(np.min(bd)), float(np.max(bd))
    if bd_max - bd_min < 1e-8:
        bd_vis = np.zeros_like(bd)
    else:
        bd_vis = (bd - bd_min) / (bd_max - bd_min)
    bd_vis_u8 = to_uint8(bd_vis)
    bd_img = Image.fromarray(bd_vis_u8, mode="L")
    bd_out = os.path.join(OUTPUT_DIR, f"{base}__bar_delta_small.png")
    bd_img.save(bd_out)

    # PDQ final reporting between small x and small x+bar_delta
    dist_cum, sim_cum, qorig_cum, qpert_cum = pdq_distance_gray(x, bar_delta)
    dist_full, sim_full, qorig_full, qpert_full = pdq_distance_rgb(
            orig_rgb, perturbed_rgb)

    # Save naming summary (printed)
    print(f"\nProcessed {name}")
    print(f"Outputs saved to {OUTPUT_DIR}:")
    print(f"  original full-res     : {orig_out}")
    print(f"  perturbed full-res    : {pert_resized_out}")
    print(f"  small stacked (L/R)   : {stacked_small_out}")
    print(f"  bar_delta visualization: {bd_out}")
    print(f"PDQ final (small): PDQdist={int(dist_cum)} PDQsim={sim_cum:.3f} Qorig={qorig_cum:.1f} Qpert={qpert_cum:.1f}")
    print(f"PDQ final (original): PDQdist={int(dist_full)} PDQsim={sim_full:.3f} Qorig={qorig_full:.1f} Qpert={qpert_full:.1f}")
    print(f"  Final L2 per pixel for original: {l2_per_pixel_rgb(perturbed_rgb - orig_rgb)}")
    print(f"  Final L2 norm for original: {l2_norm_rgb(perturbed_rgb - orig_rgb)}")
    # Optionally return paths & history
    return {
        "orig": orig_out,
        "pert_resized": pert_resized_out,
        "smallstack": stacked_small_out,
        "bar_delta_vis": bd_out,
        "history": history
    }

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
