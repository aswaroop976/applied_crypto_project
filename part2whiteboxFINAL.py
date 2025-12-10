import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import os

# ⭐️ Import the necessary functions from the first script file
from part1whitebox import calculate_phash_binary_matrix, generate_dct_and_phash 

# --- Core Logic Functions (No Changes Here) ---

def calculate_required_bit_flips(source_hash: np.ndarray, target_hash: np.ndarray) -> np.ndarray:
    """Calculates where the bits in the source hash need to be flipped (XOR)."""
    return np.bitwise_xor(source_hash, target_hash)

def apply_modification_via_direct_idct(target_dct_matrix: np.ndarray) -> np.ndarray:
    """
    Directly converts a 32x32 target DCT matrix (C_target) into the corresponding 
    32x32 modified image array (I_modified) using the Inverse DCT (IDCT).
    """
    modified_image_float = idct(idct(target_dct_matrix.T, norm='ortho').T, norm='ortho')
    modified_image_array = np.clip(modified_image_float, 0, 255).astype(np.uint8)
    return modified_image_array

def get_mean_threshold(dct_matrix_32x32: np.ndarray) -> float:
    """Helper to get the current pHash mean threshold."""
    top_left_8x8 = dct_matrix_32x32[0:8, 0:8]
    coeffs_to_avg = top_left_8x8.flatten()[1:]
    return np.mean(coeffs_to_avg)

# --- ITERATIVE MODIFICATION FUNCTION (Modified to return C_modification) ---
def iteratively_target_modification(source_image_array, source_dct, target_dct, target_phash, max_iterations=200):
    current_image = source_image_array.copy()
    
    # 1. INITIAL LARGE MODIFICATION
    C_delta_8x8 = target_dct[0:8, 0:8] - source_dct[0:8, 0:8]
    C_modification = np.zeros((32, 32))
    C_modification[0:8, 0:8] = C_delta_8x8
    I_delta = idct(idct(C_modification.T, norm='ortho').T, norm='ortho')
    current_image += I_delta
    
    print("\nInitial modification applied. Refinement starting...")

    for i in range(max_iterations):
        # ⭐️ STEP A: Simulate the "Real" Image (Clip & Round to Integer)
        # This is where we break the floating-point fragile match.
        real_image_state = np.clip(current_image, 0, 255).astype(np.uint8)
        
        # ⭐️ STEP B: Recalculate hash on the REAL integer pixels
        centered = real_image_state.astype(float) - 128.0
        current_dct = dct(dct(centered.T, norm='ortho').T, norm='ortho')
        current_phash = calculate_phash_binary_matrix(current_dct)
        
        # ⭐️ STEP C: Match Check
        required_flips = calculate_required_bit_flips(current_phash, target_phash)
        total_flips = np.sum(required_flips)
        
        # Only exit if the hash matches on the ROUNDED image
        if total_flips == 0:
            print(f"Success! Final match achieved on iteration {i}.")
            return real_image_state, C_modification, I_delta

        # ⭐️ STEP D: Diagnostics (Run every 20 loops)
        if i % 20 == 0:
            print(f"Iteration {i}: {total_flips} bits left on integer-rounded image.")

        # ⭐️ STEP E: APPLY NUDGES
        # Use a significant nudge magnitude (5.0) to "hammer" the bits
        # across the threshold so rounding doesn't pull them back.
        NUDGE_MAGNITUDE = 5.0
        
        delta_dct_refinement = np.zeros((32, 32))
        flip_indices = np.argwhere(required_flips > 0)
        
        for u, v in flip_indices:
            if u == 0 and v == 0: continue
            # Nudge direction based on current bit vs target
            delta_dct_refinement[u, v] = -NUDGE_MAGNITUDE if current_phash[u, v] == 1 else NUDGE_MAGNITUDE

        # Invert nudge matrix and apply to the floating-point current_image
        delta_image_refinement = idct(idct(delta_dct_refinement.T, norm='ortho').T, norm='ortho')
        current_image += delta_image_refinement

    print(f"--- FAILED TO CONVERGE: {total_flips} bits remain. ---")
    return np.clip(current_image, 0, 255).astype(np.uint8), C_modification, I_delta
    
    
    
def apply_modification_to_full_size(original_image_path, modified_32_array, output_path):
    """
    Takes the converged 32x32 array and applies the change to the high-res original image.
    """
    # 1. Load the original color high-res image
    original_color = Image.open(original_image_path).convert('RGB')
    orig_w, orig_h = original_color.size
    
    # 2. Re-create the 32x32 baseline grayscale image
    # Note: We match your preprocessing EXACTLY (L-mode, 32x32, Lanczos)
    img_baseline = Image.open(original_image_path).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
    baseline_32_array = np.array(img_baseline, dtype=float)
    
    # 3. Calculate Delta I (The iterative change mask)
    # This is the "nudge" that forced the pHash to match.
    delta_i_32 = modified_32_array.astype(float) - baseline_32_array
    
    # 4. Upsample Delta I to the high-res original size
    # Stretching the 32x32 mask smoothly over the full image resolution.
    delta_i_pil = Image.fromarray(delta_i_32)
    delta_upsampled_pil = delta_i_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    delta_upsampled = np.array(delta_upsampled_pil, dtype=float)
    
    # 5. Apply to Original Color Array
    original_array = np.array(original_color, dtype=float)
    
    # Apply the intensity shift to R, G, and B channels equally.
    # This preserves original colors while changing the local luminance values.
    for i in range(3):
        original_array[:, :, i] += delta_upsampled
        
    # 6. Final cleanup: Clip values to [0, 255] and convert to uint8
    final_full_res = np.clip(original_array, 0, 255).astype(np.uint8)
    
    # Save the result
    final_image = Image.fromarray(final_full_res)
    final_image.save(output_path)
    print(f"\n[+] Modification reintegrated. Full-size image saved as: {output_path}")
       
        
def final_system_verification(full_size_modified_path, target_phash):
    """
    Passes the final high-res file through the full pHash pipeline 
    to verify the match survived the upsampling.
    """
    print("\n" + "="*50)
    print("## FINAL SYSTEM VERIFICATION")
    print(f"File: {full_size_modified_path}")
    
    # Run Script 1 logic on the final modified file
    # We call generate_dct_and_phash from your imported module
    _, final_phash = generate_dct_and_phash(full_size_modified_path)
    
    # Comparison
    match = np.array_equal(final_phash, target_phash)
    
    print("\n--- Match Results ---")
    print(f"Target Hash Matrix:\n{target_phash}")
    print(f"Final Hash Matrix:\n{final_phash}")
    print(f"\nFinal Collision Status: {'✅ SUCCESS - COLLISION ACHIEVED' if match else '❌ FAILURE'}")
    print("="*50)
    
    return match
    
    
    
# --- Main Execution ---
if __name__ == "__main__":
    SOURCE_IMAGE_PATH = "IMG_5737.jpeg" 
    TARGET_IMAGE_PATH = "IMG_4818.jpeg" 
    
    # 1. Process Source Image
    source_dct, source_phash = generate_dct_and_phash(SOURCE_IMAGE_PATH)
    
    # 2. Process Target Image
    target_dct, target_phash = generate_dct_and_phash(TARGET_IMAGE_PATH)
    
    if source_dct is None or target_dct is None:
        exit()
    
    # Check initial hash difference
    initial_flips = np.sum(calculate_required_bit_flips(source_phash, target_phash))
    print("\n" + "="*50)
    print(f"INITIAL HASH DIFFERENCE: {initial_flips} bits")
    print("Starting targeted iterative modification...")
    print("="*50)
    
    try:
        source_img_obj = Image.open(SOURCE_IMAGE_PATH).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
    except Exception:
        print("Error loading image for manipulation.")
        exit()
        
    source_img_array = np.array(source_img_obj, dtype=float)
    
    # 3. Apply the Targeted Iterative Modification
    modified_image_array, c_modification_matrix, i_delta_matrix = iteratively_target_modification(
        source_img_array, 
        source_dct, 
        target_dct, 
        target_phash
    )
    
    # 4. Final Verification
    modified_array_float = modified_image_array.astype(float) - 128.0
    modified_dct = dct(dct(modified_array_float.T, norm='ortho').T, norm='ortho')
    verified_phash = calculate_phash_binary_matrix(modified_dct)
    
    # --- MANUAL RMSE (Showing off the math) ---
    mse = np.mean((source_img_array - modified_image_array)**2)
    rmse = np.sqrt(mse)
    
      
    apply_modification_to_full_size(
        SOURCE_IMAGE_PATH, 
        modified_image_array, 
        "MODIFIED_FULLSIZE.png"
    )
    
    
    FINAL_PATH = "IMG_5718_MODIFIED_FULLSIZE.png"
    final_system_verification(FINAL_PATH, target_phash)
    
    
    # Save the modified image
    modified_img = Image.fromarray(modified_image_array)
    modified_img.save("modified_image_final.png")
    
    print("\n" + "="*50)
    print(f"VISUAL DISTORTION REPORT")
    print(f"RMSE (Root Mean Square Error): {rmse:.2f} levels")
    print(f"A lower RMSE means the source and modified images look similar.")
    print("="*50)
    
    # --- ADDED DEBUG AND PROCESS DISPLAY ---
    print("\n" + "—"*50)
    print("## MODIFICATION PROCESS MATRICES")
    print(f"Initial Hash Difference: {initial_flips} bits")
    print("\n### Calculated Modification DCT Matrix (C_modification - Low Freq Delta)")
    print(c_modification_matrix[0:8, 0:8]) # Show only the 8x8 relevant part
    print(f"\nMax C Change: {np.max(np.abs(c_modification_matrix)):.2f}")
    
    print("\n### Initial Pixel Change Matrix (I_delta - The applied image change)")
    # Print the mean and range of the pixel changes
    print(f"Mean Delta: {np.mean(i_delta_matrix):.2f}, Range: [{np.min(i_delta_matrix):.2f}, {np.max(i_delta_matrix):.2f}]")
    print(i_delta_matrix[16:20, 16:20]) # Sample a small center block
    print("—"*50)
    
    print("\n## Final Results")
    print(f"Target pHash:\n{target_phash}")
    print(f"\nVerified Final pHash:\n{verified_phash}")
    print(f"\nModification Successful: {np.array_equal(target_phash, verified_phash)}")
    print("Modified image saved as 'modified_image_final.png'.")