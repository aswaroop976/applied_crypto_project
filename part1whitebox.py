import numpy as np
from scipy.fftpack import dct
from PIL import Image
import os

def calculate_phash_binary_matrix(dct_matrix_32x32: np.ndarray) -> np.ndarray:
    """
    Calculates the 8x8 binary pHash from the 32x32 DCT matrix.
    Note: This function assumes the input data was centered (subtracted 128).
    """
    
    # 1. Select the top-left 8x8 block (low-frequency components)
    top_left_8x8 = dct_matrix_32x32[0:8, 0:8]

    # 2. Calculate the average threshold (excluding the DC term at (0,0))
    # This exclusion is vital for pHash robustness.
    coeffs_to_avg = top_left_8x8.flatten()[1:] # Excludes the first element C(0,0)
    mean_dct_value = np.mean(coeffs_to_avg)

    # 3. Generate the 8x8 binary hash
    # 1 if coefficient >= mean, 0 otherwise.
    binary_matrix = (top_left_8x8 >= mean_dct_value).astype(int)
    
    return binary_matrix

def generate_dct_and_phash(image_path: str):
    """
    Loads, preprocesses an image, centers the data, generates the 32x32 DCT and 8x8 pHash.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None

    # 1. Load, Resize to 32x32, and Greyscale
    # LANCZOS resampling maintains sharp details during downscaling.
    try:
        img = Image.open(image_path).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"Error processing image file: {e}")
        return None, None
        
    image_array = np.array(img, dtype=float)

    # **CRITICAL FIX: Subtract 128.0 to center the data around zero.**
    # This ensures the AC coefficients are not skewed negative by the large DC term.
    centered_array = image_array - 128.0 

    # 2. Apply the 32x32 DCT
    # Using type-II DCT (norm='ortho')
    dct_matrix_32x32 = dct(dct(centered_array.T, norm='ortho').T, norm='ortho')

    # 3. Generate the 8x8 Binary Matrix (pHash)
    binary_matrix_8x8 = calculate_phash_binary_matrix(dct_matrix_32x32)

    # --- Output Results ---
    print(f"\n--- Results for {os.path.basename(image_path)} ---")
    print("\n## 32x32 Full DCT Matrix (Floating Point):")
    print(dct_matrix_32x32)
    # Show key diagnostics from the new matrix
    print(f"  DC Term C(0,0) is now much smaller: {dct_matrix_32x32[0, 0]:.2f}")
    
    print("\n## 8x8 pHash Binary Matrix (Corrected):")
    print(binary_matrix_8x8)
    
    return dct_matrix_32x32, binary_matrix_8x8

if __name__ == "__main__":
    # Use the specified image file. Ensure 'testimage.jpeg' exists!
    SOURCE_IMAGE = "IMG_5718.jpeg"
    
    # Note: If 'testimage.jpeg' doesn't exist, create a sample image for testing.
    
    source_dct, source_phash = generate_dct_and_phash(SOURCE_IMAGE)
    #print(source_dct)
    
    if source_dct is not None:
        print("\n--- Diagnostics ---")
        C_8x8 = source_dct[0:8, 0:8]
        C_flat_no_DC = C_8x8.flatten()[1:]
        print(f"New Mean (Threshold) of 63 coefficients: {np.mean(C_flat_no_DC):.2f}")
        print(f"Range of 63 coefficients: [{np.min(C_flat_no_DC):.2f}, {np.max(C_flat_no_DC):.2f}]")