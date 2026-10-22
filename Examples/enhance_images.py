"""
Image Enhancement Script for DIC Images
Applies various enhancement techniques and shows before/after comparison.
Edit the parameters below to tune each enhancement.
"""
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from skimage import exposure, filters, restoration, img_as_float, img_as_ubyte

# =============================================================================
# INPUT - Edit these paths to point to your images
# =============================================================================
IMAGE_PATH = r'../AgedCell/SmallPouch_Fresh.tif'

# =============================================================================
# ENHANCEMENT PARAMETERS - Edit these to tune the enhancements
# =============================================================================

# 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP_LIMIT = 0.03        # Higher = more contrast (0.01 - 0.1)
CLAHE_KERNEL_SIZE = 64         # Size of local regions (smaller = more local contrast)

# 2. Denoising (Non-local means)
DENOISE_PATCH_SIZE = 5         # Size of patches for comparison (3-7)
DENOISE_PATCH_DISTANCE = 6     # Search distance for similar patches (5-11)
DENOISE_H = 0.08               # Filter strength - higher removes more noise (0.01 - 0.2)

# 3. Sharpening (Unsharp mask)
SHARPEN_RADIUS = 2.0           # Blur radius for mask (1.0 - 5.0)
SHARPEN_AMOUNT = 1.0           # Sharpening strength (0.5 - 3.0)

# 4. Background normalization
BG_SIGMA = 50                  # Gaussian sigma for background estimation (20 - 100)

# 5. Histogram stretching
STRETCH_LOW_PERCENTILE = 2     # Lower percentile to clip (0 - 5)
STRETCH_HIGH_PERCENTILE = 98   # Upper percentile to clip (95 - 100)

# =============================================================================
# LOAD IMAGE
# =============================================================================
print("Loading image...")
raw = np.array(imageio.imread(IMAGE_PATH), dtype=float)
if raw.ndim == 3:
    raw = raw[0]  # Take first page of multi-page TIFF

# Normalize to 0-1
img = (raw - raw.min()) / (raw.max() - raw.min())
print(f"Image shape: {img.shape}, dtype: {img.dtype}")

# =============================================================================
# ENHANCEMENT FUNCTIONS
# =============================================================================

def apply_clahe(image, clip_limit, kernel_size):
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    return exposure.equalize_adapthist(image, clip_limit=clip_limit,
                                       kernel_size=kernel_size)

def apply_denoise(image, patch_size, patch_distance, h):
    """Apply non-local means denoising."""
    return restoration.denoise_nl_means(image, patch_size=patch_size,
                                         patch_distance=patch_distance,
                                         h=h, fast_mode=True)

def apply_sharpen(image, radius, amount):
    """Apply unsharp mask sharpening."""
    return filters.unsharp_mask(image, radius=radius, amount=amount)

def apply_bg_normalization(image, sigma):
    """Remove uneven background lighting using Gaussian blur subtraction."""
    background = filters.gaussian(image, sigma=sigma)
    normalized = image - background
    # Rescale to 0-1
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    return normalized

def apply_histogram_stretch(image, low_pct, high_pct):
    """Stretch histogram to use full dynamic range."""
    p_low, p_high = np.percentile(image, (low_pct, high_pct))
    return exposure.rescale_intensity(image, in_range=(p_low, p_high))


# =============================================================================
# APPLY ENHANCEMENTS
# =============================================================================
print("\nApplying enhancements...")

results = {}

print("  1/5 CLAHE...")
results['CLAHE'] = apply_clahe(img, CLAHE_CLIP_LIMIT, CLAHE_KERNEL_SIZE)

print("  2/5 Denoising...")
results['Denoised'] = apply_denoise(img, DENOISE_PATCH_SIZE, DENOISE_PATCH_DISTANCE, DENOISE_H)

print("  3/5 Sharpening...")
results['Sharpened'] = apply_sharpen(img, SHARPEN_RADIUS, SHARPEN_AMOUNT)

print("  4/5 Background normalization...")
results['BG Normalized'] = apply_bg_normalization(img, BG_SIGMA)

print("  5/5 Histogram stretch...")
results['Hist Stretched'] = apply_histogram_stretch(img, STRETCH_LOW_PERCENTILE, STRETCH_HIGH_PERCENTILE)

# =============================================================================
# COMBINED: Apply all enhancements in sequence
# =============================================================================
print("  Applying all combined...")
combined = img.copy()
combined = apply_bg_normalization(combined, BG_SIGMA)
combined = apply_denoise(combined, DENOISE_PATCH_SIZE, DENOISE_PATCH_DISTANCE, DENOISE_H)
combined = apply_clahe(combined, CLAHE_CLIP_LIMIT, CLAHE_KERNEL_SIZE)
combined = apply_sharpen(combined, SHARPEN_RADIUS, SHARPEN_AMOUNT)
combined = apply_histogram_stretch(combined, STRETCH_LOW_PERCENTILE, STRETCH_HIGH_PERCENTILE)
results['All Combined'] = combined

# =============================================================================
# VISUALIZATION - Before/After comparison
# =============================================================================
print("\nShowing results...")

n_results = len(results)
fig, axes = plt.subplots(n_results + 1, 1, figsize=(16, 3 * (n_results + 1)))

# Original
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Each enhancement
for i, (name, enhanced) in enumerate(results.items()):
    axes[i + 1].imshow(enhanced, cmap='gray')
    axes[i + 1].set_title(name, fontsize=12, fontweight='bold')
    axes[i + 1].axis('off')

plt.tight_layout()
plt.suptitle('Image Enhancement Comparison', fontsize=14, fontweight='bold', y=1.01)
plt.show()

# =============================================================================
# SAVE ENHANCED IMAGE (optional - uncomment to save)
# =============================================================================
# output_path = r'../AgedCell/SmallPouch_Fresh_enhanced.tif'
# imageio.imwrite(output_path, img_as_ubyte(combined))
# print(f"Saved enhanced image to: {output_path}")

print("\nDone!")
