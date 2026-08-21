"""
Rigid Body Motion Removal Comparison for DIC Results
=====================================================
Compares three approaches to remove rigid body motion from DIC displacement fields:
1. Subtract mean displacement (removes bulk translation)
2. Subtract best-fit rigid body (translation + rotation)
3. Pre-align images before DIC using feature-based registration

Also adds a finer mesh step to check convergence.
"""
import sys
from os.path import abspath
sys.path.extend([abspath(".")])

import muDIC as dic
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from skimage.transform import warp, AffineTransform
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
import logging

logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# =============================================================================
# PARAMETERS - Edit these
# =============================================================================
IMAGE_PATH = r'../FromPaper/'
IMAGE_1 = '1.tif'
IMAGE_2 = '2.png'

# Mesh configuration (use medium-fine for this comparison)
N_ELY = 8
N_ELX = 8
MARGIN = 15  # pixels from edge

# =============================================================================
# LOAD IMAGES
# =============================================================================
def to_grayscale(img):
    """Convert RGBA/RGB image to grayscale, handle 2D passthrough."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return np.dot(img[:, :, :3], [0.2989, 0.5870, 0.1140])

print("Loading images...")
img1 = to_grayscale(np.array(imageio.imread(IMAGE_PATH + IMAGE_1), dtype=float))
img2 = to_grayscale(np.array(imageio.imread(IMAGE_PATH + IMAGE_2), dtype=float))

# Crop to matching size
min_rows = min(img1.shape[0], img2.shape[0])
min_cols = min(img1.shape[1], img2.shape[1])
img1 = img1[:min_rows, :min_cols]
img2 = img2[:min_rows, :min_cols]
print(f"Image shape: {img1.shape}")

Xc1, Xc2 = MARGIN, min_cols - MARGIN
Yc1, Yc2 = MARGIN, min_rows - MARGIN

# =============================================================================
# RIGID BODY REMOVAL FUNCTIONS
# =============================================================================

def remove_mean_translation(disp_x, disp_y):
    """Remove mean displacement (bulk translation only)."""
    return disp_x - np.mean(disp_x), disp_y - np.mean(disp_y)


def remove_rigid_body(disp_x, disp_y, xs, ys):
    """
    Remove best-fit rigid body motion (translation + rotation).
    Fits: u_x = tx - theta * (y - y_c)
          u_y = ty + theta * (x - x_c)
    """
    x_flat = xs.flatten()
    y_flat = ys.flatten()
    ux_flat = disp_x.flatten()
    uy_flat = disp_y.flatten()

    x_c = np.mean(x_flat)
    y_c = np.mean(y_flat)

    # Build linear system: [1, 0, -(y-yc)] [tx]     [ux]
    #                       [0, 1,  (x-xc)] [ty]  =  [uy]
    #                                       [theta]
    n = len(x_flat)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)

    A[:n, 0] = 1.0       # tx for ux
    A[:n, 2] = -(y_flat - y_c)  # -theta*(y-yc) for ux
    A[n:, 1] = 1.0       # ty for uy
    A[n:, 2] = (x_flat - x_c)   # theta*(x-xc) for uy

    b[:n] = ux_flat
    b[n:] = uy_flat

    # Least squares fit
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    tx, ty, theta = params

    # Compute rigid body field
    rb_x = tx - theta * (ys - y_c)
    rb_y = ty + theta * (xs - x_c)

    print(f"  Rigid body: tx={tx:.3f}px, ty={ty:.3f}px, theta={np.degrees(theta):.4f}°")

    return disp_x - rb_x, disp_y - rb_y


def align_images_feature_based(img_ref, img_def):
    """
    Align images using ORB feature matching + RANSAC affine estimation.
    Returns the aligned deformed image.
    """
    from skimage import img_as_ubyte
    from skimage.transform import EuclideanTransform

    # Normalize to uint8 for feature detection
    ref_u8 = img_as_ubyte((img_ref - img_ref.min()) / (img_ref.max() - img_ref.min()))
    def_u8 = img_as_ubyte((img_def - img_def.min()) / (img_def.max() - img_def.min()))

    # Detect ORB features
    detector = ORB(n_keypoints=500)
    detector.detect_and_extract(ref_u8)
    kp_ref = detector.keypoints
    desc_ref = detector.descriptors

    detector.detect_and_extract(def_u8)
    kp_def = detector.keypoints
    desc_def = detector.descriptors

    # Match descriptors
    matches = match_descriptors(desc_ref, desc_def, cross_check=True)
    src = kp_ref[matches[:, 0]][:, ::-1]  # (x, y) format
    dst = kp_def[matches[:, 1]][:, ::-1]

    print(f"  Found {len(matches)} feature matches")

    # RANSAC for robust rigid body estimation
    model, inliers = ransac(
        (dst, src),  # transform dst -> src (align deformed to reference)
        EuclideanTransform,
        min_samples=3,
        residual_threshold=2.0,
        max_trials=1000,
    )

    print(f"  RANSAC inliers: {inliers.sum()}/{len(inliers)}")
    print(f"  Transform: tx={model.translation[0]:.3f}, ty={model.translation[1]:.3f}, "
          f"rotation={np.degrees(model.rotation):.4f}°")

    # Warp deformed image to align with reference
    aligned = warp(img_def, model.inverse, output_shape=img_def.shape,
                   preserve_range=True)

    return aligned, model


# =============================================================================
# RUN DIC ON ORIGINAL IMAGES
# =============================================================================
print("\n" + "=" * 60)
print("Running DIC on original (unaligned) images...")
print("=" * 60)

images_orig = dic.IO.image_stack_from_list([img1, img2])
mesher = dic.Mesher(deg_e=3, deg_n=3, type="q4")
mesh = mesher.mesh(images_orig, Xc1=Xc1, Xc2=Xc2, Yc1=Yc1, Yc2=Yc2,
                   n_ely=N_ELY, n_elx=N_ELX, GUI=False)

settings = dic.DICInput(mesh, images_orig)
settings.max_nr_im = 500
settings.maxit = 20
settings.tol = 1.e-6
settings.interpolation_order = 4
settings.store_internals = True
settings.noconvergence = "ignore"

job = dic.DICAnalysis(settings)
dic_results = job.run()
fields_orig = dic.post.viz.Fields(dic_results, upscale=10)

# Extract original displacements
disp_x_raw = fields_orig.disp()[0, 0, :, :, -1]
disp_y_raw = fields_orig.disp()[0, 1, :, :, -1]
xs = fields_orig.coords()[0, 0, :, :, -1]
ys = fields_orig.coords()[0, 1, :, :, -1]

# =============================================================================
# METHOD 1: Subtract mean translation
# =============================================================================
print("\nMethod 1: Subtracting mean displacement...")
disp_x_mean, disp_y_mean = remove_mean_translation(disp_x_raw, disp_y_raw)

# =============================================================================
# METHOD 2: Subtract best-fit rigid body
# =============================================================================
print("\nMethod 2: Subtracting rigid body (translation + rotation)...")
disp_x_rb, disp_y_rb = remove_rigid_body(disp_x_raw, disp_y_raw, xs, ys)

# =============================================================================
# METHOD 3: Pre-align images, then run DIC
# =============================================================================
print("\nMethod 3: Pre-aligning images with feature matching...")
img2_aligned, transform_model = align_images_feature_based(img1, img2)

print("  Running DIC on aligned images...")
images_aligned = dic.IO.image_stack_from_list([img1, img2_aligned])
mesh_aligned = mesher.mesh(images_aligned, Xc1=Xc1, Xc2=Xc2, Yc1=Yc1, Yc2=Yc2,
                           n_ely=N_ELY, n_elx=N_ELX, GUI=False)

settings_aligned = dic.DICInput(mesh_aligned, images_aligned)
settings_aligned.max_nr_im = 500
settings_aligned.maxit = 20
settings_aligned.tol = 1.e-6
settings_aligned.interpolation_order = 4
settings_aligned.store_internals = True
settings_aligned.noconvergence = "ignore"

job_aligned = dic.DICAnalysis(settings_aligned)
dic_results_aligned = job_aligned.run()
fields_aligned = dic.post.viz.Fields(dic_results_aligned, upscale=10)

disp_x_aligned = fields_aligned.disp()[0, 0, :, :, -1]
disp_y_aligned = fields_aligned.disp()[0, 1, :, :, -1]
xs_al = fields_aligned.coords()[0, 0, :, :, -1]
ys_al = fields_aligned.coords()[0, 1, :, :, -1]

# =============================================================================
# VISUALIZATION - 4 rows x 3 columns (magnitude heatmap + vector field + stats)
# =============================================================================
print("\nGenerating comparison figure...")

methods = [
    ("Raw (no correction)", disp_x_raw, disp_y_raw, xs, ys, img2),
    ("Mean subtracted", disp_x_mean, disp_y_mean, xs, ys, img2),
    ("Rigid body removed", disp_x_rb, disp_y_rb, xs, ys, img2),
    ("Pre-aligned images", disp_x_aligned, disp_y_aligned, xs_al, ys_al, img2_aligned),
]

n, m = img2.shape
fig, axes = plt.subplots(len(methods), 3, figsize=(18, 5 * len(methods)))

for i, (label, dx, dy, x_coords, y_coords, bg_img) in enumerate(methods):
    mag = np.sqrt(dx**2 + dy**2)

    # Column 1: Magnitude heatmap
    ax = axes[i, 0]
    ax.imshow(bg_img, cmap='gray', origin='lower', extent=(0, m, 0, n), aspect='equal')
    cf = ax.contourf(x_coords, y_coords, mag, levels=50, cmap='jet', alpha=0.5)
    plt.colorbar(cf, ax=ax, shrink=0.8, label='|displacement| [px]')
    ax.set_title(f"{label}\nMagnitude", fontsize=11, fontweight='bold')
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)

    # Column 2: Vector field
    ax = axes[i, 1]
    ax.imshow(bg_img, cmap='gray', origin='lower', extent=(0, m, 0, n), aspect='equal')
    skip = max(1, x_coords.shape[0] // 12)
    ax.quiver(x_coords[::skip, ::skip], y_coords[::skip, ::skip],
              dx[::skip, ::skip], dy[::skip, ::skip],
              mag[::skip, ::skip], cmap='jet', alpha=0.8,
              angles='xy', scale_units='xy', scale=0.5)
    ax.set_title(f"{label}\nVector Field", fontsize=11, fontweight='bold')
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)

    # Column 3: Statistics text
    ax = axes[i, 2]
    ax.axis('off')
    stats_text = (
        f"Method: {label}\n"
        f"{'─' * 35}\n"
        f"X-displacement:\n"
        f"  mean = {np.mean(dx):.4f} px\n"
        f"  std  = {np.std(dx):.4f} px\n"
        f"  max  = {np.max(np.abs(dx)):.4f} px\n\n"
        f"Y-displacement:\n"
        f"  mean = {np.mean(dy):.4f} px\n"
        f"  std  = {np.std(dy):.4f} px\n"
        f"  max  = {np.max(np.abs(dy)):.4f} px\n\n"
        f"Magnitude:\n"
        f"  mean = {np.mean(mag):.4f} px\n"
        f"  max  = {np.max(mag):.4f} px\n"
        f"  min  = {np.min(mag):.4f} px\n\n"
        f"Mesh: {N_ELX}x{N_ELY} elements"
    )
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      alpha=0.9, edgecolor='gray'))

fig.suptitle("Rigid Body Motion Removal - Comparison\n"
             "(1) Mean subtraction  (2) Rigid body fit  (3) Image pre-alignment",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

print("\nDone!")
