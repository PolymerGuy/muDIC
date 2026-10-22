# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath
from pathlib import Path
sys.path.extend([abspath(".")])

import muDIC as dic
import logging

# Set the amount of info printed to terminal during analysis
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

# # --- Original example run (commented out) ---
# path = r'./example_data/'
# images = dic.IO.image_stack_from_folder(path, file_type='.tif')
# mesher = dic.Mesher(deg_e=3, deg_n=3,type="q4")
# mesh = mesher.mesh(images,Xc1=316,Xc2=523,Yc1=209,Yc2=1055,n_ely=36,n_elx=9, GUI=False)
# settings = dic.DICInput(mesh, images)
# settings.max_nr_im = 500
# settings.ref_update = [15]
# settings.maxit = 20
# settings.tol = 1.e-6
# settings.interpolation_order = 4
# settings.store_internals = True
# settings.noconvergence = "ignore"
# job = dic.DICAnalysis(settings)
# dic_results = job.run()
# fields = dic.post.viz.Fields(dic_results,upscale=10)
# viz = dic.Visualizer(fields,images=images)
# viz.show(field="displacement", component = (1,1), frame=-1)

# --- Registered mid-slice aging run ---
import numpy as np
import imageio.v2 as imageio

# =============================================================================
# INPUT AND RIGID BODY CORRECTION
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
ALIGNED_IMAGE_DIR = (
    REPO_ROOT
    / "AgedCells"
    / "slices_for_aging"
    / "jpg_file_converted_by_FIJI"
    / "aligned_results"
)
REFERENCE_IMAGE = ALIGNED_IMAGE_DIR / "mid_slice_before_aligned_crop.tif"
DEFORMED_IMAGE = ALIGNED_IMAGE_DIR / "mid_slice_after_aligned_crop.tif"

# The input images have already had their global rotation and translation corrected
# by align_mid_slices.py. This removes any small residual translation and rotation
# fitted to each DIC result.
REMOVE_RIGID_BODY = True
# =============================================================================


def to_grayscale(img):
    """Convert RGBA/RGB image to grayscale, handle 2D passthrough."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:  # RGBA - drop alpha, convert RGB
        img = img[:, :, :3]
    # Luminance weights for RGB -> grayscale
    return np.dot(img[:, :, :3], [0.2989, 0.5870, 0.1140])


def remove_rigid_body(disp_x, disp_y, xs, ys):
    """Remove best-fit rigid body motion (translation + rotation) from displacement fields."""
    x_flat = xs.flatten()
    y_flat = ys.flatten()
    ux_flat = disp_x.flatten()
    uy_flat = disp_y.flatten()
    x_c, y_c = np.mean(x_flat), np.mean(y_flat)

    n = len(x_flat)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[:n, 0] = 1.0
    A[:n, 2] = -(y_flat - y_c)
    A[n:, 1] = 1.0
    A[n:, 2] = (x_flat - x_c)
    b[:n] = ux_flat
    b[n:] = uy_flat

    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    tx, ty, theta = params
    print(
        f"  Rigid body removed: tx={tx:.3f}px, ty={ty:.3f}px, "
        f"rotation={np.degrees(theta):.4f} deg"
    )

    rb_x = tx - theta * (ys - y_c)
    rb_y = ty + theta * (xs - x_c)
    return disp_x - rb_x, disp_y - rb_y


if not REFERENCE_IMAGE.exists() or not DEFORMED_IMAGE.exists():
    raise FileNotFoundError(
        "Aligned images are missing. Run Examples\\align_mid_slices.py first."
    )

img1 = to_grayscale(np.array(imageio.imread(REFERENCE_IMAGE), dtype=float))
img2 = to_grayscale(np.array(imageio.imread(DEFORMED_IMAGE), dtype=float))

# Crop to matching size (images may differ by a pixel or two)
min_rows = min(img1.shape[0], img2.shape[0])
min_cols = min(img1.shape[1], img2.shape[1])
img1 = img1[:min_rows, :min_cols]
img2 = img2[:min_rows, :min_cols]
print(f"Image shape: {img1.shape}")
print(f"Reference image: {REFERENCE_IMAGE}")
print(f"Deformed image: {DEFORMED_IMAGE}")

images = dic.IO.image_stack_from_list([img1, img2])

# Generate mesh - GUI broken with newer matplotlib, set coordinates manually
# Image ~342 rows x 335 cols. Xc=columns, Yc=rows. Leave small margin for interpolation.
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

mesh_configs = [
    {"label": "Coarse", "n_ely": 2, "n_elx": 2},
    {"label": "Medium", "n_ely": 4, "n_elx": 4},
    {"label": "Fine", "n_ely": 8, "n_elx": 8},
    {"label": "Very Fine", "n_ely": 12, "n_elx": 12},
]

# Extend mesh to cover nearly the entire image
Xc1, Xc2 = 15, min_cols - 15   # columns with small margin
Yc1, Yc2 = 15, min_rows - 15   # rows with small margin

# Store results for combined figure
all_results = []

for cfg in mesh_configs:
    print(f"\n{'='*60}")
    print(f"Running: {cfg['label']} (n_ely={cfg['n_ely']}, n_elx={cfg['n_elx']})")
    print(f"{'='*60}")

    mesher = dic.Mesher(deg_e=3, deg_n=3, type="q4")
    mesh = mesher.mesh(images, Xc1=Xc1, Xc2=Xc2, Yc1=Yc1, Yc2=Yc2,
                       n_ely=cfg["n_ely"], n_elx=cfg["n_elx"], GUI=False)

    settings = dic.DICInput(mesh, images)
    settings.max_nr_im = 500
    settings.maxit = 20
    settings.tol = 1.e-6
    settings.interpolation_order = 4
    settings.store_internals = True
    settings.noconvergence = "ignore"

    job = dic.DICAnalysis(settings)
    dic_results = job.run()

    fields = dic.post.viz.Fields(dic_results, upscale=10)
    all_results.append((cfg, fields, settings))

# =============================================================================
# VISUALIZATION - One two-panel figure for each mesh configuration
# =============================================================================
HEATMAP_ALPHA = 0.5
VECTOR_LENGTH_PIXELS = 100.0
VECTOR_GRID_TARGET = 12

correction_label = ["globally registered input"]
if REMOVE_RIGID_BODY:
    correction_label.append("residual rigid body removed")
correction_str = " + ".join(correction_label)

n, m = img2.shape

for cfg, fields, settings in all_results:
    # Extract displacement components
    disp_x = fields.disp()[0, 0, :, :, -1]  # x-displacement
    disp_y = fields.disp()[0, 1, :, :, -1]  # y-displacement
    xs = fields.coords()[0, 0, :, :, -1]
    ys = fields.coords()[0, 1, :, :, -1]

    # Apply rigid body correction if enabled
    if REMOVE_RIGID_BODY:
        disp_x, disp_y = remove_rigid_body(disp_x, disp_y, xs, ys)

    disp_mag = np.sqrt(disp_x**2 + disp_y**2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Total displacement magnitude
    ax = axes[0]
    ax.imshow(img2, cmap='gray', origin='upper', extent=(0, m, n, 0), aspect='equal')
    cf = ax.contourf(
        xs,
        ys,
        disp_mag,
        levels=50,
        cmap='jet',
        alpha=HEATMAP_ALPHA,
    )
    plt.colorbar(cf, ax=ax, shrink=0.8, label='Displacement magnitude [px]')
    ax.set_title("Total Displacement Magnitude", fontsize=12, fontweight='bold')
    ax.set_xlim(0, m)
    ax.set_ylim(n, 0)

    # Vector field panel
    ax = axes[1]
    ax.imshow(img2, cmap='gray', origin='upper', extent=(0, m, n, 0), aspect='equal')

    # Normalize every nonzero vector to a fixed display length. Arrow direction
    # remains physical, while color represents the actual displacement magnitude.
    valid_vectors = np.isfinite(disp_mag) & (disp_mag > np.finfo(float).eps)
    normalized_x = np.zeros_like(disp_x)
    normalized_y = np.zeros_like(disp_y)
    normalized_x[valid_vectors] = disp_x[valid_vectors] / disp_mag[valid_vectors]
    normalized_y[valid_vectors] = disp_y[valid_vectors] / disp_mag[valid_vectors]
    display_x = normalized_x * VECTOR_LENGTH_PIXELS
    display_y = normalized_y * VECTOR_LENGTH_PIXELS

    skip = max(1, min(xs.shape) // VECTOR_GRID_TARGET)
    quiver = ax.quiver(
        xs[::skip, ::skip],
        ys[::skip, ::skip],
        display_x[::skip, ::skip],
        display_y[::skip, ::skip],
        disp_mag[::skip, ::skip],
        cmap='jet',
        alpha=1.0,
        edgecolor='black',
        linewidth=0.5,
        angles='xy',
        scale_units='xy',
        scale=1.0,
        width=0.006,
        headwidth=5.5,
        headlength=6.5,
        pivot='middle',
    )
    plt.colorbar(quiver, ax=ax, shrink=0.8, label='Displacement [px]')
    ax.quiverkey(
        quiver,
        X=0.74,
        Y=1.04,
        U=VECTOR_LENGTH_PIXELS,
        label=f'Normalized {VECTOR_LENGTH_PIXELS:.0f} px display length',
        labelpos='E',
    )

    ax.set_title(
        "Normalized Vector Directions\n(color = actual magnitude)",
        fontsize=12,
        fontweight='bold',
    )
    ax.set_xlim(0, m)
    ax.set_ylim(n, 0)

    param_text = (
        f"mesh={cfg['n_elx']}x{cfg['n_ely']} | deg_e=3, deg_n=3, Q4 | "
        f"ROI: X=[{Xc1},{Xc2}], Y=[{Yc1},{Yc2}]\n"
        f"tol={settings.tol:.0e}, maxit={settings.maxit}, "
        f"interp_order={settings.interpolation_order} | "
        f"normalized vector display length={VECTOR_LENGTH_PIXELS:.0f}px"
    )
    fig.suptitle(
        f"{cfg['label']} DIC Results ({cfg['n_elx']}x{cfg['n_ely']} mesh)\n"
        f"{correction_str}",
        fontsize=15,
        fontweight='bold',
    )
    fig.text(
        0.5,
        0.015,
        param_text,
        ha='center',
        va='bottom',
        fontsize=9,
        fontfamily='monospace',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            alpha=0.9,
            edgecolor='gray',
        ),
    )
    plt.tight_layout(rect=(0, 0.07, 1, 0.94))

plt.show()
