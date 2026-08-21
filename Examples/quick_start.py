# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath
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

# --- AgedCell run ---
# Load images from AgedCell folder (fresh = reference, 100cycles = deformed)
import numpy as np
import imageio.v2 as imageio

# =============================================================================
# RIGID BODY CORRECTION FLAGS - Toggle these
# =============================================================================
PRE_ALIGN_IMAGES = False        # True = align images with feature matching before DIC
REMOVE_RIGID_BODY = True        # True = subtract best-fit rigid body (translation + rotation)
# Note: If PRE_ALIGN_IMAGES is True, REMOVE_RIGID_BODY is applied to the aligned result
# =============================================================================

path = r'../FromPaper/'
# These images are RGBA - convert to grayscale using luminance weights
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
    print(f"  Rigid body removed: tx={tx:.3f}px, ty={ty:.3f}px, θ={np.degrees(theta):.4f}°")

    rb_x = tx - theta * (ys - y_c)
    rb_y = ty + theta * (xs - x_c)
    return disp_x - rb_x, disp_y - rb_y


def align_images(img_ref, img_def):
    """Align deformed image to reference using ORB feature matching + RANSAC."""
    from skimage import img_as_ubyte
    from skimage.feature import ORB, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import EuclideanTransform, warp

    ref_u8 = img_as_ubyte((img_ref - img_ref.min()) / (img_ref.max() - img_ref.min()))
    def_u8 = img_as_ubyte((img_def - img_def.min()) / (img_def.max() - img_def.min()))

    detector = ORB(n_keypoints=500)
    detector.detect_and_extract(ref_u8)
    kp_ref, desc_ref = detector.keypoints, detector.descriptors

    detector.detect_and_extract(def_u8)
    kp_def, desc_def = detector.keypoints, detector.descriptors

    matches = match_descriptors(desc_ref, desc_def, cross_check=True)
    src = kp_ref[matches[:, 0]][:, ::-1]
    dst = kp_def[matches[:, 1]][:, ::-1]

    model, inliers = ransac(
        (dst, src), EuclideanTransform,
        min_samples=3, residual_threshold=2.0, max_trials=1000)

    print(f"  Alignment: tx={model.translation[0]:.3f}, ty={model.translation[1]:.3f}, "
          f"θ={np.degrees(model.rotation):.4f}° ({inliers.sum()}/{len(inliers)} inliers)")

    aligned = warp(img_def, model.inverse, output_shape=img_def.shape, preserve_range=True)
    return aligned


img1 = to_grayscale(np.array(imageio.imread(path + '1.tif'), dtype=float))
img2 = to_grayscale(np.array(imageio.imread(path + '2.png'), dtype=float))

# Crop to matching size (images may differ by a pixel or two)
min_rows = min(img1.shape[0], img2.shape[0])
min_cols = min(img1.shape[1], img2.shape[1])
img1 = img1[:min_rows, :min_cols]
img2 = img2[:min_rows, :min_cols]
print(f"Image shape: {img1.shape}")

# Pre-align images if flag is set
if PRE_ALIGN_IMAGES:
    print("Pre-aligning images with feature matching...")
    img2 = align_images(img1, img2)

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
# COMBINED VISUALIZATION - All meshes in a single figure
# =============================================================================
n_rows = len(all_results)
n_cols = 4  # X-disp, Y-disp, magnitude, vector field
fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows))
if n_rows == 1:
    axes = axes[np.newaxis, :]

n, m = img2.shape  # rows, cols of the deformed image

for i, (cfg, fields, settings) in enumerate(all_results):
    # Extract displacement components
    disp_x = fields.disp()[0, 0, :, :, -1]  # x-displacement
    disp_y = fields.disp()[0, 1, :, :, -1]  # y-displacement
    xs = fields.coords()[0, 0, :, :, -1]
    ys = fields.coords()[0, 1, :, :, -1]

    # Apply rigid body correction if enabled
    if REMOVE_RIGID_BODY:
        disp_x, disp_y = remove_rigid_body(disp_x, disp_y, xs, ys)

    disp_mag = np.sqrt(disp_x**2 + disp_y**2)  # magnitude

    plot_data = [
        (disp_x, "X-displacement"),
        (disp_y, "Y-displacement"),
        (disp_mag, "Magnitude (√(x²+y²))"),
    ]

    # Heatmap columns
    for j, (comp_data, comp_label) in enumerate(plot_data):
        ax = axes[i, j]

        # Show the deformed image (image 2) as background
        ax.imshow(img2, cmap='gray', origin='lower', extent=(0, m, 0, n), aspect='equal')

        # Overlay transparent displacement heatmap
        cf = ax.contourf(xs, ys, comp_data, levels=50, cmap='jet', alpha=0.5)
        plt.colorbar(cf, ax=ax, shrink=0.8, label='Displacement [px]')

        ax.set_title(f"{cfg['label']} - {comp_label}", fontsize=10, fontweight='bold')
        ax.set_xlim(0, m)
        ax.set_ylim(0, n)

        # Parameter text box
        param_text = (
            f"n_elx={cfg['n_elx']}, n_ely={cfg['n_ely']}\n"
            f"deg_e=3, deg_n=3, type=Q4\n"
            f"ROI: X=[{Xc1},{Xc2}], Y=[{Yc1},{Yc2}]\n"
            f"tol={settings.tol:.0e}, maxit={settings.maxit}\n"
            f"interp_order={settings.interpolation_order}"
        )
        props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray')
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', fontfamily='monospace', bbox=props)

    # Vector field column
    ax = axes[i, 3]
    ax.imshow(img2, cmap='gray', origin='lower', extent=(0, m, 0, n), aspect='equal')

    # Subsample for readable arrows
    skip = max(1, xs.shape[0] // 12)
    ax.quiver(xs[::skip, ::skip], ys[::skip, ::skip],
              disp_x[::skip, ::skip], disp_y[::skip, ::skip],
              disp_mag[::skip, ::skip], cmap='jet', alpha=0.8,
              angles='xy', scale_units='xy', scale=0.5)

    ax.set_title(f"{cfg['label']} - Vector Field", fontsize=10, fontweight='bold')
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)

    # Parameter text box for vector field
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', fontfamily='monospace', bbox=props)

correction_label = []
if PRE_ALIGN_IMAGES:
    correction_label.append("pre-aligned")
if REMOVE_RIGID_BODY:
    correction_label.append("rigid body removed")
correction_str = " + ".join(correction_label) if correction_label else "no correction"

fig.suptitle(f"DIC Displacement Field - Mesh Size Comparison\n"
             f"(overlay on deformed image | {correction_str})",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()

# =============================================================================
# FIGURE 2 - Fine mesh (8x8) only: magnitude + vector field
# =============================================================================
# Find the 8x8 result
fine_result = None
for cfg, fields, settings in all_results:
    if cfg['n_ely'] == 8 and cfg['n_elx'] == 8:
        fine_result = (cfg, fields, settings)
        break

if fine_result is not None:
    cfg, fields, settings = fine_result
    disp_x = fields.disp()[0, 0, :, :, -1]
    disp_y = fields.disp()[0, 1, :, :, -1]
    xs = fields.coords()[0, 0, :, :, -1]
    ys = fields.coords()[0, 1, :, :, -1]

    if REMOVE_RIGID_BODY:
        disp_x, disp_y = remove_rigid_body(disp_x, disp_y, xs, ys)

    disp_mag = np.sqrt(disp_x**2 + disp_y**2)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Magnitude heatmap
    ax = axes2[0]
    ax.imshow(img2, cmap='gray', origin='lower', extent=(0, m, 0, n), aspect='equal')
    cf = ax.contourf(xs, ys, disp_mag, levels=50, cmap='jet', alpha=0.5)
    plt.colorbar(cf, ax=ax, shrink=0.8, label='|displacement| [px]')
    ax.set_title("Total Displacement Magnitude (8×8 mesh)", fontsize=12, fontweight='bold')
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)

    # Vector field
    ax = axes2[1]
    ax.imshow(img2, cmap='gray', origin='lower', extent=(0, m, 0, n), aspect='equal')
    skip = max(1, xs.shape[0] // 15)
    ax.quiver(xs[::skip, ::skip], ys[::skip, ::skip],
              disp_x[::skip, ::skip], disp_y[::skip, ::skip],
              disp_mag[::skip, ::skip], cmap='jet', alpha=0.8,
              angles='xy', scale_units='xy', scale=0.5)
    ax.set_title("Vector Field (8×8 mesh)", fontsize=12, fontweight='bold')
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)

    fig2.suptitle(f"Fine Mesh (8×8) DIC Results | {correction_str}",
                  fontsize=13, fontweight='bold')
    plt.tight_layout()

plt.show()
