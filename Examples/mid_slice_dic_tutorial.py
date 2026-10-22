r"""Create a beginner-friendly visual guide to the mid-slice DIC workflow.

Run from the repository root:

    .\.venv\Scripts\python.exe Examples\mid_slice_dic_tutorial.py

The script creates two figures:
1. global image registration and valid-overlap preparation;
2. DIC correlation, rigid-body removal, and final deformation.
"""

from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rotate


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import muDIC as dic
from align_mid_slices import (
    INPUT_DIR,
    REFERENCE_NAME,
    DEFORMED_NAME,
    apply_registration,
    crop_to_valid_overlap,
    estimate_rigid_registration,
    remove_residual_rigid_body,
    to_grayscale,
)


OUTPUT_DIR = INPUT_DIR / "tutorial_results"

# DIC settings used for the teaching example
N_ELX = 8
N_ELY = 8
ROI_MARGIN = 15
MAX_ITERATIONS = 20
TOLERANCE = 1.0e-6
INTERPOLATION_ORDER = 4
FIELD_UPSCALE = 10

# Display settings. Vector lengths are normalized for visibility; their colors
# retain the measured displacement magnitude.
HEATMAP_ALPHA = 0.50
VECTOR_DISPLAY_LENGTH = 100.0
VECTOR_GRID_TARGET = 12


def add_step_title(axis, step, title):
    """Add a consistent numbered title to a tutorial panel."""
    axis.set_title(
        f"STEP {step}\n{title}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )


def show_image(axis, image, title, step=None):
    """Show an image using its native top-left coordinate system."""
    axis.imshow(image, cmap="gray", origin="upper")
    if step is None:
        axis.set_title(title, fontsize=12, fontweight="bold")
    else:
        add_step_title(axis, step, title)
    axis.axis("off")


def normalized_vectors(disp_x, disp_y):
    """Return fixed-length display vectors and their physical magnitudes."""
    magnitude = np.hypot(disp_x, disp_y)
    valid = np.isfinite(magnitude) & (magnitude > np.finfo(float).eps)
    vector_x = np.zeros_like(disp_x)
    vector_y = np.zeros_like(disp_y)
    vector_x[valid] = disp_x[valid] / magnitude[valid]
    vector_y[valid] = disp_y[valid] / magnitude[valid]
    return (
        vector_x * VECTOR_DISPLAY_LENGTH,
        vector_y * VECTOR_DISPLAY_LENGTH,
        magnitude,
    )


def plot_heatmap(axis, background, x_coords, y_coords, values, title, step):
    """Overlay a displacement heatmap in image coordinates."""
    rows, columns = background.shape
    axis.imshow(
        background,
        cmap="gray",
        origin="upper",
        extent=(0, columns, rows, 0),
    )
    contour = axis.contourf(
        x_coords,
        y_coords,
        values,
        levels=50,
        cmap="jet",
        alpha=HEATMAP_ALPHA,
    )
    plt.colorbar(contour, ax=axis, shrink=0.78, label="Displacement [px]")
    add_step_title(axis, step, title)
    axis.set_xlim(0, columns)
    axis.set_ylim(rows, 0)


def plot_vectors(axis, background, x_coords, y_coords, disp_x, disp_y, title, step):
    """Plot normalized vector directions colored by actual magnitude."""
    rows, columns = background.shape
    display_x, display_y, magnitude = normalized_vectors(disp_x, disp_y)
    skip = max(1, min(x_coords.shape) // VECTOR_GRID_TARGET)

    axis.imshow(
        background,
        cmap="gray",
        origin="upper",
        extent=(0, columns, rows, 0),
    )
    vectors = axis.quiver(
        x_coords[::skip, ::skip],
        y_coords[::skip, ::skip],
        display_x[::skip, ::skip],
        display_y[::skip, ::skip],
        magnitude[::skip, ::skip],
        cmap="jet",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="middle",
        width=0.006,
        headwidth=5.5,
        headlength=6.5,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.colorbar(vectors, ax=axis, shrink=0.78, label="Actual magnitude [px]")
    add_step_title(axis, step, title)
    axis.set_xlim(0, columns)
    axis.set_ylim(rows, 0)


def load_images():
    """Load and crop the source images to a common shape."""
    reference = to_grayscale(imageio.imread(INPUT_DIR / REFERENCE_NAME))
    deformed = to_grayscale(imageio.imread(INPUT_DIR / DEFORMED_NAME))
    rows = min(reference.shape[0], deformed.shape[0])
    columns = min(reference.shape[1], deformed.shape[1])
    return reference[:rows, :columns], deformed[:rows, :columns]


def run_dic(reference, deformed):
    """Correlate the registered images and return raw field data."""
    images = dic.IO.image_stack_from_list([reference, deformed])
    rows, columns = reference.shape

    mesher = dic.Mesher(deg_e=3, deg_n=3, type="q4")
    mesh = mesher.mesh(
        images,
        Xc1=ROI_MARGIN,
        Xc2=columns - ROI_MARGIN,
        Yc1=ROI_MARGIN,
        Yc2=rows - ROI_MARGIN,
        n_ely=N_ELY,
        n_elx=N_ELX,
        GUI=False,
    )

    settings = dic.DICInput(mesh, images)
    settings.max_nr_im = 2
    settings.maxit = MAX_ITERATIONS
    settings.tol = TOLERANCE
    settings.interpolation_order = INTERPOLATION_ORDER
    settings.store_internals = True
    settings.noconvergence = "ignore"

    results = dic.DICAnalysis(settings).run()
    fields = dic.post.viz.Fields(results, upscale=FIELD_UPSCALE)
    return (
        fields.disp()[0, 0, :, :, -1],
        fields.disp()[0, 1, :, :, -1],
        fields.coords()[0, 0, :, :, -1],
        fields.coords()[0, 1, :, :, -1],
    )


def create_registration_figure(
    reference,
    deformed,
    rotated,
    aligned,
    reference_crop,
    aligned_crop,
    angle,
    shift_yx,
    error,
    crop_bounds,
):
    """Explain how global capture misalignment is removed."""
    figure, axes = plt.subplots(2, 3, figsize=(17, 11))
    axes = axes.ravel()

    show_image(axes[0], reference, "Reference: before aging", 1)
    axes[0].text(
        0.5,
        -0.06,
        "This is the coordinate system used for all later results.",
        transform=axes[0].transAxes,
        ha="center",
    )

    show_image(axes[1], deformed, "Moving image: after aging", 2)
    axes[1].text(
        0.5,
        -0.06,
        "It contains material deformation plus capture rotation/translation.",
        transform=axes[1].transAxes,
        ha="center",
    )

    show_image(axes[2], rotated, "Correct global rotation", 3)
    axes[2].text(
        0.5,
        -0.06,
        f"Applied correction: {angle:.3f} deg",
        transform=axes[2].transAxes,
        ha="center",
    )

    axes[3].imshow(reference, cmap="gray", origin="upper")
    axes[3].imshow(aligned, cmap="magma", origin="upper", alpha=0.45)
    add_step_title(axes[3], 4, "Correct translation and verify overlay")
    axes[3].axis("off")
    axes[3].text(
        0.5,
        -0.06,
        f"Shift: x={shift_yx[1]:.2f}px, y={shift_yx[0]:.2f}px; "
        f"registration error={error:.4f}",
        transform=axes[3].transAxes,
        ha="center",
    )

    show_image(axes[4], reference_crop, "Crop reference to valid overlap", 5)
    show_image(axes[5], aligned_crop, "Crop aligned moving image", 6)
    top, bottom, left, right = crop_bounds
    axes[5].text(
        0.5,
        -0.06,
        f"Common analysis area: {right-left}x{bottom-top} pixels",
        transform=axes[5].transAxes,
        ha="center",
    )

    figure.suptitle(
        "PART 1 - Remove global capture misalignment before DIC",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Only a rigid transform is used here: rotation + translation. "
        "No scaling, shear, or local deformation is removed.",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    figure.savefig(OUTPUT_DIR / "01_registration_workflow.png", dpi=200)


def create_dic_figure(
    reference,
    deformed,
    raw_x,
    raw_y,
    corrected_x,
    corrected_y,
    rigid_x,
    rigid_y,
    x_coords,
    y_coords,
    translation_x,
    translation_y,
    rotation_deg,
):
    """Explain correlation, residual rigid fitting, and final deformation."""
    raw_magnitude = np.hypot(raw_x, raw_y)
    rigid_magnitude = np.hypot(rigid_x, rigid_y)
    corrected_magnitude = np.hypot(corrected_x, corrected_y)
    figure, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    rows, columns = reference.shape
    axes[0].imshow(
        reference,
        cmap="gray",
        origin="upper",
        extent=(0, columns, rows, 0),
    )
    axes[0].plot(x_coords, y_coords, color="cyan", linewidth=0.35, alpha=0.7)
    axes[0].plot(
        np.swapaxes(x_coords, 0, 1),
        np.swapaxes(y_coords, 0, 1),
        color="cyan",
        linewidth=0.35,
        alpha=0.7,
    )
    add_step_title(axes[0], 7, "Place the DIC mesh on the overlap")
    axes[0].set_xlim(0, columns)
    axes[0].set_ylim(rows, 0)
    axes[0].text(
        0.5,
        -0.08,
        f"{N_ELX}x{N_ELY} Q4 elements; subsets are correlated between images.",
        transform=axes[0].transAxes,
        ha="center",
    )

    plot_heatmap(
        axes[1],
        deformed,
        x_coords,
        y_coords,
        raw_magnitude,
        "Calculate raw displacement",
        8,
    )
    axes[1].text(
        0.5,
        -0.08,
        "Raw DIC still contains any small remaining global motion.",
        transform=axes[1].transAxes,
        ha="center",
    )

    plot_vectors(
        axes[2],
        deformed,
        x_coords,
        y_coords,
        raw_x,
        raw_y,
        "Inspect raw displacement directions",
        9,
    )

    plot_vectors(
        axes[3],
        deformed,
        x_coords,
        y_coords,
        rigid_x,
        rigid_y,
        "Fit residual rigid motion",
        10,
    )
    axes[3].text(
        0.5,
        -0.08,
        f"Best fit: tx={translation_x:.3f}px, ty={translation_y:.3f}px, "
        f"rotation={rotation_deg:.4f} deg",
        transform=axes[3].transAxes,
        ha="center",
    )

    plot_heatmap(
        axes[4],
        deformed,
        x_coords,
        y_coords,
        corrected_magnitude,
        "Subtract rigid motion",
        11,
    )
    axes[4].text(
        0.5,
        -0.08,
        "Remaining magnitude is interpreted as local material deformation.",
        transform=axes[4].transAxes,
        ha="center",
    )

    plot_vectors(
        axes[5],
        deformed,
        x_coords,
        y_coords,
        corrected_x,
        corrected_y,
        "Report final deformation vectors",
        12,
    )
    axes[5].text(
        0.5,
        -0.08,
        f"Mean={np.mean(corrected_magnitude):.3f}px; "
        f"maximum={np.max(corrected_magnitude):.3f}px",
        transform=axes[5].transAxes,
        ha="center",
    )

    figure.suptitle(
        "PART 2 - Correlate the registered images and isolate deformation",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        f"DIC settings: mesh={N_ELX}x{N_ELY}, margin={ROI_MARGIN}px, "
        f"max iterations={MAX_ITERATIONS}, tolerance={TOLERANCE:.0e}, "
        f"interpolation order={INTERPOLATION_ORDER}. "
        f"Displayed arrows are normalized to {VECTOR_DISPLAY_LENGTH:.0f}px; "
        "color shows actual magnitude.",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    figure.savefig(OUTPUT_DIR / "02_dic_workflow.png", dpi=200)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference, deformed = load_images()

    print("Step 1: estimating global rotation and translation...")
    angle, shift_yx, error = estimate_rigid_registration(reference, deformed)
    rotated = rotate(
        deformed,
        angle,
        resize=False,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    )
    aligned = apply_registration(deformed, angle, shift_yx)
    reference_crop, aligned_crop, crop_bounds = crop_to_valid_overlap(
        reference,
        deformed,
        angle,
        shift_yx,
    )

    create_registration_figure(
        reference,
        deformed,
        rotated,
        aligned,
        reference_crop,
        aligned_crop,
        angle,
        shift_yx,
        error,
        crop_bounds,
    )

    print("Step 2: running DIC on the registered overlap...")
    raw_x, raw_y, x_coords, y_coords = run_dic(reference_crop, aligned_crop)
    (
        corrected_x,
        corrected_y,
        translation_x,
        translation_y,
        rotation_deg,
    ) = remove_residual_rigid_body(raw_x, raw_y, x_coords, y_coords)
    rigid_x = raw_x - corrected_x
    rigid_y = raw_y - corrected_y

    create_dic_figure(
        reference_crop,
        aligned_crop,
        raw_x,
        raw_y,
        corrected_x,
        corrected_y,
        rigid_x,
        rigid_y,
        x_coords,
        y_coords,
        translation_x,
        translation_y,
        rotation_deg,
    )

    print(
        f"Registration: rotation={angle:.3f} deg, "
        f"x shift={shift_yx[1]:.2f}px, y shift={shift_yx[0]:.2f}px"
    )
    print(
        f"Residual rigid fit: tx={translation_x:.3f}px, "
        f"ty={translation_y:.3f}px, rotation={rotation_deg:.4f} deg"
    )
    print(f"Tutorial figures saved to: {OUTPUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
