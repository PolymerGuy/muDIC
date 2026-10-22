"""Register the mid-slice aging images and run rigid-corrected DIC.

Run from any directory:

    python Examples\align_mid_slices.py

The script:
1. estimates global rotation and translation,
2. aligns the after-aging image to the before-aging image,
3. crops both images to the largest rectangular valid overlap,
4. runs muDIC on the aligned images,
5. removes any residual best-fit translation and rotation, and
6. saves and displays registration and displacement figures.
"""

from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndi_shift
from skimage.registration import phase_cross_correlation
from skimage.transform import resize, rotate


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import muDIC as dic


# Input and output
INPUT_DIR = REPO_ROOT / "AgedCells" / "slices_for_aging" / "jpg_file_converted_by_FIJI"
REFERENCE_NAME = "mid_slice_before_aging.jpg"
DEFORMED_NAME = "mid_slice_after_aging.jpg"
OUTPUT_DIR = INPUT_DIR / "aligned_results"

# Registration parameters
SEARCH_MIN_DEG = -30.0
SEARCH_MAX_DEG = 30.0
COARSE_STEP_DEG = 0.5
FINE_HALF_RANGE_DEG = 0.75
FINE_STEP_DEG = 0.05
REGISTRATION_SIZE = 500
BACKGROUND_SIGMA = 20.0
UPSAMPLE_FACTOR = 10

# DIC parameters
N_ELX = 8
N_ELY = 8
ROI_MARGIN = 15
MAX_ITERATIONS = 20
TOLERANCE = 1.0e-6
INTERPOLATION_ORDER = 4
FIELD_UPSCALE = 10

# Plot parameters
HEATMAP_ALPHA = 0.50
VECTOR_SCALE = 0.5
VECTOR_GRID_TARGET = 18


def to_grayscale(image):
    """Convert an RGB/RGBA image to a floating-point grayscale array."""
    image = np.asarray(image, dtype=float)
    if image.ndim == 2:
        return image
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return np.dot(image[:, :, :3], [0.2989, 0.5870, 0.1140])


def normalize_for_registration(image, output_size):
    """Downsample and remove slowly varying illumination."""
    scaled = resize(
        image,
        (output_size, output_size),
        preserve_range=True,
        anti_aliasing=True,
    )
    high_pass = scaled - gaussian_filter(scaled, BACKGROUND_SIGMA)
    standard_deviation = np.std(high_pass)
    if standard_deviation == 0:
        raise ValueError("Registration image has no intensity variation")
    return high_pass / standard_deviation


def registration_error(reference, moving, correction_angle):
    """Return phase-correlation error and shift for one rotation candidate."""
    rotated = rotate(
        moving,
        correction_angle,
        resize=False,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    )
    shift_yx, error, _ = phase_cross_correlation(
        reference,
        rotated,
        upsample_factor=UPSAMPLE_FACTOR,
        normalization=None,
    )
    return float(error), shift_yx


def estimate_rigid_registration(reference, moving):
    """Estimate the rotation and translation needed to align moving to reference."""
    reference_small = normalize_for_registration(reference, REGISTRATION_SIZE)
    moving_small = normalize_for_registration(moving, REGISTRATION_SIZE)

    coarse_angles = np.arange(
        SEARCH_MIN_DEG,
        SEARCH_MAX_DEG + COARSE_STEP_DEG / 2.0,
        COARSE_STEP_DEG,
    )
    coarse_results = [
        (*registration_error(reference_small, moving_small, angle), angle)
        for angle in coarse_angles
    ]
    _, _, coarse_angle = min(coarse_results, key=lambda result: result[0])

    fine_angles = np.arange(
        coarse_angle - FINE_HALF_RANGE_DEG,
        coarse_angle + FINE_HALF_RANGE_DEG + FINE_STEP_DEG / 2.0,
        FINE_STEP_DEG,
    )
    fine_results = [
        (*registration_error(reference_small, moving_small, angle), angle)
        for angle in fine_angles
    ]
    error, _, correction_angle = min(fine_results, key=lambda result: result[0])

    rotated_full = rotate(
        moving,
        correction_angle,
        resize=False,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    )
    shift_yx, _, _ = phase_cross_correlation(
        reference,
        rotated_full,
        upsample_factor=UPSAMPLE_FACTOR,
        normalization=None,
    )
    return correction_angle, shift_yx, error


def apply_registration(image, correction_angle, shift_yx, cval=0.0):
    """Rotate about the image center, then translate."""
    rotated = rotate(
        image,
        correction_angle,
        resize=False,
        preserve_range=True,
        mode="constant",
        cval=cval,
    )
    return ndi_shift(
        rotated,
        shift=shift_yx,
        order=3,
        mode="constant",
        cval=cval,
        prefilter=True,
    )


def largest_true_rectangle(mask):
    """Find the largest axis-aligned rectangle containing only True pixels."""
    heights = np.zeros(mask.shape[1], dtype=int)
    best_area = 0
    best_bounds = None

    for bottom, row in enumerate(mask):
        heights = np.where(row, heights + 1, 0)
        stack = []

        for column in range(mask.shape[1] + 1):
            current_height = heights[column] if column < mask.shape[1] else 0
            start = column

            while stack and stack[-1][1] > current_height:
                left, height = stack.pop()
                area = height * (column - left)
                if area > best_area:
                    best_area = area
                    best_bounds = (
                        bottom - height + 1,
                        bottom + 1,
                        left,
                        column,
                    )
                start = left

            if not stack or stack[-1][1] < current_height:
                stack.append((start, current_height))

    if best_bounds is None:
        raise ValueError("Registration produced no valid overlapping rectangle")
    return best_bounds


def crop_to_valid_overlap(reference, moving, correction_angle, shift_yx):
    """Align moving and crop both images to their largest fully valid overlap."""
    aligned = apply_registration(moving, correction_angle, shift_yx)
    valid_mask = apply_registration(
        np.ones_like(moving),
        correction_angle,
        shift_yx,
    ) > 0.999
    top, bottom, left, right = largest_true_rectangle(valid_mask)
    return (
        reference[top:bottom, left:right],
        aligned[top:bottom, left:right],
        (top, bottom, left, right),
    )


def remove_residual_rigid_body(disp_x, disp_y, x_coords, y_coords):
    """Subtract the least-squares translation and infinitesimal rotation."""
    x_flat = x_coords.ravel()
    y_flat = y_coords.ravel()
    ux_flat = disp_x.ravel()
    uy_flat = disp_y.ravel()
    x_center = np.mean(x_flat)
    y_center = np.mean(y_flat)
    point_count = x_flat.size

    design = np.zeros((2 * point_count, 3))
    target = np.concatenate((ux_flat, uy_flat))
    design[:point_count, 0] = 1.0
    design[:point_count, 2] = -(y_flat - y_center)
    design[point_count:, 1] = 1.0
    design[point_count:, 2] = x_flat - x_center

    translation_x, translation_y, rotation_rad = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )[0]
    rigid_x = translation_x - rotation_rad * (y_coords - y_center)
    rigid_y = translation_y + rotation_rad * (x_coords - x_center)

    return (
        disp_x - rigid_x,
        disp_y - rigid_y,
        translation_x,
        translation_y,
        np.degrees(rotation_rad),
    )


def run_dic(reference, deformed):
    """Run DIC and return corrected displacement fields and coordinates."""
    images = dic.IO.image_stack_from_list([reference, deformed])
    rows, columns = reference.shape
    if rows <= 2 * ROI_MARGIN or columns <= 2 * ROI_MARGIN:
        raise ValueError("Valid overlap is too small for the configured ROI margin")

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
    disp_x = fields.disp()[0, 0, :, :, -1]
    disp_y = fields.disp()[0, 1, :, :, -1]
    x_coords = fields.coords()[0, 0, :, :, -1]
    y_coords = fields.coords()[0, 1, :, :, -1]
    return remove_residual_rigid_body(disp_x, disp_y, x_coords, y_coords) + (
        x_coords,
        y_coords,
    )


def save_registration_figure(reference, deformed, aligned, angle, shift_yx, bounds):
    """Display the input images and a transparent alignment overlay."""
    figure, axes = plt.subplots(1, 3, figsize=(17, 6))
    axes[0].imshow(reference, cmap="gray")
    axes[0].set_title("Before aging (reference)")
    axes[1].imshow(deformed, cmap="gray")
    axes[1].set_title("After aging (original)")
    axes[2].imshow(reference, cmap="gray")
    axes[2].imshow(aligned, cmap="magma", alpha=0.45)
    axes[2].set_title("Aligned overlay")

    for axis in axes:
        axis.axis("off")

    top, bottom, left, right = bounds
    figure.suptitle(
        "Global rigid registration\n"
        f"correction angle={angle:.3f} deg, "
        f"shift=(x={shift_yx[1]:.2f}, y={shift_yx[0]:.2f}) px, "
        f"crop=({left}:{right}, {top}:{bottom})"
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "mid_slice_registration.png", dpi=200)


def save_displacement_figure(
    deformed,
    disp_x,
    disp_y,
    x_coords,
    y_coords,
    residual_translation_x,
    residual_translation_y,
    residual_rotation_deg,
):
    """Display total displacement and its vector field."""
    magnitude = np.hypot(disp_x, disp_y)
    rows, columns = deformed.shape
    figure, axes = plt.subplots(1, 2, figsize=(15, 7))

    for axis in axes:
        axis.imshow(
            deformed,
            cmap="gray",
            origin="lower",
            extent=(0, columns, 0, rows),
        )
        axis.set_xlim(0, columns)
        axis.set_ylim(0, rows)

    contour = axes[0].contourf(
        x_coords,
        y_coords,
        magnitude,
        levels=50,
        cmap="jet",
        alpha=HEATMAP_ALPHA,
    )
    figure.colorbar(contour, ax=axes[0], label="Displacement magnitude [px]")
    axes[0].set_title("Residual total displacement")

    skip = max(1, min(x_coords.shape) // VECTOR_GRID_TARGET)
    vector = axes[1].quiver(
        x_coords[::skip, ::skip],
        y_coords[::skip, ::skip],
        disp_x[::skip, ::skip],
        disp_y[::skip, ::skip],
        magnitude[::skip, ::skip],
        cmap="jet",
        angles="xy",
        scale_units="xy",
        scale=VECTOR_SCALE,
    )
    figure.colorbar(vector, ax=axes[1], label="Displacement magnitude [px]")
    axes[1].set_title("Residual displacement vectors")

    parameters = (
        f"mesh={N_ELX}x{N_ELY}, margin={ROI_MARGIN}px\n"
        f"maxit={MAX_ITERATIONS}, tol={TOLERANCE:.0e}, "
        f"interpolation={INTERPOLATION_ORDER}\n"
        "residual rigid fit removed: "
        f"tx={residual_translation_x:.3f}px, "
        f"ty={residual_translation_y:.3f}px, "
        f"rotation={residual_rotation_deg:.4f} deg"
    )
    figure.suptitle("Rigid-corrected DIC after image registration\n" + parameters)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "mid_slice_displacement.png", dpi=200)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference = to_grayscale(imageio.imread(INPUT_DIR / REFERENCE_NAME))
    deformed = to_grayscale(imageio.imread(INPUT_DIR / DEFORMED_NAME))
    if reference.shape != deformed.shape:
        rows = min(reference.shape[0], deformed.shape[0])
        columns = min(reference.shape[1], deformed.shape[1])
        reference = reference[:rows, :columns]
        deformed = deformed[:rows, :columns]

    print(f"Image shape: {reference.shape}")
    print("Estimating global rigid registration...")
    angle, shift_yx, error = estimate_rigid_registration(reference, deformed)
    print(
        f"Registration correction: angle={angle:.3f} deg, "
        f"x shift={shift_yx[1]:.2f}px, y shift={shift_yx[0]:.2f}px, "
        f"normalized error={error:.4f}"
    )

    reference_crop, aligned_crop, bounds = crop_to_valid_overlap(
        reference,
        deformed,
        angle,
        shift_yx,
    )
    aligned_full = apply_registration(deformed, angle, shift_yx)
    print(f"Valid overlap crop: {reference_crop.shape}")

    imageio.imwrite(
        OUTPUT_DIR / "mid_slice_before_aligned_crop.tif",
        reference_crop.astype(np.float32),
    )
    imageio.imwrite(
        OUTPUT_DIR / "mid_slice_after_aligned_crop.tif",
        aligned_crop.astype(np.float32),
    )

    save_registration_figure(
        reference,
        deformed,
        aligned_full,
        angle,
        shift_yx,
        bounds,
    )

    print("Running DIC on registered, cropped images...")
    (
        disp_x,
        disp_y,
        residual_translation_x,
        residual_translation_y,
        residual_rotation_deg,
        x_coords,
        y_coords,
    ) = run_dic(reference_crop, aligned_crop)
    print(
        "Residual rigid correction: "
        f"tx={residual_translation_x:.3f}px, "
        f"ty={residual_translation_y:.3f}px, "
        f"rotation={residual_rotation_deg:.4f} deg"
    )

    save_displacement_figure(
        aligned_crop,
        disp_x,
        disp_y,
        x_coords,
        y_coords,
        residual_translation_x,
        residual_translation_y,
        residual_rotation_deg,
    )
    print(f"Results saved to: {OUTPUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
