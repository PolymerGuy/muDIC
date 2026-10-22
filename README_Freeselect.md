# Freeselect DIC Workflow

This document describes the image-registration, Digital Image Correlation
(DIC), rigid-body correction, image-enhancement, and teaching workflows added
to this branch.

The main workflow is intended for two microscopy images of the same slice:

- a reference image captured before aging;
- a deformed image captured after aging.

The scripts first remove global capture misalignment (rotation and
translation), then use muDIC to estimate local material displacement.

## 1. Requirements

### Python environment

From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If TIFF files use LZW or another compressed TIFF encoding, also install:

```powershell
python -m pip install imagecodecs
```

The optional `noise` package is only used for synthetic Perlin speckle
generation. A message saying that `noise` is unavailable does not prevent
registration or DIC analysis of existing images.

### Input images

The default mid-slice workflow expects:

```text
AgedCells/
  slices_for_aging/
    jpg_file_converted_by_FIJI/
      mid_slice_before_aging.jpg
      mid_slice_after_aging.jpg
```

The scripts accept grayscale, RGB, or RGBA images. RGB and RGBA images are
converted to grayscale before registration and correlation.

The image data and generated result folders are experiment-specific and should
normally not be committed to the repository.

## 2. Recommended workflow

Run the following scripts in this order.

### Step 1: Register and crop the images

```powershell
.\.venv\Scripts\python.exe Examples\align_mid_slices.py
```

`align_mid_slices.py`:

1. loads the before-aging image as the reference;
2. searches for the global rotation that best aligns the after-aging image;
3. estimates the remaining X/Y translation using phase correlation;
4. applies rotation and translation to the after-aging image;
5. finds the largest rectangular region containing valid pixels in both
   images;
6. crops both images to this common overlap;
7. runs an 8x8 DIC check;
8. fits and removes any small residual rigid translation and rotation.

Important editable parameters are near the top of the file:

| Parameter | Purpose |
|---|---|
| `SEARCH_MIN_DEG`, `SEARCH_MAX_DEG` | Rotation search interval |
| `COARSE_STEP_DEG` | Initial rotation-search resolution |
| `FINE_STEP_DEG` | Final rotation-search resolution |
| `REGISTRATION_SIZE` | Downsampled registration image size |
| `N_ELX`, `N_ELY` | DIC mesh element counts |
| `ROI_MARGIN` | Distance between the DIC region and image edges |
| `MAX_ITERATIONS` | Maximum solver iterations |
| `TOLERANCE` | DIC convergence tolerance |

Outputs are written to:

```text
AgedCells/slices_for_aging/jpg_file_converted_by_FIJI/aligned_results/
```

Generated files:

| File | Contents |
|---|---|
| `mid_slice_before_aligned_crop.tif` | Cropped reference image |
| `mid_slice_after_aligned_crop.tif` | Registered and cropped after-aging image |
| `mid_slice_registration.png` | Before, after, and alignment-overlay report |
| `mid_slice_displacement.png` | Rigid-corrected displacement magnitude and vectors |

The terminal also reports:

- correction rotation in degrees;
- X and Y translation in pixels;
- registration error;
- valid overlap dimensions;
- residual rigid translation and rotation measured after DIC.

### Step 2: Compare different DIC mesh sizes

After running `align_mid_slices.py`, run:

```powershell
.\.venv\Scripts\python.exe Examples\quick_start.py
```

`quick_start.py` reads the two corrected TIFF files from `aligned_results` and
runs four mesh configurations:

- 2x2 elements;
- 4x4 elements;
- 8x8 elements;
- 12x12 elements.

For each configuration it:

1. runs DIC on the registered image pair;
2. calculates X and Y displacement;
3. fits the residual best-fit rigid translation and rotation;
4. subtracts this rigid field;
5. calculates total displacement:

   ```text
   magnitude = sqrt(x_displacement^2 + y_displacement^2)
   ```

6. opens a separate two-panel figure containing:
   - total displacement magnitude as a transparent `jet` heatmap;
   - normalized displacement directions as vectors.

The vector arrows are normalized to a fixed display length so small
displacements remain visible. Arrow direction shows the measured displacement
direction, while arrow color shows the actual displacement magnitude in
pixels. The displayed arrow length is therefore not the physical magnitude.

Editable visualization parameters:

| Parameter | Purpose |
|---|---|
| `HEATMAP_ALPHA` | Heatmap transparency |
| `VECTOR_LENGTH_PIXELS` | Display length of normalized arrows |
| `VECTOR_GRID_TARGET` | Approximate vector density |

The plots use the native image coordinate convention:

- origin at the top-left;
- positive X points right;
- positive Y points down.

### Step 3: Create a teaching report

```powershell
.\.venv\Scripts\python.exe Examples\mid_slice_dic_tutorial.py
```

This script creates a beginner-friendly explanation of the complete workflow
using 12 numbered panels.

The first figure explains:

1. the before-aging reference;
2. the uncorrected after-aging image;
3. rotation correction;
4. translation correction and overlay verification;
5. reference overlap cropping;
6. aligned-image overlap cropping.

The second figure explains:

7. DIC mesh placement;
8. raw displacement magnitude;
9. raw displacement vectors;
10. the fitted residual rigid-body field;
11. rigid-corrected displacement magnitude;
12. final local-deformation vectors.

Outputs are written to:

```text
AgedCells/slices_for_aging/jpg_file_converted_by_FIJI/tutorial_results/
```

Generated files:

| File | Contents |
|---|---|
| `01_registration_workflow.png` | Registration and overlap-cropping tutorial |
| `02_dic_workflow.png` | Correlation and rigid-body-removal tutorial |

The figures report all major parameters and measured transformations.

## 3. Optional image enhancement

Run:

```powershell
.\.venv\Scripts\python.exe Examples\enhance_images.py
```

Before running, set `IMAGE_PATH` near the top of the script.

The script compares:

- CLAHE local-contrast enhancement;
- non-local means denoising;
- unsharp-mask sharpening;
- background illumination normalization;
- percentile histogram stretching;
- all methods applied in sequence.

All enhancement strengths are editable near the top of the file. The script
opens a before/after comparison figure. Uncomment the output lines at the
bottom to save the combined enhanced image.

Image enhancement should be applied consistently to both DIC images. Applying
different nonlinear processing to the reference and deformed images can
degrade correlation or create artificial displacement patterns.

## 4. Optional rigid-body-method comparison

Run:

```powershell
.\.venv\Scripts\python.exe Examples\rigid_body_comparison.py
```

This older comparison script uses paths configured through:

```python
IMAGE_PATH = r'../FromPaper/'
IMAGE_1 = '1.tif'
IMAGE_2 = '2.png'
```

Edit these values before use with another image pair.

It compares:

1. raw DIC displacement;
2. mean-displacement subtraction (translation only);
3. least-squares rigid-body subtraction (translation and rotation);
4. feature-based image pre-alignment followed by DIC.

The output figure contains displacement magnitude, vector fields, and summary
statistics for each method.

For the mid-slice aging images, the intensity-registration method in
`align_mid_slices.py` is preferred. Feature matching can fail when aging
substantially changes the local image texture.

## 5. Interpretation of the corrections

There are two separate rigid-body corrections:

### Global image registration

This correction is performed before DIC. It removes camera/sample
misalignment between captures:

- global rotation;
- global X translation;
- global Y translation.

It does not intentionally remove scaling, shear, or local deformation.

### Residual DIC rigid-body removal

After DIC, a least-squares rigid field is fitted to the measured displacement:

```text
u_x = translation_x - rotation * (y - center_y)
u_y = translation_y + rotation * (x - center_x)
```

This fitted field is subtracted from the raw DIC result. The remaining field is
interpreted as local deformation.

If genuine material motion includes a global rotation, subtracting the fitted
rigid field will also remove that component. The raw and corrected results
should therefore both be reviewed.

## 6. Mesh convergence and solver warnings

A finer mesh provides higher spatial resolution but also:

- contains fewer pixels per element;
- is more sensitive to noise and weak texture;
- may converge less reliably;
- takes longer to solve.

Compare the 2x2, 4x4, 8x8, and 12x12 results. A useful result should remain
qualitatively stable as the mesh is refined. If a fine mesh develops large,
irregular changes or reports large final increments, it has probably not
converged reliably.

The scripts currently use:

```python
settings.noconvergence = "ignore"
```

This allows visualization even when the strict tolerance is not reached. A
"Did not converged" message means that the output exists, not that it is
automatically trustworthy. Review:

- the reported final increment;
- consistency between mesh sizes;
- registration overlay quality;
- displacement smoothness;
- whether the image texture is suitable for correlation.

## 7. Other repository change

`setup.py` now opens `README.md` explicitly with UTF-8 encoding. This avoids a
Windows `UnicodeDecodeError` when installing the project in editable mode.

