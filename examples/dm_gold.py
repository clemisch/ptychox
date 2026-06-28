"""Single-mode Difference Map reconstruction of the cdtools gold-ball data.

This is a standalone ptychox example.  It reads the CXI file directly with
h5py; cdtools and PyTorch are not required.
"""

from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import ptychox as px


data_path = Path(
    "/home/clem/git/ptychox/ext/cdtools/"
    "examples/example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi"
)

# Reconstruction settings
n_iterations = 500
scan_stride = 1
fixed_probe_iterations = 15
beta = 1.0
random_seed = 0
plot_every = 5
show_plot = True
output_path = Path("dm_gold_result.npz")


def amplitude_residual(obj, probe, amplitudes, shifts):
    exit_waves = px.dm.get_exit_waves(obj, probe, shifts)
    predicted = jnp.abs(jax.vmap(px.prop.to_farfield)(exit_waves))
    return jnp.linalg.norm(predicted - amplitudes) / jnp.linalg.norm(amplitudes)


def percentile_limits(image):
    lo, hi = np.nanpercentile(np.asarray(image), [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi <= lo:
        delta = max(abs(float(lo)) * 1e-6, 1e-6)
        return lo - delta, hi + delta
    return float(lo), float(hi)


if scan_stride < 1:
    raise ValueError("scan_stride must be at least 1")

with h5py.File(data_path, "r") as cxi:
    detector = cxi["entry_1/instrument_1/detector_1"]
    patterns = detector["data"][()].astype(np.float32)
    n_frames = len(patterns)
    dark = detector["data_dark"][()].mean(axis=0, dtype=np.float32)

    # CXI stores sample translations; ptychox and cdtools use motion of
    # the illumination relative to the sample, hence the minus sign.
    translations = -cxi[
        "entry_1/sample_1/geometry_1/translation"
    ][()].astype(np.float64)

    energy = float(cxi["entry_1/instrument_1/source_1/energy"][()])
    wavelength = 1.9864459e-25 / energy
    distance = float(detector["distance"][()])

    if "basis_vectors" in detector:
        detector_basis = detector["basis_vectors"][()]
        if detector_basis.shape == (2, 3):
            detector_basis = detector_basis.T
    else:
        dx = float(detector["x_pixel_size"][()])
        dy = float(detector["y_pixel_size"][()])
        detector_basis = np.array(
            [[0, -dy, 0], [-dx, 0, 0]], dtype=np.float64
        ).T

# Select every Nth nominal raster coordinate along both scan axes.
selected = np.ones(n_frames, dtype=bool)
for axis in range(2):
    values = np.round(translations[:, axis], decimals=12)
    ranks = np.searchsorted(np.unique(values), values)
    selected &= ranks % scan_stride == 0
patterns = patterns[selected]
translations = translations[selected]

# The dataset has no detector mask.  Treat the mean recorded dark as a
# fixed background estimate and clip negative photon estimates.
intensities = np.maximum(patterns - dark[None], 0)
amplitudes = np.sqrt(intensities, dtype=np.float32)
probe_shape = amplitudes.shape[-2:]

# This reproduces cdtools.initializers.exit_wave_geometry.
detector_extent_basis = detector_basis * np.asarray(
    probe_shape, dtype=np.float64
)
object_basis = (
    np.linalg.pinv(detector_extent_basis).T * wavelength * distance
)
positions = translations @ np.linalg.pinv(object_basis).T

padding = 4
position_min = positions.min(axis=0) - padding
position_range = positions.max(axis=0) - positions.min(axis=0)
object_shape = (
    np.ceil(position_range).astype(np.int32)
    + np.asarray(probe_shape, dtype=np.int32)
    + 2 * padding
)

# cdtools addresses crops by their upper-left corner.  ptychox addresses
# them as offsets from the centered crop, with opposite sign.
crop_starts = positions - position_min
centered_start = object_shape // 2 - np.asarray(probe_shape) // 2
shifts = centered_start[None] - crop_starts

# Follow the cdtools example's raster-grid-pathology workaround.
rng = np.random.default_rng(random_seed)
shifts = shifts.astype(np.float32)
shifts -= rng.normal(scale=0.7, size=shifts.shape).astype(np.float32)

pixel_sizes = np.linalg.norm(object_basis, axis=0)

# cdtools' SHARP-style probe guess, including the 2 micrometre propagation
# and radius-50 support used in gold_ball_ptycho.py.
mean_intensity = np.mean(np.square(amplitudes), axis=0)
probe = px.prop.from_farfield(
    jnp.asarray(np.sqrt(mean_intensity), dtype=jnp.complex64)
)
kernel = px.prop.get_kernel_angular_spectrum(
    probe.shape, float(pixel_sizes.mean()), wavelength, 2e-6
)
probe = px.prop.to_nearfield(probe, kernel)

yy, xx = np.ogrid[:probe.shape[0], :probe.shape[1]]
radius = np.sqrt(
    (yy - (probe.shape[0] - 1) / 2) ** 2
    + (xx - (probe.shape[1] - 1) / 2) ** 2
)
probe = (probe * jnp.asarray(radius < 50)).astype(jnp.complex64)

object_shape = tuple(int(n) for n in object_shape)
obj = jnp.ones(object_shape, dtype=jnp.complex64)
amplitudes = jnp.asarray(amplitudes)
shifts = jnp.asarray(shifts)
psi = px.dm.get_exit_waves(obj, probe, shifts)

print(f"frames: {len(amplitudes)} of {n_frames}")
print(f"detector/probe shape: {probe_shape}")
print(f"object shape: {object_shape}")
print(f"object pixel size: {pixel_sizes.mean() * 1e9:.3f} nm")
print(f"initial residual: {float(amplitude_residual(obj, probe, amplitudes, shifts)):.4f}")

fig, axes = plt.subplots(2, 2, figsize=(9, 8))
images = []
for axis, image, title, cmap in zip(
    axes.ravel(),
    (jnp.abs(obj), jnp.angle(obj), jnp.abs(probe), jnp.angle(probe)),
    ("|object|", "arg(object)", "|probe|", "arg(probe)"),
    ("gray", "twilight", "gray", "twilight"),
):
    lo, hi = percentile_limits(image)
    artist = axis.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
    images.append(artist)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
fig.tight_layout()

for iteration in tqdm(range(n_iterations)):
    update_probe = iteration >= fixed_probe_iterations
    psi, obj, probe = px.dm.step(
        psi,
        obj,
        probe,
        amplitudes,
        shifts,
        beta=beta,
        update_probe=update_probe,
    )

    if images is not None and (iteration + 1) % plot_every == 0:
        for artist, image in zip(
            images,
            (jnp.abs(obj), jnp.angle(obj), jnp.abs(probe), jnp.angle(probe)),
        ):
            artist.set_data(image)
            artist.set_clim(*percentile_limits(image))
        fig.suptitle(
            f"DM iteration {iteration + 1}"
            f" ({'blind probe' if update_probe else 'fixed probe'})"
        )
        fig.canvas.draw_idle()
        plt.pause(1e-3)

residual = float(amplitude_residual(obj, probe, amplitudes, shifts))
print(f"final residual: {residual:.4f}")
np.savez_compressed(
    output_path,
    object=np.asarray(obj),
    probe=np.asarray(probe),
    shifts=np.asarray(shifts),
    object_basis=object_basis,
    dark=dark,
    residual=residual,
)
print(f"saved: {output_path}")

if show_plot:
    plt.show()
