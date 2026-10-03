"""
Computes nail width/length from a segmentation polygon using its principal
axis, rather than an axis-aligned bounding box (which is wrong for rotated
or curved nails).
"""
from __future__ import annotations
import numpy as np


def principal_axis(polygon: np.ndarray):
    """Return (centroid, unit_axis_vector, unit_perp_vector) via PCA on the
    polygon vertices. axis_vector points along the nail's long (length)
    direction; perp_vector is the transverse (width) direction.
    """
    pts = np.asarray(polygon, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending eigenvalues; the largest-variance axis is last.
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)
    perp = np.array([-axis[1], axis[0]])
    return centroid, axis, perp


def measure_width_length(polygon: np.ndarray, mm_per_px: float,
                          n_slices: int = 40):
    """Project polygon onto its principal axis, slice it into bands along
    the length, and take the MAXIMUM transverse (width) extent across
    slices as the nail width. Length is the full extent along the axis.

    Returns dict: width_mm, length_mm, width_px, length_px, axis, centroid.
    """
    pts = np.asarray(polygon, dtype=np.float64)
    centroid, axis, perp = principal_axis(pts)
    centered = pts - centroid

    along = centered @ axis   # position along length axis
    across = centered @ perp  # position along width axis

    length_px = along.max() - along.min()

    # Slice along the length axis and compute local transverse extent,
    # to be robust to a handful of outlier vertices, then take the max
    # reliable transverse distance (per spec: not a single global bbox).
    edges = np.linspace(along.min(), along.max(), n_slices + 1)
    max_width_px = 0.0
    slice_widths = []
    for i in range(n_slices):
        mask = (along >= edges[i]) & (along <= edges[i + 1])
        if not np.any(mask):
            continue
        band_across = across[mask]
        w = band_across.max() - band_across.min()
        slice_widths.append(w)
        if w > max_width_px:
            max_width_px = w

    # Use the 95th percentile slice width as the "reliable" width, guarding
    # against a single noisy slice spiking the result.
    if slice_widths:
        reliable_width_px = float(np.percentile(slice_widths, 95))
    else:
        reliable_width_px = max_width_px

    return {
        "width_px": reliable_width_px,
        "length_px": length_px,
        "width_mm": round(reliable_width_px * mm_per_px, 2),
        "length_mm": round(length_px * mm_per_px, 2),
        "centroid": centroid,
        "axis": axis,
    }
