from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def _avg_time_bins(max_ew, max_uvw, longest_baseline_bins=2):
    return np.ceil(max_ew / max_uvw).astype(np.int32) * longest_baseline_bins

def baseline_average_scan(time, ant1, ant2, uvw, data, flag_row, max_uvw):
    # Find unique baselines
    baselines = np.stack([ant1, ant2], axis=1)
    ubl, inv = np.unique(baselines, return_inverse=True, axis=0)

    # Generate a mask for each unflagged row
    # containing the unique baseline
    bl_mask = np.arange(ubl.shape[0])[:, None] == inv[None, :]
    # Remove flagged data
    unflagged = flag_row == False # noqa
    bl_mask[:, :] &= unflagged[None, :]

    # Compute the maximum EW distance and
    # number of rows for each baseline
    max_ew_dist = np.empty(bl_mask.shape[0], dtype=uvw.dtype)
    nrows = np.empty(bl_mask.shape[0], dtype=np.int32)

    for i, mask in enumerate(bl_mask):
        max_ew_dist[i] = np.abs(uvw[mask, 0]).sum()
        nrows[i] = np.count_nonzero(mask)

    # Compute the average time bins for each baseline
    avg_time_bins = _avg_time_bins(max_ew_dist, max_uvw)
    # Clamp number of bins to number of rows
    avg_time_bins = np.minimum(avg_time_bins, nrows)
    # Number of output rows for each baseline
    out_nrows = nrows // avg_time_bins
    out_rem = nrows % avg_time_bins

    for mask, in_rows, out_rows, rem, bins in zip(bl_mask, nrows, out_nrows, out_rem, avg_time_bins):
        # No averaging required
        if out_rows == in_rows:
            continue

        bl_uvw = uvw[mask, ...]
        assert bl_uvw.shape[0] == in_rows

        tot_rows = out_rows if rem == 0 else out_rows + 1

        avg_uvw = np.empty((tot_rows, 3), dtype=uvw.dtype)
        avg_uvw[:out_rows, :] = bl_uvw[:out_rows*bins, :].reshape(out_rows, bins, 3).mean(axis=1)

        if rem > 0:
            avg_uvw[out_rows:, :] = bl_uvw[out_rows*bins:, :].mean(axis=0)

    return data
