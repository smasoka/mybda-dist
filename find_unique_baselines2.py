import argparse

import dask
import dask.array as da
import numpy as np
from xarrayms import xds_from_ms


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    return p

args = create_parser().parse_args()


# Find unique baselines in this Measurement Set
# Get data from the Measurement Set
xds = list(xds_from_ms(args.ms,
                       # We only need the antenna columns
                       columns=("ANTENNA1", "ANTENNA2"),
                       group_cols=[],
                       index_cols=[],
                       chunks={"row": 1e6}))

# Should only have one dataset
assert len(xds) == 1
ds = xds[0]

# Need ant1 and ant2 to be int32 for the compound int64 below
# to work
assert ds.ANTENNA1.dtype == ds.ANTENNA2.dtype == np.int32

bl = da.stack([ds.ANTENNA1.data, ds.ANTENNA2.data], axis=1)
# convert array to dtype int64 from int32
bl = bl.rechunk(-1, 2).view(np.int64)
# get the unique values
ubl = da.unique(bl)
# dask compute, convert back to int32 and reshape
ubl = da.compute(ubl)[0].view(np.int32).reshape(-1, 2)

print "%s unique baselines" % ubl.shape[0]
# We have unique baselines

# Now find number of averaged rows per scan, per baseline
# Get data from the Measurement Set again
xds = list(xds_from_ms(args.ms,
                       columns=["TIME", "ANTENNA1", "ANTENNA2",
                                "UVW", "FLAG_ROW"],
                       group_cols=["SCAN_NUMBER"],
                       index_cols=[],
                       chunks={"row": 1e9}))


def _averaging_rows(time, ant1, ant2, uvw, flagged_rows, ubl=None):
    # TODO(smasoka)
    # Understand this
    # Need to index passed argument lists,
    # as atop contracts over dimensions not present in the output.
    ant1 = ant1[0]
    ant2 = ant2[0]
    uvw = uvw[0][0]
    flagged_rows = flagged_rows[0]

    print ant1.shape
    print ant2.shape
    print uvw.shape
    print flagged_rows.shape

    # Create empty array container with shape of the unique baselines
    baseline_avg_rows = np.empty(ubl.shape[0], dtype=np.int32)
    print baseline_avg_rows.shape
    print

    unflagged = flagged_rows is False
    print unflagged
    print type(unflagged)

    print "UBL"
    print "UBL SHAPE " +str(ubl.shape)
    print "UBL SIZE " +str(ubl.size)
    print

    # Foreach baseline
    for bl, (a1, a2) in enumerate(ubl):
        print "bl : %s, a1 : %s, a2 : %s" %(bl, a1, a2)
        # Find rows associated with each baseline
        # also removing flagged rows
        valid_rows = (ant1 == a1) & (ant2 == a2) & unflagged
        # depending on the unflagged, valid_rows can be reduced, smaller
        print "VALID_ROWS " +str(valid_rows.shape)
        # print
        # Maximum EW distance for each baseline
        bl_max_ew = np.abs(uvw[valid_rows, 0]).sum(axis=0)
        print "MAX EW DISTANCE " +str(bl_max_ew)

        # Figure out what the averaged number of rows will be
        # I think np.divmod is the way to do this
        # baseline_avg_rows[i] = np.divmod(valid_rows)

    return baseline_avg_rows

scan_baseline_avg_rows = []

# Print the xarray Dataset
print xds[0]
print xds[1]

print "Before atop for loop"
for ds in xds:
    # calls _averaging_rows
    # output block pattern ("bl",) --> array of baselines
    # TIME data (row array)
    # ANTENNA1 data (row array)
    # ANTENNA2 data (row array)
    # UVW data (row array and all columns) ???
    # FLAG_ROW data (row array)
    # creates a new_axis for the number of baselines
    # pass the actual array of unique baselines
    # dtype of results
    avg_rows = da.core.atop(_averaging_rows, ("bl",),
                            ds.TIME.data, ("row",),
                            ds.ANTENNA1.data, ("row",),
                            ds.ANTENNA2.data, ("row",),
                            ds.UVW.data, ("row", "(u,v,w)"),
                            ds.FLAG_ROW.data, ("row",),
                            # Must tell dask about the number of baselines
                            new_axes={"bl": ubl.shape[0]},
                            # Pass through ubl to all instances of
                            ubl=ubl,
                            dtype=np.int32)

    scan_baseline_avg_rows.append(avg_rows)



dask.compute(scan_baseline_avg_rows)
