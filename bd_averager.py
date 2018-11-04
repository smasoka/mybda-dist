from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys


def get_tables_spw(sc, spw, scan, desc, flag, A0, A1, vis_data, uvw=None, flag_row=None, interval=None, exposure=None, timestamps=None, time_centroid=None, weights=None):
    """
    """
    indexes_spw = np.where((scan == sc)&(desc == spw))[0]
    if uvw is not None:
        uvw = uvw[indexes_spw]
    if flag_row is not None:
        flag_row = flag_row[indexes_spw]
    flag = flag[indexes_spw]
    A0 = A0[indexes_spw]
    A1 = A1[indexes_spw]
    if interval is not None:
        interval = interval[indexes_spw]
    if exposure is not None:
        exposure = exposure[indexes_spw]
    if timestamps is not None:
        timestamps = timestamps[indexes_spw]
    if time_centroid is not None:
        time_centroid = time_centroid[indexes_spw]
    vis_data = vis_data[indexes_spw]
    if weights is not None:
        weights = weights[indexes_spw]

    return uvw, flag_row, flag, A0, A1, interval, exposure,\
        timestamps, time_centroid, vis_data, weights


def remove_flagged_row(uvw, flag_row, flag, A0, A1, interval, exposure, timestamps, time_centroid, vis_data, weights):
    """
    """
    indexes_flagged_row = np.where(flag_row == False)
    uvw_spw = uvw[indexes_flagged_row]
    flag_row_spw = flag_row[indexes_flagged_row]
    flag_spw = flag[indexes_flagged_row]
    A0_spw = A0[indexes_flagged_row]
    A1_spw = A1[indexes_flagged_row]
    interval_spw = interval[indexes_flagged_row]
    exposure_spw = exposure[indexes_flagged_row]
    timestamps_spw = timestamps[indexes_flagged_row]
    time_centroid_spw = time_centroid[indexes_flagged_row]
    vis_data_spw = vis_data[indexes_flagged_row]
    weight_spw = weights[indexes_flagged_row]

    return uvw_spw, flag_row_spw, flag_spw, A0_spw, A1_spw, interval_spw, exposure_spw,\
        timestamps_spw, time_centroid_spw, vis_data_spw, weight_spw


def remove_flagged_channels(flagpq, vis_datapq):
    """
    """
    vis = vis_datapq[:, np.where(flagpq[0, :, 0] == False), :]
    ntime, _, nfreq, ncorr = vis.shape

    return vis.reshape(ntime, nfreq, ncorr)


def get_longest_ew_distance(uvw, A0, A1):
    """
    """
    na = np.max(A0)+1
    max_baseline = 0.
    for p in range(na):
        for q in range(p+1, na):
            uvwpq = uvw[(A0 == p) & (A1 == q)]
            baseline_dist = np.abs(uvwpq[:, 0]).sum(axis=0)
            max_baseline = baseline_dist if baseline_dist > max_baseline else max_baseline
    return max_baseline


def get_longest_baseline(uvw):
    """
    """
    max_baseline = np.sqrt((uvw**2).sum(axis=1)).max()
    return max_baseline


def evaluate_cf_over_longuest_ew_or_baseline(uvw, longest_ew_distance, max_baseline):
    """
    """
    if max_baseline is None:

        baseline_dist = np.abs(uvw[:, 0]).sum(axis=0)
    else:
        baseline_dist = np.sqrt((uvw**2).sum(axis=1))[0]

    distance_compression_ration = int(longest_ew_distance//baseline_dist)
    nbr_av_vispq = distance_compression_ration

    return nbr_av_vispq


def get_tables_baseline(A0, A1, p, q, flag_row, vis_data, weights=None, interval=None, exposure=None, timestamps=None, time_centroid=None, uvw=None, flag=None):
    """
    """
    indicepq = (A0 == p) & (A1 == q)
    flag_row = flag_row[indicepq]
    vis_data = vis_data[indicepq]
    if weights is not None:
        weights = weights[indicepq]
    if time_centroid is not None:
        time_centroid = time_centroid[indicepq]
    if interval is not None:
        interval = interval[indicepq]
    if exposure is not None:
        exposure = exposure[indicepq]
    if timestamps is not None:
        timestamps = timestamps[indicepq]
    if uvw is not None:
        uvw = uvw[indicepq]
    if flag is not None:
        flag = flag[indicepq]

    return flag_row, vis_data, weights, interval, exposure, timestamps, time_centroid, uvw, flag


def get_averaging_blocks(index_head, nbr_timebins, uvw, flag, interval, exposure, timestamps, time_centroid, vis_data, weights):
    """
    """
    time_slice = slice(index_head, index_head+nbr_timebins)
    uvwpq_av_block = uvw[time_slice, ...]
    flagpq_av_block = flag[time_slice, ...]
    intervalpq_av_block = interval[time_slice]
    exposurepq_av_block = exposure[time_slice]
    timestampspq_av_block = timestamps[time_slice]
    time_centroidpq_av_block = time_centroid[time_slice]
    vis_datapq_av_block = vis_data[time_slice, ...]
    weightpq_av_block = weights[time_slice, ...]

    return uvwpq_av_block, flagpq_av_block, intervalpq_av_block, exposurepq_av_block, timestampspq_av_block, time_centroidpq_av_block, vis_datapq_av_block, weightpq_av_block


def do_bd_averaging_time(uvw, interval, exposure, timestamps, time_centroid, weights, flag, vis_data, nbr_av_vispq=None, nbr_timebins=None):
    """
    """
    # number of correlation and frequency for this baseline
    nbr_vis, nbr_freqs, nbr_corr = vis_data.shape
    if nbr_timebins and nbr_av_vispq is not None:
        # reshape the defferent tables accordingly
        uvwpq_av_block = uvw.reshape(nbr_timebins, nbr_av_vispq, 3)
        flagpq_av_block = flag.reshape(
            nbr_timebins, nbr_av_vispq, nbr_freqs, nbr_corr)
        intervalpq_av_block = interval.reshape(nbr_timebins, nbr_av_vispq)
        exposurepq_av_block = exposure.reshape(nbr_timebins, nbr_av_vispq)
        timestampspq_av_block = timestamps.reshape(nbr_timebins, nbr_av_vispq)
        time_centroidpq_av_block = time_centroid.reshape(
            nbr_timebins, nbr_av_vispq)
        vis_datapq_av_block = vis_data.reshape(
            nbr_timebins, nbr_av_vispq, nbr_freqs, nbr_corr)
        weightpq_av_block = weights.reshape(
            nbr_timebins, nbr_av_vispq, nbr_corr)

        # work out the averaging requierement for the tables

        uvwpq_av_block = uvwpq_av_block.mean(axis=1)
        intervalpq_av_block = intervalpq_av_block.sum(axis=1)
        exposurepq_av_block = exposurepq_av_block.sum(axis=1)
        timestampspq_av_block = timestampspq_av_block.mean(axis=1)
        time_centroidpq_av_block = time_centroidpq_av_block.mean(axis=1)
        weightpq_av_block = weightpq_av_block.mean(axis=1)
        vis_datapq_av_block = np.ma.masked_where(
            flagpq_av_block == True, vis_datapq_av_block)
        vis_datapq_av_block = vis_datapq_av_block.mean(axis=1)

        #
        if isinstance(vis_datapq_av_block.mask, np.ndarray):
            # the flaggin arrays is equal to the mask in dataBDA see function: np.ma.masked_where
            flagpq_av_block = vis_datapq_av_block.mask
        else:
            # there is no flagging entries, contruct flagging array contaiing "False" everywher
            flagpq_av_block = np.zeros_like(
                vis_datapq_av_block.data, dtype=bool)

        vis_datapq_av_block = vis_datapq_av_block.data
    else:
        uvwpq_av_block = uvw.mean(axis=0)[np.newaxis, ...]
        intervalpq_av_block = [interval.sum(axis=0)]
        exposurepq_av_block = [exposure.sum(axis=0)]
        timestampspq_av_block = [timestamps.mean(axis=0)]
        time_centroidpq_av_block = [time_centroid.mean(axis=0)]
        weightpq_av_block = weights.mean(axis=0)[np.newaxis, ...]
        vis_datapq_av_block = np.ma.masked_where(flag == True, vis_data)
        vis_datapq_av_block = vis_datapq_av_block.mean(axis=0)[np.newaxis, ...]

        if isinstance(vis_datapq_av_block.mask, np.ndarray):
            # the flaggin arrays is equal to the mask in dataBDA see function: np.ma.masked_where
            flagpq_av_block = vis_datapq_av_block.mask
        else:
            # there is no flagging entries, contruct flagging array contaiing "False" everywhere
            flagpq_av_block = np.zeros_like(
                vis_datapq_av_block.data, dtype=bool)

        vis_datapq_av_block = vis_datapq_av_block.data

    return uvwpq_av_block, intervalpq_av_block, exposurepq_av_block, timestampspq_av_block, time_centroidpq_av_block, weightpq_av_block, flagpq_av_block, vis_datapq_av_block


def do_bd_averaging_freq(flag, vis_data, nbr_av_vispq=None, nbr_freqbins=None):
    """
    """
    # number of correlation and frequency for this baseline
    nbr_vis, nbr_freqs, nbr_corr = vis_data.shape
    if nbr_freqbins and nbr_av_vispq is not None:
        # reshape the defferent tables accordingly
        flagpq_av_block = flag.reshape(
            nbr_vis, nbr_freqbins, nbr_av_vispq, nbr_corr)
        vis_datapq_av_block = vis_data.reshape(
            nbr_vis, nbr_freqbins, nbr_av_vispq, nbr_corr)

        # work out the averaging requierement for the tables
        vis_datapq_av_block = np.ma.masked_where(
            flagpq_av_block == True, vis_datapq_av_block)
        vis_datapq_av_block = vis_datapq_av_block.mean(axis=2)
        #
        if isinstance(vis_datapq_av_block.mask, np.ndarray):
            # the flaggin arrays is equal to the mask in dataBDA see function: np.ma.masked_where
            flagpq_av_block = vis_datapq_av_block.mask
        else:
            # there is no flagging entries, contruct flagging array contaiing "False" everywher
            flagpq_av_block = np.zeros_like(
                vis_datapq_av_block.data, dtype=bool)

        vis_datapq_av_block = vis_datapq_av_block.data
    else:
        vis_datapq_av_block = np.ma.masked_where(flag == True, vis_data)
        vis_datapq_av_block = vis_datapq_av_block.mean(axis=1)[
            :, np.newaxis, ...]

        if isinstance(vis_datapq_av_block.mask, np.ndarray):
            # the flaggin arrays is equal to the mask in dataBDA see function: np.ma.masked_where
            flagpq_av_block = vis_datapq_av_block.mask
        else:
            # there is no flagging entries, contruct flagging array contaiing "False" everywhere
            flagpq_av_block = np.zeros_like(
                vis_datapq_av_block.data, dtype=bool)

        vis_datapq_av_block = vis_datapq_av_block.data

    return flagpq_av_block, vis_datapq_av_block


# ===============
# MAIN FUNCTIONS
# ===============


def bda_time(scan, desc, uvw, flag_row, flag, A0, A1, interval, exposure, timestamps, time_centroid,
             vis_data, weights, nbr_bins_av_longest_baseline=None, starttimebin_index=0, nbr_timebins=None, ew=False):
    """ baseline dependent averaging in time  

        Parameters
        ----------

        desc : 
        vis_data : array of complex, shape (num_vis_samples, num_channels, num_pols)
                Array containing complex visibility data in Janskys
        desc: 
        uvw : array of float, shape (num_vis_samples, 3)
                Array containing (u,v,w) coordinates in metres
        flag_row : 
        flag : array of boolean, shape same as vis_data
        A0:
        A1:
        interval:
        exposure:
        timestamps:
        time_centroid:
        weights:


        Return
        ------

        desc : 
        vis_data : array of complex, shape (num_vis_samples, num_channels, num_pols)
                Array containing complex visibility data in Janskys
        desc: 
        uvw : array of float, shape (num_vis_samples, 3)
                Array containing (u,v,w) coordinates in metres
        flag_row : 
        flag : array of boolean, shape same as vis_data
        A0:
        A1:
        interval:
        exposure:
        timestamps:
        time_centroid:
        weights:



        Raises
        ------

        exit : 
        KeyError :      

    """

    if ew is False:
        max_baseline = get_longest_baseline(uvw)
    else:
        max_baseline = None

    # initialise output arrays
    data_scan_id_output = data_desc_id_output = a0_output = a1_output = np.array([], dtype=A0.dtype)

    flag_row_output = flag_output = np.array([], dtype=flag_row.dtype)
    uvw_output = interval_output = exposure_output = timestamps_output = time_centroid_output = vis_data_output = weight_output = np.array([
    ])
    # get the indexes of spectral window for this observation
    index_spw = list(set(desc))
    # get the indexes of antennas for this configuration
    na = np.max(A0)+1
    # get the number of scan for this observation
    index_scan = list(set(scan))
    # iterate across the number of band for this observation
    for sc in index_scan:
     for spw in index_spw:
        # get all the informations (tables) for this band
        uvw_spw, flag_row_spw, flag_spw, A0_spw, A1_spw, interval_spw, exposure_spw, timestamps_spw, time_centroid_spw, vis_data_spw, weight_spw = \
            get_tables_spw(sc, spw, scan, desc, flag, A0, A1, vis_data, uvw, flag_row,
                           interval, exposure, timestamps, time_centroid, weights)

        # get the indexes for the timeslots flagged for this band and
        # remove these timeslots from the informations (tables) for this band
        uvw_spw, flag_row_spw, flag_spw, A0_spw, A1_spw, interval_spw, exposure_spw, timestamps_spw, time_centroid_spw, vis_data_spw, weight_spw = \
            remove_flagged_row(uvw_spw, flag_row_spw, flag_spw, A0_spw, A1_spw, interval_spw,
                               exposure_spw, timestamps_spw, time_centroid_spw, vis_data_spw, weight_spw)

        # evaluate the longest east west baseline for this band, remember the longest east west baseline can be different in the different bands but one can still use the lonest east west baseline for a single band as an approximation
        longest_ew_distance = max_baseline or get_longest_ew_distance(
            uvw_spw, A0_spw, A1_spw)

        # loop across all the baselines
        for p in range(na+1):
            for q in range(p+1, na+1):
                # get uvw for this baseline
                uvwpq = uvw_spw[(A0_spw == p) & (A1_spw == q)]
                # get the time number of bins to average for this baseline
                nbr_av_vispq = evaluate_cf_over_longuest_ew_or_baseline(
                    uvwpq, longest_ew_distance, max_baseline) * nbr_bins_av_longest_baseline
                # check shape compatibilities
                nbr_timeslotspq = nbr_timebins or uvwpq.shape[0] - \
                    starttimebin_index

                if nbr_av_vispq > nbr_timeslotspq:
                   print(" the number of time bins %d to average for baseline  (%d, %d) is larger than the total number of timeslots  %d. All the bins for this baseline have been compressed to one " % (
                        nbr_av_vispq, p, q, nbr_timeslotspq))
		   nbr_av_vispq = nbr_timeslotspq

                # printing information
                from time import gmtime, strftime
                print (" %s +++ BDA: Baseline (%d,%d), Ntime bin avg = %d " % (
                    strftime("%Y-%m-%d %H:%M:%S", gmtime()), p, q, nbr_av_vispq))

                # this is the output shape for all the tables for this baseline
                nbr_timeslots_outputpq = nbr_timeslotspq//nbr_av_vispq

                nbr_timeslots_actual = nbr_timeslots_outputpq*nbr_av_vispq
                nbr_timeslots_remaining = nbr_timeslotspq - \
                    nbr_timeslots_actual if nbr_timeslotspq-nbr_timeslots_actual != 0 else 0

                # make resulting ANTENNA1/ANTENNA2 indices
                a0 = np.zeros(nbr_timeslots_outputpq, dtype=A0.dtype)
                a1 = np.zeros(nbr_timeslots_outputpq, dtype=A0.dtype)
                a0[:] = p
                a1[:] = q

                # get information (tables) for this baseline
                flag_rowpq, vis_datapq, weightpq, intervalpq, exposurepq, timestampspq, time_centroidpq, _, _ = get_tables_baseline(
                    A0_spw, A1_spw, p, q, flag_spw, vis_data_spw, weight_spw, interval_spw, exposure_spw, timestamps_spw, time_centroid_spw)

                # work out new uvw for bda
                uvwpq_av_block, flag_rowpq_av_block, intervalpq_av_block, exposurepq_av_block, timestampspq_av_block, time_centroidpq_av_block, vis_datapq_av_block, weightpq_av_block = \
                    get_averaging_blocks(starttimebin_index, nbr_timeslots_actual, uvwpq, flag_rowpq,
                                         intervalpq, exposurepq, timestampspq, time_centroidpq, vis_datapq, weightpq)

                uvwpq_av_block, intervalpq_av_block, exposurepq_av_block, timestampspq_av_block, time_centroidpq_av_block, weightpq_av_block, flagpq_av_block, vis_datapq_av_block = \
                    do_bd_averaging_time(uvwpq_av_block, intervalpq_av_block, exposurepq_av_block, timestampspq_av_block,
                                         time_centroidpq_av_block, weightpq_av_block, flag_rowpq_av_block, vis_datapq_av_block, nbr_av_vispq, nbr_timeslots_outputpq)

                if nbr_timeslots_remaining > 0:
                    uvwpq_av_block_r, flag_rowpq_av_block_r, intervalpq_av_block_r, exposurepq_av_block_r, timestampspq_av_block_r, time_centroidpq_av_block_r, vis_datapq_av_block_r, weightpq_av_block_r = \
                        get_averaging_blocks(starttimebin_index+nbr_timeslots_actual, nbr_timeslots_remaining, uvwpq,
                                             flag_rowpq, intervalpq, exposurepq, timestampspq, time_centroidpq, vis_datapq, weightpq)

                    uvwpq_av_block_r, intervalpq_av_block_r, exposurepq_av_block_r, timestampspq_av_block_r, time_centroidpq_av_block_r, weightpq_av_block_r, flagpq_av_block_r, vis_datapq_av_block_r = \
                        do_bd_averaging_time(uvwpq_av_block_r, intervalpq_av_block_r, exposurepq_av_block_r, timestampspq_av_block_r,
                                             time_centroidpq_av_block_r, weightpq_av_block_r, flag_rowpq_av_block_r, vis_datapq_av_block_r)

                    # concatenation of the two set of visibilities
                    uvwpq_av_block = np.append(
                        uvwpq_av_block, uvwpq_av_block_r, axis=0)
                    intervalpq_av_block = np.append(
                        intervalpq_av_block, intervalpq_av_block_r, axis=0)
                    exposurepq_av_block = np.append(
                        exposurepq_av_block,  exposurepq_av_block_r, axis=0)
                    timestampspq_av_block = np.append(
                        timestampspq_av_block,  timestampspq_av_block_r, axis=0)
                    time_centroidpq_av_block = np.append(
                        time_centroidpq_av_block, time_centroidpq_av_block_r, axis=0)
                    weightpq_av_block = np.append(
                        weightpq_av_block, weightpq_av_block_r, axis=0)
                    vis_datapq_av_block = np.append(
                        vis_datapq_av_block, vis_datapq_av_block_r, axis=0)
                    flagpq_av_block = np.append(
                        flagpq_av_block, flagpq_av_block_r, axis=0)
                    a0 = np.append(a0, [p], axis=0)
                    a1 = np.append(a1, [q], axis=0)

                # work out output tables
		data_scan_id_output = np.append(
		    data_scan_id_output, np.ones_like(a0)*sc)
                data_desc_id_output = np.append(
                    data_desc_id_output, np.ones_like(a0)*spw)
                flag_row_output = np.append(flag_row_output, np.zeros(
                    intervalpq_av_block.shape, dtype=flag_row.dtype))
                uvw_output = np.append(
                    uvw_output, uvwpq_av_block, axis=0) if uvw_output.size else uvwpq_av_block
                flag_output = np.append(
                    flag_output, flagpq_av_block, axis=0) if flag_output.size else flagpq_av_block
                a0_output = np.append(a0_output, a0, axis=0)
                a1_output = np.append(a1_output, a1, axis=0)
                interval_output = np.append(
                    interval_output, intervalpq_av_block, axis=0)
                exposure_output = np.append(
                    exposure_output, exposurepq_av_block, axis=0)
                timestamps_output = np.append(
                    timestamps_output, timestampspq_av_block, axis=0)
                time_centroid_output = np.append(
                    time_centroid_output, time_centroidpq_av_block, axis=0)
                vis_data_output = np.append(
                    vis_data_output, vis_datapq_av_block, axis=0) if vis_data_output.size else vis_datapq_av_block
                weight_output = np.append(
                    weight_output, weightpq_av_block, axis=0) if weight_output.size else weightpq_av_block

    return data_scan_id_output, data_desc_id_output, flag_row_output, uvw_output, flag_output, a0_output, a1_output, interval_output, exposure_output, timestamps_output, time_centroid_output, vis_data_output, weight_output


def bda_freq(scan, desc, uvw, flag_row, flag, A0, A1, interval, exposure, timestamps, time_centroid, vis_data, weights, chan_width, nbr_bins_av_longest_baseline=None, startfreqbin_index=0, nbr_freqbins=None, ew=False):
    """
    """

    if ew is False:
        max_baseline = get_longest_baseline(uvw)
    else:
        max_baseline = None

    # initialise output arrays
    data_desc_id_output = data_scan_id_output = a0_output = a1_output = np.array([], dtype=A0.dtype)

    flag_output = np.array([], dtype=flag.dtype)
    vis_data_output = np.array([])
    index_spw = list(set(desc))
    # get the indexes of antennas for this configuration
    na = np.max(A0)+1
    # get the number of scan for this observation
    index_scan = list(set(scan))

    # prepare the result output
    output_dict = {}
    # iterate across the number of scan and band for this observation
    for sc in index_scan:
     # output for this scan
     output_result_sc = []  
     scan_dict = {}
     for spw in index_spw:
        # get all the informations (tables) for this band
        uvw_spw, flag_row_spw, flag_spw, A0_spw, A1_spw, interval_spw, exposure_spw, timestamps_spw, time_centroid_spw, vis_data_spw, weight_spw = \
            get_tables_spw(sc, spw, scan, desc, flag, A0, A1, vis_data, uvw, flag_row,
                           interval, exposure, timestamps, time_centroid, weights)

        # evaluate the longest east west baseline for this band, remember the longest east west baseline can be different in the different bands but one can still use the lonest east west baseline for a single band as an approximation
        longest_ew_distance = max_baseline or get_longest_ew_distance(
            uvw_spw, A0_spw, A1_spw)
        # get the channels width for this spectral window
        chan_width_spw = chan_width[spw, 0]
        
        # loop across all the baselines
        for p in range(na+1):
            for q in range(p+1, na+1):
		
		subtab_dict = {}
                # get information (tables) for this baseline
                flag_rowpq, vis_datapq, weightpq, intervalpq, exposurepq, timestampspq, time_centroidpq, uvwpq, flagpq = get_tables_baseline(
                    A0_spw, A1_spw, p, q, flag_row_spw, vis_data_spw, weight_spw, interval_spw, exposure_spw, timestamps_spw, time_centroid_spw, uvw_spw, flag_spw)

                # get the time number of bins to average for this baseline
                nbr_av_vispq = evaluate_cf_over_longuest_ew_or_baseline(
                    uvwpq, longest_ew_distance, max_baseline) * nbr_bins_av_longest_baseline

                # remove flag along the frequency axis, note that all flags were removed along the time axis, this was not important for bda in frequency but still we wanna clean up the data and free space
                # vis_datapq = remove_flagged_channels(flagpq, vis_datapq)

                # check shape compatibilities along frequency. This may be done just after the spw loop but after flagging channels are removed this may be different at each baseline
                nbr_freqpq = nbr_freqbins or vis_datapq.shape[1] - \
                    startfreqbin_index

                if nbr_av_vispq > nbr_freqpq:
		    print(" The number of frequency bins %d to average for baseline  (%d, %d) is larger than the total number of unflagged channels for this baseline  %d. all the samples are averaged to 1 " % (
                        nbr_av_vispq, p, q, nbr_freqpq))
		    nbr_av_vispq = nbr_freqpq

                # printing information
                from time import gmtime, strftime
                print (" %s +++ BDA: Baseline (%d,%d), Ntime freq bin avg = %d " % (
                    strftime("%Y-%m-%d %H:%M:%S", gmtime()), p, q, nbr_av_vispq))

                # this is the output shape for after frequency BDA
                nbr_freq_outputpq = nbr_freqpq//nbr_av_vispq

                nbr_freq_actual = nbr_freq_outputpq*nbr_av_vispq
                nbr_freq_remaining = nbr_freqpq - \
                    nbr_freq_actual if nbr_freqpq-nbr_freq_actual != 0 else 0

                # get the block data to average,
                freq_slice = slice(startfreqbin_index,
                                   startfreqbin_index+nbr_freq_actual)

                vis_datapq_av_block = vis_datapq[:, freq_slice, ...]
                flagpq_av_block = flagpq[:, freq_slice, ...]
                
                # do average in freq
                flagpq_av_block, vis_datapq_av_block = do_bd_averaging_freq(
                    flagpq_av_block, vis_datapq_av_block, nbr_av_vispq, nbr_freq_outputpq)

                if nbr_freq_remaining > 0:
                    freq_slice = slice(startfreqbin_index+nbr_freq_actual,
                                       startfreqbin_index+nbr_freq_actual+nbr_freq_remaining)

                    vis_datapq_av_block_r = vis_datapq[:, freq_slice, ...]

                    flagpq_av_block_r = flagpq[:, freq_slice, ...]

                    flagpq_av_block_r, vis_datapq_av_block_r = do_bd_averaging_freq(
                        flagpq_av_block_r, vis_datapq_av_block_r)

                    # concatenation of the two set of visibilities
                    vis_datapq_av_block = np.append(
                        vis_datapq_av_block, vis_datapq_av_block_r, axis=1)
                    flagpq_av_block = np.append(
                        flagpq_av_block, flagpq_av_block_r, axis=1)
                # get the new channels width for this compression factor
                chan_widthpq = chan_width_spw*nbr_av_vispq
                # work out output tables

		
		a0 = np.zeros(vis_datapq_av_block.shape[0], dtype=np.int32)
         	a1 = np.zeros_like(a0)
         	a0[:] = p
                a1[:] = q
                #print("*********************\n")
                #print(nbr_av_vispq, vis_datapq_av_block.shape)
                
                if nbr_av_vispq in scan_dict:		   
	   	   subtab_dict['A0'] = np.append(
                            scan_dict[nbr_av_vispq]['A0'], a0, axis=0)
                   subtab_dict['A1'] = np.append(
                            scan_dict[nbr_av_vispq]['A1'], a1, axis=0)
                   
                   subtab_dict['DATA'] = np.append(scan_dict[nbr_av_vispq]['DATA'], vis_datapq_av_block, axis=0) if scan_dict[nbr_av_vispq]['DATA'].size else vis_datapq_av_block
                   subtab_dict['FLAG'] = np.append(scan_dict[nbr_av_vispq]['FLAG'], flagpq_av_block, axis=0) if scan_dict[nbr_av_vispq]['FLAG'].size else flagpq_av_block

                   subtab_dict['FLAG_ROW'] = np.append(
                            scan_dict[nbr_av_vispq]['FLAG_ROW'], flag_rowpq, axis=0)
                   subtab_dict['UVW'] = np.append(
                            scan_dict[nbr_av_vispq]['UVW'], uvwpq, axis=0) if scan_dict[nbr_av_vispq]['UVW'].size else uvwpq
                   subtab_dict['INTERVAL'] = np.append(
                            scan_dict[nbr_av_vispq]['INTERVAL'], intervalpq, axis=0)
                   subtab_dict['EXPOSURE'] = np.append(
                            scan_dict[nbr_av_vispq]['EXPOSURE'], exposurepq, axis=0)
                   subtab_dict['TIME'] = np.append(
                            scan_dict[nbr_av_vispq]['TIME'], timestampspq, axis=0)
                   subtab_dict['TIME_CENTROID'] = np.append(
                            scan_dict[nbr_av_vispq]['TIME_CENTROID'], time_centroidpq, axis=0)
                   subtab_dict['WEIGHT'] = np.append(
                            scan_dict[nbr_av_vispq]['WEIGHT'], weightpq, axis=0) if scan_dict[nbr_av_vispq]['WEIGHT'].size else weightpq
 		   scan_dict[nbr_av_vispq] = subtab_dict
		else:
	            subtab_dict['A0'] = a0
                    subtab_dict['A1'] = a1
                    subtab_dict['DATA'] = vis_datapq_av_block
                    subtab_dict['FLAG'] = flagpq_av_block
                    subtab_dict['FLAG_ROW'] = flag_rowpq
                    subtab_dict['UVW'] = uvwpq
                    subtab_dict['INTERVAL'] = intervalpq
                    subtab_dict['EXPOSURE'] = exposurepq
                    subtab_dict['TIME'] = timestampspq
                    subtab_dict['TIME_CENTROID'] = time_centroidpq
                    subtab_dict['WEIGHT'] = weightpq
                    scan_dict[nbr_av_vispq] = subtab_dict
                   		    
     # for outpout for this scan
     output_dict[sc] = scan_dict
     
    return output_dict;
