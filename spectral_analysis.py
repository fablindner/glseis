from obspy import read, UTCDateTime, Stream, Trace
import numpy as np
from obspy.signal import PPSD
import matplotlib.pyplot as plt
from matplotlib import mlab
import matplotlib.dates as mdates
import datetime as dtime
import warnings
import numpy.ma as ma
import sys


def medianPSD(stream, win_len, overlap, t1, t2):
    """
    Calculates the median PSD of data contained in stream. Single PSDs are calculated
    from overlapping windows of length 'win_len'.

    :type stream: ObsPy stream object. Must contain exactly on Trace
    :param stream: ObsPy stream that contains data to process and stats information
    :type win_len: float
    :param win_len: window length in seconds used for calculating single PSDs
    :type overlap: float
    :param overlap: overlap of sliding window in percent
    :type t1, t2: ObsPy UTCDateTime
    :param t1, t2: starttime and entime of the window of consideration. The timestamp
        returned is t1 + (t2 - t1) / 2.
    :rtype: numpy array; numpy array; float
    :return: frequency vector to median PSD; median PSD; ObsPy timestamp corresponting
        to stime + (etime - stime) / 2, where stime and etime are the starttime and
        endtime of the stream, respectivley
    """
    dt = stream[0].stats.delta
    fs = stream[0].stats.sampling_rate
    start = stream[0].stats.starttime 
    end = stream[0].stats.endtime
    nfft = int(win_len * fs)
    while start < end - win_len:
        part = stream.slice(start, start + win_len - dt)
        data = part[0].data
        pxx, freq = mlab.psd(x=data, NFFT=nfft, pad_to=nfft, Fs=fs, scale_by_freq=True)
        pxx = pxx[:, None]
        if start == stream[0].stats.starttime:
            psds = pxx
        else:
            psds = np.concatenate((psds, pxx), axis=-1)
        start += overlap * win_len
    median = np.median(psds, axis=-1)
    timestamp = (t1 + (t2 - t1) / 2.).timestamp
    return freq, median, timestamp


class spectral_analysis():
    """
    Class to perform spectral analysis on given data.
    """
    def __init__(self, filist, path, stn, chn, paz, dec_fact):
        """
        Initialize the spectral_analysis class. This comprises
        all information for reading and preprocessing the data.

        :type filist: string
        :param filist: Text file containing the files which got into the ppsd
            routine. Just one station-channel combination!
        :type path: string
        :param path: Data path to the files specified in 'filist'
        :type stn: string
        :param stn: The station to use. For safety only, discards files
            recorded at different stations (if contained in 'filist')
        :type chn: string
        :param chn: The channel to use. For safety only: discards files
            recorded on different channels (if contained in 'filist')
        :type paz: dictionary
        :param paz: Poles, zeros and sensitivity of the recording device
        :type dec_fact: integer
        :param dec_fact: Decimation factor applied to the data series
        """
        self.filist = filist
        self.path = path
        self.stn = stn
        self.chn = chn
        self.paz = paz
        self.dec_fact = dec_fact
        

    def ppsd(self, fmin=1., fmax=100.):
        """
        Function that calculates the probabilistic power spectral density
        of a given station-channel combination.

        :type fmin: float
        :param fmin: Minimum frequency to show in PPSD plot
        :type fmax: float
        :param fmax: Maximum frequency to show in PPSD plot
        """

        # read list of files
        files = np.genfromtxt(self.path + self.filist, dtype=str)
        n = files.size
        # if no paz information is given, divide by 1.0
        if self.paz == None:
            self.paz = {"sensitivity": 1.0}
        # loop over files
        for i in range(n):
            st = read(self.path + files[i])
            st.decimate(self.dec_fact)
            if len(st) > 1:
                warnings.warn("more than one trace in st")
            tr = st.select(station=self.stn, channel=self.chn)[0]
            # at first run, initialize PPSD instance
            if i == 0:
                # "is_rotational_data" is set in order not to differentiate that data
                inst = PPSD(tr.stats, paz=self.paz, is_rotational_data=True)
            # add trace
            inst.add(tr)
        print("number of psd segments:", len(inst.times))
        inst.plot(show_noise_models=False, period_lim=(1./fmax, 1./fmin))


    def spectrogram(self, nfft, overlap, fmin=1, fmax=50, downsample=None, starttime=None, endtime=None, show=True, verbose=True):
        """
        This routine computes spectrograms of all files contained in 'filist'.
        The number of returned spectrograms may differ from the number of files
        given - files without a data gap in between are merged.

        :type nfft: integer
        :param nfft: data is split into nfft length segments - see matplotlibs
            mlab.specgram documentation
        :type overlap: float
        :param overlap: overlap between the nfft length segments in percent
        :type fmin, fmax: float
        :param fmin, fmax: frequency interval for which spectrogram is shown
        :type downsample: integer
        :param downsample: If not None, the spectrogram is downsampled along the
            frequency axis; Just every downsample value is plotted/returned
        :type starttime: obspy UTCDateTime
        :param starttime: if given, data is trimmed. Works only if endtime is also
            specified
        :type endtime: obspy UTCDateTime
        :param endtime: if given, data is trimmed. Works only if starttime is also
            specified
        :type show: boolean
        :param show: If True, spectrogram is displayed. Otherwise,  times, fruencies and 
            spectrogram are returned.
        :type verbose: boolen
        :param verbose: if True, information on processing is printed

        :return: list, np.array, list
            1. list contains arrays of timestamps corresponding to the spectrograms
            array contains frequencies of spectrograms
            2. list contains nd.arrays of spectrograms
        """

        # read list of files
        files = np.genfromtxt(self.path + self.filist, dtype=str)
        n = files.size
        # initialize arrays
        # array that keeps track of to which continuous segment a file belongs to
        seg = np.zeros(n)
        # start and endtimes of all files
        stimes_files = np.zeros(n)
        etimes_files = np.zeros(n)
        # read files and get start and endtimes
        for i in range(n):
            st = read(self.path + files[i])
            st.select(station=self.stn, channel=self.chn)
            delta = st[0].stats.delta
            if len(st) > 1:
                raise ValueError("more than one trace in st")
            stimes_files[i] = st[0].stats.starttime.timestamp
            etimes_files[i] = st[0].stats.endtime.timestamp

        # first continuous segment is associated with 0
        interval = 0
        # list containing the starttimes of the continuous segments
        stimes_seg = [stimes_files[0]]
        # loop over files to detect gaps between files and store starttimes of
        # continuous segments
        for i in range(n-1):
            if stimes_files[i+1] > etimes_files[i] + delta:
                interval +=1
                stimes_seg.append(stimes_files[i+1])
            seg[i+1] = interval
        stimes_seg = np.asarray(stimes_seg)

        # calculate overlap in samples
        nlap = float(nfft) * overlap
        # number of segments
        nseg = int(seg[-1])
        # empty list for spectrogram and corresponding timestamps - for each
        # cont. segment, an array is stored into the list
        specs = []
        times = []
        # loop over cont. segments and compute spectrograms
        if verbose:
            print("calculating spectrograms ...")
        for s in range(nseg + 1):
            # empty stream, where single files are added to
            master = Stream()
            # get filenames of current cont. segment and ...
            fs = files[np.where(seg == s)]
            # ... loop over these files, read, decimate, add to master stream and merge 
            for f in range(len(fs)):
                st = read(self.path + fs[f])
                st.decimate(self.dec_fact)
                if starttime is not None and endtime is not None:
                    st.trim(starttime, endtime)
                master += st[0]
                master.merge()
            if verbose:
                print(master[0].stats.starttime)
            # data array of current cont. segment
            data = master[0].data
            # sampling rate
            fs = master[0].stats.sampling_rate
            # finally...calculate spectrogram of current cont. segment
            spectrogram, freqs, time = mlab.specgram(data, nfft, fs, noverlap=nlap, mode="magnitude")
            # add timestamp of stime of current cont. segment in order to obtain absolute time
            time += stimes_seg[s]
            # discard frequencies which are out of range fmin - fmax
            ind = np.where((freqs >= fmin) & (freqs <= fmax))[0]
            freqs = freqs[ind]
            spectrogram = spectrogram[ind, :]
            # append spectrogram and corresponding times to the lists
            specs.append(spectrogram)
            times.append(time)

        if show:
            if verbose:
                print("plotting spectrograms ...")
            # for plotting proper timestring
            dateconv = np.vectorize(dtime.datetime.utcfromtimestamp)
            xfmt = mdates.DateFormatter("%m/%d %H:%M")
            # initialize arrays for min and max values of all spectrogram
            mins = np.zeros(len(specs))
            maxs = np.zeros(len(specs))
            # convert spectrograms to dB scale and obtain min and max values
            for ii in range(len(specs)):
                specs[ii] = 10*np.log10(specs[ii])
                mins[ii], maxs[ii] = specs[ii].min(), specs[ii].max()
                times[ii] = dateconv(times[ii])
            # delete nans and infs
            nans_mins = np.where(np.isnan(mins) | np.isinf(mins))
            nans_maxs = np.where(np.isnan(maxs) | np.isinf(maxs))
            mins = np.delete(mins, nans_mins[0])
            maxs = np.delete(maxs, nans_maxs[0])
            # get min and max values
            absmin = mins.min()
            absmax = maxs.max()
            # sample down
            if downsample is not None:
                freqs = freqs[0::downsample]
                for ii in range(len(specs)):
                    specs[ii] = specs[ii][0::downsample, :]
            # create figure
            fig = plt.figure(figsize=(27,4))
            ax = fig.add_subplot(111)
            for ii in range(len(specs)):
                im = ax.pcolormesh(times[ii], freqs, specs[ii], cmap="jet", vmin=absmin, vmax=absmax)
            ax.set_ylim(fmin, fmax)
            ax.set_xlim(times[0][0], times[-1][-1])
            ax.set_ylabel("Frequency (Hz)")
            ax.xaxis.set_major_formatter(xfmt)
            cbar = fig.colorbar(im)
            cbar.set_label("Power (dB)")
            ax.set_title("%s..%s" % (self.stn, self.chn))
            plt.show()
        else:
            return times, freqs, specs


class tremor():
    """
    Class to calculate and save tremor amplitude as described in Bartholomaus et al., 2015
    """
    def __init__(self, stn, chn, win_long, win_short, overlap, fmin, fmax, t1, t2, ts=None, Vs=None, errors=None):
        """
        Initialize class tremor.

        :type stn: string
        :param stn: station
        :type chn: string
        :param chn: channel
        :type win_long: float
        :param win_long: window length for which a single MGV value is calculated
        :type win_short: float
        :param win_short: window length used to calculate a median PSD
        :type overlap: float
        :param overlap: overlap of sliding windows. applies to win_long and win_short
        :type fmin, fmax: float
        :param fmin, fmax: frequency range for MGV values have been calculated
        :type t1, t2: obspy UTCDateTime object
        :param t1, t2: start- and endtime of interval of consideration. t1 (t2) should be
            00:00 - win_long / 2 (t1 + 24h + win_long / 2)
        :param ts: empty field for tremor amplitudes' timestamps
        :param Vs: empty field for tremor amplitudes
        :param errors: empty field for matrix containing error-prone values

        """

        self.stn = stn
        self.chn = chn
        self.win_long = win_long
        self.win_short = win_short
        self.overlap = overlap
        self.fmin = fmin
        self.fmax = fmax
        self.t1 = t1
        self.t2 = t2


    def tremor_amplitude(self, st, skip_gaps=True):
        """
        Function to compute the tremor amplitude (Bartholomaus et al, 2015) in a given frequency
            band. Is expected to process one day at a time only.

        :type st: obspy stream object
        :param st: Stream that contains on trace of data
        :type skip_gaps: boolean
        :param skip_gaps: skips win_longs which have gaps or are smaller win_long / 2.

        :return: None
            Stores tremor amplitudes and corresponding timestamps.
        """

        # initalize for while loop
        t = self.t1
        Vs = []
        ts = []
        err_ts = []
        err_Vs = []
        count = 0
        # apply sliding window to data stream and calculate median PSDs
        while t < self.t2 - self.win_long + 1:
            # assume there is no gap
            gap = False
            try:
                # slice window of length win_long
                endtime = t + self.win_long
                piece = st.slice(t, endtime)
                # detect gaps
                if ma.is_masked(piece[0].data):
                    gap = True
                # detect chunks shorter than win_long / 2.
                if piece[0].data.size < ((self.win_long / 2.) * piece[0].stats.sampling_rate):
                    gap = True
                # provoke error if there is no trace in stream
                provoke = piece[0].stats.starttime
            except:
                # if no data, go on to next window
                print("No data for %s - %s" % (t, t + self.win_long))
                t += self.win_long * self.overlap
                continue
            # if gap and skip_gaps are True, skip calculation
            if gap == True and skip_gaps == True:
                pass 
            else:
                # calculate median PSD
                freq, median, timestamp = medianPSD(piece, self.win_short, self.overlap, t, endtime)
                # obtain indices of frequency band of interest
                ind = np.where((freq >= self.fmin) & (freq <= self.fmax))
                # discard all other frequencies
                median_freqband = median[ind]
                # integrate over this frequency band and take square root. see bartholomaus et al., 2015
                df = freq[1] - freq[0]
                V = np.sqrt(np.sum(median_freqband * df))
                # append values to list
                Vs.append(V)
                ts.append(timestamp)
                if gap:
                    err_ts.append(timestamp)
                    err_Vs.append(V)
            t += self.win_long * self.overlap
            # increase counter - this is equal to the indice of the current value pair (ts, Vs)
            count += 1

        # convert values from lists to arrays
        ts = np.asarray(ts)
        Vs = np.asarray(Vs)
        # create matrix containing potentially error-prone values
        if len(err_ts) > 0:
            err_ts = np.asarray(err_ts)
            err_Vs = np.asarray(err_Vs)
            errors = np.zeros((err_ts.size, 2))
            errors[:, 0] = err_ts
            errors[:, 1] = err_Vs
        else:
            errors = None

        self.ts = ts
        self.Vs = Vs
        self.errors = errors


    def write_data(self, path):
        """
        Writes median ground velocity values as ObsPy stream files. Timestamps and
        corresponting values are expected to span a full day. First sample shout be
        at 00:00 and last sample at 00:00 (+1day) - win_long / 2.

        :type path: string
        :param path: data path where output is written to
    
        :return: None
            Writes data and a file containing potentially incorrect values (due to
            data gaps) to the given path/directory.
        """

        # create empty stream
        mgv = Stream()
        # detect gaps is data
        # number of traces, starting indice is stored
        ntr = [0]
        for i in range(self.ts.size - 1):
            # gap if time between timestamps is greater (win_long * overlap)
            if self.ts[i+1] - self.ts[i] > (self.win_long * self.overlap):
                # append starting indice of new trace
                ntr.append(i+1)
        # get data for single traces
        for n in range(len(ntr)):
            # just one trace
            if len(ntr) == 1:
                data = self.Vs
                time = self.ts
            # last trace
            elif n == len(ntr) - 1:
                data = self.Vs[ntr[n]:]
                time = self.ts[ntr[n]:]
            # every other case
            else:
                data = self.Vs[ntr[n]: ntr[n+1]]
                time = self.ts[ntr[n]: ntr[n+1]]
            # create new trace and add data
            new = Trace(data=data)
            new.stats.starttime = UTCDateTime(time[0])
            new.stats.delta = self.win_long * self.overlap
            mgv += new
        # add stats to traces
        for i, tr in enumerate(mgv):
            tr.stats.network = "4D"
            tr.stats.station = self.stn
            tr.stats.channel = self.chn
        # obtain julian day
        julday = UTCDateTime(self.ts[0]).julday
        # write data
        mgv.write(path + "MGV.%s.%s.%03d_%.1f-%.1fHz.mseed" % (self.stn, self.chn, julday, self.fmin, self.fmax), format="MSEED")
        # write file containing errorneous data points. These values have been computed form data containing gaps
        if self.errors is not None:
            hdr = "Potentially incorrect values (timestamp, MGV value) due to \
                   data gaps in the window associated to the timestamps below"
            np.savetxt(path + "MGV_errval_%s.%s.%03d_%.1f-%.1fHz" % (self.stn, self.chn, julday, self.fmin, self.fmax), self.errors, header=hdr)
