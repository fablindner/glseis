from obspy import read, Stream
import numpy as np
from obspy.signal import PPSD
import matplotlib.pyplot as plt
from matplotlib import mlab
import matplotlib.dates as mdates
import datetime as dtime
import warnings


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


    def spectrogram(self, nfft, overlap, fmin=1, fmax=50, starttime=None, endtime=None, show=True):
        """
        This routine computes spectrograms of all files contained in 'filist'.
        The number of returned spectrograms may differ from the number of files
        given - files without a data gap in between are merged.

        :type nfft: integer
        :param nfft: data is split into nfft length segments - see matplotlibs
            mlab.specgram documentation
        :type overlap: float
        :param overlap: overlap between the nfft length segments in percent
        :type starttime: obspy UTCDateTime
        :param starttime: if given, data is trimmed. Works only if endtime is also
            specified
        :type endtime: obspy UTCDateTime
        :param endtime: if given, data is trimmed. Works only if starttime is also
            specified
        :type return_results: boolean
        :param return_results: If True, times, fruencies and spectrogram are returned.
            Otherwise they are stored hand can be plotted with the function 'plot_specgram'

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
            # data array of current cont. segment
            data = master[0].data
            # sampling rate
            fs = master[0].stats.sampling_rate
            # finally...calculate spectrogram of current cont. segment
            spectrogram, freqs, time = mlab.specgram(data, nfft, fs, noverlap=nlap, mode="magnitude")
            # add timestamp of stime of current cont. segment in order to obtain absolute time
            time += stimes_seg[s]
            # append spectrogram and corresponding times to the lists
            specs.append(spectrogram)
            times.append(time)

        if show:
            # for plotting proper timestring
            dateconv = np.vectorize(dtime.datetime.utcfromtimestamp)
            xfmt = mdates.DateFormatter("%m-%d")
            # initialize arrays for min and max values of all spectrogram
            mins = np.zeros(len(specs))
            maxs = np.zeros(len(specs))
            # convert spectrograms to dB scale and obtain min and max values
            for ii in range(len(specs)):
                specs[ii] = 10*np.log10(specs[ii])
                mins, maxs = specs[ii].min(), specs[ii].max()
                times[ii] = dateconv(times[ii])
            absmin = mins.min()
            absmax = maxs.max()
            # create figure
            fig = plt.figure()
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
