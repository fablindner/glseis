import numpy as np
from obspy.signal.filter import bandpass
from glseis.filter import ricker
from glseis.quantity import stretch
import scipy.signal as signal
import matplotlib.pyplot as plt
#import msnoise_move2obspy as msnoise
import warnings
import os




def preprocessing(data, path, muteP, fs, fmin=5, fmax=30):
    """
    preprocess data. only used for SES3D waveforms !!!
    :param data: data to preprocess
    :param path: path where data is stored
    :param muteP: if muteP, time domain taper is applied to mute the p-wave arrival
    :param fs: sampling frequency
    :param fmin: lower bound for bandpass filtering
    :param fmax: higher bound for bandpass filtering
    :return: preprocessed data
    """
    # mute unphysical reflections and p-wave
    rylgh_int = np.loadtxt(path + "rayleigh_interval.txt")
    pwave = rylgh_int[0]
    refl = rylgh_int[1]
    argmax = np.argmax(data)
    if muteP:
        data[:argmax - pwave] = 0
    data[argmax + refl:] = 0
    # filter
    data = bandpass(data, fmin, fmax, fs, zerophase=True)
    # demean
    data = signal.detrend(data, type="constant")
    return data




class MDD():
    def __init__(self, scenarios, fname, epssq, filt, fmin, fmax, bnds):
        self.scenarios = scenarios
        self.fname = fname
        self.epssq = epssq
        self.filt = filt
        self.fmin = fmin
        self.fmax = fmax
        self.bnds = bnds
        # empty fields
        self.freqs = None
        self.fs = None
        self.dt = None
        self.t_acaus = None
        self.t_caus = None
        self.fmin_prep = None
        self.fmax_prep = None
        self.npts = None
        self.invdata = {}
        self.res_mdd = {}
        self.res_cc = {}
        self.recs = {}
        self.srcs = {}
        self.taper = None
        # some colors
        self.eth3 = "#0069B4"   # blue
        self.eth6 = "#6F6F6E"   # grey
        self.eth7 = "#A8322D"   # red
        self.eth8 = "#007A92"   # blue



    def read_arrays(self):
        """
        read data arrays for all scenarios
        """
        count = 0
        for key in self.scenarios:
            path = self.scenarios[key] + self.fname
            arrays = np.load(path)

            C = arrays["arr_0"]
            T = arrays["arr_1"]
            freqs = arrays["arr_2"]
            fs = arrays["arr_3"][0]
            fmin_prep = arrays["arr_3"][1]
            fmax_prep = arrays["arr_3"][2]
            npts = arrays["arr_4"]
            recs = arrays["arr_5"]
            srcs = arrays["arr_6"]

            self.invdata.update({key: [C, T]})
            self.recs.update({key: recs})
            self.srcs.update({key: srcs})
            # test if values are the same as in previous scenarios
            if count > 0:
                if not np.array_equal(freqs ,self.freqs):
                    warnings.warn("Parameter 'freqs' is not the same as for " \
                                + "the previous scenario", UserWarning)
                if fs != self.fs:
                    warnings.warn("Parameter 'fs' is not the same as for the " \
                                + "previous scenario", UserWarning)
                if fmin_prep != self.fmin_prep:
                    warnings.warn("Parameter 'fmin' is not the same as for " \
                                + "the previous scenario", UserWarning)
                if fmax_prep != self.fmax_prep:
                    warnings.warn("Parameter 'fmax' is not the same as for " \
                                + "the previous scenario", UserWarning)
                if npts != self.npts:
                    warnings.warn("Parameter 'npts' is not the same as for " \
                                + "the previous scenario", UserWarning)

            self.freqs = freqs
            self.fs = fs
            self.dt = 1./self.fs
            self.fmin_prep = fmin_prep
            self.fmax_prep = fmax_prep
            self.npts = npts
            count += 1



    def make_taper(self, a, b):
        """
        Cosine taper acting on the interval [a*f0, b*f0]. Mutes all
        frequencies above b*f0
        :param a: Taper acts on frequencies higher than a*f0
            [f0 = (self.fmax_prep - self.fmin_prep) / 2]
        :param b: Frequencies higher than b*f0 are muted by the taper
        :return: no data is returned; taper is stored
        """
        taper = np.ones(self.freqs.size)
        ind_tpr = np.where((self.freqs >= a) & (self.freqs <= b))[0]
        ind_zeros = np.where(self.freqs > b)
        npts_tpr = ind_tpr.size
        x_tpr = np.linspace(0, np.pi, npts_tpr)
        tpr = np.cos(x_tpr)
        tpr += abs(tpr.min())
        tpr /= tpr.max()
        taper[ind_zeros] = 0
        taper[ind_tpr] = tpr
        self.taper = taper



    def inversion(self, fmax, fricker=0):
        """
        Perform inversion for all scenarios

        :param fmax: inversion is only performed for frequencies smaller fmax
            to speed up the inversion
        :param fricker: center frequency of ricker wavelet used to create the
            shaping filter (auto-correlation of ricer wavelet)
        :return: no data is returned; mdd time domain, cc time domain signals
            and time vector are stored
        """
        print("perform inversion ...")
        # create autocorrelation of ricker wavelet
        if fricker > 0:
            auto = ricker(self.dt, fricker, 1, self.freqs) \ 
                 * np.conj(ricker(self.dt, fricker, 1, self.freqs))
        # loop over scenarios
        for key in self.scenarios:
            print(key)
            # prepare data
            nrec = len(self.recs[key])
            C = self.invdata[key][0]
            for i in range(nrec):
                C[i, :] *= self.taper
            T = self.invdata[key][1]
            for i in range(nrec):
                for j in range(nrec):
                    T[i,j] *= self.taper
            GR = np.zeros((nrec, self.freqs.size), dtype=complex)
            # do the inversion
            freqs_ = self.freqs[self.freqs <= fmax]
            max_val = abs(T).max()
            for f in range(freqs_.size):
                T_ = T[:, :, f]
                C_ = C[:, f]
                #max_val = abs(T_).max()
                for i in range(nrec):
                    T_[i, i] += self.epssq * max_val
                if np.all(T_ == 0j + 0):
                    GR[:, f] = np.zeros(nrec, dtype=complex)
                else:
                    GR[:, f] = np.dot(C_, np.linalg.inv(T_)) * self.taper[f]
                # convolve with autorocorrelation of ricker wavelet
                if fricker > 0:
                    GR[:, f] *= auto[f]

            # plot spectra
            #rec = 7
            #plt.plot(self.freqs, abs(C[rec, :]/C[rec, :].max()), label="XCORR")
            #plt.plot(self.freqs, abs(GR[rec, :]/GR[rec, :].max()), label="MDD")
            #plt.plot(self.freqs, abs(T[rec, rec, :]), label="PSF")
            #plt.legend()
            #plt.xlim(0, 100)
            #plt.show()

            # inverse fft - obtain time series again
            GdR = np.zeros((nrec, self.npts))
            #plt.plot(self.freqs, abs(GR[7,:]))
            #plt.xlim(0,200)
            #plt.show()
            ccf = np.zeros((nrec, self.npts))
            for r in range(nrec):
                buff_mdd = np.fft.irfft(GR[r, :], norm="ortho")
                buff_cc = np.fft.irfft(C[r, :], norm="ortho")
                if self.filt:
                    GdR[r, :] = bandpass(buff_mdd, self.fmin, self.fmax,
                                         self.fs, corners=3, zerophase=True)
                    ccf[r, :] = bandpass(buff_cc, self.fmin, self.fmax,
                                         self.fs, corners=3, zerophase=True)
                else:
                    GdR[r, :] = buff_mdd
                    ccf[r, :] = buff_cc
            # corresponding time vector
            # acausal part in left half of seismogram
            time = np.linspace(-(self.npts/2.)*self.dt, (self.npts/2.)*self.dt,
                               num=self.npts)
            self.t_acaus = time
            # causal part in left half of seismogram
            time_ = time + abs(time).max() + abs(time).min()
            self.t_caus = time_
            # store results
            self.res_mdd.update({key: GdR})
            self.res_cc.update({key: ccf})
            # write results as arrays
            np.savez(self.scenarios[key] + "RESP_freq_resp_time_epssq_CCF_%s.npz"
                     % self.bnds[:4], C, self.freqs, ccf, time, self.epssq)
            np.savez(self.scenarios[key] + "RESP_freq_resp_time_epssq_MDD_%s.npz"
                     % self.bnds[:4], GR, self.freqs, GdR, time, self.epssq)



    def dvv_stretching(self, rec, nrefl, d, v, win_len, dvv_max, dvv_delta):
        """
        estimates dv/v values with cross-correlation responses and mdd responses
            using a stretching technique
        :param rec: chain receiver used for this analysis
        :param nrefl: number of reflections used to calculate mdd-dv/v's
        :param d: distance between the cross-correlation receivers
        :param v: velocity of the unperturbed medium
        :param win_len: window length used to cut the direct arrivals and
            reflections. window will be centered around the theoretical travel
            time of the homogeneous medium
        :return: array holding dv/v's + cc's obtained from MDD and array holding
            dv/v + cc obtained from CC
        """

        print("Calculate dv/v - Stretching ...")

        # load data
        time = self.t_caus
        keys = list(self.scenarios.keys())
        d_mdd_ref = self.res_mdd[keys[0]][rec, :]
        d_mdd = self.res_mdd[keys[1]][rec, :]
        d_cc_ref = self.res_cc[keys[0]][rec, :]
        d_cc = self.res_cc[keys[1]][rec, :]
        # vector holding the dv/v values
        dvv = np.arange(-dvv_max, dvv_max+dvv_delta, dvv_delta)
        ################# CC ###########################################
        # prepare for stretching the cc response of the perturbed medium
        tt = d/v
        t_win = [tt - win_len/2., tt + win_len/2.]
        #plt.plot(time, d_cc_ref)
        #plt.plot(time, d_cc)
        #plt.axvline(t_win[0])
        #plt.axvline(t_win[1])
        #plt.show()
        win = np.where((time >= t_win[0]) & (time <= t_win[1]))[0]
        # stretch!
        eps, cc = stretch(time, d_cc, d_cc_ref, dvv, win, plot=False)
        dvv_cc_cc = np.array([eps, cc])
        ################# MDD ###########################################
        # prepare for stretching the mdd response of the perturbed medium
        dvv_cc_mdd = np.zeros((nrefl, 2))
        dists = np.zeros(nrefl)
        for i in range(nrefl):
            dist = (50. + i*100.)
            if nrefl == 1:
                dist += 200.
            dists[i] = dist
            tt = dist / v
            t_win = [tt - win_len/2., tt + win_len/2.]
            win = np.where((time >= t_win[0]) & (time <= t_win[1]))[0]
            eps, cc = stretch(time, d_mdd, d_mdd_ref, dvv, win, plot=False)
            dvv_cc_mdd[i, 0] = eps
            dvv_cc_mdd[i, 1] = cc
        #################################################################
        return dists, dvv_cc_cc, dvv_cc_mdd



    def dvv_mwcs(self, rec, fmin, fmax, tmin, windL, step, nrefl, min_coh=0.65,
                 max_err=0.1):
        """...

        :type rec: int
        :param rec: receiver used to calculate mwcs
        :type fmin: float
        :param fmin: The lower frequency bound to compute the dephasing
        :type fmax: float
        :param fmax: The higher frequency bound to compute the dephasing
        :type sampRate: float
        :param sampRate: The sample rate of the input timeseries
        :type tmin: float
        :param tmin: The leftmost time lag (used to compute the "time lags array")
        :type windL: float
        :param windL: The moving window length (in seconds)
        :type step: float
        :param step: The step to jump for the moving window (in seconds)
        :rtype: :class:`numpy.ndarray`
        :returns: [Taxis,deltaT,deltaErr,deltaMcoh]. Taxis contains the central
            times of the windows. The three other columns contain dt, error and
            mean coherence for each window.
        """

        print("Calculate dv/v - MWCS ...")

        keys = list(self.scenarios.keys())
        ccReference = self.res_mdd[keys[0]][rec, :]
        ccCurrent = self.res_mdd[keys[1]][rec, :]
        sampRate = self.fs

        mwcs = msnoise.mwcs(ccCurrent, ccReference, fmin, fmax, sampRate,
                            tmin, windL, step)
        mwcs = mwcs[:nrefl, :]
        ind_coh = np.where(mwcs[:, 3] >= min_coh)[0]
        ind_err = np.where(mwcs[:, 2] <= max_err)[0]
        ind = np.intersect1d(ind_coh, ind_err)
        mwcs = mwcs[ind, :]
        tArray = mwcs[:, 0]
        dtArray = mwcs[:, 1]
        errArray = mwcs[:, 2]
        w = 1.0 / errArray
        w[~np.isfinite(w)] = 1.0

        m, a, em, ea = msnoise.linear_regression(tArray, dtArray, w, intercept=True)
        m0, em0 = msnoise.linear_regression(tArray, dtArray, w, intercept=False)

        x = np.arange(0,0.7,0.01)
        y = m0 * x
        #plt.plot(tArray, dtArray, "rx")
        #plt.plot(x, y, "k")
        #plt.show()
        return m, a, em, ea, m0, em0




    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    The following functions are work in progress and are changed very often
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """




    def write_dvv_results(self, fn, dvv_cc_cc, dvv_cc_mdd, m0, em0, m, a, em,
                          ea, dvv_mdd_stch, cc_mdd_stch):

        """
        :param dvv_cc_cc: dvv + cc value obtained for the CC responses
        :param dvv_cc_mdd: dvv + cc values obtained for the MDD (and reflection)
            responses
        :param m: slope (dt/t) obtained from the mwcs measurement of the MDD
            reflection responses
        :param a: intercept of the slope (dt/t) obtained from the mwcs
            measurement of the MDD reflection responses
        :param em: error of the slope m
        :param ea: error of the intercept a
        :param m0: as m, but forced through the origin
        :param em0: error of m0
        :return: no data is returned; writes the values to a text file
        """
        fh = open(fn, "a")
        if os.stat(fn).st_size == 0:
            ncol = dvv_cc_cc.size + dvv_cc_mdd.size
            hline1 = "# columns 01 - %02d: dvv and cc values from stretching " \
                   + "(CC, MDD dir, MDD 1st refl, ...)" % ncol
            fh. write(hline1 + "\n")
            hline2 = "# columns %02d - %02d: m0, em0, m, a, em, ea from MWCS " \
                   + "of MDD responses" % (ncol + 1, ncol + 7)
            fh.write(hline2 + "\n")
            hline3 = "# columns %02d - %02d: dvv, cc from stetching of MDD " \
                   + "reflection responses" % (ncol + 8, ncol + 9)
            fh.write(hline3 + "\n")

        # stretching
        line = "%08.5f %08.5f " % (dvv_cc_cc[0], dvv_cc_cc[1])
        for i in range(dvv_cc_mdd.size // 2):
            line += "%08.5f %08.5f " % (dvv_cc_mdd[i,0], dvv_cc_mdd[i,1])
        # MWCS
        line += "   %08.5f %010.7f %08.5f %08.5f %010.7f %010.7f " % (m0, em0, m, a, em, ea)
        # stretching full response
        line += "   %08.5f %08.5f" % (dvv_mdd_stch, cc_mdd_stch)

        fh.write(line + "\n")
        fh.close()



    ############################################################################
    # THIS FUNCTION NEEDS TO BE CLEANED UP !!!
    def plot_psf(self, sc, rec):
        for i in range(30):
            rec = i
            Tpick = self.invdata[sc][1][rec, :, :]
            tpick = np.zeros((len(self.recs), int(self.npts)))
            time = np.linspace(-(self.npts/2.)*self.dt, (self.npts/2.)*self.dt,
                               num=self.npts)
            for i in range(len(self.recs)):
                tpick[i, :] = np.fft.ifftshift(np.fft.irfft(Tpick[i, :], int(self.npts)))
            R = np.arange(1, len(self.recs) + 1) - 0.5
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            psf_td = ax1.pcolormesh(R, time, tpick.T, cmap="seismic",
                                    vmin=-abs(tpick).max(), vmax=abs(tpick).max())
            fig.colorbar(psf_td, ax=ax1)
            tpick_max = tpick.max()
            for i in range(len(self.recs)):
                data = tpick[i, :]
                data /= tpick_max
                data += R[i] + 0.5
                ax1.plot(data, time, "grey")
            ax1.set_title("Point-spread function (PSF) for receiver %i" % (rec + 1))
            ax1.set_xlabel("Receiver")
            ax1.set_ylabel("Lag Time (s)")
            ax1.set_xlim(R.min(), R.max())
            ax1.set_ylim(-1, 1)

            ax2 = fig.add_subplot(122)
            psf_fd = ax2.pcolormesh(R, self.freqs, abs(Tpick.T), cmap="viridis")
            fig.colorbar(psf_fd, ax=ax2)
            ax2.set_xlabel("Receiver")
            ax2.set_ylabel("Frequency (Hz)")
            ax2.set_title("Point-spread function (PSF) for receiver %i" % (rec + 1))
            ax2.set_xlim(R.min(), R.max())
            ax2.set_ylim(-100, 100)
            plt.show()
    ############################################################################



    def plot_all_velocities(self, rec, shift=False, norm=True):
        """
        plot the mdd responses of one receiver pair for all scenarios
        :param rec: chain receiver, for which the result is shown
        :param shift: if shift, the causal part is shown in the right hand side 
            of the seismogram
        :param norm: if norm, the seismograms are normalized by their maximum
            value
        :return: no data is returned; plot is displayed
        """
        # select time vector
        if shift:
            time = self.t_acaus
        else:
            time = self.t_caus
        # create figure
        fig = plt.figure(figsize=(8,3.4))
        ax = fig.add_subplot(111)
        colors = [self.eth3, self.eth7, self.eth6, self.eth8]
        count = 0
        # loop over scenarios and plot waveforms
        for key in self.scenarios:
            data = self.res_mdd[key][rec, :]
            if norm:
                data /= data.max()
            if shift:
                data = np.fft.ifftshift(data)
            #data = bandpass(data, 50, 80, self.fs)
            ax.plot(time, data, colors[count], label=self.scenarios[key], lw=1)
            count += 1
        ax.axvline(0.0, color="k")
        ax.set_xlabel("Lag Time (s)")
        if norm:
            ax.set_ylim(-1, 1)
            ax.set_ylabel("Norm. Amplitude")
        ax.set_xlabel("Time (s)")
        #ax.yaxis.set_visible(False)
        ax.set_ylabel("Norm. amplitude")
        #plt.savefig("/home/fabian/Desktop/Plots/epssq_%.9f.png" % epssq)
        #ax.set_xlim(-0.5, 0.5)
        #ax.axvline(0.028, color=self.eth6, lw=22, alpha=0.2)
        #ax.axvline(0.087, color=self.eth6, lw=22, alpha=0.2)
        #ax.axvline(0.147, color=self.eth6, lw=22, alpha=0.2)
        #ax.axvline(0.206, color=self.eth6, lw=22, alpha=0.2)
        #ax.axvline(0.265, color=self.eth6, lw=22, alpha=0.2)
        dx = np.arange(100., 1100., 100.)
        for i in range(10):
            ax.axvline(dx[i] / 1650., color="k", alpha=0.5)
        plt.show()
        plt.close()



    def plot_all_mdd_cc(self, sc, shift=True):
        """
        All MDD and CC results are displayed
        :param sc: scenario, which is displayed
        :param shift: if shift, causal part is shown in right half of seismogram
        :return: no data is returned; plot is displayed
        """
        # time vector and data
        if shift:
            time = self.t_acaus
        else:
            time = self.t_caus
        GdR = self.res_mdd[sc]
        ccf = self.res_cc[sc]
        # plotting
        fig = plt.figure(figsize=(14,14))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for i in range(len(self.recs[sc])):
            # MDD responses
            d_mdd = GdR[i, :]
            if shift:
                d_mdd = np.fft.ifftshift(d_mdd)
            d_mdd -= np.mean(d_mdd)
            d_mdd /= d_mdd.max() * 1.5
            d_mdd += (i + 1)
            ax1.plot(time, d_mdd, "b", lw=0.7)
            # CC responses
            d_cc = ccf[i, :]
            if shift:
                d_cc = np.fft.ifftshift(d_cc)
            d_cc /= d_cc.max() * 1.5
            d_cc -= np.mean(d_cc)
            d_cc += (i + 1)
            ax1.plot(time, d_cc, "r", lw=0.7)
        nrec = len(self.recs[sc])
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(0, nrec + 1)
        plt.yticks(np.arange(len(self.recs[sc]))+1, self.recs[sc])
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(0, nrec + 1)
        plt.yticks(np.arange(len(self.recs[sc]))+1, self.recs[sc])
        plt.show()



    def compare_mdd_cc(self, sc, rec, xdisp):
        """
        Compare MDD response with CC response
        :param sc: scenario, which is displayed
        :param rec: chain receiver, for which the result is shown in detail
        :param xdisp: x-axis is will be limited to [-xdisp, xdisp]
        :return: no data is returned; plot is displayed
        """
        # collect data for plotting
        nrec = len(self.recs[sc])
        GdR = self.res_mdd[sc]
        ccf = self.res_cc[sc]
        time = self.t_acaus
        # plot!
        fig = plt.figure(figsize=(18,12))
        # ax1: all receivers
        ax1 = fig.add_axes([0.1,0.1,0.8,0.6])
        for i in range(nrec):
            # adjust linewidth for selected receiver
            if i == rec:
                lw = 2.
            else:
                lw = 1.
            # MDD data
            d_mdd = GdR[i, :]
            d_mdd = np.fft.fftshift(d_mdd)
            d_mdd -= np.mean(d_mdd)
            d_mdd /= d_mdd.max()
            d_mdd += i
            ax1.plot(time, d_mdd, self.eth8, lw=lw)
            # CC data
            d_cc = ccf[i, :]
            d_cc = np.fft.fftshift(d_cc)
            d_cc -= np.mean(d_cc)
            d_cc /= d_cc.max()
            d_cc += i
            ax1.plot(time, d_cc, self.eth7, lw=lw)
        ax1.set_xlim(-xdisp, xdisp)
        ax1.set_ylim(-1, nrec)
        ax1.set_ylabel("Receiver")
        ax1.set_xlabel("Time (s)")

        # ax2: selected receiver
        ax2 = fig.add_axes([0.1,0.75, 0.8, 0.15])
        d_mdd = np.fft.ifftshift(GdR[rec, :])
        d_mdd -= np.mean(d_mdd)
        d_cc = np.fft.ifftshift(ccf[rec, :])
        d_cc -= np.mean(d_cc)
        ax2.plot(time, d_mdd / d_mdd.max(), "#007A92", lw=2, label="MDD")
        ax2.plot(time, d_cc / d_cc.max(), "#A8322D", lw=2, label="CC")
        ax2.set_xlim(-xdisp, xdisp)
        ax2.set_ylim(-1,1)
        #ax2.yaxis.set_ticklabels([])
        ax2.text(-xdisp+0.01, 0.3, "Receiver %i" % rec, fontsize=16)
        plt.legend()
        plt.show()
