import numpy as np
from obspy import read, Stream, UTCDateTime
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset
import obspy.signal
from glseis.array_analysis import plwave_beamformer, matchedfield_beamformer
from collections import OrderedDict
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import glob
import os
import sys



class icequake_locations():
    """
    Class to locate icequakes using beamforming and triangulation.
    """


    def __init__(self, path2mseed, path2DBs, array, r, stnlist, chn,
                 sens, jday, fs, decfact, fmin, fmax, vmin, vmax, dv):
        """
        Initialize class icequake_locations.
        :param path2mseed: path to mseed data
        :param path2DBs: path where event data base will be stored
        :param array: array of consideration
        :param r: radius of array
        :param stnlist: list containing the array stations
        :param chn: channel of consideration
        :param sens: overall sensitivity of seismometer + digitizer -> convert to m/s
        :param jday: julian day of consideration
        :param fs: sampling frequency
        :param decfact: decimation factor
        :param fmin: min frequency used for beamforming
        :param fmax: max frequency used for beamforming
        :param vmin: min velocity used for beamforming
        :param vmax: max velocity used for beamforming
        :param dv: velocity step used for beamforming
        """
        self.path2mseed = path2mseed
        self.path2DBs = path2DBs
        self.array = array
        self.r = r
        self.stnlist = stnlist
        self.chn = chn
        self.sens = sens
        self.jday = jday
        self.fs = fs
        self.decfact = decfact
        self.fmin = fmin
        self.fmax = fmax
        self.vmin = vmin
        self.vmax = vmax
        self.dv = dv


    def make_eventDB_header(self, fh):
        """
        Write eventDB header.
        :param fh: already open file (write access)
        """
        fh.write("BEAMFORMING PARAMETERS:\n")
        fh.write("fmin: %.1f\n" % self.fmin)
        fh.write("fmax: %.1f\n" % self.fmax)
        fh.write("vmin: %.3f\n" % self.vmin)
        fh.write("vmax: %.3f\n" % self.vmax)
        fh.write("dv: %.3f\n" % self.dv)
        fh.write("RESULTS (ID, t1, t2, peak ampl., peak freq., trigger duration, average delay, baz max pow, vel max pow, max pow):\n")
        fh.flush()


    def make_eventDB_entry(self, fh, n, t1, t2, pampl, pfreq, dur, avg_delay, baz, slow, beam):
        """
        Make eventDB entry for an event.
        :param fh: already open file (write access)
        :param n: event number (consecutively numbered)
        :param t1: starttime of event (trigger on time of center station minus traveltime for array radius distance) 
        :param t2: endtime of event (trigger off time of center station plus traveltime for array radius distance) 
        :param pampl: peak amplitude averaged over all array stations
        :param pfreq: peak frequency averaged over all array stations
        :param dur: trigger duration of center station
        :param avg_delay: average delay of all stations with respect to center station
        :param baz: determined back azimuth
        :param slow: determined (average) slowness in array
        :param beam: beam power
        """
        id = "%03d_%s_%04d" % (self.jday, self.array, n)
        try:
            ind_baz = np.argmax(np.amax(beam, axis=0))
            ind_slow = np.argmax(np.amax(beam, axis=1))
            max_pow = beam[ind_slow,ind_baz]
            max_slow = slow[ind_slow]
            max_baz = baz[ind_baz]
            line = "%s   %s   %s   %.2e   %05.1f   %.3f   %06.3f   %3i   %06.3f   %.3f\n" % (id, t1, t2, pampl, pfreq,
                                                                        dur, avg_delay, max_baz, 1./max_slow, max_pow)
        except:
            line = "%s   %s   %s   %.2e   %05.1f   %.3f   %06.3f   %3s   %6s   %5s\n" % (id, t1, t2, pampl, pfreq, dur,
                                                                        avg_delay, np.nan, np.nan, np.nan)
            pass
        fh.write(line)
        fh.flush()


    def calc_pfreq_pampl(self, st):
        """
        Calculates average peak frequency and peak amplitude of traces in stream.
        :param st: obspy stream object
        """
        # filter with corner frequency
        st.trim(st[0].stats.starttime + 0.4, st[0].stats.endtime - 0.4)
        st.taper(max_percentage=0.1)

        nfft = 2**11
        dt = st[0].stats.delta
        freq = np.fft.rfftfreq(nfft, dt)
        spec = np.zeros((freq.size, len(self.stnlist)))
        ampl = np.zeros(len(self.stnlist))
        for i, tr in enumerate(st):
            data = np.zeros(nfft)
            data[:tr.data.size] = tr.data
            spec[:, i] = abs(np.fft.rfft(data))
            ampl[i] = abs(tr.data).max()
        spec[0, :] = 0.
        spec = np.average(spec, axis=1)
        pfreq = freq[np.argmax(spec)]
        pampl = float(np.average(ampl)) / self.sens
        return pfreq, pampl


    def calc_average_delay(self, st):
        """
        Calculates the average delay of events with respect to center station. In case of plaine
        waves, this approaches to zero.
        :param st: obspy stream object
        """
        delays = np.zeros(len(self.stnlist))
        dt = st[0].stats.delta
        for i in range(len(st)):
            cc = np.correlate(st.select(station=self.stnlist[i])[0].data, st[-1].data, mode="full")
            delays[i] = np.argmax(cc) - len(st[-1].data) + 1
        cum_delay = np.sum(delays) * dt
        avg_delay = cum_delay / float(len(self.stnlist) - 1)
        return avg_delay


    def read_eventDB(self, array, powmin):
        """
        Read an existing event data base for a certain array and remove entries with weak beam powers.
        :param array: array of consideration
        :param powmin: minimum beam power
        """
        try:
            # load data
            path = "%s/EventDB/%s_EventDB_%03d.txt" % (array, array, self.jday)
            id, on, off = np.genfromtxt(path, skip_header=7, dtype="str", usecols=(0,1,2), unpack=True)
            baz, slow, pow = np.loadtxt(path, skiprows=7, usecols=(7,8,9), unpack=True)

            # remove restults with weak beam powers
            ind_del = np.where(pow <= powmin)[0]
            id = np.delete(id, ind_del)
            on = np.delete(on, ind_del)
            off = np.delete(off, ind_del)
            baz = np.delete(baz, ind_del)
            slow = np.delete(slow, ind_del)
            pow = np.delete(pow, ind_del)
            return [id, on, off, baz, slow, pow]
        except:
            return None


    def write_IQcat_entry(self, fh, fn, event):
        """
        Function to write icequake catalog entries
        :param fh: open file
        :param fn: corresponding file name
        :param event: event dictionary
        :return: writes entry to file fn
        """
        # write header (if file is empty)
        if os.stat(fn).st_size == 0:
            fh.write("# 1. col:  (avg) trigger onset\n")
            fh.write("# 2. col:  epicenter easting\n")
            fh.write("# 3. col:  epicenter northing\n")
            fh.write("# 4. col:  epicenter easting std\n")
            fh.write("# 5. col:  epicenter northing std\n")
            fh.write("# 6. col:  average weight\n")
            fh.write("# 7. col:  number of arrays that detected event\n")
            fh.write("# 8. col:  number of arrays that located event\n")
            fh.write("# 9. col:  event id array A0\n")
            fh.write("# 10. col:  event id array A1\n")
            fh.write("# 11. col: event id array A2\n")
            fh.write("# 12. col: event id array A3\n")
            fh.write("# =============================================\n")
            fh.flush()

        # write data to file
        id_A0 = np.nan
        id_A1 = np.nan
        id_A2 = np.nan
        id_A3 = np.nan
        for id in event["aIDs"]:
            if "A0" in id:
                id_A0 = id
            elif "A1" in id:
                id_A1 = id
            elif "A2" in id:
                id_A2 = id
            elif "A3" in id:
                id_A3 = id
        line = "%s  %8.1f  %8.1f  %6.1f  %6.1f  %5.3f  %i  %i  %11s  %11s  %11s  %11s"\
               % (UTCDateTime(event["trig_time"]), event["epicntr"][0], event["epicntr"][1],
                  event["epicntr_std"][0], event["epicntr_std"][1], event["weight"],
                  len(event["arrays"]), event["num_loc"], id_A0, id_A1, id_A2, id_A3)
        fh.write(line + "\n")


    def plot_event_loc(self, event, fn_coords):
        """
        Function to plot the event location result.
        :param event: dictionary containing event information
        :param fn_coords: file containing coordinates of array centers
        :return: plot
        """

        # coordinates of array centers
        e_coords, n_coords = np.loadtxt(fn_coords, usecols=(1,2), unpack=True)
        array_ids = np.genfromtxt(fn_coords, usecols=(0,), dtype=str)

        # coordinate bounds of image
        e_min = 602909.
        e_max = 608674.
        n_min = 135425.
        n_max = 138507.

        # read epicenter data
        epic_east = event["epicntr"][0]
        epic_north = event["epicntr"][1]
        epic_east_std = event["epicntr_std"][0]
        epic_north_std = event["epicntr_std"][1]

        # data of center stations
        t1 = UTCDateTime(event["trig_time"]) - 0.5
        t2 = UTCDateTime(event["trig_time"]) + event["trig_dur"] + 0.8
        # read data
        arrays = event["arrays"]
        st = Stream()
        for arr in arrays:
            ind_arr = int(arr[1])
            st += read(self.path2mseed + "PM%i5/%s.D/4D.PM%i5..%s.D.2016.%03d" % (ind_arr, self.chn, ind_arr,
                                                                                  self.chn, self.jday),
                       starttime=t1 - 5, endtime=t2 + 5)
        # adjust sampling rate
        for tr in st:
            if tr.stats.sampling_rate != self.fs:
                tr.resample(self.fs)
        st.filter("highpass", freq=1., zerophase=True)
        st.trim(t1, t2)
        # calc hour of day
        hour = t1.hour + t1.minute / 60. + t1.second / 3600.

        # determine icequake locations from travel time differences
        # calculate envelopes and determine relative time lags
        lags = np.zeros(len(arrays))
        for k, tr in enumerate(st):
            env = obspy.signal.filter.envelope(tr.data)
            lags[k] = np.argmax(env) * st[0].stats.delta
        # calculate travel time differences
        dlags = np.zeros((len(arrays), len(arrays)))
        for k in range(len(arrays)):
            for l in range(len(arrays)):
                dlags[k, l] = lags[k] - lags[l]
                dlags[k, k] = 0.

        # setup grid and velocity
        dx = 25
        east = np.arange(np.round(e_min, -2), np.round(e_max, -2), dx)
        north = np.arange(np.round(n_min, -2), np.round(n_max, -2), dx)
        v = 1670.

        ## calcualte traveltimes for all grid points and stations
        #dists = np.zeros((north.size, east.size, len(arrays)))
        #for n in range(north.size):
        #    for e in range(east.size):
        #        for k, arr in enumerate(arrays):
        #            ind_arr = np.where(arr == array_ids)[0]
        #            dists[n, e, k] = np.sqrt((north[n] - n_coords[ind_arr]) ** 2
        #                                     + (east[e] - e_coords[ind_arr]) ** 2)
        #tt = dists / v
        ## calculate travel time differences
        #dtt = np.zeros((tt.shape[0], tt.shape[1], len(arrays), len(arrays)))
        #for k in range(len(arrays)):
        #    for l in range(len(arrays)):
        #        dtt[:, :, k, l] = tt[:, :, k] - tt[:, :, l]

        ## calculate residuals (L1-norm!) for all travel time difference combinations and them up
        #res = np.zeros((tt.shape[0], tt.shape[1]))
        #count = 0.
        #for k in range(len(arrays)):
        #    for l in range(len(arrays)):
        #        if k > l:
        #            res[:, :] += abs(dtt[:, :, k, l] - dlags[k, l])
        #            count += 1.
        #res /= count

        # data for seismogram plotting
        data = np.zeros((st[0].data.size, 4))
        dt = st[0].stats.delta
        t = np.arange(0, st[0].data.size) * dt
        for tr in st:
            ind = int(tr.stats.station[2])
            tr.data /= abs(tr.data).max()
            data[:, ind] = tr.data

        # convert coordinates for plotting
        e_coords_ = (e_coords - e_min) / 2.
        n_coords_ = (n_coords - n_min) / 2.
        east = (east - e_min) / 2.
        north = (north - n_min) / 2.
        epic_east_ = (epic_east - e_min) / 2.
        epic_north_ = (epic_north - n_min) / 2.
        epic_east_std_ = epic_east_std / 2.
        epic_north_std_ = epic_north_std / 2.

        # radius for plotting beams as wedges and estimated beam error (+-)
        r = e_max - e_min
        err = 1.

        # plot map, beamforming results and travel time results
        # cols = ["m", "g", "r", "c"]
        cols = ["#0069B4", "#A8322D", "#6F6F6E", "#72791C"]
        img = mpimg.imread("/media/fabian/Data/PhD/PlaineMorte/AirialImage/PM_2m_res.png")
        # reverse image y axis
        for k in range(img.shape[1]):
            for l in range(img.shape[2]):
                img[:, k, l] = img[:, k, l][::-1]

        # figure
        fig = plt.figure(figsize=(20, 8))

        # axis for map and beamforming results
        ax1 = fig.add_axes([0.01, 0, 0.7, 1])
        ids = event["aIDs"]
        ax1.set_title("Event IDs:    " + "  -  ".join(ids))
        ax1.imshow(img, origin="lower")
        for arr in arrays:
            if not np.isnan(event["bazs"][arr]):
                ind_arr = np.where(arr == array_ids)[0][0]
                ax1.add_patch(patches.Wedge((e_coords_[ind_arr], n_coords_[ind_arr]), r,
                                            -(event["bazs"][arr] - 90 + err), -(event["bazs"][arr] - 90 - err),
                                            color=cols[ind_arr], alpha=0.5))
        ax1.errorbar(epic_east_, epic_north_, epic_north_std_, epic_east_std_, color="k")
        #tmin = res.min()
        #tmax = res.max() * 0.5
        #if tmax < tmin:
        #    tmax = res.max()
        #bp = ax1.pcolormesh(east + dx / 2., north + dx / 2., res, alpha=0.3, cmap="CMRmap", vmin=tmin, vmax=tmax)
        #cb = fig.colorbar(bp, shrink=0.78, pad=0.02)
        #cb.set_label("Residual (s)")
        ax1.set_xlim(0, (e_max - e_min) / 2. - 20)
        ax1.set_ylim(0, (n_max - n_min) / 2. - 20)
        plt.axis("off")

        # axis for seismograms
        ax2 = fig.add_axes([0.72, 0.090, 0.25, 0.875])
        for arr in arrays:
            ind_arr = np.where(arr == array_ids)[0][0]
            if data[:, ind_arr].all() == 0.:
                data[:, ind_arr] = np.nan
            ax2.plot(t, data[:, ind_arr] - (ind_arr + 1) * 1.5, cols[ind_arr])
        ax2.set_ylim(-7, 0)
        ax2.set_xlim(t[0], t[-1])
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.set_xlabel("Time (s)")

        # plt.savefig("/home/fabian/Desktop/%03d_%04d.png" % (self.jday, i+1))
        plt.show()
        plt.close()


    def trigger_events(self, ftmin, ftmax, nsta, nlta, thrsh1, thrsh2, num_trig, plot=True):
        """
        Function to trigger events from continuous data.
        :param ftmin: lower corner frequency used for filtering prior to triggering
        :param ftmax: upper corner frequency used for filtering prior to triggering
        :param nsta: number of samples short term average
        :param nlta: number of samples long term average
        :param thrsh1: nsta/nlta threshold to trigger
        :param thrsh2: nsta/nlta threshold to stop trigger
        :param num_trig: number of stations required to eventually trigger an event
        :param plot: if true, gives an overview of triggered events
        """
        print("TRIGGER EVENTS ...")
        print("STA: %i samples" % nsta)
        print("LTA: %i samples" % nlta)
        print("-------------------------------------------------")

        # trigger on continuous data from all array stations
        trig_times = {}
        dict_env_maxs = {}
        # read data
        for stn in self.stnlist:
            try:
                st = read(self.path2mseed + "%s/%s.D/4D.%s..%s.D.2016.%i" % (stn, self.chn, stn, self.chn, self.jday))
            except:
                print("%s: no data!!!" % stn)
                print("skip this day!")
                sys.exit(1)
            # adjust sampling rate
            for tr in st:
                if tr.stats.sampling_rate != self.fs:
                    tr.resample(self.fs)
            if self.decfact > 1:
                st.decimate(self.decfact)
            # trim and filter
            t1 = UTCDateTime(2016, 1, 1)
            t1.julday = self.jday
            t2 = t1 + 24. * 60. * 60.
            st.trim(t1, t2)
            dt = st[0].stats.delta
            st.filter("bandpass", freqmin=ftmin, freqmax=ftmax, zerophase=True)

            # for station, trigger all traces and count number of events
            n_ev = 0
            for tr in st:
                cft = classic_sta_lta(tr.data, nsta, nlta)
                on_off = trigger_onset(cft, thrsh1, thrsh2)
                if plot:
                    plot_trigger(tr, cft, thrsh1, thrsh2)
                if len(on_off) > 0:
                    n_ev += on_off.shape[0]

            # trigger again and convert trigger time to timestring format
            ons = np.zeros(n_ev)
            offs = np.zeros(n_ev)
            ind = 0
            for i, tr in enumerate(st):
                cft = classic_sta_lta(tr.data, nsta, nlta)
                trig = trigger_onset(cft, thrsh1, thrsh2)
                if len(trig) > 0:
                    ons[ind:ind+trig.shape[0]] = trig.astype(float)[:,0] * dt + tr.stats.starttime.timestamp
                    offs[ind:ind+trig.shape[0]] = trig.astype(float)[:,1] * dt + tr.stats.starttime.timestamp
                    ind += trig.shape[0]

            # remove events which are triggered within one second after an event
            tt = 3.
            ind_del = []
            for i in range(len(ons) - 1):
                print(ons[i+1] - ons[i])
                if (ons[i+1] - ons[i]) < tt:
                    ind_del.append(i+1)
            ons = np.delete(ons, ind_del)
            offs = np.delete(offs, ind_del)
            print("%s: %i events detected!" % (stn, len(ons)))

            # recalculate on/off times (defined as the interval where the envelope is greater than 0.2 times its max)
            n = 0.4
            p = 0.6
            env_maxs = np.zeros(len(ons))
            for i in range(len(ons)):
                ts = UTCDateTime(ons[i]) - n
                te = UTCDateTime(offs[i]) + p
                st_ = read(self.path2mseed + "%s/%s.D/4D.%s..%s.D.2016.%i" % (stn, self.chn, stn, self.chn, self.jday),
                           starttime=ts-5, endtime=te+5)
                # adjust sampling rate
                for tr in st_:
                    if tr.stats.sampling_rate != self.fs:
                        tr.resample(self.fs)
                st_.filter("bandpass", freqmin=self.fmin, freqmax=self.fmax, zerophase=True)
                st_.trim(ts, te)
                # calc envelope
                env = obspy.signal.filter.envelope(st_[0].data)
                env_max = np.max(env)
                env_maxs[i] = ts.timestamp + np.argmax(env) * dt
                ind_greater = np.where(env >= 0.2*env_max)[0]
                ediff = np.ediff1d(ind_greater)
                ind_greater = np.split(ind_greater, np.where(ediff != 1)[0] + 1)

                for j in range(len(ind_greater)):
                    if np.argmax(env) in ind_greater[j]:
                        ons[i] = ts.timestamp + ind_greater[j][0]*dt
                        offs[i] = ts.timestamp + ind_greater[j][-1]*dt
                        #plt.plot(st_[0].data)
                        #plt.plot(env, "k:")
                        #plt.plot(ind_greater[j], env[ind_greater[j]], "r")
                        #plt.show()
            on_off_ = np.zeros((len(ons),2))
            on_off_[:,0] = ons
            on_off_[:,1] = offs
            trig_times[stn] = on_off_
            dict_env_maxs[stn] = env_maxs

        # take only events which are triggered on at least num_trig stations (take travel time for very slow velocities
        #  across the array radius as limit)
        tt = self.r / 1000.
        self.stnlist.sort()
        cstn = self.stnlist[4]

        on_off_ = []
        env_max_PMx5 = dict_env_maxs[cstn]
        for i in range(len(env_max_PMx5)):
            env_PMx5 = env_max_PMx5[i]
            count = 0
            for j in range(len(self.stnlist)-1):
                stn = self.stnlist[j]
                ind = np.where(((dict_env_maxs[stn] - tt) < env_PMx5) & (env_PMx5 < (dict_env_maxs[stn] + tt)))[0]
                if len(ind) == 1:
                    count += 1
            if count >= (num_trig - 1):
                add = np.array(trig_times[cstn][i])
                on_off_.append(add)
        print("--> %i events detected on >= %i stations!" % (len(on_off_), num_trig))
        print("------------------------------------------")

        # convert to numpy array
        on_off = np.zeros((len(on_off_),2))
        for i in range(len(on_off_)):
            on_off[i,:] = on_off_[i]

        return on_off



    def beamform_icequakes(self, on_off, coords, select_iq=False, show_res=False):
        """
        Function to perform plaine wave beamforming on triggerd event. Also calculates the peak frequency and peak
        amplitude averaged over the array. Writes results to file.
        :param on_off: trigger on/off times
        :param coords: coordinates of array stations (consecutively numbered station name required)
        :param select_iq: if True, event will be displayed and user can decide, whether event will be further processed
        :param show_res: if True, beamforming result (along with waveforms) will be plotted and saved.
        """
        # open file and create header
        path_ = self.path2DBs + "%s/EventDB/" % (self.array)
        if not os.path.exists(path_):
            os.makedirs(path_)
        fh = open(path_ + "%s_EventDB_%03d.txt" % (self.array, self.jday), "w")
        icequake_locations.make_eventDB_header(self, fh)

        # loop over triggered events events
        print("BEAMFORM EVENTS ...")
        for k in range(len(on_off)):
            print("event %i/%i ..." % (k, len(on_off)-1))
            try:
                # icequake interval
                intvl = on_off[k]
                dur = intvl[1] - intvl[0]
                # calc travel time from margin to center of array (use rayleigh wave as slowest phase)
                tt = self.r / 1600.
                ts = UTCDateTime(intvl[0] - tt)
                te = UTCDateTime(intvl[1] + tt)
                # read and process data
                cont = Stream()
                for stn in self.stnlist:
                    cont += read(self.path2mseed + "%s/%s.D/4D.%s..%s.D.2016.%i*" % (stn, self.chn, stn,
                                 self.chn, self.jday), starttime=ts-1, endtime=te+1)
                if len(cont) < len(self.stnlist):
                    print("only %i stations recorded event - skipped event !" % len(cont))
                    sys.exit(1)
                # adjust sampling rate
                for tr in cont:
                    if tr.stats.sampling_rate != self.fs:
                        tr.resample(self.fs)
                if self.decfact > 1:
                    cont.decimate(self.decfact)
                df = cont[0].stats.sampling_rate
                cont.detrend("linear")
                cont.taper(max_percentage=0.1)
                cont.filter("highpass", freq=1., zerophase=True)
                cont_filt = cont.copy()
                cont_filt.filter("bandpass", freqmin=self.fmin, freqmax=self.fmax, zerophase=True)

                # calc peak frequency and peak amplitude
                pfreq, pampl = icequake_locations.calc_pfreq_pampl(self, cont)
                cont_filt.trim(ts, te)
                cont.trim(ts, te)

                # calc time shifts of seismograms with respect to center station
                avg_delay = icequake_locations.calc_average_delay(self, cont_filt)
                #dt = 1. / df
                #if self.array == "A0":
                #    thrsh_dt = 4*dt
                #else:
                #    thrsh_dt = 1*dt
                #if abs(avg_delay) > thrsh_dt:
                #    print("... not located!")
                #    baz = None
                #    s = None
                #    beam = None
                #    icequake_locations.make_eventDB_entry(self, fh, (k+1), ts, te, pampl, pfreq,
                #                                          dur, avg_delay, baz, s, beam)
                #    sys.exit(1)

                # if true, show icequake before applying beamforming
                if select_iq:
                    all = Stream()
                    for tr in cont:
                        all += tr
                    for tr in cont_filt:
                        tr.stats.network = "4D_"
                        all += tr
                    all.plot(method="full")
                    input_var = str(input("Use this IQ (y/n)?"))
                    if input_var == "n":
                        # force quit
                        sys.exit(1)

                # prepare for beamforming
                npts = len(cont[0].data)
                data = np.zeros((npts, len(self.stnlist)))
                for i, tr in enumerate(cont):
                    if tr.stats.npts != npts:
                        raise ValueError("different number of samples !!!")
                    data[:, i] = tr.data.astype(float)
                # adjust window length and delay
                w_frac = 0.85
                w_length = w_frac * (te - ts)
                w_delay = (1 - w_frac) * (te - ts) / 10.

                # beamforming
                baz, s, beam = plwave_beamformer(data, coords, self.vmin, self.vmax, self.dv, False, 0,
                                                 self.fmin, self.fmax, df, w_length, w_delay, df=0.25)
                # write entry to eventDB 
                icequake_locations.make_eventDB_entry(self, fh, (k+1), ts, te, pampl, pfreq,
                                                      dur, avg_delay, baz, s, beam)

                # if true, visualize icequake and beamforming result
                if show_res:
                    fig = plt.figure(figsize=(20,10))
                    ax1 = fig.add_subplot(121)
                    t = np.arange(0, npts*dt, dt)
                    for i in range(len(self.stnlist)):
                        d = data[:, i]
                        d /= abs(data).max()
                        d -= i * 1.5
                        ax1.plot(t, d, "k")
                        ax1.text(t[3], np.mean(d) + 0.3, self.stnlist[i], fontsize=16)
                    ax1.set_xlabel("seconds")
                    ax1.set_yticklabels([])
                    ax1.set_title("starttime: %s" % ts)
                    ax2 = fig.add_subplot(122, projection="polar")
                    ax2.set_theta_direction(-1)
                    ax2.set_theta_zero_location("N")
                    im = ax2.pcolormesh(np.radians(baz), s, beam, cmap="viridis")
                    ind_baz = np.argmax(np.amax(beam, axis=0))
                    ind_slow = np.argmax(np.amax(beam, axis=1))
                    ax2.plot(np.radians(baz[ind_baz]), s[ind_slow], "ko", markersize=2)
                    ax2.set_title("pow = %.3f - vel = %.3f m/s - baz = %i deg"
                                  % (beam.max(), 1./s[ind_slow], baz[ind_baz]))
                    pth = "./%s/%03d/" % (self.array, self.jday)
                    id = "%03d_%s_%04d" % (self.jday, self.array, k+1)
                    plt.savefig(pth + id, dpi=50)
                    plt.close()
            except:
                pass
        fh.close()



    def associate_icequakes(self, arrays, powmin, narr):
        """
        Function to associate events recorded on different arrays.
        :param arrays: list containing arrays which are used for association
        :param powmin: minimum beam power required to use event for association
        :param narr: number of arrays required for triangulation
        :return: list containing dictionaries with information of associated events
        """
        # read eventDB of all arrays for jday
        dict_cat = OrderedDict()
        for arr in arrays:
            DB_arr = icequake_locations.read_eventDB(self, arr, powmin)
            if DB_arr is not None:
                dict_cat[arr] = DB_arr
        # update array list
        arrays = list(dict_cat.keys())

        # test if at least 2 arrays have detected icequakes for specified julday
        if len(dict_cat) < 2:
            print("Only %i arrays detected icequakes on DOY %03d - exit!!!" % (len(dict_cat), self.jday))
            sys.exit(1)
        else:
            print("%i arrays detected icequakes on DOY %03d - associate events ..." % (len(dict_cat), self.jday))


        # read trigger on times and convert to timestamp format
        ons = []
        for arr in arrays:
            on_x = dict_cat[arr][1]
            for i in range(on_x.size):
                on_x[i] = UTCDateTime(on_x[i]).timestamp
            on_x = on_x.astype(float)
            ons.append(on_x)

        # max travel time: PM35 -> PM05 assume vel=1500 m/s
        tt = 2377. / 1500.

        # prepare for association
        assoc_iqs = []

        # loop over ons, compare to other ons
        for i in range(len(ons)-1):
            on = ons[i]
            for j in range(on.size):
                dict_ind = {}
                dict_ind[arrays[i]] = j

                # select other next on to compare
                ind_on2_ = i+1
                if ind_on2_ >= len(ons):
                    ind_on2_ -= len(ons)
                ind2 = np.where((on[j]-tt < ons[ind_on2_]) & (ons[ind_on2_] < on[j]+tt))[0]
                if len(ind2) == 1:
                    dict_ind[arrays[ind_on2_]] = ind2[0]

                # select second next on to compare
                if len(dict_cat) > 2:
                    ind_on3_ = i+2
                    if ind_on3_ >= len(ons):
                        ind_on3_ -= len(ons)
                    ind3 = np.where((on[j]-tt < ons[ind_on3_]) & (ons[ind_on3_] < on[j]+tt))[0]
                    if len(ind3) == 1:
                        dict_ind[arrays[ind_on3_]] = ind3[0]

                # select third next on to compare
                if len(dict_cat) > 3:
                    ind_on4_ = i+3
                    if ind_on4_ >= len(ons):
                        ind_on4_ -= len(ons)
                    ind4 = np.where((on[j]-tt < ons[ind_on4_]) & (ons[ind_on4_] < on[j]+tt))[0]
                    if len(ind4) == 1:
                        dict_ind[arrays[ind_on4_]] = ind4[0]

                # append only to associated event, if dict_ind has at least narr entries
                if len(dict_ind) >= narr:
                    assoc_iqs.append(dict_ind)


        # delete multiple entries
        assoc_iqs_ = assoc_iqs
        assoc_iqs = []
        for i in range(0, len(assoc_iqs_)):
            if assoc_iqs_[i] not in assoc_iqs_[i+1:]:
                assoc_iqs.append(assoc_iqs_[i])
        print("Associated %i events!" % len(assoc_iqs))

        # create dictionary gathering some information for each associated event
        assoc_iqs_ = assoc_iqs
        assoc_iqs = []
        for i in range(len(assoc_iqs_)):
            event = OrderedDict()
            # event ID
            event["EventID"] = "%03d_%04d" % (self.jday, i)
            # arrays
            arrays_ = list(assoc_iqs_[i].keys())
            arrays_.sort()
            event["arrays"] = arrays_
            # array IDs, avg. trigger on time, avg. trigger duration, baz
            ids = []
            tons = []
            toffs = []
            bazs = {}
            for arr in arrays_:
                ids.append(dict_cat[arr][0][assoc_iqs_[i][arr]])
                tons.append(float(dict_cat[arr][1][assoc_iqs_[i][arr]]))
                toffs.append(UTCDateTime(dict_cat[arr][2][assoc_iqs_[i][arr]]).timestamp)
                bazs[arr] = dict_cat[arr][3][assoc_iqs_[i][arr]]
            ids.sort()
            event["aIDs"] = ids
            event["trig_time"] = np.mean(tons)
            event["trig_dur"] = np.mean(np.array(toffs) - np.array(tons))
            event["bazs"] = bazs
            assoc_iqs.append(event)

        return assoc_iqs


    def locate_events(self, assoc_iqs, fn_cat, fn_coords, plot=False):
        """
        Function to locate associated icequakes using triangulation.
        :param assoc_iqs: list containing containing event dictionaries
        :param fn_cat: file to which results are written
        :param fn_coords: file containing array center coordinates
        :param plot: if True, shows a map with the location result for each event
        :return: writes results to icequake catalog
        """

        # coordinates of array centers
        e_coords, n_coords = np.loadtxt(fn_coords, usecols=(1,2), unpack=True)
        array_ids = np.genfromtxt(fn_coords, usecols=(0,), dtype=str)

        # open icequake catalog file
        fh = open(fn_cat, "a")

        # loop over associated events and calculate intersection of beams
        for i in range(len(assoc_iqs)):
            # slope and y-axis intersection of beams
            event = assoc_iqs[i]
            arrays = event["arrays"]
            slopes = {}
            intcpts = {}
            for arr in arrays:
                ind_arr = np.where(arr == array_ids)[0]
                slopes[arr] = 1. / np.tan(np.radians(event["bazs"][arr]))
                intcpts[arr] = n_coords[ind_arr][0] - slopes[arr] * e_coords[ind_arr][0]
            event["slopes"] = slopes
            event["intcpts"] = intcpts

            # convert baz values to degree range [0, 180]
            event["bazs_"] = dict(event["bazs"])
            for arr in arrays:
                if not np.isnan(event["bazs_"][arr]):
                    if event["bazs_"][arr] > 180.:
                        event["bazs_"][arr] -= 180.


            # prepare for calculation of beam intersections
            epics_east = []     # epicenters east coordinate
            epics_north = []    # epicenters north coordinate
            weights = []

            # calculate intersection of first beam with all following beam
            for j in range(1,len(arrays)):
                if event["bazs_"][arrays[0]] != event["bazs_"][arrays[j]]:
                    iq_east = (event["intcpts"][arrays[j]] - event["intcpts"][arrays[0]])\
                            / (event["slopes"][arrays[0]] - event["slopes"][arrays[j]])
                else:
                    iq_east = np.nan
                epics_east.append(iq_east)
                iq_north = event["slopes"][arrays[0]] * iq_east + event["intcpts"][arrays[0]]
                epics_north.append(iq_north)
                weights.append(abs(event["bazs_"][arrays[j]] - event["bazs_"][arrays[0]]))

            # calculate intersection of second beam with all following beams (if applicable)
            if len(arrays) > 2:
                for l in range(2,len(arrays)):
                    if event["bazs_"][arrays[1]] != event["bazs_"][arrays[l]]:
                        iq_east = (event["intcpts"][arrays[l]] - event["intcpts"][arrays[1]]) \
                                  / (event["slopes"][arrays[1]] - event["slopes"][arrays[l]])
                    else:
                        iq_east = np.nan
                    epics_east.append(iq_east)
                    iq_north = event["slopes"][arrays[1]] * iq_east + event["intcpts"][arrays[1]]
                    epics_north.append(iq_north)
                    weights.append(abs(event["bazs_"][arrays[l]] - event["bazs_"][arrays[1]]))

            # calculate intersection of third beam with the last beam (if applicable)
            if len(arrays) == 4:
                if event["bazs_"][arrays[2]] != event["bazs_"][arrays[3]]:
                    iq_east = (event["intcpts"][arrays[3]] - event["intcpts"][arrays[2]])\
                            / (event["slopes"][arrays[2]] - event["slopes"][arrays[3]])
                else:
                    iq_east = np.nan
                epics_east.append(iq_east)
                iq_north = event["slopes"][arrays[2]] * iq_east + event["intcpts"][arrays[2]]
                epics_north.append(iq_north)
                weights.append(abs(event["bazs_"][arrays[3]] - event["bazs_"][arrays[2]]))
            del event["bazs_"]


            # remove nans
            epics_east = np.array(epics_east)
            epics_north = np.array(epics_north)
            weights = np.array(weights)
            ind_del = np.where(np.isnan(epics_east))
            epics_east = np.delete(epics_east, ind_del)
            epics_north = np.delete(epics_north, ind_del)
            weights = np.delete(weights, ind_del)

            # convert weights such that they are in the interval [0,90]
            weights[weights > 90.] -= 180
            weights = abs(weights)
            weights /= 90.

            # calculate average and standard deviation
            if len(epics_east) > 0:
                epic_east = np.average(epics_east, weights=weights)
                epic_north = np.average(epics_north, weights=weights)
                epic_east_std = np.std(epics_east)
                epic_north_std = np.std(epics_north)
                weights_mean = np.average(weights)
            else:
                epic_east = np.nan
                epic_north = np.nan
                epic_east_std = np.nan
                epic_north_std = np.nan
                weights_mean = np.nan

            # update event dictionary
            event["epicntr"] = [epic_east, epic_north]
            event["epicntr_std"] = [epic_east_std, epic_north_std]
            event["weight"] = weights_mean
            bazs = np.array(event["bazs"].values())
            bazs = np.delete(bazs, np.nan)
            event["num_loc"] = len(bazs)

            # wirte to file
            icequake_locations.write_IQcat_entry(self, fh, fn_cat, event)

            # show map
            if plot:
                icequake_locations.plot_event_loc(self, event, fn_coords)

        fh.close()
