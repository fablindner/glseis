from scipy import signal
import scipy
import numpy as np
import matplotlib.pyplot as plt
import warnings
from obspy import UTCDateTime


def nearest_powof2(number):
    """
    This function determines the next smaller value to "number" which is of power 2
    :type number: interger
    :param number: number of samples

    :return: next smaller value of power 2 which is smaller than number
    """
    x = np.arange(0, 27)
    pow2 = 2**x
    ind = np.argmin(abs(number - pow2))
    if pow2[ind] < number:
        res = pow2[ind]
    else:
        res = pow2[ind - 1]
    return res


def transfer_function(u, freq, easting, northing, elevation):
    """
    Function to calculate the response of an array.
    :type u: numpy.array
    :param u: array containing slowness values of consideration
    :type freq: float
    :param freq: frequency for which the array response is calculated
    :type easting: numpy.array
    :param easting: coordinates of stations in x-direction in meters
    :type northing: numpy.array
    :param northing: coordinates of stations in y-direction in meters
    :type elevation: numpy.array
    :param elevation: elevation of stations 

    """
    # array coordinate mean as array reference point
    meanarrayeast = np.mean(easting)
    meanarraynorth = np.mean(northing)
    # distance statations to reference point
    x = np.zeros(easting.size)
    y = np.zeros(northing.size)
    for i in range(northing.size):
        x[i] = (easting[i] - meanarrayeast) / 1000.
        y[i] = (northing[i] - meanarraynorth) / 1000.

    theo_backazi = np.radians(np.arange(0, 361, 1))
    theo_backazi = theo_backazi[:, None]
    # transfer function of input array geometry 2D
    nstats = x.shape
    beamres = np.zeros((theo_backazi.size, u.size))
    R = np.ones((nstats[0], nstats[0]))
    for vel in range(len(u)):
        kx =  np.cos(theo_backazi) * u[vel]
        ky =  np.sin(theo_backazi) * u[vel]
        e_steer = np.exp(1j * 2. * np.pi * freq * (kx * x + ky * y))
        w = e_steer
        wT = w.T.copy()
        beamres[:, vel] = (1. / nstats[0]**2) * abs((np.conjugate(w) * np.dot(R, wT).T).sum(1))

    # plotting
    fig = plt.figure()
    ax_array = fig.add_subplot(211)
    elev = ax_array.scatter(easting-meanarrayeast, northing-meanarraynorth, c=elevation,
            s=150, marker="^", vmin=elevation.min(), vmax=elevation.max())
    ax_array.set_xlabel("Easting [m] rel. to array center")
    ax_array.set_ylabel("Northing [m] rel. to array center")
    ax_array.set_title("Array Configuration")
    cbar_array = plt.colorbar(elev)
    cbar_array.set_label("Elevation (m)")
    ax = fig.add_subplot(212, projection='polar')
    theo_backazi = theo_backazi[:, 0]
    CONTF = ax.contourf((theo_backazi), u, beamres.T, 100, cmap='jet', antialiased=True, linstyles='dotted')
    ax.set_rmax(u[-1])
    cbar = plt.colorbar(CONTF)
    cbar.set_label('Rel. Power')
    ax.grid(True)
    ax.text(np.radians(32), u.max() + 0.01, 's/km', color='k')
    ax.set_title('Array Response Function, f=%.1f Hz' % freq)
    plt.tight_layout()
    plt.show()


def array_response_wathelet(easting, northing, kmax, kstep, show_greater_thresh=False,
        outfile=None):
    """
    Function to calculate the array response function as in Wathelet et al., 2008 (J.Seismol.).
    Also calculates the resolution limits in terms of wavelengths as discussed in the paper.

    :param easting: Easting coordinates of stations.
    :param northing: Northing coordinates of stations.
    :param kmax: Maximum wavenumber considered.
    :param kstep: Step in wavenumber.
    :param show_greater_thresh:
    :param outfile: File name for saving plot.
    """
    # runtime
    t1 = UTCDateTime()
    # initialize kx and ky
    kxmax = kmax
    kymax = kmax
    kx = np.arange(-kxmax, kxmax + kstep, kstep)
    ky = np.arange(-kymax, kymax + kstep, kstep)
    KX, KY = np.meshgrid(kx, ky)
    # threshold for kmin and kmax
    thresh = 0.5
    # initialize and calculate the array response
    # -> compare with Wathelet (2008), equation 3
    sum_rth = np.zeros(KX.shape, dtype=complex) 
    for i in range(easting.size):
        sum_rth += np.exp(-1j * (KX * easting[i] + KY * northing[i]))
    Rth = abs(sum_rth) ** 2 / float(easting.size) ** 2
    Rth = np.round_(Rth, decimals=2)

    # get distance to smallest 0.5 value from center
    KX_thresh = np.ma.masked_where(Rth != thresh, KX)
    KY_thresh = np.ma.masked_where(Rth != thresh, KY)
    kmin = np.sqrt(KX_thresh**2 + KY_thresh**2).min()

    ## get response values greater than the threshold and calclate corresponding k values
    #gr_thresh = np.where(Rth > thresh)
    #k_gr_thresh = np.zeros(gr_thresh[0].size)
    #for l in range(gr_thresh[0].size):
    #    k_gr_thresh[l] = np.sqrt(kx[gr_thresh[1][l]] ** 2 + ky[gr_thresh[0][l]] ** 2)
    ## distribution of k values greater than the threshold
    #hist, bins = np.histogram(k_gr_thresh, bins=int(kx.size / 2))
    ## obtain the two threshold values (see Wathelet, 2008)
    #k_thresh = []
    #first_peak = True  # corresponds to zero slowness
    #for b in range(hist.size - 1):
    #    if hist[b] > 0 and hist[b + 1] == 0:
    #        k_thresh.append((bins[b] + bins[b + 1]) / 2.)
    #        first_peak = False
    #    if hist[b] == 0 and hist[b + 1] > 0 and not first_peak:
    #        k_thresh.append((bins[b] + bins[b + 1]) / 2.)
    #min_wvnmbr = k_thresh[0]
    #max_wvnmbr = k_thresh[1]
    ## calculate corresponding wavelengths
    #lambdamin = 2. * np.pi / max_wvnmbr
    #lambdamax = 2. * np.pi / min_wvnmbr

    # distance between stations
    d = np.zeros((easting.size, easting.size))
    for i in range(easting.size):
        for j in range(easting.size):
            d[i,j] = np.sqrt((easting[i] - easting[j])**2 + (northing[i] - northing[j])**2)
    # resoluion limints according to Tokimatsu (1997) -> see Wathelet et al. (2008)
    dmin = 2 * d[d>0].min()
    dmax = 3 * d[d>0].max()

    # plot
    fig = plt.figure(figsize=(10, 3.65))

    # array geometry
    ax1 = fig.add_subplot(121)
    ax1.plot(easting, northing, "kv", markersize=12, mec="silver")
    ax1.text(0.05, 0.9, "Tokimatsu (1997): %.1f m < $\lambda$ < %.1f m" % (dmin, dmax),
             transform=ax1.transAxes, fontsize=10)
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    ax1.set_title("Array Geometry")
    ax1.set_xlim(easting.min()-50., easting.max()+50.)
    ax1.set_ylim(northing.min()-50., northing.max()+50.)
    plt.axis("equal")

    # array response
    ax2 = fig.add_subplot(122)
    im = ax2.pcolormesh(kx - kstep / 2., ky - kstep / 2., Rth, vmin=0.0, vmax=1., cmap="viridis",
                        rasterized=True)
    ax2.contour(KX, KY, Rth, colors="k", levels=[0.5], linewidths=0.8)
    levels = 2. * np.pi / np.linspace(kmin, kmax, 6)
    cs = ax2.contour(KX, KY, 2. * np.pi / np.sqrt(KX**2 + KY**2), colors="w",
            levels=levels[::-1], linewidths=0.8, linestyles="--")
    ax2.clabel(cs, fontsize=9, inline=True, fmt="%i")
    if show_greater_thresh:
        ax2.plot(ky[gr_thresh[1]], kx[gr_thresh[0]], "r.", alpha=0.6, label="> %.2f" % thresh)
    cbar = plt.colorbar(im)
    cbar.set_label("Beam Power")
    ax2.set_xlabel("wavenumber $k_x$ (rad/m)")
    ax2.set_ylabel("wavenumber $k_y$ (rad/m)")
    an = np.linspace(0, 2 * np.pi, 100)
    #ax2.plot(max_wvnmbr * np.cos(an), max_wvnmbr * np.sin(an), "w--",
    #        label="$k_{max}$: $\lambda$=%im" % lambdamin)
    #ax2.plot(min_wvnmbr * np.cos(an), min_wvnmbr * np.sin(an), "w",
    #        label="$k_{min}$: $\lambda$=%im" % lambdamax)
    #legend = plt.legend(loc=1, labelspacing=0, borderpad=0.2)
    #frame = legend.get_frame()
    #frame.set_facecolor('0.70')
    ax2.set_xlim(-kmax, kmax)
    ax2.set_ylim(-kmax, kmax)
    ax2.set_title("Theoretical array response", fontsize=10)
    plt.axis("equal")
    # runtime
    t2 = UTCDateTime()
    print("runtime: %.1f s" % (t2 - t1))
    if outfile is not None:
        plt.savefig(outfile, format=outfile.split(".")[-1], bbox_inches="tight")
    plt.show()



def annul_dominant_interferers(CSDM, neig, data):
    """
    This routine cancels the strong interferers from the data by projecting the
    dominant eigenvectors of the cross-spectral-density matrix out of the data.
    :type CSDM: numpy.ndarray
    :param CSDM: cross-spectral-density matrix obtained from the data.
    :type neig: integer
    :param neig: number of dominant CSDM eigenvectors to annul from the data.
    :type data: numpy.ndarray
    :param data: the data which was used to calculate the CSDM. The projector is
        applied to it in order to cancel the strongest interferer.

    :return: numpy.ndarray
        csdm: the new cross-spectral-density matrix calculated from the data after
        the projector was applied to eliminate the strongest source.
    """

    # perform singular value decomposition to CSDM matrix
    u, s, vT = np.linalg.svd(CSDM)
    # chose only neig strongest eigenvectors
    u_m = u[:, :neig]   # columns are eigenvectors
    v_m = vT[:neig, :]  # rows (!) are eigenvectors
    # set-up projector
    proj = np.identity(CSDM.shape[0]) - np.dot(u_m, v_m)
    # apply projector to data - project largest eigenvectors out of data
    data = np.dot(proj, data)
    # calculate projected cross spectral density matrix
    csdm = np.dot(data, data.conj().T)
    return csdm


def csdm_eigvals(matr, fmin, fmax, Fs, w_length, w_delay, df=0.2, norm=True):
    """
    This routine estimates the back azimuth and phase velocity of incoming waves
    based on the algorithm presented in Corciulo et al., 2012 (in Geophysics).

    :type matr: numpy.ndarray
    :param matr: time series of used stations (dim: [number of samples, number of stations])
    :type fmin, fmax: float
    :param fmin, fmax: frequency range for which the beamforming result is calculated
    :type Fs: float
    :param Fs: sampling rate of data streams
    :type w_length: float
    :param w_length: length of sliding window in seconds. result is "averaged" over windows
    :type w_delay: float
    :param w_delay: delay of sliding window in seconds with respect to previous window
    :type df: float
    :param df: frequency step between fmin and fmax
    :type norm: boolean
    :param norm: if True (default), beam power is normalized

    :return: array holding the eigenvalues of the CSDM matrix

    Note: the body of this function is taken from the function "plwave_beamformer"
        as of Jun 15 2018.
    """

    data = matr
    # number of stations
    n_stats = data.shape[1]
    # extract number of data points
    Nombre = data[:, 1].size
    # construct time window
    time = np.arange(0, Nombre) / Fs
    # construct analysis frequencies
    indice_freq = np.arange(fmin, fmax+df, df)
    # construct analysis window for entire hour and delay
    interval = np.arange(0, np.ceil(w_length * Fs) + 1, dtype=int)
    delay = int(w_delay * Fs)
    # number of analysis windows ('shots')
    if delay > 0:
        numero_shots = (Nombre - len(interval)) // delay + 1
    elif delay == 0:
        numero_shots = 1
    
    # initialize data steering vector:
    # dim: [number of frequencies, number of stations, number of analysis windows]
    vect_data_adaptive = np.zeros((len(indice_freq), n_stats, numero_shots), dtype=np.complex)
    
    # construct matrix for DFT calculation
    # dim: [number time points, number frequencies]
    matrice_int = np.exp(2. * np.pi * 1j * np.dot(time[interval][:, None], indice_freq[:, None].T))

    # loop over stations
    for ii in range(n_stats):
        toto = data[:, ii]
        # now loop over shots
        for jj in range(numero_shots):
        #while (numero * delay + len(interval)) <= len(toto):
            # calculate DFT
            # dim: [number frequencies]
            adjust = np.dot(toto[jj * delay + interval][:, None], np.ones((1, len(indice_freq))))
            test = np.mean(np.multiply(adjust, matrice_int), axis=0)  # mean averages over time axis
            # fill data steering vector: ii'th station, numero'th shot.
            # normalize in order not to bias strongest seismogram.
            # dim: [number frequencies, number stations, number shots]
            vect_data_adaptive[:, ii, jj] = (test / abs(test)).conj().T
            #numero += 1

    eigvals = np.zeros(n_stats)
    # loop over frequencies
    for ll in range(len(indice_freq)):
        # calculate cross-spectral density matrix
        # dim: [number of stations X number of stations]
        K = np.dot(vect_data_adaptive[ll, :, :], vect_data_adaptive[ll, :, :].conj().T)
    
        if np.linalg.matrix_rank(K) < n_stats:
            warnings.warn("Warning! Poorly conditioned cross-spectral-density matrix.")

        vals = abs(scipy.linalg.eigvals(K)).astype(float)
        eigvals += np.sort(vals)[::-1]
    
    return eigvals / len(indice_freq)



def calculate_CSDM(dft_array, neig=0, norm=True):
    """
    Calculate CSDM matrix for beamforming.
    :param dft_array: 2-Dim array containing DFTs of all stations
        and for multiple time windows. dim: [number stations, number windows]
    :param neig: Number of eigenvalues to project out.
    :param norm: If True, normalize CSDM matrix.
    """
    # CSDM matrix
    K = np.dot(dft_array, dft_array.conj().T)
    if np.linalg.matrix_rank(K) < dft_array.shape[0]:
        warnings.warn("Warning! Poorly conditioned cross-spectral-density matrix.")

    # annul dominant source 
    if neig > 0:
        K = annul_dominant_interferers(K, neig, dft_array)

    # normalize
    if norm:
        K /= np.linalg.norm(K)

    return K



def phase_matching(replica, K, processor):
    """
    Do phase matching of the replica vector with the CSDM matrix.
    :param replica: 2-D array containing the replica vectors of all parameter
        combinations (dim: [n_stats, n_param])
    :param K: 2-D array CSDM matrix (dim: [n_stats, n_stats])
    :param processor: Processor used for phase matching. bartlett or adaptive.
    """
    # calcualte inverse of CSDM matrix for adaptive processor
    if processor == "adaptive":
        K = np.linalg.inv(K)

    # reshape K matrix (or inverse of K) and append copy of it n_param times
    # along third dimension
    n_stats, n_param = replica.shape
    K = np.reshape(K, (n_stats, n_stats, 1))
    K = np.tile(K, (1, 1, n_param))

    # bartlett processor
    if processor == "bartlett":
        # initialize array for dot product
        dot1 = np.zeros((n_stats, n_param), dtype=complex)
        # first dot product - replica.conj().T with K
        for i in range(n_stats):
            dot1[i] = np.sum(np.multiply(replica.conj(), K[:,i,:]), axis=0)
        # second dot product - dot1 with replica
        beam = abs(np.sum(np.multiply(dot1, replica), axis=0))

    # adaptive processor
    elif processor == "adaptive":
        # initialize array for dot product
        dot1 = np.zeros((n_stats, n_param), dtype=complex)
        # first dot product - replica.conj().T with K_inv 
        for i in range(n_stats):
            dot1[i] = np.sum(np.multiply(replica.conj(), K[:,i,:]), axis=0)
        # second dot product - dot1 with replica
        dot2 = np.sum(np.multiply(dot1, replica), axis=0)
        beam = abs((1. + 0.j) / dot2)

    return beam
    


def plwave_beamformer(data, scoord, svmin, svmax, dsv, slow, fmin, fmax, Fs, w_length,
        w_delay, baz=None, processor="bartlett", df=0.2, neig=0, norm=True):
    """
    This routine estimates the back azimuth and phase velocity of incoming waves
    based on the algorithm presented in Corciulo et al., 2012 (in Geophysics).

    :type data: numpy.ndarray
    :param matr: time series of used stations (dim: [number of samples, number of stations])
    :type scoord: numpy.ndarray
    :param scoord: UTM coordinates of stations (dim: [number of stations, 2])
    :type svmin, svmax: float
    :param svmin, svmax: slowness/velocity interval used to calculate replica vector
    :type dsv: float
    :param dsv: slowness/velocity step used to calculate replica vector
    :type slow: boolean 
    :param slow: if true, svmin, svmax, dsv are slowness values. if false, velocity values
    :type fmin, fmax: float
    :param fmin, fmax: frequency range for which the beamforming result is calculated
    :type Fs: float
    :param Fs: sampling rate of data streams
    :type w_length: float
    :param w_length: length of sliding window in seconds. result is "averaged" over windows
    :type w_delay: float
    :param w_delay: delay of sliding window in seconds with respect to previous window
    :type baz: float
    :param baz: Back azimuth. If given, the beam is calculated only for this specific back azimuth
    :type processor: string
    :param processor: processor used to match the cross-spectral-density matrix to the
        replica vecotr. see Corciulo et al., 2012
    :type df: float
    :param df: frequency step between fmin and fmax
    :type neig: integer
    :param neig: number of dominant CSDM eigenvectors to annul from the data.
        enables to suppress strong sources.
    :type norm: boolean
    :param norm: if True (default), beam power is normalized

    :return: three numpy arrays:
        teta: back azimuth (dim: [number of bazs, 1])
        c: phase velocity (dim: [number of cs, 1])
        beamformer (dim: [number of bazs, number of cs])
    """

    # number of stations
    n_stats = data.shape[1]

    # grid for search over backazimuth and apparent velocity
    if baz is None:
        teta = np.arange(1, 363, 2) + 180
    else:
        teta = np.array([baz + 180])
    if slow:
        s = np.arange(svmin, svmax + dsv, dsv) / 1000.
    else:
        v = np.arange(svmin, svmax + dsv, dsv) * 1000.
        s = 1. / v

    # create meshgrids
    teta_, s_ = np.meshgrid(teta, s)
    n_param = teta_.size
    # reshape
    teta_ = teta_.reshape(n_param)
    s_ = s_.reshape(n_param)
    # reshape for efficient calculation
    xscoord = np.tile(scoord[:,0].reshape(n_stats, 1), (1, n_param)) 
    yscoord = np.tile(scoord[:,1].reshape(n_stats, 1), (1, n_param))
    teta_ = np.tile(teta_, (n_stats, 1))
    s_ = np.tile(s_, (n_stats, 1))

    # extract number of data points
    npts = data[:, 1].size
    # construct analysis frequencies
    freq = np.arange(fmin, fmax+df, df)
    # construct time vector for sliding window 
    w_time = np.arange(0, w_length, 1./Fs)
    npts_win = w_time.size
    npts_delay = int(w_delay * Fs)
    # number of analysis windows ('shots')
    nshots = int(np.floor((npts - w_time.size) / npts_delay)) + 1

    # initialize data steering vector:
    # dim: [number of frequencies, number of stations, number of analysis windows]
    vect_data = np.zeros((freq.size, n_stats, nshots), dtype=np.complex)
    
    # construct matrix for DFT calculation
    # dim: [number time points, number frequencies]
    matrice_int = np.exp(2. * np.pi * 1j * np.dot(w_time[:, None], freq[:, None].T))

    # initialize beamformer
    # dim: [n_param]
    beamformer = np.zeros(n_param)

    # calculate DFTs
    for ii in range(n_stats):
        toto = data[:, ii]
        # now loop over shots
        n = 0
        while (n * npts_delay + npts_win) <= npts:
            # calculate DFT
            # dim: [number frequencies]
            adjust = np.dot(toto[n*npts_delay: n*npts_delay+npts_win][:,None],
                     np.ones((1, len(freq))))
            # mean averages over time axis
            data_freq = np.mean(np.multiply(adjust, matrice_int), axis=0)
            # fill data steering vector: ii'th station, n'th shot.
            # normalize in order not to bias strongest seismogram.
            # dim: [number frequencies, number stations, number shots]
            vect_data[:, ii, n] = (data_freq / abs(data_freq)).conj().T
            n += 1

    # loop over frequencies and do phase matching
    for ll in range(len(freq)):
        # calculate cross-spectral density matrix
        # dim: [number of stations X number of stations]
        K = calculate_CSDM(vect_data[ll,:,:], neig, norm)

        # calculate replica vector
        replica = np.exp(-1j * (xscoord * np.cos(np.radians(90 - teta_)) \
                              + yscoord * np.sin(np.radians(90 - teta_))) \
                              * 2. * np.pi * freq[ll] * s_)
        replica /= np.linalg.norm(replica, axis=0)
        replica = np.reshape(replica, (n_stats, n_param))

        # do phase matching
        beamformer += phase_matching(replica, K, processor)

    # normalize by deviding through number of discrete frequencies
    beamformer /= freq.size
    # reshape, dim: [number baz, number slowness]
    beamformer = np.reshape(beamformer, (s.size, teta.size))
    teta -= 180
    return teta, s*1000., beamformer


def matchedfield_beamformer(data, scoord, xrng, yrng, zrng, dx, dy, dz, svrng, ds,
        slow, fmin, fmax, Fs, w_length, w_delay,  processor="bartlett", df=0.2,
        neig=0, norm=True):
    """
    This routine estimates the back azimuth and phase velocity of incoming waves
    based on the algorithm presented in Corciulo et al., 2012 (in Geophysics).
    Can also be used to focus the beam to a certain coordinate, which must be
    specified with xmax, ymax, zmax. In this case, dx, dy, and dz need to be set
    to zero!
    
    :type data: numpy.ndarray
    :param data: time series of used stations (dim: [number of samples, number of stations])
    :type scoord: numpy.ndarray
    :param scoord: UTM coordinates of stations (dim: [number of stations, 2])
    :type xrng, yrng, zrng: tuple
    :param xrng, yrng, zrng: parameters for spatial grid search. Grid ranges
        from xrng[0] to xrng[1], yrng[0] to yrng[1], and zrng[0] to zrng[1].
    :type dx, dy, dz: float
    :param dx, dy, dz: grid resolution; increment from xrng[0] to xrng[1],
        yrng[0] to yrng[1], zrng[0] to zrng[1]
    :type svrng: tuple
    :param svrng: slowness interval used to calculate replica vector
    :type ds: float
    :param ds: slowness step used to calculate replica vector
    :type slow: boolean 
    :param slow: if true, svmin, svmax, dsv are slowness values. if false, velocity values
    :type fmin, fmax: float
    :param fmin, fmax: frequency range for which the beamforming result is calculated
    :type Fs: float
    :param Fs: sampling rate of data streams
    :type w_length: float
    :param w_length: length of sliding window in seconds. result is "averaged" over windows
    :type w_delay: float
    :param w_delay: delay of sliding window in seconds with respect to previous window
    :type processor: string
    :param processor: processor used to match the cross-spectral-density matrix to the
        replica vecotr. see Corciulo et al., 2012
    :type df: float
    :param df: frequency step between fmin and fmax
    :type neig: integer
    :param neig: number of dominant CSDM eigenvectors to annul from the data.
        enables to suppress strong sources.
    :type norm: boolean
    :param norm: if True (default), beam power is normalized

    :return: four numpy arrays:
        xcoord: grid coordinates in x-direction (dim: [number x-grid points, 1])
        ycoord: grid coordinates in y-direction (dim: [number y-grid points, 1])
        c: phase velocity (dim: [number of cs, 1])
        beamformer (dim: [number y-grid points, number x-grid points, number cs])
    """

    # number of stations
    n_stats = data.shape[1]

    # grid for search over location
    # if beam is fixed to a coordinate in x, y, or z
    if yrng[0] == yrng[1]:
        ycoord = np.array([yrng[0]])
    # if beam is calculated for a regular grid
    else:
        ycoord = np.arange(yrng[0], yrng[1] + dy, dy)
    # same for x ... 
    if xrng[0] == xrng[1]:
        xcoord = np.array([xrng[0]])
    else:
        xcoord = np.arange(xrng[0], xrng[1] + dx, dx)
    # and for z 
    if zrng[0] == zrng[1]:
        zcoord = np.array([zrng[0]])
    else:
        zcoord = np.arange(zrng[0], zrng[1] + dz, dz)
    # create meshgrids
    ygrid, xgrid = np.meshgrid(ycoord, xcoord)
    zgrid = np.zeros(xgrid.shape)
    ygrid = ygrid.reshape(ygrid.size)
    xgrid = xgrid.reshape(xgrid.size)
    zgrid = zgrid.reshape(zgrid.size)
    if zcoord.size > 1:
        ygrid = np.tile(ygrid, zcoord.size)
        xgrid = np.tile(xgrid, zcoord.size)
        zgrid_ = np.copy(zgrid)
        for i in range(zcoord.size - 1):
            zgrid = np.concatenate((zgrid, zgrid_ + zcoord[i+1]))

    # grid for search over slowness
    if svrng[0] == svrng[1]:
        s = np.array([svrng[0]]) / 1000.
    else:
        s = np.arange(svrng[0], svrng[1] + ds, ds) / 1000.
    if not slow:
        s = 1. / (s * 1.e6)
    # extend coordinate grids and slowness grid
    sgrid = np.zeros(xgrid.size) + s[0]
    ssize = sgrid.size
    if s.size > 1:
        ygrid = np.tile(ygrid, s.size)
        xgrid = np.tile(xgrid, s.size)
        zgrid = np.tile(zgrid, s.size)
        for i in range(s.size - 1):
            sgrid = np.concatenate((sgrid, np.zeros(ssize) + s[i+1]))
    # reshape for efficient calculation
    xscoord = np.tile(scoord[:,0].reshape(n_stats, 1), (1, xgrid.size))
    yscoord = np.tile(scoord[:,1].reshape(n_stats, 1), (1, ygrid.size))
    ygrid = np.tile(ygrid, (n_stats, 1))
    xgrid = np.tile(xgrid, (n_stats, 1))
    zgrid = np.tile(zgrid, (n_stats, 1))
    sgrid = np.tile(sgrid, (n_stats, 1))
    # number of parameter combinations
    n_param = xgrid.shape[1]

    # extract number of data points
    npts = data[:, 1].size
    # construct analysis frequencies
    freq = np.arange(fmin, fmax + df, df)
    # construct time vector for sliding window 
    w_time = np.arange(0, w_length, 1./Fs)
    npts_win = w_time.size
    npts_delay = int(w_delay * Fs)
    # number of analysis windows ('shots')
    nshots = int(np.floor((npts - w_time.size) / npts_delay)) + 1

    # initialize data steering vector:
    # dim: [number of frequencies, number of stations, number of analysis windows]
    vect_data = np.zeros((freq.size, n_stats, nshots), dtype=np.complex)

    # construct matrix for DFT calculation
    # dim: [number w_time points, number frequencies]
    matrice_int = np.exp(2. * np.pi * 1j * np.dot(w_time[:, None], freq[:, None].T))

    # initialize array for beamformer 
    beamformer = np.zeros(n_param)

    # calculate DFTs 
    for ii in range(n_stats):
        toto = data[:, ii]
        # now loop over shots
        n = 0
        while (n * npts_delay + npts_win) <= npts:
            # calculate DFT
            # dim: [number frequencies]
            adjust = np.dot(toto[n*npts_delay: n*npts_delay+npts_win][:, None],
                            np.ones((1, freq.size)))
            # mean averages over time axis
            data_freq = np.mean(np.multiply(adjust, matrice_int), axis=0)
            # fill data steering vector: ii'th station, n'th shot.
            # normalize in order not to bias strongest seismogram.
            # dim: [number frequencies, number stations, number shots]
            vect_data[:, ii, n] = (data_freq / abs(data_freq)).conj().T
            n += 1


    # loop over frequencies and perform beamforming
    for ll in range(freq.size):
        # calculate cross-spectral density matrix
        # dim: [number of stations X number of stations]
        K = calculate_CSDM(vect_data[ll,:,:], neig, norm)

        # calculate replica vector
        replica = np.exp(-1j * np.sqrt((xscoord - xgrid)**2 \
            + (yscoord - ygrid)**2 + zgrid**2) * 2. * np.pi * freq[ll] * sgrid)
        replica /= np.linalg.norm(replica, axis=0)
        replica = np.reshape(replica, (n_stats, n_param))

        # do phase matching
        beamformer += phase_matching(replica, K, processor)

    # normalize beamformer and reshape
    beamformer /= freq.size
    beamformer = np.reshape(beamformer, (ycoord.size, xcoord.size,
        zcoord.size, s.size), order="F")
    return ycoord, xcoord, zcoord, s*1000., beamformer
