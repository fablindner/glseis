from scipy import signal
import numpy as np
import scipy.io
import warnings


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


def preprocess(matr, prepr, Fs, fc_min, fc_max, taper_fract):
    """
    :type matr: numpy.ndarray
    :param matr: time series of used stations (dim: [number of samples, number of stations])
    :type prepr: integer
    :param prepr: type of preprocessing. 0=None, 1=bandpass filter, 2=spectral whitening
    :type Fs: float
    :param Fs: sampling rate of data streams
    :type fc_min, fc_max: float
    :param fc_min, fc_max: corner frequencies used for preprocessing
    :type taper_fract: float
    :param taper_fract: percentage of frequency band which is tapered after spectral whitening

    :return: preprocessed data (dim: [number of samples, number of stations])
    """
    if prepr == 0:
        data = signal.detrend(matr, axis=0)

    elif prepr == 1:
        # generate frequency vector and butterworth filter
        b, a = signal.butter(4, np.array([fc_min, fc_max]) / Fs * 2, btype="bandpass")
        # filter data and normalize it by maximum energy
        data = signal.filtfilt(b, a, signal.detrend(matr, axis=0), axis=0)
        fact = np.sqrt(np.dot(np.ones((data.shape[0], 1)), np.sum(data**2, axis=0).reshape((1, data.shape[1]))))
        data = np.divide(data, fact)

    elif prepr == 2:
        nfft = nearest_powof2(matr.shape[0])
        Y = np.fft.fft(matr, n=nfft, axis=0)
        f = np.fft.fftfreq(nfft, 1./float(Fs))

        # whiten: discard all amplitude information within range fc
        Y_white = np.zeros(Y.shape)
        J = np.where((f > fc_min) & (f < fc_max))
        Y_white[J, :] = np.exp(1j * np.angle(Y[J, :]))

        # now taper within taper_fract
        deltaf = (fc_max - fc_min) * taper_fract
        Jdebut = np.where((f > fc_min) & (f < (fc_min + deltaf)))
        Jfin = np.where((f > (fc_max - deltaf)) & (f < fc_max))
        for ii in range(Y.shape[1]):
            if len(Jdebut[0]) > 1:
                Y_white[Jdebut, ii] = np.multiply(Y_white[Jdebut, ii],
                            np.sin(np.pi / 2 * np.arange(0, len(Jdebut[0])) / len(Jdebut[0]))**2)
            if len(Jfin[0]) > 1:
                Y_white[Jfin, ii] = np.multiply(Y_white[Jfin, ii],
                            np.cos(np.pi / 2 * np.arange(0, len(Jfin[0])) / len(Jfin[0]))**2)

        # perform inverse fft to obtain time signal
        # data = 2*np.real(np.fft.ifft(Y_white, n=nfft, axis=0))
        data = np.fft.ifft(Y_white, n=nfft, axis=0)
        # normalize it by maximum energy
        fact = np.sqrt(np.dot(np.ones((data.shape[0], 1)), np.sum(data**2, axis=0).reshape((1, data.shape[1]))))
        data = np.divide(data, fact)
    return data


def plwave_beamformer(matr, scoord, prepr, fmin, fmax, Fs, w_length, w_delay,
                      processor="bartlett", df=0.2, fc_min=1, fc_max=10, taper_fract=0.1, norm=True):
    """
    This routine estimates the back azimuth and phase velocity of incoming waves
    based on the algorithm presented in Corciulo et al., 2012 (in Geophysics).
    Singular value decomposition is not implemented, yet.

    :type matr: numpy.ndarray
    :param matr: time series of used stations (dim: [number of samples, number of stations])
    :type scoord: numpy.ndarray
    :param scoord: UTM coordinates of stations (dim: [number of stations, 2])
    :type prepr: integer
    :param prepr: type of preprocessing. 0=None, 1=bandpass filter, 2=spectral whitening
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
    :type fc_min, fc_max: float
    :param fc_min, fc_max: corner frequencies used for preprocessing
    :type taper_fract: float
    :param taper_fract: percentage of frequency band which is tapered after spectral whitening
    :type norm: boolean
    :param norm: if True (default), beam power is normalized

    :return: three numpy arrays:
        teta: back azimuth (dim: [number of bazs, 1])
        c: phase velocity (dim: [number of cs, 1])
        beamformer (dim: [number of bazs, number of cs])
    """

    data = preprocess(matr, prepr, Fs, fc_min, fc_max, taper_fract)
    # select a data chunk
    data_chunk = 200
    data = data[data_chunk * 500: 2 * data_chunk * 500, :]
    # number of stations
    n_stats = data.shape[1]

    # grid for search over backazimuth and apparent velocity
    teta = np.arange(0, 365, 5) + 180
    c = np.arange(800, 4200, 200)
    # extract number of data points
    Nombre = data[:, 1].size
    # construct time window
    time = np.arange(0, Nombre) / Fs
    # construct analysis frequencies
    indice_freq = np.arange(fmin, fmax+df, df)
    # construct analysis window for entire hour and delay
    interval = np.arange(0, int(w_length * Fs))
    delay = int(w_delay * Fs)
    # number of analysis windows ('shots')
    numero_shots = np.floor((Nombre - len(interval)) / delay) + 1
    
    # initialize data steering vector:
    # dim: [number of frequencies, number of stations, number of analysis windows]
    vect_data_adaptive = np.zeros((len(indice_freq), n_stats, numero_shots), dtype=np.complex)
    
    # initialize beamformer
    # dim: [number baz, number app. vel.]
    beamformer = np.zeros((len(teta), len(c)))
    
    # construct matrix for DFT calculation
    # dim: [number time points, number frequencies]
    matrice_int = np.exp(2. * np.pi * 1j * np.dot(time[interval][:, None], indice_freq[:, None].T))

    # loop over stations
    for ii in range(n_stats):
        toto = data[:, ii]
        # now loop over shots
        numero = 0
        while (numero * delay + len(interval)) < len(toto):
            # calculate DFT
            # dim: [number frequencies]
            adjust = np.dot(toto[numero * delay + interval][:, None], np.ones((1, len(indice_freq))))
            test = np.mean(np.multiply(adjust, matrice_int), axis=0)  # mean averages over time axis
            # fill data steering vector: ii'th station, numero'th shot.
            # normalize in order not to bias strongest seismogram.
            # dim: [number frequencies, number stations, number shots]
            vect_data_adaptive[:, ii, numero] = (test / abs(test)).conj().T
            numero += 1

    # loop over frequencies
    for ll in range(len(indice_freq)):
        # calculate cross-spectral density matrix
        # dim: [number of stations X number of stations]
        if numero == 1:
            K = np.dot(vect_data_adaptive[ll, :, :].conj().T, vect_data_adaptive[ll, :, :])
        else:
            K = np.dot(vect_data_adaptive[ll, :, :], vect_data_adaptive[ll, :, :].conj().T)
    
        if np.linalg.matrix_rank(K) < n_stats:
            warnings.warn("Warning! Poorly conditioned cross-spectral-density matrix.")

        if norm:
            K /= np.linalg.norm(K)

    
        K_inv = np.linalg.inv(K)
    
        # loop over backazimuth
        for bb in range(len(teta)):
            # loop over apparent velocity
            for cc in range(len(c)):
    
                # define and normalize replica vector (neglect amplitude information)
                omega = np.exp(-1j * (scoord[:, 0] * np.cos(np.radians(90 - teta[bb])) \
                                      + scoord[:, 1] * np.sin(np.radians(90 - teta[bb]))) \
                               * 2. * np.pi * indice_freq[ll] / c[cc])
                omega /= np.linalg.norm(omega)
    
                # calculate processors and save results
                replica = omega[:, None]
                # bartlett
                if processor == "bartlett":
                    beamformer[bb, cc] += abs(np.dot(np.dot(replica.conj().T, K), replica))
                # adaptive - Note that replica.conj().T * replica = 1. + 0j
                elif processor == "adaptive":
                    beamformer[bb, cc] += abs(np.dot(replica.conj().T, replica) \
                                              / (np.dot(np.dot(replica.conj().T, K_inv), replica)))
                else:
                    raise ValueError("No processor called '%s'" % processor)
    beamformer /= indice_freq.size
    teta -= 180
    return teta, c, beamformer.T


def matchedfield_beamformer(matr, scoord, xmax, ymax, dx, dy, cmin, cmax, dc, prepr, fmin, fmax, fc_min, fc_max,
                            Fs, w_length, w_delay, processor="bartlett", df=0.2, taper_fract=0.1, norm=True):
    """
    This routine estimates the back azimuth and phase velocity of incoming waves
    based on the algorithm presented in Corciulo et al., 2012 (in Geophysics).
    Singular value decomposition is not implemented, yet.

    :type matr: numpy.ndarray
    :param matr: time series of used stations (dim: [number of samples, number of stations])
    :type scoord: numpy.ndarray
    :param scoord: UTM coordinates of stations (dim: [number of stations, 2])
    :type xmax, ymax: float
    :param xmax, ymax: spatial grid search: grid ranges from x - xmax to x + xmax and
        y - ymax to y + ymax, where (x,y) are the coordinates of first station (scoord[0, :])
    :type dx, dy: float
    :param dx, dy: grid resolution; increment from x - xmax to x + xmax and y - ymax to y + ymax,
        respectively. (x,y) are the coordinates of the first station (scoord[0, :])
    :type cmin, cmax: float
    :param cmin, cmax: velocity interval used to calculate replica vector
    :type dc: float
    :param dc: velocity step used to calculate replica vector
    :type prepr: integer
    :param prepr: type of preprocessing. 0=None, 1=bandpass filter, 2=spectral whitening
    :type fmin, fmax: float
    :param fmin, fmax: frequency range for which the beamforming result is calculated
    :type fc_min, fc_max: float
    :param fc_min, fc_max: corner frequencies used for preprocessing
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
    :type taper_fract: float
    :param taper_fract: percentage of frequency band which is tapered after spectral whitening
    :type norm: boolean
    :param norm: if True (default), beam power is normalized

    :return: four numpy arrays:
        xcoord: grid coordinates in x-direction (dim: [number x-grid points, 1])
        ycoord: grid coordinates in y-direction (dim: [number y-grid points, 1])
        c: phase velocity (dim: [number of cs, 1])
        beamformer (dim: [number y-grid points, number x-grid points, number cs])
    """

    data = preprocess(matr, prepr, Fs, fc_min, fc_max, taper_fract)
    # number of stations
    n_stats = data.shape[1]

    # grid for search over location and apparent velocity
    xcoord = np.arange(-xmax, xmax + dx, dx) + scoord[0, 0]
    ycoord = np.arange(-ymax, ymax + dy, dy) + scoord[0, 1]
    c = np.arange(cmin, cmax + dc, dc)
    # extract number of data points
    Nombre = data[:, 1].size
    # construct time window
    time = np.arange(0, Nombre) / Fs
    # construct analysis frequencies
    indice_freq = np.arange(fmin, fmax+df, df)
    # construct analysis window for entire hour and delay
    interval = np.arange(0, int(w_length * Fs))
    delay = int(w_delay * Fs)
    # number of analysis windows ('shots')
    numero_shots = np.floor((Nombre - len(interval)) / delay) + 1

    # initialize data steering vector:
    # dim: [number of frequencies, number of stations, number of analysis windows]
    vect_data_adaptive = np.zeros((len(indice_freq), n_stats, numero_shots), dtype=np.complex)

    # initialize beamformer
    # dim: [number xcoord, number ycoord, number app. vel.]
    beamformer = np.zeros((len(ycoord), len(xcoord), len(c)))

    # construct matrix for DFT calculation
    # dim: [number time points, number frequencies]
    matrice_int = np.exp(2. * np.pi * 1j * np.dot(time[interval][:, None], indice_freq[:, None].T))

    # loop over stations
    for ii in range(n_stats):
        toto = data[:, ii]
        # now loop over shots
        numero = 0
        while (numero * delay + len(interval)) < len(toto):
            # calculate DFT
            # dim: [number frequencies]
            adjust = np.dot(toto[numero * delay + interval][:, None], np.ones((1, len(indice_freq))))
            test = np.mean(np.multiply(adjust, matrice_int), axis=0)  # mean averages over time axis
            # fill data steering vector: ii'th station, numero'th shot.
            # normalize in order not to bias strongest seismogram.
            # dim: [number frequencies, number stations, number shots]
            vect_data_adaptive[:, ii, numero] = (test / abs(test)).conj().T
            numero += 1

    # loop over frequencies
    for ll in range(len(indice_freq)):
        # calculate cross-spectral density matrix
        # dim: [number of stations X number of stations]
        if numero == 1:
            K = np.dot(vect_data_adaptive[ll, :, :].conj().T, vect_data_adaptive[ll, :, :])
        else:
            K = np.dot(vect_data_adaptive[ll, :, :], vect_data_adaptive[ll, :, :].conj().T)

        if np.linalg.matrix_rank(K) < n_stats:
            warnings.warn("Warning! Poorly conditioned cross-spectral-density matrix.")

        if norm:
            K /= np.linalg.norm(K)


        K_inv = np.linalg.inv(K)

        # loop over spatial grid
        for yy in range(len(ycoord)):
            for xx in range(len(xcoord)):
                # loop over apparent velocity
                for cc in range(len(c)):

                    # define and normalize replica vector (neglect amplitude information)
                    omega = np.exp(-1j * np.sqrt((scoord[:, 0] - xcoord[xx])**2 + (scoord[:, 1] - ycoord[yy])**2) \
                                   * 2. * np.pi * indice_freq[ll] / c[cc])
                    omega /= np.linalg.norm(omega)

                    # calculate processors and save results
                    replica = omega[:, None]
                    # bartlett
                    if processor == "bartlett":
                        beamformer[yy, xx, cc] += abs(np.dot(np.dot(replica.conj().T, K), replica))
                    # adaptive - Note that replica.conj().T * replica = 1. + 0j
                    elif processor == "adaptive":
                        beamformer[yy, xx, cc] += abs(np.dot(replica.conj().T, replica) \
                                                   / (np.dot(np.dot(replica.conj().T, K_inv), replica)))
                    else:
                        raise ValueError("No processor called '%s'" % processor)

    beamformer /= indice_freq.size
    return xcoord, ycoord, c, beamformer
