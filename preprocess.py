from scipy import signal
import numpy as np


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