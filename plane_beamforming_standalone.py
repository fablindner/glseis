import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from preprocess import preprocess
import warnings


def plwave_beamformer(s, scoord, prepr, fmin, fmax, Fs, w_length, w_delay,
                      processor="bartlett", df=0.2, fc_min=1, fc_max=10, taper_fract=0.1):

    data = preprocess(s, prepr, Fs, fc_min, fc_max, taper_fract)
    # parameters ?
    data_chunk = 200
    
    # grid for search over backazimuth and apparent velocity
    teta = np.arange(0, 365, 5) + 180
    c = np.arange(800, 4200, 200)
    
    data = data[data_chunk * 500: 2 * data_chunk * 500, :]
    
    # extract number of data points
    Nombre = data[:, 1].size
    # construct time window
    time = np.arange(0, Nombre) / Fs
    # construct analysis frequencies
    indice_freq = np.arange(fmin, fmax+df, df)
    # construct analysis window for entire hour and delay
    interval = np.arange(0, w_length * Fs)
    delay = w_delay * Fs
    # number of analysis windows ('shots')
    numero_shots = np.floor((Nombre - len(interval)) / delay) + 1
    
    # initialize data steering vector:
    # dim: [number of frequencies, number of stations, number of analysis windows]
    vect_data_adaptive = np.zeros((len(indice_freq), Nstats, numero_shots), dtype=np.complex)
    
    # initialize beamformer
    # dim: [number baz, number app. vel.]
    beamformer = np.zeros((len(teta), len(c)))
    
    # construct matrix for DFT calculation
    # dim: [number time points, number frequencies]
    matrice_int = np.exp(2. * np.pi * 1j * np.dot(time[interval][:, None], indice_freq[:, None].T))

    # loop over stations
    for ii in range(Nstats):
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
        if numero == 1:
            K = np.dot(vect_data_adaptive[ll, :, :].conj().T, vect_data_adaptive[ll, :, :])
        else:
            K = np.dot(vect_data_adaptive[ll, :, :], vect_data_adaptive[ll, :, :].conj().T)
    
        if np.linalg.matrix_rank(K) < Nstats:
            warnings.warn("Warning! Poorly conditioned cross-spectral-density matrix.")
    
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
                    raise ValueError("Processor '%s' not found" % processor)
    teta -= 180
    return teta, c, beamformer.T


# import .mat data
mat = scipy.io.loadmat("/home/flindner/Beamforming/greenland_data10.mat")
s = mat["s"][0, 0]  # array of subarrays -> index 0 = data matrix, ...
# station ids
statid = np.arange(0, 8)
# number of sensors
Nstats = len(statid)
# sampling frequency
Fs = s[2][0][0]
# station UTM coordinates - easting and northing
scoord = s[4][statid, :2]

# Preprocessing: 0 (none), 1 (bandpass filter), 2 (prewhitening)
prepr = 2

# interval for whitening
fc_min = 1
fc_max = 10
# taper percentage ?
taper_fract = 0.1
# frequency interval
fmin = 3
fmax = 8
df = 0.2  # frequency step
# window length in seconds
w_length = 30
w_delay = 5
# get data and apply preprocessing


t, c, b = plwave_beamformer(s, scoord, prepr, fmin, fmax, Fs, w_length, w_delay,
                            processor="adaptive", df=0.2, taper_fract=0.1)

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
im1 = ax1.pcolormesh(t, c, b, cmap='YlGnBu_r')
ax1.set_title("Bartlett")
ax1.set_xlabel("Angle (deg, CCW from North)")
ax1.set_ylabel("Apparent velocity (m/s)")
ax1.set_xlim(t[0], t[-1])
ax1.set_ylim(c[0], c[-1])
cbar1 = fig1.colorbar(im1)
plt.show()
