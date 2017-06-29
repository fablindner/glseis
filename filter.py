#!/usr/bin/env python

"""
This module contains frequency domain filters.
"""

__author__ = "Fabian Lindner, Kees Weemstra, Joachim Wassermann"





import numpy as np
from scipy.fftpack import fft, ifft, fftfreq


def ricker(dt, f0, ampmax, f):
    """
    create frequency domain ricker wavelet
    :param dt: time domain sample spacing
    :param f0: center frequency
    :param ampmax: maximal amplitude
    :param f: frequency vector
    :return: frequency domain ricker wavelet
    """
    w0 = 2. * np.pi * f0
    w = 2. * np.pi * f
    wavel = 4 * ampmax * np.exp((1j * w)**2 / w0**2) * np.sqrt(np.pi) * (w**2 / w0**3) / dt
    return wavel


def gaussianfilter(sigarray, delta, bandwidth, freq0):
    """
    Filters a given data array with a guassian.
    :param sigarray: signal array (much faster if the length of this is a power of 2)
    :param delta: time sampling interval (seconds)
    :param bandwidth: filter df (>0.)
    :param freq0: center frequency (Hz)
    """
    #1 : prepare the frequency domain filter
    n = len(sigarray)  #number of samples
    freq = fftfreq(n, delta) #exact (!) frequency array
    # we construct our gaussian according the constQ criterion of Archambeau et al.
    #beta = np.log(2.)/2.
    beta = abs(np.log(1. / np.sqrt(2.))) / 0.5**2
    g = np.sqrt(beta/np.pi)*np.exp(-beta * (np.abs(freq - freq0) / bandwidth) ** 2.) #do not forget negative frequencies

    #2 : convolve your signal by the filter in frequency domain
    sigarray_fourier = fft(sigarray)
    sigarray_fourier_filtered = sigarray_fourier * g

    #3 : back to time domain
    sigarray_filtered = np.real(ifft(sigarray_fourier_filtered))
    return sigarray_filtered
