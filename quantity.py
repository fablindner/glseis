#!/usr/bin/env python
"""
This module contains functions thought to extract physical meaningful
quantities from data.
"""
__author__ = "Fabian Lindner"


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import interpolate





def stretch(t, data, ref_data, dvv, win, plot=False):

    """
    stretch data to determine dv/v
    :param t: time vector
    :param data: data, which is stretched
    :param ref: reference seismogram: streched data is compared to this time series
    :param dvv: array containing all epsilon values used to stretch the data
    :param win: window used for stretching
    :param plot: if plot, stretching animation will be shown
    :return: best match epsilon and corresponding correlation coefficient
    """

    # initialize cubic spline interpolation
    cs = interpolate.CubicSpline(t, data)
    # prepare for plotting
    if plot:
        fig = plt.figure()
        res = []
    # prepare for looping over all epsilon values
    ndvv = len(dvv)
    cc = np.zeros(ndvv)
    for e in range(ndvv):
        t_ = t * (1 + dvv[e])
        data_ = cs(t_)
        cc_ = np.corrcoef(data_[win], ref_data[win])
        cc[e] = cc_[0,1]
        if plot:
            res.append(plt.plot(ref_data[win], "g", data_[win], "r"),)
    if plot:
        im_ani = animation.ArtistAnimation(fig, res, interval=1, repeat=False, blit=True)
        plt.show()
    # obtain dvv
    eps = dvv[np.argmax(cc)]
    cc = cc.max()
    return eps, cc



def disp_curves(z_theo, z_func, dx_sensor):

    """
    Calculates dispersion curves based on Aki (1957). Fits the zeros of
    cross-spectra (e.g. CC, MDD) to that of the desired target function
    (e.g. Bessel, Struve). See Ekström et al. (2009), Boschi et al. (2013),
    Weemstra et al. (2017)
    :param z_theo: zeros of the target function
    :param z_func: zeros of the cross-spectrum
    :param dx_sensor: station separation of the station pair used to calc
        the cross-spectrum
    :return: matrix containing all possible zero-crossing combinations of
        target function and cross-spectrum (i.e. dispersion curves)
    """

    nzt = z_theo.size
    nzf = z_func.size
    disp_ = np.zeros((nzt, nzf))
    for i in range(nzt):
        disp_[i,:] = z_func * dx_sensor / z_theo[i]

    # convert matrix: diagonals (dispersion curves) --> rows of new matrix
    ndisp = nzt + nzf - 1
    disp = np.zeros((ndisp, nzf))
    if nzt >= nzf:
        offset = -nzt + 1
        s = 1
    else:
        offset = nzf - 1
        s = -1
    for i in range(ndisp):
        offset_ = offset + s*i
        diag = np.diagonal(disp_, offset=offset_)
        if offset_ < 0:
            disp[i, :len(diag)] = diag
        else:
            disp[i, -len(diag):] = diag
    disp[disp == 0.] = np.nan
    freq = z_func / (2. * np.pi)
    return freq, disp



def pick_disp_curve(freq, disp, vmin, vmax, freq_theo=None, disp_theo=None):

    """
    Displays the output of disp_curves and allow to pick on of the curves
    :param freq: frequency vector of dispersion curves (returned by disp_curves)
    :param disp: matrix containing possible dispersion curves (returned by
        disp_curves)
    :param vmin: minimum phase velocity to display
    :param vmax: maximum phase vleocity to display
    :param freq_theo: frequency vector of a theoretical dispersion curve
    :param disp_theo: theoretical dispersion curve which will also be drawn
    :return: the picked dispersion curve
    """

    fig = plt.figure()
    for i in range(disp.shape[0]):
        plt.plot(freq, disp[i,:], "^", linestyle="--", markersize=9)
    if freq_theo is not None and disp_theo is not None:
        plt.plot(freq_theo, disp_theo, "k")
    plt.ylim(vmin, vmax)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase velocity (m/s)")
    xclicks = []
    yclicks = []
    def onclick(event):
        x = event.xdata
        y = event.ydata
        xclicks.append(x)
        yclicks.append(y)
    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    # determine curve with minimum distance to mouse click location
    disp_ = np.copy(disp)
    disp_[np.isnan(disp_) == True] = 1.e9
    ind_freq = np.argmin(abs(freq - xclicks[0]))
    ind_disp = np.argmin(abs(disp_[:, ind_freq] - yclicks[0]))
    return disp[ind_disp, :]



def fit_smith_dahlen(baz, vel, four_theta=True):
    """
    Fits the Smith & Dahlen (1973) anisotropy model through the
    given data points using linear least squares.
    :param baz: backazimuth values
    :param vel: corresponding phase velocity values
    :param four_theta: if True, five parameter fit (including 4 theta
        component). Otherwise only three parameter fit.
    :return: the model parameters describing anisotropy 
    """
    nvals = baz.size
    if four_theta:
        G = np.zeros((nvals, 5))
    else:
        G = np.zeros((nvals, 3))
    d = np.zeros((nvals, 1))

    G[:,0] = 1.
    G[:,1] = np.cos(2. * np.radians(baz))
    G[:,2] = np.sin(2. * np.radians(baz))
    if four_theta:
        G[:,3] = np.cos(4. * np.radians(baz))
        G[:,4] = np.sin(4. * np.radians(baz))

    d[:,0] = vel

    # inversion
    GTG_inv = np.linalg.inv(np.dot(G.T, G))
    GTd = np.dot(G.T, d)
    m = np.dot(GTG_inv, GTd)
    return np.squeeze(m)
