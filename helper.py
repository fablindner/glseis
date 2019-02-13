#!/usr/bin/env python
"""
This module contains small helper functions.
"""
__author__ = "Fabian Lindner"


import numpy as np
import sys
import datetime
from scipy.special import struve
from obspy import UTCDateTime




def zero_crossings(x, func, xmin=None, xmax=None):
    """
    returns location of func's zero-crossings.
    :param x: array holding the x-values
    :param func: array holding the corresponding function values
    :param xmin: smallest zero crossing of interest
    :param xmax: biggest zero crossing of interest
    """
    xi = []
    dx = x[1] - x[0]
    for i in range(x.size - 1):
        if func[i] == 0 and func[i+1] != 0:
            xi.append(x[i])
        test = func[i] * func[i+1]
        if test < 0.:
            dx_ = abs(func[i]) / (abs(func[i]) + abs(func[i+1])) * dx
            f = x[i] + dx_
            xi.append(f)
    xi.sort()
    if xmin is None:
    	return np.array(xi)
    else:
        xi = np.array(xi)
        ind = np.where((xi >= xmin) & (xi <= xmax))[0]
        return xi[ind]


def Hv_zeros(order, n):
    """
    Calculates the first 20 zeros of the Struve function of arbitrary order
    :param order: order of the Struve function
    :param n: number of zeros returned (max 20)
    """
    # x-vector - use only positive part, Struve functions are 
    # even (if order is odd) or odd( if order is even)
    x = np.arange(0., 70., 0.0001)
    H = struve(order, x)
    xi = zero_crossings(x, H)
    xi[0] = 0.
    return xi[1:n+1]


def interstation_dist(stn1, stn2, path2file, hor=True):
    """
    Calculates the the (horizontal) distance between two stations.
    :param stn1: first station
    :param stn2: second station
    :param path_fn: path + file name of file containing the coordinates of stn1 and stn2
    :param hor: if True, calculates the horizontal distance. If False, calculates the
        distance by considering also changes in altitude.
    """
    stns = np.genfromtxt(path_fn, usecols=(0,), dtype=str)
    easts, norths, elevs = np.loadtxt(fn, usecols=(1,2,3), unpack=True)

    ind1 = np.where(stns == stn1)[0]
    ind2 = np.where(stns == stn2)[0]
    
    if hor:
        dist = np.sqrt((easts[ind1] - easts[ind2])**2 + (norths[ind1] - norths[ind2])**2)
    else:
        dist = np.sqrt((easts[ind1] - easts[ind2])**2 + (norths[ind1] - norths[ind2])**2 
                     + (elevs[ind1] - elevs[ind2])**2)
    return dist


def ascii_header():
    fn = sys.argv[0]
    date = datetime.datetime.now() 
    header = "# %s\n# %s \n" % (fn, date)
    return header


def load_beams(fn, t1=None, t2=None, powmin=0, rpow=False):
    """
    Function to load beamforming results.
    :param fn: file name
    :param t1: starttime
    :param t2: endtime
    :param powmin: beam power threshold
    :param rpow: if True, power values are returned
    """

    # dictionary containing all data
    path = "/scratch/flindner/PlaineMorte/Beamforming/"

    try:
        # load relvant data
        data = np.load(path + fn)["arr_0"].item()
        times = data["times"]
        beams = data["beams"]
        baz = data["baz"]
        vel = data["sv"]

        # max beam power, slowness/velocity and baz values
        pows = [np.max(beam) for beam in beams]
        inds_max = [np.unravel_index(np.argmax(beam), beam.shape) \
                for beam in beams]
        vels = [vel[ind[0]] for ind in inds_max]
        bazs = [baz[ind[1]] for ind in inds_max]

        # remove nans
        ind = np.where(np.isnan(pows) == False)[0]
        pows = np.array(pows)
        pows = pows[ind]
        times = np.array(times)[ind]
        vels = np.array(vels)[ind]
        bazs = np.array(bazs)[ind]

        # get only results in specified time period
        if t1 is not None:
            ind = np.where((times >= t1.timestamp) & (times <= t2.timestamp))
            pows = pows[ind]
            times = times[ind]
            vels = vels[ind]
            bazs = bazs[ind]

        # get only results with beampower >= powmin
        ind = np.where(pows >= powmin)[0]
        pows = pows[ind] 
        times = times[ind]
        vels = vels[ind]
        bazs = bazs[ind]

        if rpow:
            return times, bazs, vels, pows
        else:
            return times, bazs, vels

    except:
        return None, None, None

