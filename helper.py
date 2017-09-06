#!/usr/bin/env python
"""
This module contains small helper functions.
"""
__author__ = "Fabian Lindner"


import numpy as np
from scipy.special import struve



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
