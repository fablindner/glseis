"""Perform parallel I/O operations.

This module containes the following functions:

    * read_mseed - returns an ObsPy Stream object
"""

from obspy import Stream
from joblib import Parallel, delayed


def read_mseed(client, network, stations, channels, t1, t2):
    """Read miniseed data and return an ObsPy Stream object.

    :param client (object): ObsPy Client object for data handling.
    :param network (str): Network code.
    :param stations (list): List of station codes.
    :param channels (list): List of channel codes.
    :param t1 (object): Starttime for data retrieval.
    :param t2 (object): Endtime for data retrieval.

    Returns
        ObsPy Stream object
    """
    # read all data
    st_list = Parallel(n_jobs=-1, backend="loky")(delayed(client.get_waveforms)\
        (network, stn, "*", chn, t1, t2) for stn in stations for chn in channels)
    # make stream
    st = Stream()
    for tr in st_list:
        st += tr

    return st
