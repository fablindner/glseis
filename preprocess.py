from scipy import signal


def preprocess(s, prepr, fc, taper_fract):
    if prepr == 0:
        data = signal.detrend(s[0], axis=0)
    return data
