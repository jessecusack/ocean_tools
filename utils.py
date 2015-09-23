# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 12:09:43 2014

@author: jc3e13

Too many little modules were cluttering the directory so I have shoved them
all into one miscellaneous 'utilities' module.

"""

import numpy as np
import datetime as dt


class Bunch(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def datenum_to_datetime(datenum):
    """
    Convert a MATLAB datenums into python datetimes.

    Parameters
    ----------
    datenum : array_like
        MATLAB datenumber which is the number of days since 0001-01-01.

    Returns
    -------
    pydt : ndarray
        Python datetime. See datetime module.

    """

    def convert(datenum):
        try:
            return dt.datetime.fromordinal(int(datenum)) + \
                dt.timedelta(days=datenum % 1) - dt.timedelta(days=366)
        except ValueError:
            return np.nan

    try:
        # If datenum is not iterable it will raise a TypeError. I could just
        # check whether it is iterable first... !
        pydt = np.array([convert(el) for el in datenum])

    except TypeError:

        pydt = convert(datenum)

    return pydt


def lldist(lon, lat):
    """Calculates the distance between longitude and latitude coordinates on a
    spherical earth with radius using the Haversine formula. Code modified from
    the MATLAB m_map toolbox function m_lldist.m.

    Parameters
    ----------
    lon : 1-D numpy.ndarray of floats.
        Longitude values. [degrees]
    lat : 1-D numpy.ndarray of floats.
        Latitude values. [degrees]

    Returns
    -------
    dist : 1-D numpy.ndarray of floats.
        Distance between lon and lat positions. [km]

    """

    lon = np.asarray(lon)
    lat = np.asarray(lat)

    pi180 = np.pi/180.
    earth_radius = 6378.137  # [km]

    lat1 = lat[:-1]*pi180
    lat2 = lat[1:]*pi180

    dlon = np.diff(lon)*pi180
    dlat = lat2 - lat1

    a = (np.sin(dlat/2.))**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2.))**2
    angles = 2.*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dist = earth_radius*angles
    return dist


def distll(lon_0, lat_0, x, y):
    """ """
    pi180 = np.pi/180.
    earth_radius = 6378.137  # [km]

    r = earth_radius*np.cos(pi180*lat_0)
    dlons = x/(r*pi180)
    dlats = y/(earth_radius*pi180)

    lons = lon_0 + dlons
    lats = lat_0 + dlats

    return lons, lats


def flip_cols(data, cols=None):
    """Input an array of data. Receive flipped array. If array is two
    dimensional then a list of columns should be provided else the whole matrix
    will be flipped. This is different from the numpy.flipud function because
    any end chunk of data won't be flipped, e.g.:

    [1, 2, 3, nan, nan] -> [3, 2, 1, nan, nan]

    also

    [1, 2, nan, 4, nan, nan] -> [4, nan, 2, 1, nan, nan]

    If data is 2D, assumes that each column contains a list of data otherwise
    please transpose the input.

    Sometimes data sets combine combine columns of different lengths and pad
    out the empty space with nans. This is useful for flipping in that
    situation.

    """
    out_data = data.copy()
    d = np.ndim(data)

    def flip(arr):
        flip_arr = np.flipud(arr)
        nnans = ~np.isnan(flip_arr)
        idx = nnans.searchsorted(True)
#        try:
#            indx = next(i for i, el, in enumerate(flip_arr) if ~np.isnan(el))
#        except StopIteration:
#            return flip_arr
        return np.concatenate((flip_arr[idx:], flip_arr[:idx]))

    if d == 1 and cols is None:

        out_data = flip(data)

    elif d == 2 and cols is not None:

        for col_indx, col in zip(cols, data[:, cols].T):
            out_data[:, col_indx] = flip(col)

    elif d == 2 and cols is None:

        for col_indx, col in enumerate(data.T):
            out_data[:, col_indx] = flip(col)

    else:
        raise RuntimeError('Inputs are probably wrong.')

    return out_data


def finite_diff(x, y, ivar=None, order=1, acc=1):
    """Differentiate a curve and then interpolate back onto x positions.

    Parameters
    ----------
    x : array_like
        Numbers.
    y : array_like
        Numbers, same size as x.
    ivar : array_like
        Numbers, same size as x. Alternative variable to use as the
        interpolant. This could be useful if x is sometimes not monotonically
        increasing and another variable (e.g. time) is.
    order : int
        Order of the derivative to calculate e.g. 2 will be the second
        derivative. finite_diff calls itself recursively.
    acc : int
        Accuracy of the finite difference approximation to use. Currently
        second order with first order interpolation.

    Returns
    -------
    dydx : numpy.array
        Differential of y.

    """

    dydx_out = np.nan*np.zeros_like(x)

    x_nans = np.isnan(x)
    y_nans = np.isnan(y)
    nans = x_nans | y_nans

    x_nn = x[~nans]
    y_nn = y[~nans]

    if ivar is not None:
        ivar_nn = ivar[~nans]

    if acc == 1:
        dydx = np.diff(y_nn)/np.diff(x_nn)

        # Option to use alternative interpolant (e.g. time).
        if ivar is not None:
            mid = (ivar_nn[1:] + ivar_nn[:-1])/2.
        else:
            mid = (x_nn[1:] + x_nn[:-1])/2.

        dydx_i = np.interp(x[~x_nans], mid, dydx)
    elif acc > 1:
        raise ValueError('Accuracies higher than 1 not yet implimented.')

    dydx_out[~x_nans] = dydx_i

    # Differentiate again if order greater than 1.
    for i in range(order - 1):
        dydx_out = finite_diff(x, dydx_out, i, acc=acc)

    return dydx_out


def nan_interp(x, xp, fp, left=None, right=None):
    """ """

    y = np.nan*np.zeros_like(x)

    x_nans = np.isnan(x)
    xp_nans = np.isnan(xp) | np.isnan(fp)

    y[~x_nans] = np.interp(x[~x_nans], xp[~xp_nans], fp[~xp_nans], left, right)

    return y


def nan_polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """ """

    nans = np.isnan(x) | np.isnan(y)
    return np.polyfit(x[~nans], y[~nans], deg, rcond, full, w, cov)


def nan_polyvalfit(x, y, deg):
    """ """

    p = nan_polyfit(x, y, deg)
    nans = np.isnan(x) | np.isnan(y)
    y_out = np.nan*np.zeros_like(y)
    y_out[~nans] = np.polyval(p, x[~nans])
    return y_out


def nan_detrend(x, y, deg=1):
    """ """
    y_out = np.nan*np.zeros_like(y)

    if np.ndim(x) == 1:
        nans = np.isnan(x) | np.isnan(y)
        p = nan_polyfit(x, y, deg)
        y_out[~nans] = y[~nans] - np.polyval(p, x[~nans])
    elif np.ndim(x) == 2:
        for i in xrange(x.shape[1]):
            nans = np.isnan(x[:, i]) | np.isnan(y[:, i])
            p = nan_polyfit(x[:, i], y[:, i], deg)
            y_out[~nans, i] = y[~nans, i] - np.polyval(p, x[~nans, i])
    else:
        raise RuntimeError('Arguments must be 1 or 2 dimensional arrays.')

    return y_out
