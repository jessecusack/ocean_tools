# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 12:09:43 2014

@author: jc3e13

Too many little modules were cluttering the directory so I have shoved them
all into one miscellaneous 'utilities' module.

To keep things simple this should only import modules from the python standard
library or numpy and scipy.

"""

import numpy as np
import scipy.signal as sig
import scipy.io as io
import scipy.stats as stats
from datetime import datetime, timedelta


class Bunch(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def datenum_to_datetime(datenum):
    """
    Convert a MATLAB datenums into python datetimes.

    Parameters
    ----------
    datenum : array_like
        MATLAB datenumber which is the number of days since 0000-01-00.

    Returns
    -------
    dt : ndarray
        Python datetime. See datetime module.

    """

    def convert(datenum):
        try:
            return datetime.fromordinal(int(datenum)) + \
                timedelta(days=datenum % 1) - timedelta(days=366)
        except ValueError:
            return np.nan

    if np.iterable(datenum):
        datenumar = np.asarray(datenum)
        shape = datenumar.shape
        dt = np.array([convert(el) for el in datenumar.flat])
        dt = dt.reshape(shape)
    else:
        dt = convert(datenum)

    return dt


def datetime_to_datenum(dt):
    """
    Convert a python datetime object into a MATLAB datenum.

    Parameters
    ----------
    dt : array_like
        Python datetime. See datetime module.

    Returns
    -------
    datenum : ndarray
        MATLAB datenumber which is the number of days since 0000-01-00.

    """

    def convert(dt):
        try:
            mdn = dt + timedelta(days=366)
            frac_seconds = (dt - datetime(dt.year, dt.month, dt.day)).seconds/86400.
            frac_microseconds = dt.microsecond/8.64e10
            return mdn.toordinal() + frac_seconds + frac_microseconds
        except ValueError:
            return np.nan

    if np.iterable(dt):
        dtar = np.asarray(dt)
        shape = dtar.shape
        datenum = np.array([convert(el) for el in dtar.flat])
        datenum = datenum.reshape(shape)
    else:
        datenum = convert(dt)

    return datenum


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

    Notes
    -----
    This functionality does exist in the Gibbs seawater toolbox as gsw.dist.

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


def mid(x, axis=0):
    """Returns mid point values along given axis."""
    ndim = np.ndim(x)
    if ndim == 1:
        return 0.5*(x[1:] + x[:-1])
    elif ndim > 1:
        x_ = np.swapaxes(x, axis, 0)
        xmid_ = 0.5*(x_[1:, ...] + x_[:-1, ...])
        return np.swapaxes(xmid_, 0, axis)
    else:
        raise ValueError


def rotate(x, y, a):
    """Rotate vector (x, y) by an angle a."""
    return x*np.cos(a) + y*np.sin(a), -x*np.sin(a) + y*np.cos(a)


def flip_padded(data, cols=None):
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


def nansort(a, axis=-1, kind='quicksort'):
    """Sort but leave NaN values untouched in place."""
    if axis not in [-1, 0, 1]:
        raise ValueError('The axis may be only -1, 0 or 1.')

    ndim = np.ndim(a)

    if ndim > 2:
        raise ValueError('Only 1 or 2 dimensional arrays are supported.')

    nans = np.isnan(a)
    a_sorted = np.full_like(a, np.nan)
    if ndim == 1:
        a_valid = a[~nans]
        a_sorted[~nans] = np.sort(a_valid, kind=kind)
    if ndim == 2:
        nr, nc = a.shape
        if axis == 0:
            for i in range(nc):
                a_valid = a[~nans[:, i], i]
                a_sorted[~nans[:, i], i] = np.sort(a_valid, kind=kind)
        if axis == -1 or axis == 1:
            for i in range(nr):
                a_valid = a[i, ~nans[i, :]]
                a_sorted[i, ~nans[i, :]] = np.sort(a_valid, kind=kind)

    return a_sorted


def nantrapz(y, x=None, dx=1.0, axis=0, xave=False):
    """
    Integrate along the given axis using the composite trapezoidal rule.

    Integrate `y` (`x`) along given axis. NaN values are removed.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.
    xave : boolean, optional
        If True then the integral average is estimated by dividing the final
        integral by the range of x values. Default is False. Behaviour is
        inaccurate for very gappy data.

    Returns
    -------
    yint : float
        Definite integral as approximated by trapezoidal rule.

    """
    if x is None:
        x = np.full_like(y, dx)

    ndimy = np.ndim(y)
    ndimx = np.ndim(x)

    if ndimy == ndimx == 1:
        nans = np.isnan(x) | np.isnan(y)
        nnans = ~nans
        yint = np.trapz(y[nnans], x[nnans])

        if xave:
            yint /= np.diff(x[nnans]).sum()

    if ndimy == 2 and ndimx == 1:
        ni, nj = y.shape
        nx = x.size

        if ni == nx:
            x = np.tile(x[:, np.newaxis], (1, nj))
            axis = 0
        elif nj == nx:
            x = np.tile(x[np.newaxis, :], (ni, 1))
            axis = 1
        else:
            raise ValueError('Size of x does not match any axis size of y.')

        ndimx = 2

    if ndimy == ndimx == 2:
        nans = np.isnan(x) | np.isnan(y)
        nnans = ~nans

        if axis is 0:
            nj = y.shape[1]
            yint = np.full((nj,), np.nan)
            for j in range(nj):
                if nans[:, j].all():
                    continue
                y_ = y[nnans[:, j], j]
                x_ = x[nnans[:, j], j]
                yint[j] = np.trapz(y_, x_)

        if axis is 1 or axis is -1:
            ni = y.shape[0]
            yint = np.full((ni,), np.nan)
            for i in range(ni):
                if nans[i, :].all():
                    continue
                y_ = y[i, nnans[i, :]]
                x_ = x[i, nnans[i, :]]
                yint[i] = np.trapz(y_, x_)

        if xave:
            yint /= np.nansum(np.diff(x, axis=axis), axis=axis)

    return yint


def finite_diff(x, y, ivar=None, order=1, acc=1):
    """Differentiate a curve and then interpolate back onto x positions.
    NOTE: Why use this when there is a np.gradient function? Because this deals
    with NaN values differently which may be preferable.

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


def nan_interp(x, xp, fp, left=None, right=None, axis=0, squeeze_me=True):
    """See numpy.interp documentation. This does the same thing but ignores NaN
    values in the data. It can accept 2D arrays.

    Parameters
    ----------
    x : float or 1D array
        The x-coordinates of the interpolated values. No NaNs please!
    xp : 1D or 2D array of floats
        The x-coordinates of the data points, must be increasing along the
        dimension along which the interpolation is being performed.
    fp : 1D or 2D array of floats or complex
        The y-coordinates of the data points, same shape as `xp`.
    left : optional float or complex corresponding to fp
        Value to return for `x < xp[0]`, default is `fp[0]`.
    right : optional float or complex corresponding to fp
        Value to return for `x > xp[-1]`, default is `fp[-1]`.
    axis : [-1, 0, 1] int
        Default is 0. The axis along which to perform the interpolation.
    squeeze_me : boolean
        Default is True. Squeeze output to remove singleton dimensions.
    Returns
    -------
    y : ndarray
        The interpolated values.
    """

    if axis not in [-1, 0, 1]:
        raise ValueError('The axis may be only -1, 0 or 1.')

    if xp.shape != fp.shape:
        raise ValueError('xp and fp have different shapes.')

    ndim = np.ndim(xp)
    if ndim > 2:
        raise ValueError('Only 1 or 2 dimensional arrays are supported.')

    nans = np.isnan(xp) | np.isnan(fp)

    if ndim == 1:
        y = np.full_like(x, np.nan)
        y = np.interp(x, xp[~nans], fp[~nans], left, right)
    if ndim == 2:
        nr, nc = xp.shape

        if axis == 0:
            if np.iterable(x):
                y = np.full((len(x), nc), np.nan)
            else:
                y = np.full((1, nc), np.nan)

            for i in range(nc):
                xp_ = xp[~nans[:, i], i]
                fp_ = fp[~nans[:, i], i]
                y[:, i] = np.interp(x, xp_, fp_, left, right)

        if axis == -1 or axis == 1:
            if axis == 0:
                if np.iterable(x):
                    y = np.full((nr, len(x)), np.nan)
                else:
                    y = np.full((nr, 1), np.nan)

            for i in range(nr):
                xp_ = xp[i, ~nans[i, :]]
                fp_ = fp[i, ~nans[i, :]]
                y[i, :] = np.interp(x, xp_, fp_, left, right)

    if squeeze_me:
        return np.squeeze(y)
    else:
        return y


def nan_polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """See numpy.polyfit documentation. This does the same thing but ignores
    NaN values.

    """
    nans = np.isnan(x) | np.isnan(y)
    return np.polyfit(x[~nans], y[~nans], deg, rcond, full, w, cov)


def nan_polyvalfit(x, y, deg):
    """Fit a polynomial to data and return polynomial values, ignoring NaNs.

    Parameters
    ----------
    x : array_like
        x data.
    y : array_like
        Data to fit.
    deg : int
        Degree of polynomial to fit. (Can be zero i.e. constant)


    Returns
    -------
    y_out : numpy.array
        Values of the polynomial at x positions.

    """
    p = nan_polyfit(x, y, deg)
    nans = np.isnan(x) | np.isnan(y)
    y_out = np.nan*np.zeros_like(y)
    y_out[~nans] = np.polyval(p, x[~nans])
    return y_out


def nan_detrend(x, y, deg=1):
    """Subtract a polynomial fit from the data, ignoring NaNs.

    Parameters
    ----------
    x : array_like
        x data.
    y : array_like
        Data to detrend.
    deg : int
        Degree of polynomial to subtract. (Can be zero i.e. constant)


    Returns
    -------
    y_out : numpy.array
        Detrended data.

    """
    y_out = np.nan*np.zeros_like(y)

    if np.ndim(x) == 1:
        nans = np.isnan(x) | np.isnan(y)
        p = nan_polyfit(x, y, deg)
        y_out[~nans] = y[~nans] - np.polyval(p, x[~nans])
    elif np.ndim(x) == 2:
        for i in range(x.shape[1]):
            nans = np.isnan(x[:, i]) | np.isnan(y[:, i])
            p = nan_polyfit(x[:, i], y[:, i], deg)
            y_out[~nans, i] = y[~nans, i] - np.polyval(p, x[~nans, i])
    else:
        raise RuntimeError('Arguments must be 1 or 2 dimensional arrays.')

    return y_out


def nan_binned_statistic(x, values, statistic='mean', bins=10, range=None):
    """See help for scipy.stats.binned_statistic. This is the same but removes
    NaN values."""
    x = x.flatten()
    values = values.flatten()
    nnans = ~(np.isnan(values) | np.isnan(x))
    return stats.binned_statistic(x[nnans], values[nnans], statistic=statistic,
                                  bins=bins, range=range)


def std_spike_detector(x, N):
    """Returns boolean for values in exceed the mean by more than N standard
    deviations.

    Parameters
    ----------
    x : array_like
        Numbers.
    N : array_like
        Number of standard deivations.


    Returns
    -------
    tf : numpy.array
        Array of true and false flags.

    """
    x_mean = np.mean(x)
    x_std = np.std(x)
    tf = np.abs(x - x_mean) > N*x_std
    return tf


def interp_nonmon(x, xp, fp, left=None, right=None):
    """See documentation for numpy.interp. This does the same thing, however,
    if it detects that xp is not monotonically increasing it attempts to flip
    xp and fp before doing the interpolation. This should work for the case
    where xp is monotonically decreasing instead but not much else.

    """
    if np.mean(np.diff(xp)) < 0.:
        xpf = np.flipud(xp)
        fpf = np.flipud(fp)
        return np.interp(x, xpf, fpf, left, right)
    else:
        return np.interp(x, xp, fp, left, right)


def spherical_polar_gradient(f, lon, lat, r=6371000.):
    """Extension of the np.gradient function to spherical polar coordinates.
    Only gradients on a surface of constant radius (i.e. 2 dimensional) are
    currently supported. The grid must be evenly spaced in latitude and
    longitude.

    Important
    ---------
    For f(i, j), this function assumes i denotes position in latitude and j the
    position in longitude.

    Parameters
    ----------
    f : 2d array
        Scalar to calculate gradient.
    lon : 1d array
        Longitude points. [Degrees]
    lat : 1d array
        Latitude points. [Degrees]
    r : float
        Radius of sphere, defaults to Earth radius, 6371000 m.


    Returns
    -------
    dflon: 2d array
        Derivative in longitudinal direction.
    dflat: 2d array
        Derivative in the latitudinal direction.

    """
    nr, nc = f.shape
    if (nr != len(lat)) or (nc != len(lon)):
        raise ValueError('Latitude and longitude are expected to be rows and'
                         'columns respectively')

    lon, lat = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))

    dfi = np.gradient(f, axis=0)
    dfj = np.gradient(f, axis=1)
    dlon = np.gradient(lon, axis=1)
    dlat = np.gradient(lat, axis=0)

    # Cosine because latitude from -90 to 90. Not 0 to pi.
    dfdlon = dfj/(r*dlon*np.cos(lat))
    dfdlat = dfi/(r*dlat)

    return dfdlon, dfdlat


def spherical_polar_gradient_ts(f, lon, lat, r=6371000.):
    """Gradient of a two dimensional time series.

    Important
    ---------
    For f(i, j, k), this function assumes i denotes time, j latitude and k
    longitude.

    See spherical_polar_gradient for details.

    """
    nt, nr, nc = f.shape
    if (nr != len(lat)) or (nc != len(lon)):
        raise ValueError

    lon, lat = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))

    dfi = np.gradient(f, axis=1)
    dfj = np.gradient(f, axis=2)
    dlon = np.gradient(lon, axis=1)
    dlat = np.gradient(lat, axis=0)

    dlon = np.tile((r*dlon*np.cos(lat))[np.newaxis, ...], (nt, 1, 1))
    dlat = np.tile((r*dlat)[np.newaxis, ...], (nt, 1, 1))

    dfdlon = dfj/dlon
    dfdlat = dfi/dlat

    return dfdlon, dfdlat


def spherical_polar_area(r, lon, lat):
    """Calculates the area bounding an array of latitude and longitude points.

    Parameters
    ----------
    r : float
        Radius of sphere.
    lon : 1d array
        Longitude points. [Degrees]
    lat : 1d array
        Longitude points. [Degrees]

    Returns
    -------
    areas: 2d array

    """

    mid_dlon = (lon[2:] - lon[:-2])/2.
    s_dlon = lon[1] - lon[0]
    e_dlon = lon[-1] - lon[-2]
    dlon = np.hstack((s_dlon, mid_dlon, e_dlon))

    mid_dlat = (lat[2:] - lat[:-2])/2.
    s_dlat = lat[1] - lat[0]
    e_dlat = lat[-1] - lat[-2]
    dlat = np.hstack((s_dlat, mid_dlat, e_dlat))

    dlon, dlat = np.deg2rad(dlon), np.deg2rad(dlat)

    gdlon, gdlat = np.meshgrid(dlon, dlat)

    solid_angle = gdlon.T*gdlat.T*np.cos(np.deg2rad(lat))

    return solid_angle.T*r**2


def loadmat(filename, check_arrays=False, **kwargs):
    '''
    Big thanks to mergen on stackexchange for this:
        http://stackoverflow.com/a/8832212

    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.
    '''
    kwargs['struct_as_record'] = False
    kwargs['squeeze_me'] = True
    data = io.loadmat(filename, **kwargs)
    return _check_keys(data, check_arrays)


def _check_keys(dict, check_arrays):
    '''
    Checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries.
    '''
    for key in dict:
        if isinstance(dict[key], io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
        if isinstance(dict[key], np.ndarray) and check_arrays:
            shape = dict[key].shape
            array = dict[key].flatten()
            for i, item in enumerate(array):
                if isinstance(item, io.matlab.mio5_params.mat_struct):
                    array[i] = _todict(item)
            dict[key] = array.reshape(shape)
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries.
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def periodogram2D(z, fs=(1., 1.), window=None, detrend=None):
    """Calculate the two dimensional power spectral density.

    Parameters
    ----------
    z : 2D numpy array
        Data to compute spectral density from.
    fs : 2 element sequence
        The sampling frequency along the dimensions of z.
    window : optional, None, string
        Default is 'None'. Power spectral density should be modified by scaling
        factor if a window is chosen, this is not yet implemented.
    detrend : optional, None, string
        Default is 'constant' in which case the mean is subtracted from z. You
        can also use 'None' in which case no detrending is performed. Other
        types of detrending are not yet supported.

    Returns
    -------
    fi : numpy array
        Sampling frequencies/wavenumbers along the first dimension. Cyclical
        units, not angular.
    fj : numpy array
        Sampling frequencies/wavenumbers along the second dimension. Cyclical
        units, not angular.
    result : 2D numpy array
        Power spectral density of z. If z has units of 'V', and is sampled in
        units of 's', then the output has units V^2 / s^-2 (or V^2 / Hz^2).

    """

    fsi, fsj = fs
    Ni, Nj = z.shape

    if detrend == 'constant':
        z = z - np.mean(z)

    if window is None:
        window = 'boxcar'

    if isinstance(window, str) or type(window) is tuple:
        wini = sig.windows.get_window(window, Ni)
        winj = sig.windows.get_window(window, Nj)
        win = np.outer(wini, winj)
    else:
        raise ValueError("Value for window kwarg not valid.")

    FTz = np.fft.fftshift(np.fft.fft2(win*z))
    fi = np.fft.fftshift(np.fft.fftfreq(Ni, d=1./fsi))
    fj = np.fft.fftshift(np.fft.fftfreq(Nj, d=1./fsj))

    # The power spectrum.
    result = (FTz*FTz.conj()).real/(Ni*fsi*Nj*fsj)

    return fi, fj, result


def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""
    # Stole this off stack exchange...
    # https://stackoverflow.com/a/4495197
    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def butter(cutoff, fs, btype='low', order=4):
    """Return Butterworth filter coefficients. See scipy.signal.butter for a
    more thorough documentation.

    Parameters
    ----------
    cutoff : array
        Cutoff frequency, e.g. roughly speaking, the frequency at which the
        filter acts. Units should be same as for fs paramter.
    fs : float
        Sampling frequency of signal. Units should be same as for cutoff
        parameter.
    btype : {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
        Default is 'low'.
    order : optional, int
        Default is 4. The order of the Butterworth filter.

    Returns
    -------
    b : numpy array
        Filter b coefficients.
    a : numpy array
        Filter a coefficients.

    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a


def butter_filter(x, cutoff, fs, btype='low', order=4, **kwargs):
    """Apply Butterworth filter to data using scipy.signal.filtfilt.

    Parameters
    ----------
    x : array
        The data to be filtered. Should be evenly sampled.
    cutoff : array
        Cutoff frequency, e.g. roughly speaking, the frequency at which the
        filter acts. Units should be same as for fs paramter.
    fs : float
        Sampling frequency of signal. Units should be same as for cutoff
        parameter.
    btype : optional, string
        Default is 'low'. Filter type can be 'low', 'high' or 'band'.
    order : optional, int
        Default is 4. The order of the Butterworth filter.

    Returns
    -------
    y : numpy array
        The filtered data.

    """
    b, a = butter(cutoff, fs, btype=btype, order=order)
    y = sig.filtfilt(b, a, x, **kwargs)
    return y


def bin_data(x, bins, x_monotonic=True):
    """Bin data into given bins, which can be irregular sizes and even
    separated from one another.

    Parameters
    ----------
    x : array
        Index of the data to bin in the same units as the bins e.g. time stamp
        of sampling.
    bins : 2d array (N, 2)
        A size (N, 2) array of bins where the column 0 specifies the left bin
        edges and column 1 the right edges.
    x_monotonic : boolean
        Default is True. If false, boolean indexing is used to bin data (slow).

    Returns
    -------
    idxs : numpy array
        List of indexes corresponding to each bin.

    """
    idxs = []
    if x_monotonic:
        for bin in bins:
            idxs.append(np.arange(*np.searchsorted(x, bin)))
    else:
        for bin in bins:
            inbin = (x > bin[0]) & (x < bin[1])
            idxs.append(np.argwhere(inbin))
    return idxs
