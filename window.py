# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 14:16:02 2014

@author: jc3e13

Functions for chopping up arrays using moving windows and smoothing data.
"""

import numpy as np
import utils


def chunk(x, x_range, y):
    """Chunk returns slices of arrays x and y given some range of x values.

    Parameters
    ----------
    x : array_like
        Monotonically increasing values.
    x_range : sequence
        Should contain (min, max) value at which to slice x and y.
    y : array_like
        Arbitrary values.

    Returns
    -------
    x_chunk : array_like
        Values of x that fall in the range x_range.
    y_chunk : array_like
        values of y that fall in the range x_range.

    """

    if len(x_range) != 2:
        raise ValueError('x_range must be a sequence of two numbers only.')

    s = slice(*np.searchsorted(x, x_range))

    return x[s], y[s]


def window(x, y, width, overlap=0., x_0=None, expansion=None, cap_left=True,
           cap_right=True, ret_x=True):
    """Break arrays x and y into slices.

    Parameters
    ----------
    x : array_like
        Monotonically increasing numbers. If x is not monotonically increasing
        then it will be flipped, beware that this may not have the desired
        effect.
    y : array_like
        Arbitrary values, same size as x.
    width : float
        Window width in the same units as x.
    overlap : float, optional
        Overlap of windows in the same units as x. If negative, the window
        steps along x values rather than binning.
    x_0 : float, optional
        Position in x at which to start windowing. (untested)
    expansion : polynomial coefficients, optional
        Describes the rate of change of window size with x. (not implimented)
        The idea is that width = width*np.polyval(expansion, x). Overlap is
        similarly increased.
    cap_left : boolean, optional
        Stop window exceeding left most (minimum) value of x. Only applies when
        overlap is positive.
    cap_right : boolean, optional
        Stop window exceeding right most (maximum) value of x. Only applies
        when overlap is positive.

    Returns
    -------
    vals : numpy.array
        Contains all the windowed chunks of x and y.

    Notes
    -----
    The current check on monotonicity is whether more than 20% of points in
    x are are not monotonic. This is a sort of hack to avoid flipping for the
    occasional erroneous non-monotonic point.

    """

    if x.size != y.size:
        raise ValueError('x and y must be of equal size.')

    if overlap > width:
        raise ValueError('The overlap cannot be larger than the width.')

    # Incredibly bad check for monotonicity.
    not_monotonic = np.sum(np.diff(x) < 0) > 0.2*len(x)
    if not_monotonic:
        x = utils.flip_cols(x)
        y = utils.flip_cols(y)

    if x_0 is not None:
        idxs = ~np.isnan(x) & (x >= x_0)
    else:
        idxs = ~np.isnan(x)

    x = x[idxs]
    y = y[idxs]

    if overlap < 0.:
        left = x - width/2.
        right = left + width

    elif overlap >= 0.:
        step = width - overlap

        if cap_left:
            xmin = x[0]
        else:
            xmin = x[0] - width

        if cap_right:
            # Take away slightly less than the full width to allow for the last
            # bin to complete the full range.
            xmax = x[-1] - 0.99*width
        else:
            xmax = x[-1]

        left = np.arange(xmin, xmax, step)
        right = left + width

    bins = np.transpose(np.vstack((left, right)))

    if ret_x:
        vals = np.asarray([chunk(x, b, y) for b in bins])
    else:
        vals = np.asarray([chunk(x, b, y)[1] for b in bins])

    if not_monotonic:
        vals = np.flipud(vals)

    return vals


def moving_polynomial_smooth(x, y, width=25., deg=1, expansion=None):
    """Smooth y using a moving polynomial fit.

    Parameters
    ----------
    x : array_like
        Monotonically increasing numbers. If x is not monotonically increasing
        then it will be flipped, beware that this may not have the desired
        effect.
    y : array_like
        Arbitrary values, same size as x.
    width : float
        Window width in the same units as x.
    deg : int
        Degree of the polynomial with which to smooth.
    expansion : polynomial coefficients
        Describes the rate of change of window size with x. (not implimented)
        The idea is that width = width_0*np.polyval(expansion, x)

    Returns
    -------
    y_out : numpy.array
        Smoothed y.

    """

    vals = window(x, y, width=width, overlap=-1, expansion=expansion)

    idxs = ~np.isnan(x)
    y_out = np.nan*np.zeros_like(x)
    xp = x[idxs]
    yp = y_out[idxs]

    for i, val in enumerate(vals):
        p = np.polyfit(val[0], val[1], deg)
        yp[i] = np.polyval(p, xp[i])

    y_out[idxs] = yp

    return y_out


def moving_mean_smooth(x, y, width=25., expansion=None):
    """Smooth y using a moving mean.

    Parameters
    ----------
    x : array_like
        Monotonically increasing numbers. If x is not monotonically increasing
        then it will be flipped, beware that this may not have the desired
        effect.
    y : array_like
        Arbitrary values, same size as x.
    width : float
        Window width in the same units as x.
    expansion : polynomial coefficients
        Describes the rate of change of window size with x. (not implimented)
        The idea is that width = width_0*np.polyval(expansion, x)

    Returns
    -------
    y_out : numpy.array
        Smoothed y.

    """

    vals = window(x, y, width=width, overlap=-1, expansion=expansion)

    idxs = ~np.isnan(x)
    y_out = np.nan*np.zeros_like(x)
    yp = y_out[idxs]

    for i, val in enumerate(vals):
        yp[i] = np.mean(val[1])

    y_out[idxs] = yp

    return y_out
