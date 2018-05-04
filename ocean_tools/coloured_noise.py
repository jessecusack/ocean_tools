# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:53:40 2016

@author: jc3e13
"""

import numpy as np


def noise(N, dx, slope, c=1., ret_spec=False):
    """Generate a noise with a given spectral slope.

    Parameters
    ----------
    N : int
        Number of noise points to generate.
    dx : float
        Sample spacing, i.e. if time series in seconds, the number of seconds
        between consecutive measurements.
    slope :float
        Power spectral slope of the noise. Negative for red noise, positive for
        blue noise.
    c : float, optional
        Multiply the spectrum by this factor the change the magnitude of the
        noise.
    ret_spec : boolean, optional
        Default False. If True, also return the fft spectrum.

    Returns
    -------
    x : 1D array
        Noise.
    Fx : optional
        Noise spectrum.

    Notes
    -----
    The function first generates a spectrum with given slope slope, with random
    phase, then performs an inverse FFT to generate the noise series.

    """
    # Quick fix for odd N.
    if N % 2 == 1:
        oddN = True
        N += 1
    else:
        oddN = False

    f = np.fft.fftfreq(N, dx)[1:N/2]
    fNy = (1./(2.*dx))
    b = slope/2.  # Half because power spectrum is fourier spectrum squared.
    mag = f**b

    # Normalise magnitude so series std roughly 0.1
    magmax = 0.1*np.max(mag)
    mag = c*mag/magmax
    magNy = c*np.sqrt(2.)*fNy**b/magmax

    # Construct the spectrum with random phase.
    phase = np.random.rand(N/2-1)*2.*np.pi
    real = np.zeros(N)
    imag = np.zeros(N)
    real[1:N/2] = mag*np.cos(phase)
    imag[1:N/2] = mag*np.sin(phase)
    real[:N/2:-1] = real[1:N/2]
    imag[:N/2:-1] = -imag[1:N/2]
    real[N/2] = magNy
    Fx = real + 1j*imag
    # Inverse transform to generate series.
    x = np.fft.ifft(Fx).real

    # Quick fix for odd N.
    if oddN:
        x = x[:-1]

    if ret_spec:
        return x, Fx
    else:
        return x


def more_noise(N, dx, slopes, fc, c=1., ret_spec=False):
    """Generate a noise with a given spectrum.

    Parameters
    ----------
    N : int
        Number points to generate.
    dx : float
        Sample spacing, i.e. if time series in seconds, the number of seconds
        between consecutive measurements.
    slopes : array of floats
        Power spectral slopes of the noise. Negative for red noise and positive
        for blue. e.g. [0., -2, 0., 1]. There should be one more slope than
        change frequency.
    fc : array
        Frequencies at which the slope changes, specified in order of
        increasing frequency. e.g. [1, 10, 50]. There should be one less change
        frequency than slope!
    c : float, optional
        Multiply the spectrum by this factor the change the magnitude of the
        noise.
    ret_spec : boolean, optional
        Default False. If True, also return the fft spectrum.

    Returns
    -------
    x : 1D array
        Noise.
    Fx : optional
        Noise spectrum.

    """
    slopes = np.asarray(slopes)
    fc = np.asarray(fc)

    if np.ndim(fc) == 0:
        fc = fc[np.newaxis]

    Ns = slopes.size
    Nfc = fc.size

    if Ns < 2:
        raise ValueError("Please specify more than one slope or use the noise"
                         " function")

    if not Ns == (Nfc+1):
        raise ValueError("There should be one more slope specified than change"
                         " frequency.")

    if N % 2 == 1:
        oddN = True
        N += 1
    else:
        oddN = False

    slopes /= 2.  # Half because power spectrum is fourier spectrum squared.
    f = np.fft.fftfreq(N, dx)[1:N/2]
    fNy = (1./(2.*dx))  # Nyquist frequency

    Nf = N/2 - 1
    idx = np.searchsorted(f, fc)
    idxs = np.hstack((0, idx, Nf))
    mag = np.zeros(Nf)

    coefs = np.ones(Ns)
    for i in range(Nfc):
        coefs[i+1] = coefs[i]*fc[i]**(slopes[i]-slopes[i+1])

    for i in range(Ns):
        i1, i2 = idxs[i], idxs[i+1]
        mag[i1:i2] = coefs[i]*f[i1:i2]**slopes[i]

    # Normalise magnitude so series std roughly 0.1
    magmax = 0.1*np.max(mag)
    mag = c*mag/magmax
    magNy = c*np.sqrt(2.)*coefs[-1]*fNy**slopes[-1]/magmax

    # Construct the spectrum with random phase.
    phase = np.random.rand(N/2-1)*2.*np.pi
    real = np.zeros(N)
    imag = np.zeros(N)
    real[1:N/2] = mag*np.cos(phase)
    imag[1:N/2] = mag*np.sin(phase)
    real[:N/2:-1] = real[1:N/2]
    imag[:N/2:-1] = -imag[1:N/2]
    real[N/2] = magNy
    Fx = real + 1j*imag
    # Inverse transform to generate series.
    x = np.fft.ifft(Fx).real

    # Quick fix for odd N.
    if oddN:
        x = x[:-1]

    if ret_spec:
        return x, Fx
    else:
        return x
