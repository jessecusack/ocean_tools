# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:53:40 2016

@author: jc3e13
"""

import numpy as np
import numpy.random as random


def noise(N, dx, beta, mu=1., std=0.2):
    """Generate a noise with a given spectral slope, beta.

    Parameters
    ----------
    N : scalar
        Number of noise points to generate.
    dx : scalar
        Sample spacing, i.e. if time series in seconds, the number of seconds
        between consecutive measurements.
    beta :scalar
        Power spectral slope of the noise. Negative for red noise, positive for
        blue noise.
    mu : scalar
        Mean of the random array that multiplies the spectrum magnitude.
    std: scalar
        Standard deviation of the random array that multiplies the spectrum
        magnitude.

    Returns
    -------
    y : 1D array
        Noise.

    Notes
    -----
    The function first generates a spectrum with given slope beta, with random
    phase, then performs an inverse FFT to generate the series.

    """

    # Quick fix for odd N.
    if N % 2 == 1:
        Nisodd = True
        N += 1
    else:
        Nisodd = False

    f = np.fft.fftfreq(N, dx)[1:N/2]
    fNy = (1./(2.*dx))
    b = beta/2.  # Half because power spectrum is fourier spectrum squared.

    # Not sure if multiplying magnitude by normally distributed random numbers
    # is approprimate... maybe a poisson distribution is better?
    mag = (std*random.randn(N/2-1) + mu)*f**b
    magNy = np.sign(random.randn())*(std*random.randn() + mu)*fNy**b

    # Normalise the spectra so the intagrel is 1. There might be a bug here in
    # that the Nyquist freq doesn't want to be multiplied by 2.
    I = 2*np.trapz(np.hstack((mag, magNy)), np.hstack((f, fNy)))
    mag /= I
    magNy /= I

    phase = random.rand(N/2-1)*2.*np.pi

    real = np.zeros(N)
    imag = np.zeros(N)

    real[1:N/2] = mag*np.cos(phase)
    imag[1:N/2] = mag*np.sin(phase)
    real[:N/2:-1] = real[1:N/2]
    imag[:N/2:-1] = -imag[1:N/2]

    # Real part of Nyquist frequency has magnitude too!
    real[N/2] = magNy

    Fy = real + 1j*imag

    out = np.fft.ifft(Fy).real

    # Quick fix for odd N.
    if Nisodd:
        out = out[:-1]

    return out


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import scipy.signal as sig
    from scipy.optimize import curve_fit

    dx = 1.
    N = 2048
    beta = -2.

    x = np.arange(0., N*dx, dx)
    y = noise(N, dx, beta)

    f, Py = sig.periodogram(y, 1./dx)
    Py[0] = 0.

    Pyfit = lambda x, a, b: a*x + b
    popt, __ = curve_fit(Pyfit, np.log10(f[1:]), np.log10(Py[1:]), p0=[1., 1.])
    a, b = popt

    fig, axs = plt.subplots(1, 2)
    axs[0].loglog(f, Py, 'k', label=None)
    axs[0].loglog(f[1:], f[1:]**a*10**b, 'r',
                  label="Fit exponent: {:1.0f}".format(popt[0]))
    axs[1].plot(x, y, 'k')

    axs[0].set_xlabel('Frequency')
    axs[0].set_ylabel('Variance')
    axs[1].set_xlabel('Time')

    axs[0].legend(loc=0)
