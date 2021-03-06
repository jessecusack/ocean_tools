# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:45:36 2014

A place for turbulent kinetic energy dissipation parameterisation functions.

@author: jc3e13
"""

import numpy as np
import gsw
import seawater as sw
import scipy.signal as sig
import matplotlib.pyplot as plt
from . import window
from . import GM
from . import utils


# Define some standard parameters.
default_corrections = {
    'use_range': False,
    'use_diff': False,
    'use_interp': True,
    'use_tilt': False,
    'use_bin': False,
    'use_volt': True,
    'dzt': 8.,
    'dzr': 8.,
    'dzfd': 8.,
    'dzg': 8.,
    'ddash': 5.4,
    'dzs': 8.,
    'vfi': 50.,
    'mfr': 0.12
    }

default_periodogram_params = {
    'window': 'hanning',
    'nfft': 256,
    'detrend': 'linear',
    'scaling': 'density',
    }

default_params = {
    'dz': 4.,
    'zmin': None,
    'zmax': -300.,
    'bin_width': 300.,
    'bin_overlap': 200.,
    'fine_grid_spectra': False,
    'print_diagnostics': False,
    'plot_profiles': False,
    'plot_spectra': False,
    'plot_results': False,
    'plot_dir': '../figures/finescale',
    'm_0': 1./150.,
    'm_c': 1./15.,
    'apply_corrections': False,
    'corrections': default_corrections,
    'periodogram_params': default_periodogram_params,
    'mixing_efficiency': 0.2
    }


def sin2taper(L):
    """A boxcar window that tapers the last 10% of points of both ends using a
    sin^2 function."""
    win = np.ones(L)
    idx10 = int(np.ceil(L/10.))
    idxs = np.arange(idx10)
    win[:idx10] = np.sin(np.pi*idxs/(2.*idx10))**2
    win[-idx10:] = np.cos(np.pi*(idxs + 1 - L)/(2.*idx10))**2
    return win


def adiabatic_level(P, SA, T, lat, P_bin_width=200., deg=1):
    """Generate smooth buoyancy frequency profile by applying the adiabatic
    levelling method of Bray and Fofonoff (1981).

    THIS FUNCTION HAS BEEN SUPERSEEDED BY:
        adiabatic_level_sw
        adiabatic_level_gsw

    Both of which are significantly faster by over a factor of 10.

    Parameters
    ----------
    P : 1-D ndarray
        Pressure [dbar]
    SA : 1-D ndarray
        Absolute salinity [g/kg]
    T : 1-D ndarray
        Temperature [degrees C]
    lat : float
        Latitude [-90...+90]
    p_bin_width : float, optional
        Pressure bin width [dbar]
    deg : int, optional
        Degree of polynomial fit. (DEGREES HIGHER THAN 1 NOT YET TESTED)

    Returns
    -------
    N2_ref : 1-D ndarray
        Reference buoyancy frequency [s-2]

    Notes
    -----
    Calls to the gibbs seawater toolbox are slow and therefore this function
    is quite slow.

    """

    N2_ref = np.NaN*P.copy()
    nans = np.isnan(P) | np.isnan(SA) | np.isnan(T)

    # If there are nothing but NaN values don't waste time.
    if np.sum(nans) == nans.size:
        return N2_ref

    P = P[~nans]
    SA = SA[~nans]
    T = T[~nans]
    P_min, P_max = np.min(P), np.max(P)

    shape = (P.size, P.size)
    Pm = np.NaN*np.empty(shape)
    SAm = np.NaN*np.empty(shape)
    Tm = np.NaN*np.empty(shape)

    # Populate bins.
    for i in range(len(P)):

        P_bin_min = np.maximum(P[i] - P_bin_width/2., P_min)
        P_bin_max = np.minimum(P[i] + P_bin_width/2., P_max)
        in_bin = np.where((P >= P_bin_min) & (P <= P_bin_max))[0]

        Pm[in_bin, i] = P[in_bin]
        SAm[in_bin, i] = SA[in_bin]
        Tm[in_bin, i] = T[in_bin]

    P_bar = np.nanmean(Pm, axis=0)
    T_bar = np.nanmean(Tm, axis=0)
    SA_bar = np.nanmean(SAm, axis=0)

    # Perform thermodynamics once only...
    rho_bar = gsw.pot_rho_t_exact(SA_bar, T_bar, P_bar, P_bar)
    sv = 1./gsw.pot_rho_t_exact(SAm, Tm, Pm, P_bar)

    p = []
    for P_bin, sv_bin in zip(Pm.T, sv.T):
        bnans = np.isnan(P_bin)
        p.append(np.polyfit(P_bin[~bnans],
                            sv_bin[~bnans] - np.nanmean(sv_bin),
                            deg))

    p = np.asarray(p)

    g = gsw.grav(lat, P_bar)
    # The factor 1e-4 is needed for conversion from dbar to Pa.
    N2_ref[~nans] = -1e-4*rho_bar**2*g**2*p[:, 0]

    return N2_ref


def adiabatic_level_gsw(P, S, T, lon, lat, bin_width=100., order=1,
                        ret_coefs=False, cap=None):
    """Generate smooth buoyancy frequency profile by applying the adiabatic
    levelling method of Bray and Fofonoff (1981). This function uses the newest
    theormodynamic toolbox, 'gsw'.

    Parameters
    ----------
    P : 1-D ndarray
        Pressure [dbar]
    S : 1-D ndarray
        Practical salinity [-]
    T : 1-D ndarray
        Temperature [degrees C]
    lon : float
        Longitude [-180...+360]
    lat : float
        Latitude [-90...+90]
    bin_width : float, optional
        Pressure bin width [dbar]
    order : int, optional
        Degree of polynomial fit. (DEGREES HIGHER THAN 1 NOT PROPERLY TESTED)
    ret_coefs : bool, optional
        Flag to return additional argument pcoefs. False by default.
    cap : optional
        Flag to change proceedure at ends of array where bins may be partially
        filled. None by default, meaning they are included. Can also specify
        'left', 'right' or 'both' to cap method before partial bins.

    Returns
    -------
    N2_ref : 1-D ndarray
        Reference buoyancy frequency [s-2]
    pcoefs : 2-D ndarray
        Fitting coefficients, returned only when the flag ret_coefs is set
        True.

    """
    valid = np.isfinite(P) & np.isfinite(S) & np.isfinite(T)
    valid = np.squeeze(np.argwhere(valid))
    P_, S_, T_ = P[valid], S[valid], T[valid]

    flip = False
    if (np.diff(P_) < 0).all():
        flip = True
        P_ = np.flipud(P_)
        S_ = np.flipud(S_)
        T_ = np.flipud(T_)
    elif (np.diff(P_) < 0).any():
        raise ValueError('P must be monotonically increasing/decreasing.')

    i1 = np.searchsorted(P_, P_ - bin_width/2.)
    i2 = np.searchsorted(P_, P_ + bin_width/2.)

    if cap is None:
        Nd = P_.size
    elif cap == 'both':
        icapl = i2[0]
        icapr = i1[-1]
    elif cap == 'left':
        icapl = i2[0]
        icapr = i2[-1]
    elif cap == 'right':
        icapl = i1[0]
        icapr = i1[-1]
    else:
        raise ValueError("The argument cap must be either None, 'both', 'left'"
                         " or 'right'")

    if cap is not None:
        i1 = i1[icapl:icapr]
        i2 = i2[icapl:icapr]
        valid = valid[icapl:icapr]
        Nd = icapr - icapl

    dimax = np.max(i2 - i1)

    Pb = np.full((dimax, Nd), np.nan)
    Sb = np.full((dimax, Nd), np.nan)
    Tb = np.full((dimax, Nd), np.nan)

    for i in range(Nd):
        imax = i2[i]-i1[i]
        Pb[:imax, i] = P_[i1[i]:i2[i]]
        Sb[:imax, i] = S_[i1[i]:i2[i]]
        Tb[:imax, i] = T_[i1[i]:i2[i]]

    Pbar = np.nanmean(Pb, axis=0)

    SAb = gsw.SA_from_SP(Sb, Pb, lon, lat)

    rho = gsw.pot_rho_t_exact(SAb, Tb, Pb, Pbar)
    sv = 1./rho

    rhobar = np.nanmean(rho, axis=0)

    p = np.full((order+1, Nd), np.nan)

    for i in range(Nd):
        imax = i2[i]-i1[i]
        p[:, i] = np.polyfit(Pb[:imax, i], sv[:imax, i], order)

    g = gsw.grav(lat, Pbar)
    # The factor 1e-4 is needed for conversion from dbar to Pa.
    if order == 1:
        N2 = -1e-4*rhobar**2*g**2*p[0, :]
    elif order == 2:
        N2 = -1e-4*rhobar**2*g**2*(p[1, :]+2*Pbar*p[0,:])
    elif order == 3:
        N2 = -1e-4*rhobar**2*g**2*(p[2, :]+2*Pbar*p[1,:]+3*Pbar**2*p[0,:])
    else:
        raise ValueError('Fits are only included up to 3rd order.')

    N2_ref = np.full_like(P, np.nan)
    pcoef = np.full((order+1, P.size), np.nan)
    if flip:
        N2_ref[valid] = np.flipud(N2)
        pcoef[:, valid] = np.fliplr(p)
    else:
        N2_ref[valid] = N2
        pcoef[:, valid] = p

    if ret_coefs:
        return N2_ref, pcoef
    else:
        return N2_ref


def adiabatic_level_sw(P, S, T, lat, bin_width=100., order=1,
                       ret_coefs=False, cap=None):
    """Generate smooth buoyancy frequency profile by applying the adiabatic
    levelling method of Bray and Fofonoff (1981). This function uses the older
    theormodynamic toolbox, 'seawater'.

    Parameters
    ----------
    P : 1-D ndarray
        Pressure [dbar]
    S : 1-D ndarray
        Practical salinity [-]
    T : 1-D ndarray
        Temperature [degrees C]
    lat : float
        Latitude [-90...+90]
    bin_width : float, optional
        Pressure bin width [dbar]
    deg : int, optional
        Degree of polynomial fit. (DEGREES HIGHER THAN 1 NOT PROPERLY TESTED)
    ret_coefs : bool, optional
        Flag to return additional argument pcoefs. False by default.
    cap : optional
        Flag to change proceedure at ends of array where bins may be partially
        filled. None by default, meaning they are included. Can also specify
        'left', 'right' or 'both' to cap method before partial bins.

    Returns
    -------
    N2_ref : 1-D ndarray
        Reference buoyancy frequency [s-2]
    pcoefs : 2-D ndarray
        Fitting coefficients, returned only when the flag ret_coefs is set
        True.

    """
    valid = np.isfinite(P) & np.isfinite(S) & np.isfinite(T)
    valid = np.squeeze(np.argwhere(valid))
    P_, S_, T_ = P[valid], S[valid], T[valid]

    flip = False
    if (np.diff(P_) < 0).all():
        flip = True
        P_ = np.flipud(P_)
        S_ = np.flipud(S_)
        T_ = np.flipud(T_)
    elif (np.diff(P_) < 0).any():
        raise ValueError('P must be monotonically increasing/decreasing.')

    i1 = np.searchsorted(P_, P_ - bin_width/2.)
    i2 = np.searchsorted(P_, P_ + bin_width/2.)

    if cap is None:
        Nd = P_.size
    elif cap == 'both':
        icapl = i2[0]
        icapr = i1[-1]
    elif cap == 'left':
        icapl = i2[0]
        icapr = i2[-1]
    elif cap == 'right':
        icapl = i1[0]
        icapr = i1[-1]
    else:
        raise ValueError("The argument cap must be either None, 'both', 'left'"
                         " or 'right'")

    if cap is not None:
        i1 = i1[icapl:icapr]
        i2 = i2[icapl:icapr]
        valid = valid[icapl:icapr]
        Nd = icapr - icapl

    dimax = np.max(i2 - i1)

    Pb = np.full((dimax, Nd), np.nan)
    Sb = np.full((dimax, Nd), np.nan)
    Tb = np.full((dimax, Nd), np.nan)

    for i in range(Nd):
        imax = i2[i]-i1[i]
        Pb[:imax, i] = P_[i1[i]:i2[i]]
        Sb[:imax, i] = S_[i1[i]:i2[i]]
        Tb[:imax, i] = T_[i1[i]:i2[i]]

    Pbar = np.nanmean(Pb, axis=0)

    rho = sw.pden(Sb, Tb, Pb, Pbar)
    sv = 1./rho

    rhobar = np.nanmean(rho, axis=0)

    p = np.full((order+1, Nd), np.nan)

    for i in range(Nd):
        imax = i2[i]-i1[i]
        p[:, i] = np.polyfit(Pb[:imax, i], sv[:imax, i], order)

    g = sw.g(lat, -sw.dpth(Pbar, lat))
    # The factor 1e-4 is needed for conversion from dbar to Pa.
    N2 = -1e-4*rhobar**2*g**2*p[order-1, :]

    N2_ref = np.full_like(P, np.nan)
    pcoef = np.full((order+1, P.size), np.nan)
    if flip:
        N2_ref[valid] = np.flipud(N2)
        pcoef[:, valid] = np.fliplr(p)
    else:
        N2_ref[valid] = N2
        pcoef[:, valid] = p

    if ret_coefs:
        return N2_ref, pcoef
    else:
        return N2_ref


def h_gregg(R=3.):
    """Gregg 2003 implimentation."""
    return 3.*(R + 1)/(2.*R*np.sqrt(2*np.abs(R - 1)))


def h_whalen(R=3.):
    """Whalen 2012 implimentation based on Kunze 2006 implimentation (which
    is based on Gregg yet equations are different)."""
    return R*(R + 1)/(6.*np.sqrt(2*np.abs(R - 1)))


def L(f, N):
    f30 = 7.292115e-5  # rad s-1
    N0 = 5.2e-3  # rad s-1
    f = np.abs(f)
    return f*np.arccosh(N/f)/(f30*np.arccosh(N0/f30))


def CW_ps(Pxx, Pyy, Pxy):
    """Clockwise power spectrum."""
    # NOTE that in the MATLAB code QS = -Pxy.imag because they take the
    # conjugate transpose of Pxy first meaning that the imaginary parts are
    # multiplied by -1.
    QS = Pxy.imag
    return (Pxx + Pyy - 2.*QS)/2.


def CCW_ps(Pxx, Pyy, Pxy):
    """Counter clockwise power spectrum."""
    QS = Pxy.imag
    return (Pxx + Pyy + 2*QS)/2.


def integrated_ps(m, P, m_c, m_0):
    """Integrates power spectrum P between wavenumbers m_c and m_0."""

    idxs = (m < m_c) & (m > m_0)
    return np.trapz(P[idxs], x=m[idxs])


def spectral_correction(m, use_range=True, use_diff=True, use_interp=True,
                        use_tilt=True, use_bin=True, use_volt=True, dzt=8.,
                        dzr=8., dzfd=8., dzg=8., ddash=5.4, dzs=8., vfi=50.,
                        mfr=0.12):
    """
    Calculates the appropriate transfer function to scale power spectra.

    Parameters
    ----------
    m : ndarray
        Vertical wavenumber. [rad s-1]
    use_range : boolean, optional (LADCP)
        Switch for range correction.
    use_diff : boolean, optional (LADCP)
        Switch for differencing correction.
    use_interp : boolean, optional (LADCP/EM-APEX)
        Switch for interpolation correction.
    use_tilt : boolean, optional (LADCP)
        Switch for tilt correction.
    use_bin : boolean, optional (LADCP)
        Switch for binning correction.
    use_volt : boolean, optional (EM-APEX)
        Switch for voltmeter noise correction.
    dzt : float, optional (LADCP)
        Transmitted sound pulse length projected on the vertical. [m]
    dzr : float, optional (LADCP/EM-APEX)
        Receiver processing bin length. [m]
    dzfd : float, optional (LADCP)
        First-differencing interval. [m]
    dzg : float, optional (LADCP/EM-APEX)
        Interval of depth grid onto which single-ping piecewise-linear
        continuous profiles of vertical shear are binned. [m]
    ddash : float, optional (LADCP)
        ?
    dzs : float, optional (LADCP)
        Superensemble pre-averaging interval, usually chosen to be dzg. [m]
    vfi : float, optional (EM-APEX)
        ? [s-1]
    mfr : float, optional (EM-APEX)
        ? [m s-1]

    Returns
    -------
    T : ndarray
        Transfer function, which is the product of all of the individual
        transfer functions for each separate spectral correction.


    Notes
    -----
    Spectral corrections for LADCP data - see Polzin et. al. 2002.

    There is another possible correction which isn't used.

    Notes from MATLAB code
    ----------------------

    A quadratic fit to the range maxima (r_max) pairs given by Polzin et al.
    (2002) yields.

    ddash = -1.2+0.0857r_max - 0.000136r_max^2 ,

    which has an intercept near r_max = 14 m. It should be noted that
    expressions (4) and (5) are semi-empirical and apply strictly only to the
    data set of Polzin et al. (2002). Estimating r_max ? 255 m as the range at
    which 80% of all ensembles have valid velocities yields d? ? 11.8 m in case
    of this data set (i.e as in Thurherr 2011 NOT DIMES - need to update!!).
    ddash is determined empirically by Polzin et al. (2002) and is dependent
    on range and the following assumptions:
        Small tilts (~ 3 deg).
        Instrument tilt and orientation are constant over measurement period.
        Instrument tilt and orientation are independent.
        Tilt attenuation is limited by bin-mapping capabilities of RDI (1996)
        processing.

    """

    pi2 = np.pi*2

    # Range averaging.
    if use_range:
        T_range = np.sinc(m*dzt/pi2)**2 * np.sinc(m*dzr/pi2)**2
    else:
        T_range = 1.

    # First differencing.
    if use_diff:
        T_diff = np.sinc(m*dzfd/pi2)**2
    else:
        T_diff = 1.

    # Interpolation.
    if use_interp:
        T_interp = np.sinc(m*dzr/pi2)**4 * np.sinc(m*dzg/pi2)**2
    else:
        T_interp = 1.

    # Tilting.
    if use_tilt:
        T_tilt = np.sinc(m*ddash/pi2)**2
    else:
        T_tilt = 1.

    # Binning
    if use_bin:
        T_bin = np.sinc(m*dzg/pi2)**2 * np.sinc(m*dzs/pi2)**2
    else:
        T_bin = 1.

    # Voltmeter
    if use_volt:
        T_volt = 1./np.sinc(m*vfi*mfr/pi2)
    else:
        T_volt = 1.

    T = T_range*T_diff*T_interp*T_tilt*T_bin*T_volt

    return T


def window_ps(dz, U, V, dUdz, dVdz, strain, N2_ref, params=default_params):
    """Calculate the power spectra for a window of data."""

    # Normalise the shear by the mean buoyancy frequency.
    ndUdz = dUdz/np.mean(np.sqrt(N2_ref))
    ndVdz = dVdz/np.mean(np.sqrt(N2_ref))

    # Compute the (co)power spectral density.
    fs = 1./dz
    m, PdUdV = sig.csd(ndUdz, ndVdz, fs=fs, **params['periodogram_params'])
    __, PdU = sig.periodogram(ndUdz, fs=fs, **params['periodogram_params'])
    __, PdV = sig.periodogram(ndVdz, fs=fs, **params['periodogram_params'])
    __, PU = sig.periodogram(U, fs=fs, **params['periodogram_params'])
    __, PV = sig.periodogram(V, fs=fs, **params['periodogram_params'])
    __, Pstrain = sig.periodogram(strain, fs=fs, **params['periodogram_params'])

    # Clockwise and counter clockwise spectra.
    PCW = CW_ps(PdU, PdV, PdUdV)
    PCCW = CCW_ps(PdU, PdV, PdUdV)
    # Shear spectra.
    Pshear = PdU + PdV

    if params['apply_corrections']:
        T = spectral_correction(m, **params['corrections'])
        PCW /= T
        PCCW /= T
        Pshear /= T
        PU /= T
        PV /= T

        if params['print_diagnostics']:
            print("T = {}".format(T))

    # Kinetic energy spectra.
    PEK = (PU + PV)/2.

    return m, Pshear, Pstrain, PCW, PCCW, PEK


def analyse(z, U, V, dUdz, dVdz, strain, N2_ref, lat, params=default_params):
    """ """

    X = [U, V, dUdz, dVdz, strain, N2_ref]

    if params['plot_profiles']:
        fig, axs = plt.subplots(1, 4, sharey=True)

        axs[0].set_ylabel('$z$ (m)')
        axs[0].plot(np.sqrt(N2_ref), z, 'k-', label='$N_{ref}$')
        axs[0].plot(np.sqrt(strain*N2_ref + N2_ref), z, 'k--', label='$N$')
        axs[0].set_xlabel('$N$ (rad s$^{-1}$)')
        axs[0].legend(loc=0)
        axs[0].set_xticklabels(axs[0].get_xticks(), rotation='vertical')
        axs[1].plot(U, z, 'k-', label='$U$')
        axs[1].plot(V, z, 'r-', label='$V$')
        axs[1].set_xlabel('$U$, $V$ (m s$^{-1}$)')
        axs[1].legend(loc=0)
        axs[1].set_xticklabels(axs[1].get_xticks(), rotation='vertical')
        axs[2].plot(dUdz, z, 'k-', label=r'$\frac{dU}{dz}$')
        axs[2].plot(dVdz, z, 'r-', label=r'$\frac{dV}{dz}$')
        axs[2].set_xlabel(r'$\frac{dU}{dz}$, $\frac{dV}{dz}$ (s$^{-1}$)')
        axs[2].legend(loc=0)
        axs[2].set_xticklabels(axs[2].get_xticks(), rotation='vertical')
        axs[3].plot(strain, z, 'k-')
        axs[3].set_xlabel(r'$\xi_z$ (-)')

    # Split varables into overlapping window segments, bare in mind the last
    # window may not be full.
    width = params['bin_width']
    overlap = params['bin_overlap']
    wdws = [wdw.window(z, x, width=width, overlap=overlap) for x in X]

    n = wdws[0].shape[0]
    z_mean = np.empty(n)
    EK = np.empty(n)
    R_pol = np.empty(n)
    R_om = np.empty(n)
    epsilon = np.empty(n)
    kappa = np.empty(n)

    for i, w in enumerate(zip(*wdws)):

        # This takes the z values from the horizontal velocity.
        wz = w[0][0]
        z_mean[i] = np.mean(wz)
        # This (poor code) removes the z values from windowed variables.
        w = [var[1] for var in w]
        N2_mean = np.mean(w[-1])
        N_mean = np.sqrt(N2_mean)

        # Get the useful power spectra.
        m, PCW, PCCW, Pshear, Pstrain, PEK = \
            window_ps(params['dz'], *w, params=params)

        # Integrate the spectra.
        I = [integrated_ps(m, P, params['m_c'], params['m_0'])
             for P in [Pshear, Pstrain, PCW, PCCW, PEK]]

        Ishear, Istrain, ICW, ICCW, IEK = I

        # Garrett-Munk shear power spectral density normalised.
        # The factor of 2 pi is there to convert to cyclical units.
        GMshear = 2.*np.pi*GM.E_she_z(2*np.pi*m, N_mean)/N_mean

        IGMshear = integrated_ps(m, GMshear, params['m_c'], params['m_0'])

        EK[i] = IEK
        R_pol[i] = ICCW/ICW
        R_om[i] = Ishear/Istrain
        epsilon[i] = GM.epsilon_0*N2_mean/GM.N0**2*Ishear**2/IGMshear**2
        # Apply correcting factors

        epsilon[i] *= L(gsw.f(lat), N_mean)*h_gregg(R_om[i])

        kappa[i] = params['mixing_efficiency']*epsilon[i]/N2_mean

        if params['print_diagnostics']:
            print("Ishear = {}".format(Ishear))
            print("IGMshear = {}".format(IGMshear))
            print("lat = {}. f = {}.".format(lat, gsw.f(lat)))
            print("N_mean = {}".format(N_mean))
            print("R_om = {}".format(R_om[i]))
            print("L = {}".format(L(gsw.f(lat), N_mean)))
            print("h = {}".format(h_gregg(R_om[i])))

        # Plotting here generates a crazy number of plots.
        if params['plot_spectra']:

            # The factor of 2 pi is there to convert to cyclical units.
            GMstrain = 2.*np.pi*GM.E_str_z(2*np.pi*m, N_mean)
            GMvel = 2.*np.pi*GM.E_vel_z(2*np.pi*m, N_mean)

            fig, axs = plt.subplots(4, 1, sharex=True)

            axs[0].loglog(m, PEK, 'k-', label="$E_{KE}$")
            axs[0].loglog(m, GMvel, 'k--', label="GM $E_{KE}$")
            axs[0].set_title("height {:1.0f} m".format(z_mean[i]))
            axs[1].loglog(m, Pshear, 'k-', label="$V_z$")
            axs[1].loglog(m, GMshear, 'k--', label="GM $V_z$")
            axs[2].loglog(m, Pstrain, 'k', label=r"$\xi_z$")
            axs[2].loglog(m, GMstrain, 'k--', label=r"GM $\xi_z$")
            axs[3].loglog(m, PCW, 'r-', label="CW")
            axs[3].loglog(m, PCCW, 'k-', label="CCW")

            axs[-1].set_xlabel('$k_z$ (m$^{-1}$)')

            for ax in axs:
                ax.vlines(params['m_c'], *ax.get_ylim())
                ax.vlines(params['m_0'], *ax.get_ylim())
                ax.grid()
                ax.legend()

    if params['plot_results']:

        fig, axs = plt.subplots(1, 5, sharey=True)

        axs[0].plot(np.log10(EK), z_mean, 'k-o')
        axs[0].set_xlabel('$\log_{10}E_{KE}$ (m$^{2}$ s$^{-2}$)')
        axs[0].set_ylabel('$z$ (m)')
        axs[1].plot(np.log10(R_pol), z_mean, 'k-o')
        axs[1].set_xlabel('$\log_{10}R_{pol}$ (-)')
        axs[1].set_xlim(-1, 1)
        axs[2].plot(np.log10(R_om), z_mean, 'k-o')
        axs[2].set_xlabel('$\log_{10}R_{\omega}$ (-)')
        axs[2].set_xlim(-1, 1)
        axs[3].plot(np.log10(epsilon), z_mean, 'k-o')
        axs[3].set_xlabel('$\log_{10}\epsilon$ (W kg$^{-1}$)')
        axs[4].plot(np.log10(kappa), z_mean, 'k-o')
        axs[4].set_xlabel('$\log_{10}\kappa$ (m$^{2}$ s$^{-1}$)')

        for ax in axs:
            ax.grid()
            ax.set_xticklabels(ax.get_xticks(), rotation='vertical')

    return z_mean, EK, R_pol, R_om, epsilon, kappa


def intermediate_profile(x, xhinge, delta):
    """Generate an intermediate profile of some quantity.
    Ferron et. al. 1998

    Returns the 'top down' and 'bottom up' profiles as well as the average
    of the two.

    """

    xf = np.flipud(x)

    xtd = np.zeros_like(x)
    xbu = np.zeros_like(x)

    ntd = np.fix(x[0]/delta - xhinge/delta)
    nbu = np.fix(xf[0]/delta - xhinge/delta)

    xtd[0] = xhinge + ntd*delta
    xbu[0] = xhinge + nbu*delta

    for i in range(len(x) - 1):
        ntd = np.fix(x[i+1]/delta - xtd[i]/delta)
        nbu = np.fix(xf[i+1]/delta - xbu[i]/delta)

        xtd[i+1] = xtd[i] + ntd*delta
        xbu[i+1] = xbu[i] + nbu*delta

    xbu = np.flipud(xbu)

    xav = (xtd + xbu)/2.

    return xtd, xbu, xav


def thorpe_scales(z, x, acc=1e-3, R0=0.25, full_output=False):
    """Thorpe et. al. 1977
    Code modified from Kurt Polzin I believe.
    Contains Garret and Garner 2008 validation ratio.
    Contains Smyth 2001 buoyancy frequency estimate.
    """

    #    flip_z = False
    flip_x = False

    # z should be increasing apparently
    if z[0] > z[-1]:
        z = np.flipud(z)
    #        flip_z = True

    # x should be increasing too...
    if x[0] > x[-1]:
        x = np.flipud(x)
        flip_x = True

    # Make sure that no overturns involve the first or last points.
    x[0] = x.min() - 1e-8
    x[-1] = x.max() + 1e-8

    # Sort the profile.
    idxs = np.argsort(x)
    x_sorted = x[idxs]

    # Calculate thorpe displacements.
    thorpe_disp = z[idxs] - z

    # Index displacements.
    idxs_disp = idxs - np.arange(len(idxs))

    # Overturn bounds where cumulative sum is zero.
    idxs_sum = np.cumsum(idxs_disp)

    # This plus 1 here seems to make the indexing work in python.
    jdxs = np.squeeze(np.argwhere(idxs_sum == 0)) + 1

    thorpe_scales = np.zeros_like(x)
    L_o = np.zeros_like(x)
    R = np.zeros_like(x)
    L_neg = np.zeros_like(x)
    L_pos = np.zeros_like(x)
    Nsq = np.zeros_like(x)

    # Calculate the RMS thorpe displacement over each overturn.
    for i in range(len(jdxs)-1):
        if jdxs[i+1] - jdxs[i] > 1:
            # Check for noise.
            q = x_sorted[jdxs[i+1]] - x_sorted[jdxs[i]]
            if q < acc:
                continue

            odxs = slice(jdxs[i], jdxs[i+1])
#            L_tot = z[jdxs[i+1]] - z[jdxs[i]]
            thorpe_disp_o = thorpe_disp[odxs]

#            zeroidx = np.searchsorted(thorpe_disp_o, 0., side='right')
#            L_neg_ = z[jdxs[i] + zeroidx] - z[jdxs[i]]
#            L_pos_ = z[jdxs[i+1]] - z[jdxs[i] + zeroidx]
#            R_ = np.minimum(L_neg_/L_tot, L_pos_/L_tot)

            L_tot = 1.*thorpe_disp_o.size
            L_neg_ = 1.*np.sum(thorpe_disp_o < 0)
            L_pos_ = 1.*np.sum(thorpe_disp_o > 0)
            R_ = np.minimum(L_neg_/L_tot, L_pos_/L_tot)
            if R_ < R0:
                continue

            L_neg[odxs] = L_neg_
            L_pos[odxs] = L_pos_
            L_o[odxs] = L_tot
            R[odxs] = R_

            zrms = np.std(thorpe_disp_o)
            thorpe_scales[odxs] = zrms

            # TODO: add a condition to allow this only when x is density.
            sigma_rms = np.std(x[odxs] - x_sorted[odxs])
            dsigmadz = sigma_rms/zrms
            sigma_0 = np.mean(x[odxs])
            g = -9.81  # Gravity.
            Nsq[odxs] = -g/sigma_0 * dsigmadz

            # This is a terrible bug fix but effective.
            if dsigmadz == 0.:
                thorpe_scales[odxs] = 0.

    # Lastly if the arrays were not increasing at the beginning and were
    # flipped they need to be put back how they were.
    if flip_x:
        thorpe_scales = np.flipud(thorpe_scales)
        thorpe_disp = np.flipud(thorpe_disp)
        x_sorted = np.flipud(x_sorted)
        idxs = np.flipud(idxs)
        L_o = np.flipud(L_o)
        L_neg = np.flipud(L_neg)
        L_pos = np.flipud(L_pos)
        R = np.flipud(R)
        Nsq = np.flipud(Nsq)

    if full_output:
        return thorpe_scales, thorpe_disp, Nsq, (L_o, L_neg, L_pos), R, x_sorted, idxs
    else:
        return thorpe_scales, thorpe_disp


def w_scales(w, x, N2, dx=1., width=10., overlap=-1., lc=30., c=1., eff=0.2,
             btype='highpass', we=1e-3, ret_noise=False, ret_w_filt=False):
    """
    Estimate turbulent kinetic energy dissipation from vertical velocity
    variance, known as the 'large eddy method'.

    Parameters
    ----------
    w : array
        Vertical velocity [m s-1]
    x : array
        Indexing variable such as height or time.
    N2 : array
        Buoyancy frequency squared, note angular units. [rad2 s-2]
    dx : float, optional
        Sample spacing, same units as x. Default is 1.
    width : float, optional
        Width of box over which to calculate variance (wider boxes use more
        measurements), same units as x. Default is 10.
    lc : float, optional
        High pass filter cutoff length, same units as x. Default is 30.
    c : float, optional
        Parameterisation coefficient that should be determined by comparison of
        results from large eddy method with independent measure of TKED.
        Default is 1.
    eff : float, optional
        Mixing efficiency, typically given a value of 0.2.
    btype : {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
        The type of filter. Default is ‘highpass’.
    we : float
        Random error in the vertical velocity [m s-1]
    ret_noise : boolean, optional
        Set as True to return additional noise information.
    ret_w_filt : boolean, optional
        Set as True to return just w_filt.

    Returns
    -------
    epsilon : array
        Turbulent kinetic energy dissipation (same length as input w). [W kg-1]
    kappa : array
        Diapycnal diffusivity. [m2 s-1]
    epsilon_noise : array, optional
        Value of the noise threshold of epsilon.
    noise_flag : array, optional
        Boolean flag, true if the epsilon value is smaller than the estimated
        noise threshold.

    References
    ----------
    Beaird et. al. 2012

    """

    # First we have to design the high pass filter the data. Beaird et. al.
    # 2012 use a forth order butterworth with a cutoff of 30m.
    xc = 1./lc  # cut off wavenumber
    normal_cutoff = xc*dx*2.  # Nyquist frequency is half 1/dx.
    b, a = sig.butter(4, normal_cutoff, btype=btype)

    # Filter the data.
    w_filt = sig.filtfilt(b, a, w)

    if ret_w_filt:
        return w_filt

    w_wdws = wdw.window(x, w_filt, width=width, overlap=overlap)
    N2_wdws = wdw.window(x, N2, width=width, overlap=overlap)

    w_rms = np.zeros_like(x)
    N2_mean = np.zeros_like(x)

    for i, (w_wdw, N2_wdw) in enumerate(zip(w_wdws, N2_wdws)):
        w_rms[i] = np.std(w_wdw[1])
        N2_mean[i] = np.mean(N2_wdw[1])

    epsilon = c*np.sqrt(N2_mean)*w_rms**2
    kappa = eff*epsilon/N2_mean

    if ret_noise:
        # True if epsilon value smaller than noise threshold.
        epsilon_noise = c*np.sqrt(N2_mean)*we**2
        noise_flag = epsilon < epsilon_noise
        return epsilon, kappa, epsilon_noise, noise_flag
    else:
        return epsilon, kappa


def VKE_method(z, w, width=320., overlap=160., c=0.021, m0=1., mc=0.1):
    """
    Estimate turbulent kinetic energy dissipation from large scale vertical
    kinetic energy (VKE).

    Parameters
    ----------
    z : array
        Height (negative depth), must be regularly spaced [m]
    w : array
        Vertical velocity [m s-1]
    width : float, optional
        Width of box over which to calculate VKE (wider boxes use more
        measurements) [m]
    overlap : float, optional
        Box overlap [m]
    c : float, optional
        Parameterisation coefficient that is quoted in reference [1] [s-0.5]
    m0 : float, optional
        Another parameterisation coefficient [rad m-1]
    mc : float, optional
        Cut off high wavernumber for fit [rad m-1]

    Returns
    -------
    z_mid : array
        Height at box middles [m]
    epsilon : array
        Turbulent kinetic energy dissipation (same length as input w). [W kg-1]


    References
    ----------
    [1] Thurnherr et. al. 2015

    """
    C = c*m0**2
    wdws = wdw.window(z, w, width, overlap, cap_left=True, cap_right=True)
    epsilon = []
    z_mid = []

    for i, (z_, w_) in enumerate(wdws):
        m, VKE = sig.periodogram(w_)
        # Convert to radian units.
        VKE /= 2*np.pi
        m *= 2*np.pi
        use = (m < mc) & (m != 0)
        VKE = VKE[use]
        m = m[use]
        B = np.polyfit(np.zeros(len(VKE)), np.log(VKE) + 2*np.log(m), 0)
        p0 = np.exp(B)
        eps = (p0/C)**2
        epsilon.append(eps)
        z_mid.append((z_[0] + z_[-1])/2.)

    return np.asarray(z_mid), np.asarray(epsilon)


def intermediate_profile1(x, hinge=1000, delta=1e-3, kind='bottom up'):
    """Generate an intermediate profile of some quantity. Ferron et. al. 1998.

    Parameters
    ----------
    x : 1D array
        Temperature or density.
    hinge : float, optional
        Hinge temperature or density.
    delta : float, optional
        Step,
    kind : string, optional
        Either 'bottom up', 'top down' or 'average'.

    Returns
    -------
    y : 1D array
        Reference buoyancy frequency [s-2]

    """
    xf = np.flipud(x)

    xtd = np.zeros_like(x)
    xbu = np.zeros_like(x)

    ntd = np.fix(x[0]/delta - hinge/delta)
    nbu = np.fix(xf[0]/delta - hinge/delta)

    xtd[0] = hinge + ntd*delta
    xbu[0] = hinge + nbu*delta

    for i in range(len(x) - 1):
        ntd = np.fix(x[i+1]/delta - xtd[i]/delta)
        nbu = np.fix(xf[i+1]/delta - xbu[i]/delta)

        xtd[i+1] = xtd[i] + ntd*delta
        xbu[i+1] = xbu[i] + nbu*delta

    xbu = np.flipud(xbu)

    xav = (xtd + xbu)/2.

    if 'up' in kind:
        return xbu
    elif 'down' in kind:
        return xtd
    elif 'av' in kind:
        return xav


def thorpe_scales1(z, x, acc=1e-3, R0=0.25, Nsq=None, full_output=False,
                   Nsq_method='bulk', use_int_prof=False, **ip_kwargs):
    """Estimate thorpe scales. Thorpe et. al. 1977
    Contains Gargett and Garner 2008 validation ratio.

    Parameters
    ----------
    z : 1D array
        Height. [m] (Negative depth!)
    x : 1D array
        Density. [kg m-3]
    acc : float, optional
        Accuracy of the x measurement.
    R0 : float, optional
        Validation ratio criteria, default 0.25.
    Nsq : 1D array, optional
        Buoyancy frequency squared. [rad2 s-2]
    full_output : boolean, optional
        Return all diagnostic variables. Also calculates N squared.
    Nsq_method : string, optional
        The method used to estimated buoyancy frequency. The options are
        'endpt' or 'bulk'. See Mater et. al. 2015 for a discussion.
    use_int_prof : boolean, optional
        Use the intermediate profile method of Ferron.
    ip_kwargs : dict, optional
        Keyword arguments for the intermediate profile method.

    Returns
    -------
    LT : 1D array
        Thorpe scales. [m]
    Td : 1D array
        Thorpe displacements. [m]
    Nsqo : 1D array, optional
        Buoyancy frequency of overturns. [rad2 s-2]
    Lo : 1D array, optional
        Overturn length. [m]
    R : 1D array, optional
        Overturn ratio.
    x_sorted : 1D array, optional
        Sorted density. [kg m-3]
    idxs : 1D array, optional
        Indexes required to sort.
    """
    g = -9.807  # Gravitational acceleration [m s-2]
    LT = np.zeros_like(x)
    Lo = np.zeros_like(x)
    R = np.zeros_like(x)
    Nsqo = np.zeros_like(x)

    # x should be increasing for this algorithm to work.
    flip_x = False
    if x[0] > x[-1]:
        x = np.flipud(x)
        z = np.flipud(z)
        if Nsq is not None:
            Nsq = np.flipud(Nsq)
        flip_x = True

    if use_int_prof:
        x = intermediate_profile(x, **ip_kwargs)

    # This is for estimating the length of the overturns.
    dz = 0.5*(z[:-2] - z[2:])
    dz = np.hstack((dz[0], dz, dz[-1]))

    # Make sure that no overturns involve the first or last points.
    x[0] = x.min() - 1e-4
    x[-1] = x.max() + 1e-4

    # Sort the profile.
    idxs = np.argsort(x)
    x_sorted = x[idxs]
    # Calculate thorpe displacements.
    Td = z[idxs] - z
    # Index displacements.
    idxs_disp = idxs - np.arange(len(idxs))
    # Overturn bounds where cumulative sum is zero.
    idxs_sum = np.cumsum(idxs_disp)
    # Find overturns.
    odxs_ = utils.contiguous_regions(idxs_sum > 0)

    if odxs_.size == 0:  # No oveturns at all!
        if full_output:
            return LT, Td, Nsqo, Lo, R, x_sorted, idxs
        else:
            return LT

    cut = (odxs_[:, 1] - odxs_[:, 0]) == 1
    if odxs_[0, 0] == 0:
        cut[0] = True

    odxs = odxs_[~cut, :]

    # Calculate the RMS thorpe displacement over each overturn.
    for j1, j2 in odxs:
        odx = slice(j1, j2)
        # Check for noise.
        q = x_sorted[j2] - x_sorted[j1]
        if q < acc:
            continue

        # Overturn ratio of Gargett & Garner
        Tdo = Td[odx]
        dzo = dz[odx]
        L_tot = np.sum(dzo)
        L_neg = np.sum(dzo[Tdo < 0])
        L_pos = np.sum(dzo[Tdo > 0])
        R_ = np.minimum(L_neg/L_tot, L_pos/L_tot)
        if R_ < R0:
            continue

        # Store data.
        Lo[odx] = L_tot
        R[odx] = R_
        LT_ = np.sqrt(np.mean(Tdo**2))
        LT[odx] = LT_
        if Nsq_method == 'endpt':
            Nsqo[odx] = -g*q/(np.mean(x_sorted[odx])*L_tot)
        elif Nsq_method == 'bulk':
            dx = x[odx] - x_sorted[odx]
            dxrms = np.sqrt(np.mean(dx**2))
            Nsqo[odx] = -g*dxrms/(np.mean(x_sorted[odx])*LT_)
        else:
            raise ValueError("Nsq_method can be either 'endpt' or 'bulk'.")

    # Lastly if the arrays were not increasing at the beginning and were
    # flipped they need to be put back how they were.
    if flip_x:
        LT = np.flipud(LT)
        Td = np.flipud(Td)
        x_sorted = np.flipud(x_sorted)
        idxs = np.flipud(idxs)
        Lo = np.flipud(Lo)
        R = np.flipud(R)
        Nsqo = np.flipud(Nsqo)

    if full_output:
        return LT, Td, Nsqo, Lo, R, x_sorted, idxs
    else:
        return LT
