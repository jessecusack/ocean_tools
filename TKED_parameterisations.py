# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:45:36 2014

A place for turbulent kinetic energy dissipation parameterisation functions.

@author: jc3e13
"""

import numpy as np
import gsw
import scipy.signal as sig
import matplotlib.pyplot as plt
import window as wdw
import GM


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
    for i in xrange(len(P)):

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
        epsilon[i] = GM.epsilon_0*N2_mean/GM.N_0**2*Ishear**2/IGMshear**2
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


def thorpe_scales(z, x):
    """ """

#    flip_z = False
    flip_x = False

    # z should be increasing apparently
    if z[0] > z[-1]:
        z = np.flipud(z)
#        flip_z = True

    # x should be increasing too... I'm personally not sure about this but ok.
    if x[0] > x[-1]:
        x = np.flipud(x)
        flip_x = True

    # Make sure that no overturns involve the first data point, why?
    x[0] = x.min() - 1.

    # Sort the profile.
    idxs = np.argsort(x)
    x_sorted = x[idxs]

    # Calculate thorpe displacements.
    thorpe_disp = z[idxs] - z

    # Indix displacements.
    idxs_disp = idxs - np.arange(0, len(idxs))

    # Overturn bounds where cumulative sum is zero.
    idxs_sum = np.cumsum(idxs_disp)

    jdxs = np.argwhere(idxs_sum == 0)

    thorpe_scales = np.zeros_like(x)

    # Calculate the RMS thorpe displacement over each overturn.
    for i in xrange(len(jdxs)-1):
        if jdxs[i+1] - jdxs[i] > 1:
            # Check for noise.
            q = x_sorted[jdxs[i+1]] - x_sorted[jdxs[i]]
            if q < 1e-3:
                print('Below measurement accuracy.')
                continue

            zrms = np.std(thorpe_disp[jdxs[i]:jdxs[i+1]])
            thorpe_scales[jdxs[i]:jdxs[i+1]] = zrms

    # Lastly if the arrays were not increasing at the beginning and were
    # flipped they need to be put back how they were.
    if flip_x:
        thorpe_scales = np.flipud(thorpe_scales)
        thorpe_disp = np.flipud(thorpe_disp)
        x_sorted = np.flipud(x_sorted)
        idxs = np.flipud(idxs)

    return thorpe_scales, thorpe_disp, x_sorted, idxs


def w_scales(w, x, N2, dx=1., width=10., lc=30., c=1., eff=0.2,
             btype='highpass', we=1e-3, ret_noise=False):
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

    w_wdws = wdw.window(x, w_filt, width=width, overlap=-1.)
    N2_wdws = wdw.window(x, N2, width=width, overlap=-1.)

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
