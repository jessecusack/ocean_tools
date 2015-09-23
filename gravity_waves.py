# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 16:31:11 2014

@author: jc3e13

This module contains functions for investigating internal gravity waves.

"""

import numpy as np


def omega(N, k, m, l=0., f=0.):
    """Dispersion relation for an internal gravity wave in a continuously
    stratified fluid.

    (Gill 1980)

    Parameters
    ----------
    N : ndarray
        Buoyancy frequency [s-1]
    k : ndarray
        Horizontal wavenumber (x) [m-1]
    m : ndarray
        Vertical wavenumber (z) [m-1]
    l : ndarray, optional
        Horizontal wavenumber (y) [m-1]
    f : ndarray, optional
        Coriolis parameter [s-1]

    Returns
    -------
    omega : ndarray
        Frequency [s-1]

    """

    N2 = N**2
    k2 = k**2
    m2 = m**2
    l2 = l**2
    f2 = f**2

    return np.sqrt((f2*m2 + N2*(k2 + l2))/(k2 + l2 + m2))


def U_0(phi_0, k, l, om, f):
    """Zonal velocity amplitude."""
    return ((k*om + 1j*l*f)/(om**2 - f**2))*phi_0


def V_0(phi_0, k, l, om, f):
    """Meridional velocity amplitude."""
    return ((l*om - 1j*k*f)/(om**2 - f**2))*phi_0


def W_0(phi_0, m, om, N):
    """Vertical velocity amplitude."""
    return (-om*m/(N**2 - om**2))*phi_0


def B_0(phi_0, m, om, N):
    """Buoyancy perturbation amplitude."""
    return (1j*m*N**2/(N**2 - om**2))*phi_0


def ETA_0(phi_0, m, om, N):
    """Isopycnal displacement amplitude. """
    return phi_0*1j*m/(N**2 - om**2)


def wave_phase(x, y, z, t, k, l, m, om, U=0., V=0., W=0., phase_0=0.):
    """Phase of complex exponential equal to:
         k * x - (om + k * U) t + phase_0
       where k is the wavevector, x the position vector, om the frequency, t is
       time, U is the mean flow vector (k * U is the doppler factor) and
       phase_0 is an arbitrary phase offset.
    """
    return 1j*(k*x + l*y + m*z - (om + k*U + l*V + m*W)*t + phase_0)


def phi(x, y, z, t, phi_0, k, l, m, om, U=0., V=0., W=0., phase_0=0.):
    """Pressure pertubation."""
    phase = wave_phase(x, y, z, t, k, l, m, om, U=U, V=V, W=W, phase_0=phase_0)
    return np.real(phi_0*np.exp(phase))


def u(x, y, z, t, phi_0, k, l, m, om, f=0., U=0., V=0., W=0., phase_0=0.):
    """Zonal velocity pertubation."""
    amplitude = U_0(phi_0, k, l, om, f)
    phase = wave_phase(x, y, z, t, k, l, m, om, U=U, V=V, W=W, phase_0=phase_0)
    return np.real(amplitude*np.exp(phase))


def v(x, y, z, t, phi_0, k, l, m, om, f=0., U=0., V=0., W=0., phase_0=0.):
    """Meridional velocity pertubation."""
    amplitude = V_0(phi_0, k, l, om, f)
    phase = wave_phase(x, y, z, t, k, l, m, om, U=U, V=V, W=W, phase_0=phase_0)
    return np.real(amplitude*np.exp(phase))


def w(x, y, z, t, phi_0, k, l, m, om, N, U=0., V=0., W=0., phase_0=0.):
    """Vertical velocity pertubation."""
    amplitude = W_0(phi_0, m, om, N)
    phase = wave_phase(x, y, z, t, k, l, m, om, U=U, V=V, W=W, phase_0=phase_0)
    return np.real(amplitude*np.exp(phase))


def b(x, y, z, t, phi_0, k, l, m, om, N, U=0., V=0., W=0., phase_0=0.):
    """Buoyancy pertubation."""
    amplitude = B_0(phi_0, m, om, N)
    phase = wave_phase(x, y, z, t, k, l, m, om, U=U, V=V, W=W, phase_0=phase_0)
    return np.real(amplitude*np.exp(phase))


def eta(x, y, z, t, phi_0, k, l, m, om, N, U=0., V=0., W=0., phase_0=0.):
    """Vertical displacement of isopycnals."""
    amplitude = ETA_0(phi_0, m, om, N)
    phase = wave_phase(x, y, z, t, k, l, m, om, U=U, V=V, W=W, phase_0=phase_0)
    return np.real(amplitude*np.exp(phase))


def wave_vel(r, t, phi_0, U, k, l, m, om, N, f, phase_0=0.):
    """Wave velocity, accepts position stack and returns velocity stack."""

    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]

    u_amp = U_0(phi_0, k, l, om, f)
    v_amp = V_0(phi_0, k, l, om, f)
    w_amp = W_0(phi_0, m, om, N)

    phase = wave_phase(x, y, z, t, k, l, m, om, U=U, phase_0=phase_0)

    u = np.real(u_amp*np.exp(phase))
    v = np.real(v_amp*np.exp(phase))
    w = np.real(w_amp*np.exp(phase))

    return (np.vstack((u, v, w))).T


def buoy(r, t, phi_0, U, k, l, m, om, N, f, phase_0=0.):
    """Wave buoyancy, accepts position stack and returns buoyancy array."""
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]

    b_amp = B_0(phi_0, m, om, N)

    phase = wave_phase(x, y, z, t, k, l, m, om, U=U, phase_0=phase_0)

    return np.real(b_amp*np.exp(phase))


def cgz(k, m, N, l=0., f=0.):
    """Vertical component of group velocity."""
    num = -m*(N**2 - f**2)*(k**2 + l**2)
    den = (k**2 + l**2 + m**2)**1.5 * (f**2*m**2 + N**2*(k**2 + l**2))**0.5
    return num/den


def phip(k, m, l=0.):
    """Angle between the wavevector and the horizontal."""
    return np.arcsin(np.sqrt(m**2/(k**2 + l**2 + m**2)))


def alpha(k, m, l=0.):
    """Ratio of vertical to horizontal wavenumber."""
    return np.sqrt((k**2 + l**2)/m**2)


def Edens(w_0, k, m, l=0., rho_0=1025.):
    """Energy density."""
    phi = phip(k, m, l=l)
    return 0.5*rho_0*(w_0/np.cos(phi))**2


def Efluxz(w_0, k, m, N, l=0., f=0., rho_0=1025.):
    """Vertical energy flux in frame of reference moving with the wave."""
    return Edens(w_0, k, m, l=l, rho_0=rho_0)*cgz(k, m, N, l=l, f=f)


def Mfluxz(phi_0, k, l, m, om, N, f=0., rho_0=1025.):
    """Absolute vertical flux of horizontal momentum."""
    u_amp = np.abs(U_0(phi_0, k, l, om, f))
    v_amp = np.abs(V_0(phi_0, k, l, om, f))
    w_amp = np.abs(W_0(phi_0, m, om, N))
    return rho_0*np.sqrt(((u_amp*w_amp)**2 + (v_amp*w_amp)**2))
