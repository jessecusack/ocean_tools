# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 14:42:44 2014

@author: jc3e13

This module contains functions for working with the Garrett Munk internal wave
spectrum. The aim is to develop a toolbox similar to the one that Jody Klymak
wrote in Matlab, but so far it is just a messy collection of functions.

"""

import numpy as np
import scipy as sp
from scipy.special import gamma


# Default parameter values and sets.
N0 = 5.2e-3  # Buoyancy frequency [rad s-1].
b = 1300.  # e-folding scale of N with depth [m].
E0 = 6.3e-5  # Internal wave energy parameter.
f_30 = 7.3e-5  # Coriolis frequency at 30N [rad s-1].
epsilon_0 = 8e-10  # GM energy dissipation rate. Polzin 2014

# Garrett and Kunze 1991 set.
GK91 = {
    's': 1.,
    't': 2.,
    'jp': 0.,
    'jstar': 3.}

# Garrett and Munk 1976 set.
GM76 = {
    's': 2.,
    't': 2.,
    'jp': 0.,
    'jstar': 3.}

# Garrett and Munk 1975 set.
GK75 = {
    's': 1.,
    't': 2.5,
    'jp': 0.,
    'jstar': 6.}


class GM(object):
    """The GM class is a tool for diagnosing the Garrett-Munk internal wave
    field for a given value of buoyancy frequency N and Coriolis parameter f.
    It contains methods for estimating spectra (e.g. displacement or velocity)
    as a funciton of wavenumber and frequency.
    """
    def __init__(self, N, f, **kwargs):
        self.N = N
        self.f = np.abs(f)

        # The default parameter values are defined at the top of module.
        self.b = kwargs.pop('b', b)
        self.N0 = kwargs.pop('N0', N0)
        self.E0 = kwargs.pop('E0', E0)

        # Necessary parameters that vary between implimentations. Use Garrett
        # and Munk 1976 set by default.
        self.s = kwargs.pop('s', 2.)
        self.t = kwargs.pop('t', 2.)
        self.jp = kwargs.pop('jp', 0.)
        self.jstar = kwargs.pop('jstar', 3.)

        self.eps = self.f/self.N

    def _B(self, om):
        """The frequency part of the GM spectrum."""
        return 2.*self.f/(np.pi*om*np.sqrt(om**2 - self.f**2))

    def _A(self, m, rolloff=False, Er=E0):
        """The vertical wavenumber part of the GM spectrum.
        m in cycles per metre!
        Set Er to a non-zero value to include high wavenumber roll off."""
        # TODO: impliment trimming low and high.
        # Roll off power, may need to be an argument.
        rop = -3

        # Normalisation factor, may need to be an argument.
        I = self.s*gamma(self.t/self.s) \
            / (gamma(1/self.s)*gamma((self.t-1)/self.s))

        delta = self.jp*self.N/(2.*self.N0*self.b)
        mstar = self.jstar*self.N/(2.*self.N0*self.b)

        A = (1/mstar)*I*(1 + ((m - delta)/mstar)**self.s)**(-self.t/self.s)

        # If this is true, then roll off to m**-3 above m > 0.1 cpm.
        # Why to the power -3? Not sure.
        if rolloff:
            # The magic 5 makes the critical wavenumber 0.1 cpm.
            mc = Er/(5*self.E0)
            r = (m/mc)**rop
            r[m < mc] = 1.
            A *= r

        return A

    def _neg_jstar(self, jstar, om):
        """Deals with negative jstar... not exactly sure about this."""
        j0 = 20.
        jinf = 10.
        om0 = self.f
        # What on earth are these numbers?
        ominf = 1.133*2.*np.pi/3600.
        omm = 0.173*2.*np.pi/3600.

        logs = 4.*(np.log10(om/self.f) - np.log10(omm/self.f)) \
            / np.log10(om0/ominf)
        tanh = np.tanh(logs)
        je = j0+0.5*(jinf - j0)*(1 - tanh)

        # What is this number?
        J = 2.1

        return je/J

    def disp(self, om, m=None):
        """Displacement."""
        return (self.b**2)*self.N0*(om**2 - self.f**2)/(self.N*om**2)

    def vel(self, om, m=None):
        """Velocity."""
        return (self.b**2)*self.N0*self.N*(om**2 + self.f**2)/om**2

#    def strain(self, om, m):
#        """Strain."""
#        return self.disp(om)*(2.*np.pi*m)**2
#
#    def shear(self, om, m):
#        """Shear."""
#        return self.vel(om)*(2.*np.pi*m)**2

    def Skm(self, k, m, Stype, rolloff=False, Er=E0):
        """Garrett-Munk spectrum as a function of horizontal wavenumber and
        vertical wavenumber.

        Parameters
        ----------
        k: array
            Horizontal wavenumber values. [cpm]
        m: array
            Vertical wavenumber values. [cpm]
        Stype: string
            Select between ['disp', 'vel', 'strain', 'shear']. The last two are
            not working yet.
        rolloff: boolean
            If True, apply a rolloff after critical vertical wavenumber.
            Default is False.
        Er: float
            Dimensionless energy of the internal wave field.

        Returns
        -------
        S : array
            Spectrum of size (len(m), len(k)).

        """
        # TODO: make this an imput parameter.
        Nz = 200

        Nk = len(k)
        Nm = len(m)

        S = np.zeros((Nm, Nk))

        # Choose the spectral function that gives dimensionality.
        Sfunc = getattr(self, Stype)

        Z = np.tile(np.linspace(0., 1., Nz), (Nm, 1))
        M = np.tile(m, (Nz, 1)).T
        A = self._A(M, rolloff, Er)

        for i, _k in enumerate(k):

            # We use the scipy sqrt function here because it gives imaginary
            # results for negative numbers, rather than NaN. I really have no
            # idea what Zmax is supposed to represent.
            Zmax = Z*sp.sqrt(M**2/_k**2 - 1).real
            omsq = _k**2/M**2*(Zmax**2+1)*(self.N**2-self.f**2) + self.f**2
            om = np.sqrt(omsq)

            B = self._B(om)

            # dom/da
            domda = _k*np.sqrt(Z**2+1)*(self.N**2-self.f**2)/(om*M**2)

            # The displacement factor, gives the spectrum a distance unit.
            R = Sfunc(om, M)

            # This needs to be all the right way around. Awkward.
            dz = Zmax[:, 1] - Zmax[:, 0]
            dZ = np.tile(dz, (Nz, 1)).T

            # Tda cancels stuff, so just do that here and save some time...
            Tda = dZ/sp.sqrt(Zmax**2+1)

            # I think all this is just to scale TT so that when integrating,
            # the trapz function does the right thing. Could simply pass x
            # values to trapz? Wouldn't that be better?
            TT = B*R*A*Tda*domda

            S[:, i] = np.trapz(TT)

        # Some more constants.
        S *= 2.*self.E0/np.pi

        return S

    def Sk(self, k, Stype, Nm=100, rolloff=False, Er=E0):
        """Garrett-Munk spectrum as a function of horizontal wavenumber.

        Parameters
        ----------
        k: array
            Horizontal wavenumber values. [cpm]
        Stype: string
            Select between ['disp', 'vel', 'strain', 'shear']. The last two are
            not working yet.
        rolloff: boolean
            If True, apply a rolloff after critical vertical wavenumber.
            Default is False.
        Er: float
            Dimensionless energy of the internal wave field.

        Returns
        -------
        S : array
            Spectrum of size (len(k),).

        """
        m = np.logspace(-4, 1, Nm)
        S = self.Skm(k, m, Stype, rolloff, Er)
        return np.trapz(S, m, axis=0)



def H(j, j_star=3., N_sum=100000):

    # The number over which to sum if j_star is not 3.
    if j_star == 3.:

        # The factor 0.468043 comes from summing denominator from j = 1 to
        # j = 1e+8 using j_star = 3.
        return (j**2 + j_star**2)**(-1)/0.468043

    else:

        j_sum = np.arrange(1, N_sum)
        return (j**2 + j_star**2)**(-1)/np.sum((j_sum**2 + j_star**2)**(-1))


def B(om, f=f_30):
    """The frequency part of the GM spectrum."""
    return 2.*f/(np.pi*om*np.sqrt(om**2 + f**2))


def E(om, j):
    return B(om)*H(j)*E0


def F_disp(om, N, j, f=f_30):
    """Displacement spectra."""
    return b**2*N0*(om**2 - f**2)*E(om, j)/(N*om**2)


def F_vel(om, N, j, f=f_30):
    """Horizontal velocity spectra."""
    return b**2*N0*N*(om**2 + f**2)*E(om, j)/om**2


def F_eng(om, N, j):
    """Energy per unit mass spectra."""
    return b**2*N0*N*E(om, j)


def F_str(om, N, j, f=f_30):
    pass


def F_she(om, N, j, f=f_30):
    pass


# case upper('Str')
#  R = (2*pi*kz).^2*(b.^2*N0/N.*(om.^2-f.^2)./om.^2);
# case upper('She')
#  R = (2*pi*kz).^2*(b.^2*N0*N*(om.^2+f.^2)./om.^2);


#def m(om, N, j):
#    """Convert from frequency space to vertical wavenumber space."""
#    return (np.pi/b)*np.sqrt((N**2 - om**2)/(N0**2 - om**2))*j
#
#
#def k(om, N, j, f=f_30):
#    """Convert from frequency space to horizontal wavenumber space."""
#    return (np.pi/b)*np.sqrt((om**2 - f**2)/(N0**2 - om**2))*j
#
#
#def Emk(k, m, E_star=E0, N=N0, f=f_30, m_star=3*np.pi/b):
#    """The GM spectra in k and m space as defined in Cushman-Roisin."""
#    num = 3*f*N*E_star*m/m_star
#    den = np.pi*(1 + m/m_star)**(2.5) * (N**2 * k**2 + f**2 * m**2)
#    return num/den


def beta_star(N, j_star=3.):
    return np.pi*j_star*N/(b*N0)


def E_vel_z(m, N, j_star=3.):
    """Horizontal velocity spectra as a function of vertical wavenumber. """
    return 3*E0*b**3*N0**2/(2*j_star*np.pi*(1 + m/beta_star(N, j_star))**2)


def E_she_z(m, N, j_star=3.):
    """Vertical shear of horizontal velocity as a function of vertical
    wavenumber. To normalise by N, divide return by N."""
    return m**2 * E_vel_z(m, N, j_star)/N


def E_disp_z(m, N, j_star=3.):
    """Vertical displacement as a function of vertical wavenumber."""
    num = E0*b**3*N0**2
    den = 2*j_star*np.pi*N**2 * (1 + m/beta_star(N, j_star))**2
    return num/den


def E_str_z(m, N, j_star=3.):
    """Vertical strain as a function of vertical wavenumber."""
    return m**2 * E_disp_z(m, N, j_star)


def E_str_omk(om, k, f, N, j_star=3, rolloff=True, Er=E0):
    """Horizontal strain as a function of frequency and horizontal wavenumber.
    Kunze et. al. 2015 Appendix
    """
    A = (om**2 + f**2)/om**5
    B = k**2/(k*N0*b + np.pi*np.sqrt(om**2 - f**2)*j_star)**2
    S = np.pi*E0*N*(N0**2)*f*(b**3)*j_star*A*B

    if rolloff:
        m = k*N/np.sqrt(om**2 - f**2)
        mc = np.pi*Er/(5.*E0)
        r = mc/m
        r[m < mc] = 1.
        S *= r

    return S


def E_str_k(k, f, N, j_star=3, rolloff=True, Er=E0):
    """Horizontal strain as a function horizontal wavenumber. It is equal to
    the function E_str_omk integrated between f and N.
    Kunze et. al. 2015 Appendix
    """
    eps = 0.0001
    om = np.logspace((1.-eps)*np.log10(f), (1.+eps)*np.log10(N), 1000)
    omg, kg = np.meshgrid(om, k)
    S = E_str_omk(omg, kg, f, N, j_star=j_star, rolloff=rolloff, Er=Er)

    return np.trapz(S, om, axis=1)


def disp_om(om, f, N):
    return b**2*N0*(om**2 - f**2)/(N*om**2)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    print('Running tests\n')

    f = 1.2e-4
    N = 1.7e-3
    k = np.logspace(-6, 0, 200)
    om = np.logspace(np.log10(f), np.log10(N), 150)
    omg, kg = np.meshgrid(om, k)

    mc = np.pi/5.
    kc = mc*np.sqrt((om**2 - f**2)/(N**2 - om**2))/(2.*np.pi)

    Somk = E_str_omk(omg, 2.*np.pi*kg, f, N, True)
    Sk = E_str_k(2.*np.pi*k, f, N, True)
    kmax = kg[np.unravel_index(Somk.argmax(), Somk.shape)]

    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios':[1, 2]})
    c = axs[1].contourf(1000.*k, 1800.*om/np.pi, 2.*np.pi*Somk.T,
                        cmap=plt.get_cmap('afmhot'))
    axs[1].plot(1000.*kc, 1800.*om/np.pi, color='b')
    axs[1].vlines(1000.*kmax, *axs[1].get_ylim(), color='b')
    axs[1].set_xlim(np.min(1000.*k), np.max(1000.*k))
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    plt.colorbar(c, orientation='horizontal')

    axs[1].set_ylabel('Frequency (cph)')
    axs[1].set_xlabel('Horizontal wavenumber $k$ (cpkm)')

    axs[0].loglog(1000.*k, 2.*np.pi*Sk)
    axs[0].set_ylabel('Horizontal strain variance (')
