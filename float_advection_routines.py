# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 14:27:01 2014

Advection routines.

@author: jc3e13
"""


import numpy as np
from scipy.integrate import odeint
import sys

lib_path = '/noc/users/jc3e13/emapex/python'
pymc3_path = '/noc/users/jc3e13/envs/my_root/lib/python2.7/site-packages/pymc-3.0-py2.7.egg'
if lib_path not in sys.path:
    sys.path.append(lib_path)

# We want pymc 2.3
if pymc3_path in sys.path:
    sys.path.remove(pymc3_path)

import gravity_waves as gw
import gsw
import utils


def U_const(z):
    """Constant velocity of 50 cm s-1."""
    return 0.5


def U_shear(z):
    """Default shear of 20 cm s-1 over 1500 m."""
    return 1.3333e-4*z + 5e-1


default_params = {
    'Ufunc': U_const,
    'f': gsw.f(-57.5),
    'N': 1.8e-3,
    'w_0': 0.17,
    'Wf_pvals': np.polyfit([0., 0.06], [0.14, 0.12], 1),
    'dt': 10.,
    't_1': 15000.,
    'x_0': 0.,
    'y_0': 0.,
    'z_0': -1500,
    'print': False
    }


def drdt(r, t, phi_0, Ufunc, Wf_pvals, k, l, m, om, N, f=0., phase_0=0.):
    x = r[0]
    y = r[1]
    z = r[2]

    Wf_g = Wf_pvals[0]
    Wf_0 = Wf_pvals[1]

    U = Ufunc(z)

    dxdt = U + gw.u(x, y, z, t, phi_0, k, l, m, om, f=f, U=U, phase_0=phase_0)
    dydt = gw.v(x, y, z, t, phi_0, k, l, m, om, f=f, U=U, phase_0=phase_0)
    dzdt = (Wf_0 + gw.w(x, y, z, t, phi_0, k, l, m, om, N, U=U, phase_0=phase_0))/(1 - Wf_g)

    return np.array([dxdt, dydt, dzdt])


def model_basic(phi_0, X, Y, Z, phase_0=0., oparams=default_params):

    Ufunc = oparams['Ufunc']
    f = oparams['f']
    N = oparams['N']

    # Float change in buoyancy with velocity.
    Wf_pvals = oparams['Wf_pvals']

    # Wave parameters
    k = 2*np.pi/X
    l = 2*np.pi/Y
    m = 2*np.pi/Z
    om = gw.omega(N, k, m, l, f)

    args = (phi_0, Ufunc, Wf_pvals, k, l, m, om, N, f, phase_0)

    # Integration parameters.
    dt = oparams['dt']
    t_0 = 0.
    t_1 = oparams['t_1']
    t = np.arange(t_0, t_1, dt)

    # Initial conditions.
    x_0 = oparams['x_0']
    y_0 = oparams['y_0']
    z_0 = oparams['z_0']
    r_0 = np.array([x_0, y_0, z_0])

    # This integrator calls FORTRAN odepack to solve the problem.
    r = odeint(drdt, r_0, t, args)

    uargs = (phi_0, Ufunc(r[:, 2]), k, l, m, om, N, f, phase_0)

    u = gw.wave_vel(r, t, *uargs)
    u[:, 0] += Ufunc(r[:, 2])
    b = gw.buoy(r, t, *uargs)

    return t, r, u, b


def model_leastsq(params, z, sub, var_name, oparams=default_params):
    """ Return individual variables given by var_name.
    Subtract data given by 'sub' to return residual.

    """
    phi_0, X, Y, Z, phase_0 = params
    __, r, u, b = model_basic(phi_0, X, Y, Z, phase_0, oparams)
    u[:, 0] -= oparams['Ufunc'](r[:, 2])

    # Variable to return.
    var_dict = {'w': u[:, 2], 'u': u[:, 0], 'v': u[:, 1], 'b': b}
    var = var_dict[var_name]
    ivar = np.interp(z, r[:, 2], var)

    return ivar - sub


def model_pymc(zf, phi_0, X, Y, Z, phase_0=0., bscale=250., oparams=default_params):
    """Return a stack of all velocity components and buoyancy."""

    __, r, u, b = model_basic(phi_0, X, Y, Z, phase_0, oparams)
    u[:, 0] -= oparams['Ufunc'](r[:, 2])

    um = np.interp(zf, r[:, 2], u[:, 0])
    vm = np.interp(zf, r[:, 2], u[:, 1])
    wm = np.interp(zf, r[:, 2], u[:, 2])
    bm = bscale*np.interp(zf, r[:, 2], b)

    return np.hstack((um, vm, wm, bm))


def model_verbose(phi_0, X, Y, Z, phase_0=0., oparams=default_params):
    """Return loads and loads of info."""

    t, r, u, b = model_basic(phi_0, X, Y, Z, phase_0, oparams)

    Ufunc = oparams['Ufunc']
    f = oparams['f']
    N = oparams['N']

    # Float change in buoyancy with velocity.
    Wf_pvals = oparams['Wf_pvals']

    # Wave parameters
    w_0 = oparams['w_0']
    k = 2*np.pi/X
    l = 2*np.pi/Y
    m = 2*np.pi/Z
    om = gw.omega(N, k, m, l, f)

    u_0 = gw.U_0(phi_0, k, l, om, f)
    v_0 = gw.V_0(phi_0, k, l, om, f)
    w_0 = gw.W_0(phi_0, m, om, N)
    b_0 = gw.B_0(phi_0, m, om, N)

    if oparams['print']:
        print("N = {:1.2E} rad s-1.\n"
              "om = {:1.2E} rad s-1.\n"
              "u_0 = {:1.2E} m s-1.\n"
              "v_0 = {:1.2E} m s-1.\n"
              "w_0 = {:1.2E} m s-1.\n"
              "phi_0 = {:1.2E} m2 s-2.\n"
              "b_0 = {:1.2E} m s-2.\n"
              "X = {:1.0f} m.\n"
              "k = {:1.2E} rad m-1.\n"
              "Y = {:1.0f} m.\n"
              "l = {:1.2E} rad m-1.\n"
              "Z = {:1.0f} m.\n"
              "m = {:1.2E} rad m-1.\n"
              "".format(N, om, u_0, v_0, w_0, phi_0, b_0, X, k, Y, l, Z, m))

    output = utils.Bunch(Ufunc=Ufunc,
                         U=Ufunc(r[:, 2]),
                         f=f,
                         N=N,
                         Wf_pvals=Wf_pvals,
                         w_0=w_0,
                         k=k,
                         l=l,
                         m=m,
                         om=om,
                         phi_0=phi_0,
                         u_0=u_0,
                         v_0=v_0,
                         b_0=b_0,
                         t=t,
                         z_0=oparams['z_0'],
                         r=r,
                         u=u,
                         b=b,
                         oparams=oparams)

    return output
