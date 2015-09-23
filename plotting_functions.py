# -*- coding: utf-8 -*-
"""
Created on Tue Apr 08 12:17:53 2014

@author: jc3e13
"""

import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.basemap as bm
import sandwell
import utils
import scipy.signal as sig
from glob import glob
import scipy.io as io
import os


def dist_section(Float, hpids, var, plot_func=plt.contourf):
    """ """
    z_vals = np.arange(-1400., -100., 10.)
    __, idxs = Float.get_profiles(hpids, ret_idxs=True)
    dists = Float.dist[idxs]
    __, __, var_grid = Float.get_interp_grid(hpids, z_vals, 'z', var)
    plt.figure()
    plot_func(dists, z_vals, var_grid)


def scatter_section(Float, hpids, var, x_var='dist', cmap=plt.get_cmap('jet')):

    z_vals = np.arange(-1500., -40., 20.)
    __, z, var = Float.get_interp_grid(hpids, z_vals, 'z', var)

    if x_var == 'dist':

        __, __, d = Float.get_interp_grid(hpids, z_vals, 'z', 'dist_ctd')
        d = d.flatten(order='F')

    elif x_var == 'time':

        __, __, d = Float.get_interp_grid(hpids, z_vals, 'z', 'UTC')
        d = d.flatten(order='F')
        d = utils.datenum_to_datetime(d)

    else:
        raise ValueError("Input x_var should be 'dist' or 'time'.")

    z = z.flatten(order='F')
    var = var.flatten(order='F')

    plt.figure()
    plt.scatter(d, z, c=var, edgecolor='none', cmap=cmap)
    plt.ylim(np.min(z), np.max(z))
    plt.xlim(np.min(d), np.max(d))
    plt.colorbar(orientation='horizontal', extend='both')


def depth_profile(Float, hpids, var, plot_func=plt.plot, hold='off',
                  dlim=[-10000., 0]):
    """ """
    profiles = Float.get_profiles(hpids)
    if np.iterable(profiles):
        for i, profile in enumerate(profiles):
            z = getattr(profile, 'z')
            x = profile.interp(z, 'z', var)
            in_lim = (z > dlim[0]) == (z < dlim[1])
            if hold == 'on':
                if i == 0:
                    plt.figure()
                plot_func(x[in_lim], z[in_lim])
            else:
                plt.figure()
                plot_func(x[in_lim], z[in_lim])

    else:
        z = getattr(profiles, 'z')
        x = profiles.interp(z, 'z', var)
        in_lim = (z > dlim[0]) == (z < dlim[1])
        plt.figure()
        plot_func(x[in_lim], z[in_lim])


def isosurface(Float, hpids, var_1_name, var_2_name, var_2_vals):
    """Plot the values of some property at constant surfaces of some other
    property.

    e.g. Depth of potential density surfaces.
    e.g. Temperature of potential density surfaces.
    """

    ig, __, var_1g = Float.get_interp_grid(hpids, var_2_vals,
                                           var_2_name, var_1_name)
    plt.figure()
    plt.plot(ig.T, var_1g.T)
    plt.legend(str(var_2_vals).strip('[]').split())


def timeseries(Float, hpids, var_name):
    """TODO: Docstring..."""

    t, var = Float.get_timeseries(hpids, var_name)
    plt.figure()
    plt.plot(t, var)


def track_on_bathy(Float, hpids, projection='cyl', bathy_file=None):
    """TODO: Docstring..."""

    __, idxs = Float.get_profiles(hpids, ret_idxs=True)
    lons = Float.lon_start[idxs]
    lats = Float.lat_start[idxs]

    llcrnrlon = np.floor(np.min(lons)) - 1.
    llcrnrlat = np.floor(np.min(lats)) - 1.
    urcrnrlon = np.ceil(np.max(lons)) + 1.
    urcrnrlat = np.ceil(np.max(lats)) + 1.

    lon_lat = np.array([llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat])

    lon_grid, lat_grid, bathy_grid = sandwell.read_grid(lon_lat, bathy_file)
    bathy_grid[bathy_grid > 0] = 0

    m = bm.Basemap(projection=projection, llcrnrlon=llcrnrlon,
                   llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
                   urcrnrlat=urcrnrlat, lon_0=0.5*(llcrnrlon+urcrnrlon),
                   lat_0=0.5*(llcrnrlat+urcrnrlat), resolution='f')

    plt.figure()
    x, y = m(lon_grid, lat_grid)
    m.pcolormesh(x, y, bathy_grid, cmap=plt.get_cmap('binary_r'))
    x, y = m(lons, lats)
    m.plot(x, y, 'r-', linewidth=2)

    m.fillcontinents()
    m.drawcoastlines()

    r = np.abs((urcrnrlon-llcrnrlon)/(urcrnrlat-llcrnrlat))

    if r > 1.:
        Nm = 8
        Np = max(3, np.round(Nm/r))
        orientation = 'horizontal'
    elif r < 1.:
        Np = 8
        Nm = max(3, np.round(Nm/r))
        orientation = 'vertical'
    else:
        Np = 4
        Nm = 4
        orientation = 'horizontal'

    parallels = np.round(np.linspace(llcrnrlat, urcrnrlat, Np), 1)
    m.drawparallels(parallels, labels=[1, 0, 0, 0])
    meridians = np.round(np.linspace(llcrnrlon, urcrnrlon, Nm), 1)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1])

    cbar = plt.colorbar(orientation=orientation)
    cbar.set_label('Depth (m)')


def bathy_along_track(Float, hpids, bathy_file=None):
    """TODO: Docstring..."""

    __, idxs = Float.get_profiles(hpids, ret_idxs=True)
    lons = Float.lon_start[idxs]
    lats = Float.lat_start[idxs]
    dist = Float.dist[idxs]
    bathy = sandwell.interp_track(lons, lats, bathy_file)

    plt.figure()
    plt.plot(dist, bathy)
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (m)')


# Source of hinton: http://wiki.scipy.org/Cookbook/Matplotlib/HintonDiagrams
def _blob(x, y, area, colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2.
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    plt.fill(xcorners, ycorners, colour, edgecolor=colour)


def hinton(W, maxWeight=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    Temporarily disables matplotlib interactive mode if it is on,
    otherwise this takes forever.
    """
    reenable = False
    if plt.isinteractive():
        plt.ioff()
    plt.clf()
    height, width = W.shape
    if not maxWeight:
        maxWeight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))

    plt.fill(np.array([0, width, width, 0]), np.array([0, 0, height, height]),
             'gray')
    plt.axis('off')
    plt.axis('equal')
    for x in xrange(width):
        for y in xrange(height):
            _x = x+1
            _y = y+1
            w = W[y, x]
            if w > 0:
                _blob(_x - 0.5, height - _y + 0.5,
                      min(1, w/maxWeight), 'white')
            elif w < 0:
                _blob(_x - 0.5, height - _y + 0.5,
                      min(1, -w/maxWeight), 'black')
    if reenable:
        plt.ion()
    plt.show()


def welch_psd(Float, hpids, var, tz='z', hold='off'):
    """Compute power spectral density of some variable using Welch method.
    Variables are first interpolated onto a regular grid in either time or
    depth which can be specified using the tz optional argument. A time
    interval of 25 seconds is used and a depth interval of 4m."""

    dz = 4.
    dt = 25./86400.

    if tz == 'z':
        df = dz
        ivar = 'z'
        m = 1.
    elif tz == 't':
        df = dt
        ivar = 'UTC'
        m = 86400.
    else:
        raise RuntimeWarning("tz should be 't' or 'z'.")

    profiles = Float.get_profiles(hpids)
    if np.iterable(profiles):
        for i, profile in enumerate(profiles):
            f = getattr(profile, ivar)
            nans = np.isnan(f)
            f = np.unique(f[~nans])
            f = np.arange(f[0], f[-1], df)

            x = profile.interp(f, ivar, var)

            if hold == 'on':
                if i == 0:
                    plt.figure()
                freq, Pxx = sig.welch(x, 1./(m*df))
                plt.loglog(freq, Pxx)
            else:
                plt.figure()
                freq, Pxx = sig.welch(x, 1./(m*df))
                plt.loglog(freq, Pxx)

    else:
        f = getattr(profiles, ivar)
        nans = np.isnan(f)
        f = np.unique(f[~nans])

        f = np.arange(f[0], f[-1], df)

        x = profiles.interp(f, ivar, var)
        plt.figure()
        freq, Pxx = sig.welch(x, 1./(m*df))
        plt.loglog(freq, Pxx)


def spew_track(Floats, dt=12., fstr='test'):
    """Given a sequence of floats this function plots a time-lapse of
    their tracks onto bathymetry

    If you want it to work for one float you must put it in a sequence.
    """

    t_mins, t_maxs = [], []
    for Float in Floats:
        t_mins.append(np.min(Float.UTC_start))
        t_maxs.append(np.max(Float.UTC_start))

    t_min = np.min(t_mins)
    t_max = np.max(t_maxs)
    ti = np.arange(t_min, t_max, dt/24.)
    lons = np.empty((ti.size, len(Floats)))
    lats = np.empty((ti.size, len(Floats)))

    for i, Float in enumerate(Floats):
        lon = Float.lon_start
        lat = Float.lat_start
        t = Float.UTC_start
        lons[:, i] = np.interp(ti, t, lon, left=np.nan, right=np.nan)
        lats[:, i] = np.interp(ti, t, lat, left=np.nan, right=np.nan)

    llcrnrlon = np.floor(np.nanmin(lons)) - 1.
    llcrnrlat = np.floor(np.nanmin(lats)) - 1.
    urcrnrlon = np.ceil(np.nanmax(lons)) + 1.
    urcrnrlat = np.ceil(np.nanmax(lats)) + 1.

    lon_lat = np.array([llcrnrlon, -30., -70, urcrnrlat])

    lon_grid, lat_grid, bathy_grid = sandwell.read_grid(lon_lat)
    bathy_grid[bathy_grid > 0] = 0

    m = bm.Basemap(projection='tmerc', llcrnrlon=llcrnrlon,
                   llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
                   urcrnrlat=urcrnrlat, lon_0=0.5*(llcrnrlon+urcrnrlon),
                   lat_0=0.5*(llcrnrlat+urcrnrlat), resolution='f')

    plt.figure(figsize=(10, 10))
    x, y = m(lon_grid, lat_grid)
    m.pcolormesh(x, y, bathy_grid, cmap=plt.get_cmap('binary_r'))
    m.fillcontinents()
    m.drawcoastlines()

    r = np.abs((urcrnrlon-llcrnrlon)/(urcrnrlat-llcrnrlat))

    if r > 1.:
        Nm = 8
        Np = max(3, np.round(Nm/r))

    elif r < 1.:
        Np = 8
        Nm = max(3, np.round(Nm/r))

    parallels = np.round(np.linspace(llcrnrlat, urcrnrlat, Np), 1)
    m.drawparallels(parallels, labels=[1, 0, 0, 0])
    meridians = np.round(np.linspace(llcrnrlon, urcrnrlon, Nm), 1)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1])

    colours = ['b', 'r', 'm', 'c', 'g', 'y']

    for i, (time, lonr, latr) in enumerate(zip(ti, lons, lats)):

        print('Creating... {i:03d}'.format(i=i))
        plt.title(utils.datenum_to_datetime(time).strftime('%Y-%m-%d %H:%M'))

        for j, (lon, lat) in enumerate(zip(lonr, latr)):
            x, y = m(lon, lat)
            m.plot(x, y, '.', color=colours[j])

        if i == 0:
            leg_str = [str(Float.floatID) for Float in Floats]
            plt.legend(leg_str, loc=0)

        save_name = '../figures/animated_tracks/{}{i:03d}.png'.format(fstr,
                                                                      i=i)
        plt.savefig(save_name, bbox_inches='tight')

    print('Finished.')


def deployments():
    """This function gets data on all float deployments and plots them up."""

    files = glob('../../data/EM-APEX/allprof*.mat')
    FIDs = np.array([])  # Stores all float IDs.
    LONs = np.array([])
    LATs = np.array([])
    DATEs = np.array([])
    var_keys = ['flid', 'lon_gps', 'lat_gps', 'utc_dep']

    for f in files:

        data = io.loadmat(f, squeeze_me=True, variable_names=var_keys)
        fids = data['flid']
        dates = data['utc_dep']
        lons = data['lon_gps']
        lats = data['lat_gps']

        ufids = np.unique(fids[~np.isnan(fids)])

        FIDs = np.hstack((FIDs, ufids))

        for fid in ufids:
            flons = lons[fids == fid]
            flats = lats[fids == fid]
            fdates = dates[fids == fid]
            # Find the index of the
            idx = list(lons).index(filter(lambda x: ~np.isnan(x), lons)[0])
            LONs = np.hstack((LONs, flons[idx]))
            LATs = np.hstack((LATs, flats[idx]))
            DATEs = np.hstack((DATEs, fdates[idx]))

#    DTs = utils.datenum_to_datetime(DATEs)

    # Plot map.

    llcrnrlon = np.floor(np.nanmin(LONs)) - 8.
    llcrnrlat = np.floor(np.nanmin(LATs)) - 1.
    urcrnrlon = np.ceil(np.nanmax(LONs)) + 1.
    urcrnrlat = np.ceil(np.nanmax(LATs)) + 8.

    lon_lat = np.array([llcrnrlon, -20., -80, urcrnrlat])

    lon_grid, lat_grid, bathy_grid = sandwell.read_grid(lon_lat)
    bathy_grid[bathy_grid > 0] = 0

    m = bm.Basemap(projection='tmerc', llcrnrlon=llcrnrlon,
                   llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
                   urcrnrlat=urcrnrlat, lon_0=0.5*(llcrnrlon+urcrnrlon),
                   lat_0=0.5*(llcrnrlat+urcrnrlat), resolution='f')

    plt.figure(figsize=(10, 10))
    x, y = m(lon_grid, lat_grid)
    m.pcolormesh(x, y, bathy_grid, cmap=plt.get_cmap('binary_r'))
    m.fillcontinents()
    m.drawcoastlines()

    parallels = np.array([-64, -54])
    m.drawparallels(parallels, labels=[1, 0, 0, 0])
    meridians = np.array([-105, -95, -85, -75, -65, -55, -45])
    m.drawmeridians(meridians, labels=[0, 0, 0, 1])

    x, y = m(LONs, LATs)
    m.scatter(x, y, s=60, c=DATEs, cmap=plt.get_cmap('spring'))

    save_name = '../figures/deployments.png'
    plt.savefig(save_name, bbox_inches='tight')


def plot_everything(Float, save_dir):
    """Plot profiles of absolutely everything possible."""

    pass


def my_savefig(fig, fid, fname, sdir, fsize=None, lock_aspect=True,
               ftype='png', font_size=None):
    """My modified version of savefig."""

    scol = 3.125  # inches
    dcol = 6.5  # inches

    if fsize is None:
        pass
    elif isinstance(fsize, tuple):
        fig.set_size_inches(fsize)
    elif fsize == 'single_col':
        cfsize = fig.get_size_inches()
        if lock_aspect:
            r = scol/cfsize[0]
            fig.set_size_inches(cfsize*r)
        else:
            fig.set_size_inches(scol, cfsize[1])
    elif fsize == 'double_col':
        cfsize = fig.get_size_inches()
        if lock_aspect:
            r = dcol/cfsize[0]
            fig.set_size_inches(cfsize*r)
        else:
            fig.set_size_inches(dcol, cfsize[1])

    if font_size is None:
        pass
    else:
        axs = fig.get_axes()
        for ax in axs:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(font_size)
        fig.canvas.draw()

    fname = str(fid) + '_' + fname
    fname = os.path.join(sdir, fname) + '.' + ftype
    plt.savefig(fname, dpi=300., bbox_inches='tight')


def step(x, y, **kwargs):
    """Because I think that the matplotlib step function is crappy at ends. """
    dx = np.diff(x)
    dxl = np.hstack((dx[-1, ...], dx))/2.
    dxr = np.hstack((dx, dx[0, ...]))/2.
    shape = np.array(x.shape)
    shape[0] *= 2
    xp = np.empty(shape)
    yp = np.empty(shape)
    xp[::2, ...] = x - dxl
    xp[1::2, ...] = x + dxr
    yp[::2, ...] = y
    yp[1::2, ...] = y
    return plt.plot(xp, yp, **kwargs)
