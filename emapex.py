# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:49:23 2013

@author: Jesse

Contains functions and classes for investigating and manipulating EM-APEX float
data.
"""

import numpy as np
import scipy.io as io
from scipy.interpolate import griddata
from scipy.integrate import trapz, cumtrapz
import gsw
import utils
import pickle
import copy
import glob
import os


class Profile(object):
    """Provide to this class its parent float and half profile number and it
    will extract its data.
    """
    def __init__(self, parent_float, hpid):

        self.hpid = hpid

        # Convert the parent float into a dictionary.
        data = vars(parent_float)
        self.floatID = data['floatID']

        self.update(parent_float)

#        print("Profile {} has been created.".format(hpid))

    def interp(self, var_2_vals, var_2_name, var_1_name):
        """Linear 1-D interpolation of variables stored by a Profile.

        Parameters
        ----------
        var_2_vals : 1-D numpy.ndarray of floats
            The values at which to return interpolant.
        var_2_name : string
            The variable to which var_2_vals correspond.
        var_1_name : string
            The variable to be interpolated.

        Returns
        -------
        var_1_vals : 1-D numpy.ndarray of floats
            The interpolated values of variable 1 at given positions in
            variable 2.

        Raises
        ------
        ValueError
            If array sizes cannot be matched with eachother, the ctd time or
            the ef time.
        RuntimeWarning
            If numpy.interp raises a value error which could be because an
            empty array was passed.
            TODO: Make this work properly without exiting program.

        Notes
        -----
        There may be issues with the orientation of the returned array
        because numpy.interp insists on increasing points a certain amount of
        sorting has to be done so that interpolation points are monotonically
        increasing.

        TODO: interp was recently changed in SciPy 0.14 and this sorting may
        no longer be necessary.

        Examples
        --------
        The following example returns interpolated temperatures every 10
        meters from -1000 m to -100 m depth.
        >>> import emapex
        >>> Float = emapex.EMApexFloat('allprofs11.mat',4977)
        >>> P = Float.get_profiles(100)
        >>> z_vals = np.arange(-1000., -100., 10)
        >>> T10 = P.interp(z_vals, 'z', 'T')
        """

        # Shorten some names.
        var_1 = getattr(self, var_1_name)
        var_2 = getattr(self, var_2_name)
        t = getattr(self, 'UTC')
        tef = getattr(self, 'UTCef')
        r_t = getattr(self, 'r_UTC', None)

        equal_size = False

        # If sizes are not equal, check for corresponding time arrays.
        if var_2.size == var_1.size:
            equal_size = True
        else:
            for time in [t, tef, r_t]:
                if time is None:
                    continue
                elif time.size == var_1.size:
                    t_1 = time
                elif time.size == var_2.size:
                    t_2 = time

        # Find NaN values.
        nans_var_1 = np.isnan(var_1)
        nans_var_2 = np.isnan(var_2)

        try:
            if equal_size:
                # Both arrays are same length.
                nans = nans_var_1 | nans_var_2
                var_1, var_2 = var_1[~nans], var_2[~nans]
                var_2_sorted, idxs = np.unique(var_2, return_index=True)
                var_1_vals = np.interp(var_2_vals, var_2_sorted, var_1[idxs])

            elif not equal_size:
                nans_1 = nans_var_1 | np.isnan(t_1)
                nans_2 = nans_var_2 | np.isnan(t_2)
                var_1, t_1 = var_1[~nans_1], t_1[~nans_1]
                var_2, t_2 = var_2[~nans_2], t_2[~nans_2]
                # np.unique is necessary to make sure inputs to interp are
                # monotonically increasing!
                var_2_sorted, idxs = np.unique(var_2, return_index=True)
                t_2_interp = np.interp(var_2_vals, var_2_sorted, t_2[idxs])
                t_1_sorted, idxs = np.unique(t_1, return_index=True)
                var_1_vals = np.interp(t_2_interp, t_1_sorted, var_1[idxs])

            else:
                raise RuntimeError('Cannot match time array and/or variable'
                                   ' array sizes.')

        except ValueError:
            var_1_vals = np.NaN*np.zeros_like(var_2_vals)

        return var_1_vals

    def update(self, parent_float):
        """Checks for new attributes of parent float and adds them as
        attributes of itself.

        Potentially useful if additional processing is done after float
        initialisation and you want to add this as profile data.
        """

        if parent_float.floatID != self.floatID:
            raise RuntimeError('You are attempting to update this profile '
                               'with the wrong parent float.')

        data = vars(parent_float)

        indx = np.where(data['hpid'] == self.hpid)

        for key in data.keys():

            dat = data[key]
            d = np.ndim(dat)

            if d < 1 or d > 2 or '__' in key:
                continue
            elif d == 1:
                setattr(self, key, dat[indx])
            elif d == 2:
                setattr(self, key, np.squeeze(dat[:, indx]))


class EMApexFloat(object):
    """Initialise this class with the path to an 'allprofs##.mat' file where
    ## denotes the last two digits of the year, e.g. 11 for 2011. Also provide
    the ID number of the particular float to extract from the file.

    Notes
    -----

    Some variables have '__' in their name to stop them being transfered into
    profile data e.g. __ddist.

    Some variables have '_ca' in their names and this means that they are on a
    centered array which has length one less than the normal ctd array. These
    variables are not put on to a regular grid because they are normally also
    contained on a ctd array

    """
    def __init__(self, filepath, floatID, post_process=True, regrid=False):

        print("\nInitialising")
        print("------------\n")

        self.floatID = floatID
        print(
            "EM-APEX float: {}\n"
            "Loading data...".format(floatID)
        )

        # Loaded data is a dictionary.
        data = io.loadmat(filepath, squeeze_me=True)

        isFloat = data.pop('flid') == floatID
        del data['ar']
        self.hpid = data.pop('hpid')[isFloat]
#        flip_indx = up_down_indices(self.hpid, 'up')

        # Figure out some useful times.
        t_ctd = data['UTC'][:, isFloat]
        self.UTC_start = t_ctd[0, :]
        self.UTC_end = np.nanmax(t_ctd, axis=0)

        # Load the data!
        for key in data.keys():

            d = np.ndim(data[key])

            if d < 1 or d > 2 or '__' in key:
                print("* Skipping: {}.".format(key))
                continue
            elif d == 1:
                setattr(self, key, data[key][isFloat])
            elif d == 2:
                # This flips ascent data first, then adds it as an attribute.
#                setattr(self, key, flip_cols(data[key][:, isFloat], flip_indx))
                setattr(self, key, data[key][:, isFloat])
            else:
                print("* Don't know what to do with {}, skipping".format(key))

            print("  Loaded: {}.".format(key))

        print("All numerical data appears to have been loaded successfully.\n")

        print("Interpolated GPS positions to starts and ends of profiles.")
        # GPS interpolation to the start and end time of each half profile.
        idxs = ~np.isnan(self.lon_gps) & ~np.isnan(self.lat_gps)
        self.lon_start = np.interp(self.UTC_start, self.utc_gps[idxs],
                                   self.lon_gps[idxs])
        self.lat_start = np.interp(self.UTC_start, self.utc_gps[idxs],
                                   self.lat_gps[idxs])
        self.lon_end = np.interp(self.UTC_end, self.utc_gps[idxs],
                                 self.lon_gps[idxs])
        self.lat_end = np.interp(self.UTC_end, self.utc_gps[idxs],
                                 self.lat_gps[idxs])

        print("Calculating heights.")
        # Depth.
        self.z = gsw.z_from_p(self.P, self.lat_start)
        self.z_ca = gsw.z_from_p(self.P_ca, self.lat_start)
        self.zef = gsw.z_from_p(self.Pef, self.lat_start)

        print("Creating array of half profiles.\n")

        self.Profiles = np.array([Profile(self, h) for h in self.hpid])

        if post_process:
            self.post_process()

        if regrid:
            self.generate_regular_grids()

    def post_process(self):

        print("\nPost processing")
        print("---------------\n")
        print("Calculating distance along trajectory.")
        # Distance along track from first half profile.
        self.__ddist = utils.lldist(self.lon_start, self.lat_start)
        self.dist = np.hstack((0., np.cumsum(self.__ddist)))

        print("Interpolating distance to measurements.")
        # Distances, velocities and speeds of each half profile.
        self.profile_ddist = np.zeros_like(self.lon_start)
        self.profile_dt = np.zeros_like(self.lon_start)
        self.profile_bearing = np.zeros_like(self.lon_start)
        lons = np.zeros((len(self.lon_start), 2))
        lats = lons.copy()
        times = lons.copy()

        lons[:, 0], lons[:, 1] = self.lon_start, self.lon_end
        lats[:, 0], lats[:, 1] = self.lat_start, self.lat_end
        times[:, 0], times[:, 1] = self.UTC_start, self.UTC_end

        self.dist_ctd = self.UTC.copy()
        nans = np.isnan(self.dist_ctd)
        for i, (lon, lat, time) in enumerate(zip(lons, lats, times)):
            self.profile_ddist[i] = utils.lldist(lon, lat)
            # Convert time from days to seconds.
            self.profile_dt[i] = np.diff(time)*86400.

            d = np.array([self.dist[i], self.dist[i] + self.profile_ddist[i]])
            idxs = ~nans[:, i]
            self.dist_ctd[idxs, i] = np.interp(self.UTC[idxs, i], time, d)

        self.dist_ef = self.__regrid('ctd', 'ef', self.dist_ctd)

        print("Estimating bearings.")
        # Pythagorian approximation (?) of bearing.
        self.profile_bearing = np.arctan2(self.lon_end - self.lon_start,
                                          self.lat_end - self.lat_start)

        print("Calculating sub-surface velocity.")
        # Convert to m s-1 calculate meridional and zonal velocities.
        self.sub_surf_speed = self.profile_ddist*1000./self.profile_dt
        self.sub_surf_u = self.sub_surf_speed*np.sin(self.profile_bearing)
        self.sub_surf_v = self.sub_surf_speed*np.cos(self.profile_bearing)

        print("Interpolating missing velocity values.")
        # Fill missing U, V values using linear interpolation otherwise we
        # run into difficulties using cumtrapz next.
        self.U = self.__fill_missing(self.U)
        self.V = self.__fill_missing(self.V)

        # Absolute velocity
        self.calculate_absolute_velocity()

        print("Calculating thermodynamic variables.")
        # Derive some important thermodynamics variables.

        # Absolute salinity.
        self.SA = gsw.SA_from_SP(self.S, self.P, self.lon_start,
                                 self.lat_start)
        # Conservative temperature.
        self.CT = gsw.CT_from_t(self.SA, self.T, self.P)

        # Potential temperature with respect to 0 dbar.
        self.PT = gsw.pt_from_CT(self.SA, self.CT)

        # In-situ density.
        self.rho = gsw.rho(self.SA, self.CT, self.P)

        # Potential density with respect to 1000 dbar.
        self.rho_1 = gsw.pot_rho_t_exact(self.SA, self.T, self.P, p_ref=1000.)

        # Buoyancy frequency regridded onto ctd grid.
        N2_ca, __ = gsw.Nsquared(self.SA, self.CT, self.P, self.lat_start)
        self.N2 = self.__regrid('ctd_ca', 'ctd', N2_ca)

        print("Calculating float vertical velocity.")
        # Vertical velocity regridded onto ctd grid.
        dt = 86400.*np.diff(self.UTC, axis=0)  # [s]
        Wz_ca = np.diff(self.z, axis=0)/dt
        self.Wz = self.__regrid('ctd_ca', 'ctd', Wz_ca)

        print("Renaming Wp to Wpef.")
        # Vertical water velocity.
        self.Wpef = self.Wp.copy()
        del self.Wp

        print("Calculating shear.")
        # Shear calculations.
        dUdz_ca = np.diff(self.U, axis=0)/np.diff(self.zef, axis=0)
        dVdz_ca = np.diff(self.V, axis=0)/np.diff(self.zef, axis=0)
        self.dUdz = self.__regrid('ef_ca', 'ef', dUdz_ca)
        self.dVdz = self.__regrid('ef_ca', 'ef', dVdz_ca)

        print("Calculating Richardson number.")
        N2ef = self.__regrid('ctd', 'ef', self.N2)
        self.Ri = N2ef/(self.dUdz**2 + self.dVdz**2)

        print("Regridding piston position to ctd.\n")
        # Regrid piston position.
        self.ppos = self.__regrid('ctd_ca', 'ctd', self.ppos_ca)

        self.update_profiles()

    def calculate_absolute_velocity(self):

        print("Estimating absolute velocity and subsurface position.")
        # Absolute velocity defined as relative velocity plus mean velocity
        # minus depth integrated relative velocity. It attempts to use the
        # profile pair integral of velocity, but if it can't then it will use
        # the single profile approximation.
        self.U_abs = self.U.copy()
        self.V_abs = self.V.copy()
        self.x_ctd = np.NaN*self.dist_ctd.copy()
        self.y_ctd = np.NaN*self.dist_ctd.copy()
        self.x_ef = np.NaN*self.dist_ef.copy()
        self.y_ef = np.NaN*self.dist_ef.copy()

        didxs = up_down_indices(self.hpid, 'down')
        successful_pairs = []
        nan_pairs = []

        for idx in didxs:
            # Check that a down up profile pair exists, if not, then skip.
            if (self.hpid[idx] + 1) != self.hpid[idx+1]:
                print('  No pair, continue.')
                continue

            hpids = self.hpid[[idx, idx+1]]
            t, U = self.get_timeseries(hpids, 'U')
            __, V = self.get_timeseries(hpids, 'V')

            nans = np.isnan(U) | np.isnan(V) | np.isnan(t)

            # If all values are NaN then skip.
            if np.sum(nans) == nans.size:
                print("  hpid pair {}, {} all NaNs.".format(self.hpid[idx],
                      self.hpid[idx+1]))
                nan_pairs.append(idx)
                nan_pairs.append(idx+1)
                continue

            # A half profile pair is bounded by a box defined by the lon-lat
            # positions at its corners.
            lon1 = self.lon_start[idx]
            lon2 = self.lon_end[idx+1]
            lat1 = self.lat_start[idx]
            lat2 = self.lat_end[idx+1]

            fX = -1 if lon1 > lon2 else 1.
            fY = -1 if lat1 > lat2 else 1.

            # Convert to m.
            X = 1000.*fX*utils.lldist((lon1, lon2), (lat1, lat1))
            Y = 1000.*fY*utils.lldist((lon1, lon1), (lat1, lat2))

            # Convert to seconds, find time difference.
            ts = 86400.*t
            dt = np.nanmax(ts) - np.nanmin(ts)

            idxs = np.array([idx, idx+1])

            # I use the += here because U_abs is initialised as a copy of U.
            self.U_abs[:, idxs] += X/dt - trapz(U[~nans], ts[~nans], axis=0)/dt
            self.V_abs[:, idxs] += Y/dt - trapz(V[~nans], ts[~nans], axis=0)/dt

            # This bit is for working out the distances.
            U_abs = U + X/dt - trapz(U[~nans], ts[~nans], axis=0)/dt
            V_abs = V + Y/dt - trapz(V[~nans], ts[~nans], axis=0)/dt

            x = cumtrapz(U_abs[~nans], ts[~nans], initial=0.)
            y = cumtrapz(V_abs[~nans], ts[~nans], initial=0.)

            for jdx in idxs:
                self.x_ef[:, jdx] = utils.nan_interp(self.UTCef[:, jdx],
                                                     t[~nans], x)
                self.y_ef[:, jdx] = utils.nan_interp(self.UTCef[:, jdx],
                                                     t[~nans], y)

            successful_pairs.append(idx)
            successful_pairs.append(idx+1)
            print("  hpid pair {}, {}.".format(hpids[0], hpids[1]))

        self.x_ctd = self.__regrid('ef', 'ctd', self.x_ef)
        self.y_ctd = self.__regrid('ef', 'ctd', self.y_ef)

        failed_idxs = list(set(np.arange(len(self.hpid))) -
                           set(successful_pairs) -
                           set(nan_pairs))
        print("Absolute velocity calculation failed for the following hpids:\n"
              "{}".format(self.hpid[failed_idxs]))

    def calculate_pressure_perturbation(self):
        """Perturbation pressure divided by density.
        Assumes hydrostatic balance.
        See Kunze 2002.
        See Nash 2005."""

        self.Pprime = self.P.copy()

        for i in xrange(len(self.hpid)):

            nans = np.isnan(self.P[:, i])

            z = self.z[~nans, i]
            b = self.b[~nans, i]

            # z should be increasing.
            if z[0] > z[-1]:
                z = np.flipud(z)
                b = np.flipud(b)

                bi = cumtrapz(b, z, initial=0.)
                bii = cumtrapz(bi, z, initial=0.)

                Pprime = bi + (bii[0] - bii[-1])/(-z[0])

                self.Pprime[~nans, i] = np.flipud(Pprime)

            else:
                bi = cumtrapz(b, z, initial=0.)
                bii = cumtrapz(bi, z, initial=0.)

                self.Pprime[~nans, i] = bi + (bii[0] - bii[-1])/(-z[0])

    def generate_regular_grids(self, zmin=-1400., dz=5.):

        print("\nGenerating regular grids")
        print("------------------------\n")

        print("Interpolating from {:1.0f} m to 0 m in {:1.0f} m increments."
              "".format(zmin, dz))

        z_vals = np.arange(zmin, 0., dz)
        self.__r_z_vals = z_vals
        self_dict = self.__dict__
        for key in self_dict.keys():

            d = np.ndim(self_dict[key])

            if d < 2 or d > 2 or '__' in key or '_ca' in key or 'r_' in key:
                continue
            elif d == 2:
                name = 'r_' + key
                __, __, var_grid = self.get_interp_grid(self.hpid, z_vals,
                                                        'z', key)
                setattr(self, name, var_grid)

            print("  Added: {}.".format(name))

        self.update_profiles()

    def apply_w_model(self, fit_info):
        """Give a W_fit_info object or path to pickle of such object."""

        print("\nApplying vertical velocity model")
        print("--------------------------------\n")

        # Initially assume path to fit.
        try:
            with open(fit_info, 'rb') as f:
                print("Unpickling fit info.")
                setattr(self, '__wfi', pickle.load(f))
        except TypeError:
            print('Copying fit info.')
            setattr(self, '__wfi', copy.copy(fit_info))

        wfi = getattr(self, '__wfi')

        if wfi.profiles == 'all':

            data = [getattr(self, data_name) for data_name in wfi.data_names]

            self.Ws = wfi.model_func(wfi.p, data, wfi.fixed)
            print("  Added: Ws.")
            self.Ww = self.Wz - self.Ws
            print("  Added: Ww.")

        elif wfi.profiles == 'updown':

            self.Ws = np.nan*np.ones_like(self.Wz)

            up = up_down_indices(self.hpid, 'up')
            data = [getattr(self, data_name)[:, up] for
                    data_name in wfi.data_names]
            self.Ws[:, up] = wfi.model_func(wfi.p[0], data, wfi.fixed)
            print("  Added: Ws. (ascents)")

            down = up_down_indices(self.hpid, 'down')
            data = [getattr(self, data_name)[:, down] for
                    data_name in wfi.data_names]
            self.Ws[:, down] = wfi.model_func(wfi.p[1], data, wfi.fixed)
            print("  Added: Ws. (descents)")

            self.Ww = self.Wz - self.Ws
            print("  Added: Ww.")

        else:
            raise RuntimeError

        self.update_profiles()

    def apply_strain(self, N2_ref_file):
        """Input the path to file that contains grid of adiabatically levelled
        N2."""

        print("\nAdding strain")
        print("-------------\n")

        with open(N2_ref_file) as f:

            N2_ref = pickle.load(f)
            setattr(self, 'N2_ref', N2_ref)
            print("  Added: N2_ref.")
            setattr(self, 'strain_z', (self.N2 - N2_ref)/N2_ref)
            print("  Added: strain_z.")

        self.update_profiles()

    def apply_isopycnal_displacement(self, srho_file, rho_1_0=1031., method=1):
        """Input the path to picked array of smoothed potential density."""

        print("\nAdding isopycnal displacements.")
        print("-------------------------------\n")

        setattr(self, 'mu', np.nan*self.rho_1.copy())

        with open(srho_file) as f:

            srho_1 = pickle.load(f)

        setattr(self, 'srho_1', srho_1)
        print("  Added: srho_1.")

        # Use smoothed profiles as they are.
        if method == 0:
            pass
        if method == 1:
            print("  Further smoothing by averaging adjecent profiles.")
            # Add initial smooth profiles to Profile objects.
            self.update_profiles()
            # Do more smoothing by averaging adjecent profiles.
            for i in xrange(len(self.hpid)):
                hpids = np.arange(self.hpid[i] - 5, self.hpid[i] + 6)
                self.srho_1[:, i] = self.get_mean_profile(hpids, 'srho_1',
                                                          self.z[:, i])

        else:
            raise ValueError('Invalid method.')

        for i in xrange(len(self.hpid)):
            srho_1dz = utils.finite_diff(self.z[:, i], srho_1[:, i],
                                         self.UTC[:, i])
            self.mu[:, i] = (self.rho_1[:, i] - srho_1[:, i])/srho_1dz

        print("  Added: mu.")

        # I added a minus because I *think* gravity should be negative.
        self.b = -gsw.grav(self.lat_start, self.P) \
             * (self.rho_1 - srho_1)/rho_1_0

        print("  Added: b.")

        self.update_profiles()

    def update_profiles(self):
        print("\nUpdating half profiles.\n")
        [profile.update(self) for profile in self.Profiles]

    def __regrid(self, grid_from, grid_to, v):
        """Take a gridded variable and put it on a ctd/ef grid using time as
        the interpolant.

        Parameters
        ----------
        grid_from : string.
            Grid type which variable v is currently on: 'ctd', 'ef', 'ctd_ca'
            or 'ef_ca'. Where _ca suffic indicates a centered array.
        grid_to : string.
            Grid type that you want that output to be on: 'ctd' or 'ef'.
        v : 2-D numpy.ndarry.
            The variable values that need regridding.

        Returns
        -------
        x : 2-D numpy.ndarray.
            The values of v at the interpolation times.

        Notes
        -----
        This will not work if flattened time is not monotonically increasing.

        """
        if grid_from == grid_to:
            return v

        if grid_from == 'ctd':
            pUTC = self.UTC.flatten(order='F')
        elif grid_from == 'ef':
            pUTC = self.UTCef.flatten(order='F')
        elif grid_from == 'ctd_ca':
            pUTC = ((self.UTC[1:, :] + self.UTC[:-1, :])/2.).flatten(order='F')
        elif grid_from == 'ef_ca':
            pUTC = ((self.UTCef[1:, :] +
                    self.UTCef[:-1, :])/2.).flatten(order='F')
        else:
            raise ValueError("Can only grid from 'ctd', 'ef', 'ctd_ca' or "
                             "'ef_ca'.")

        if grid_to == 'ctd':
            xUTC = self.UTC.flatten(order='F')
            shape = self.UTC.shape
        elif grid_to == 'ef':
            xUTC = self.UTCef.flatten(order='F')
            shape = self.UTCef.shape
        else:
            raise ValueError("Can only grid to 'ctd' or 'ef'.")

        v_flat = v.flatten(order='F')
        nans = np.isnan(pUTC) | np.isnan(v_flat)
        x = xUTC.copy()
        xnans = np.isnan(x)
        x[~xnans] = np.interp(xUTC[~xnans], pUTC[~nans], v_flat[~nans])
        return x.reshape(shape, order='F')

    def __fill_missing(self, v):
        """Fill missing values in 2D arrays using linear interpolation."""

        # Get corresponding time array.
        t = getattr(self, 'UTC')
        tef = getattr(self, 'UTCef')
        if tef.size == v.size:
            t = tef

        for v_col, t_col in zip(v.T, t.T):

            nnans = ~np.isnan(v_col)

            if np.sum(nnans) == 0:
                continue

            # Index where padding nans start.
            pidx = nnans.searchsorted(True, side='right')

            # Replace non-padding nans with linearly interpolated value.
            nans = np.isnan(v_col[:pidx])
            v_col[:pidx][nans] = np.interp(t_col[nans], t_col[~nans],
                                           v_col[~nans])

        return v

    def get_profiles(self, hpids, ret_idxs=False):
        """Will return profiles requested. Can also return indices of those
        profiles."""

        if np.ndim(hpids) == 0:
            arg = np.argwhere(self.hpid == hpids)

            if arg.size == 0:
                idx = []
            else:
                idx = int(arg)

        elif np.ndim(hpids) == 1:
            idx = [i for i in xrange(len(self.hpid)) if self.hpid[i] in hpids]
            idx = np.array(idx)

        else:
            raise ValueError('Dimensionality of hpids is wrong.')

        if ret_idxs:
            return self.Profiles[idx], idx
        else:
            return self.Profiles[idx]

    def get_mean_profile(self, hpids, var_name, z_return=None, z_interp=None):
        """Calculates an average profile of some variable from a given list of
        profiles by first interpolating on to equal depth grid and then
        averaging. Interpolation is then performed back on to a given depth
        grid."""

        pfls = self.get_profiles(hpids)

        return mean_profile(pfls, var_name, z_return, z_interp)

    def get_interp_grid(self, hpids, var_2_vals, var_2_name, var_1_name):
        """Grid data from multiple profiles into a matrix. Linear interpolation
        is performed on, but not across profiles.

        Parameters
        ----------
        hpids : 1-D numpy.ndarray of integers or floats.
            The profile ID numbers at for which to construct grid.
        var_2_vals : 1-D numpy.ndarray of floats
            The values at which to return interpolant.
        var_2_name : string
            The variable to which var_2_vals correspond.
        var_1_name : string
            The variable to be interpolated.

        Returns
        -------
        number_grid : 2-D numpy.ndarray of integers.
            A meshgrid of numbers from 0 to len(hpids). (May be less if some
            hpids are missing.)
        var_2_grid : 2-D numpy.ndarray of floats.
            A meshed grid of var_2_vals.
        var_1_grid : 2-D numpy.ndarray of floats.
            The interpolated grid of variable 1 at given positions of
        variable 2.

        Notes
        -----
        Uses the Profile.interp function.

        Examples
        --------
        The following example returns and interpolated temperature grid.
        >>> import emapex
        >>> Float = emapex.EMApexFloat('allprofs11.mat',4977)
        >>> hpids = np.arange(10,40)
        >>> z_vals = np.arange(-1000., -100., 10)
        >>> T10 = Float.get_interp_grid(hpids, z_vals, 'z', 'T')
        """

        profiles = self.get_profiles(hpids)
        number = np.arange(len(profiles))

        number_grid, var_2_grid = np.meshgrid(number, var_2_vals)
        var_1_grid = np.zeros_like(number_grid, dtype=np.float64)

        for i, profile in enumerate(profiles):
            var_1_grid[:, i] = profile.interp(var_2_vals, var_2_name,
                                              var_1_name)

        return number_grid, var_2_grid, var_1_grid

    def get_griddata_grid(self, hpids, var_1_name, var_2_name, var_3_name,
                          var_1_vals=None, var_2_vals=None, method='linear'):
        """

        Parameters
        ----------

        Returns
        -------

        Notes
        -----

        Examples
        --------

        """

        __, idxs = self.get_profiles(hpids, ret_idxs=True)

        if var_1_vals is None:
            var_1_arr = getattr(self, var_1_name)[:, idxs]
            start = np.nanmin(var_1_arr)
            stop = np.nanmax(var_1_arr)
            var_1_vals = np.linspace(start, stop, 3*idxs.size)

        if var_2_vals is None:
            var_2_arr = getattr(self, var_2_name)[:, idxs]
            start = np.nanmin(np.nanmin(var_2_arr, axis=0))
            stop = np.nanmax(np.nanmax(var_2_arr, axis=0))
            var_2_vals = np.linspace(start, stop, 400)

        x_grid, y_grid = np.meshgrid(var_1_vals, var_2_vals)

        x = getattr(self, var_1_name)[:, idxs].flatten()
        y = getattr(self, var_2_name)[:, idxs].flatten()
        z = getattr(self, var_3_name)[:, idxs].flatten()

        nans = np.isnan(x) | np.isnan(y) | np.isnan(z)

        x, y, z = x[~nans], y[~nans], z[~nans]

        z_grid = griddata((x, y), z, (x_grid, y_grid), method=method)

        return x_grid, y_grid, z_grid

    def get_timeseries(self, hpids, var_name):
        """TODO: Docstring..."""

        __, idxs = self.get_profiles(hpids, ret_idxs=True)

        def make_timeseries(t, v):
            times = t[:, idxs].flatten(order='F')
            nnans = ~np.isnan(times)
            times = times[nnans]
            vals = v[:, idxs].flatten(order='F')[nnans]
            times, jdxs = np.unique(times, return_index=True)
            vals = vals[jdxs]
#            # Convert to datetime objects.
#            times = utils.datenum_to_datetime(times)
            return times, vals

        # Shorten some names.
        var = getattr(self, var_name)
        t = getattr(self, 'UTC')
        tef = getattr(self, 'UTCef')
        r_t = getattr(self, 'r_UTC', None)

        if var.size == tef.size:
            times, vals = make_timeseries(tef, var)
        elif var.size == t.size:
            times, vals = make_timeseries(t, var)
        elif r_t is not None and r_t.size == var.size:
            times, vals = make_timeseries(r_t, var)
        else:
            raise RuntimeError('Cannot match time array and/or variable'
                               ' array sizes.')

        return times, vals


def mean_profile(pfls, var_name, z_return=None, z_interp=None):
    """Calculates an average profile of some variable from a given list of
    profiles by first interpolating on to equal depth grid and then
    averaging. Interpolation is then performed back on to a given depth
    grid."""

    dz = 1.
    if z_interp is None:
        z_interp = np.arange(-1500, 0, dz)

    if z_return is None:
        z_return = z_interp

    var = []

    for pfl in pfls:
        var.append(pfl.interp(z_interp, 'z', var_name))

    var_mean = np.mean(np.transpose(np.asarray(var)), axis=-1)
    var_return = utils.nan_interp(z_return, z_interp, var_mean)

    return var_return


def up_down_indices(hpid_array, up_or_down='up'):
    """Given an array of hpid numbers return the indices of numbers that
    correspond to either ascents or decents.
    up_or_down can either be 'up' or 'down' corresponding to ascent and
    descent. It is 'up' by default.

    """

    # Weirdly np.where in this case returns tuples so the [0] is needed.
    if up_or_down == 'up':
        return np.where(hpid_array % 2 == 0)[0]
    elif up_or_down == 'down':
        return np.where(hpid_array % 2 == 1)[0]
    else:
        raise RuntimeError('Inputs are probably wrong.')


def what_floats_are_in_here(fname):
    """Finds all unique float ID numbers from a given allprofs##.mat file."""
    fs = io.loadmat(fname, squeeze_me=True, variable_names='flid')['flid']
    return np.unique(fs[~np.isnan(fs)])


def find_file(floatID, data_dir='/noc/users/jc3e13/storage/DIMES/EM-APEX'):
    """Locate the file that contains data for the given ID number."""

    file_paths = glob.glob(os.path.join(data_dir, 'allprofs*.mat'))

    for file_path in file_paths:
        floatIDs = what_floats_are_in_here(file_path)
        if (floatIDs == floatID).any():
            return file_path

    raise ValueError("Float not found in database, check ID.")


def load(floatID, data_dir='/noc/users/jc3e13/storage/DIMES/EM-APEX',
         pp_dir='/noc/users/jc3e13/storage/processed', apply_w=True,
         apply_strain=True, apply_iso=True):
    """Given an ID number this function will attempt to load data. Use the
    optional boolean arguments to turn off additional processing if not
    required as this is performed by default. The hardcoded additional
    processing paths may fail for some floats."""

    float_path = find_file(floatID, data_dir)

    Float = EMApexFloat(float_path, floatID)

    if apply_w:
        data_file = "{:g}_fix_p0k0M_fit_info.p".format(floatID)
        Float.apply_w_model(os.path.join(pp_dir, data_file))

    if apply_strain:
        data_file = "{:g}_N2_ref_300dbar.p".format(floatID)
        Float.apply_strain(os.path.join(pp_dir, data_file))

    if apply_iso:
        data_file = "srho_{:g}_100mbin.p".format(floatID)
        Float.apply_isopycnal_displacement(os.path.join(pp_dir, data_file))

    return Float
