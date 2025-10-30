import os
import sys
import subprocess
from pathlib import Path

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()

sys.path.append(repo_root)

##########################################################################################

import numpy as np

from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid, quad, solve_ivp
from scipy.optimize import minimize

import snap_utils.snapHDF5 as ws

from tqdm import tqdm

##########################################################################################

class BaseProfile:
    '''
    Base class for the profiles of the halo.
    
    Calculations are done in kpc, M_sun, and km/s, while integrations are performed using the trapezoidal rule, 
    with the exception of the Eddington inversion, which is done using the scipy quad function.
    '''

    def __init__(self,
                 r_bin_min, r_bin_max, N_bins,
                 halo_edge = None,
                 epsilon=0,
                 G=4.30071e-6):
        '''
        Initialize the base profile class.

        Parameters:
        ----------
        r_bin_min: float
            Minimum profile radius of the halo in kpc.
        r_bin_max: float
            Maximum profile radius of the halo in kpc.
        N_bins: int
            Number of bins to use for the profiles.
        epsilon: float
            Plummer-equivalent softening length for the gravitational potential in kpc.
        G: float
            Gravitational constant in kpc M_sun^-1 (km/s)^2.
        '''

        self.r_bin_min = r_bin_min
        self.r_bin_max = r_bin_max

        self.halo_edge = halo_edge if halo_edge else r_bin_max

        self.N_bins = N_bins

        self.G = G
        self.epsilon = epsilon

    def _get_interp(self, x_bins, y_bins):
        '''
        Get the interpolated profiles of the halo.

        Parameters:
        ----------
        x_bins: array
            Array of the x-param.
        y_bins: array
            Array of the y-param.

        Returns:
        -------
        interp: CubicSpline
            Interpolated profile of the halo.
        '''

        x_order = np.argsort(x_bins)
        x_increasing_mask = np.append([True], np.diff(x_bins[x_order]) > 0)

        x_bins = x_bins[x_order][x_increasing_mask]
        y_bins = y_bins[x_order][x_increasing_mask]

        finite_mask = np.logical_and(
            np.isfinite(x_bins), np.isfinite(y_bins)
        )

        return CubicSpline(
            x_bins[finite_mask], y_bins[finite_mask]
        )

    def _get_rho_bins(self, r_bins):
        '''
        Get the density profile of the halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in kpc.

        Returns:
        -------
        rho_bins: array
            Array of density bins in M_sun / kpc^3.
        '''

        raise NotImplementedError("Not implemented in base class.")

    def _get_mass_bins(self, r_bins, rho_bins):
        '''
        Get the mass profile of the halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in kpc.
        rho_bins: array
            Array of density bins in M_sun / kpc^3.

        Returns:
        -------
        mass_bins: array
            Array of mass bins in M_sun.
        '''

        _zero_mass = 4/3*np.pi * r_bins[0]**3 * rho_bins[0]
        _mass_integrand = 4*np.pi * r_bins**2 * rho_bins
        return cumulative_trapezoid(
            _mass_integrand, r_bins, initial=0
        ) + _zero_mass

    def _get_convoluted_rho_bins(self, r_bins, epsilon):
        '''
        Softened density profile of the halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in kpc.
        epsilon: float
            Plummer-equivalent softening length for the gravitational potential in kpc.

        Returns:
        -------
        conv_rho_bins: array
            Array of convoluted density bins in M_sun / kpc^3.
        '''

        _h = epsilon * 2.8

        if _h == 0:
            return self._get_rho_bins(r_bins)

        def _A(u):
            return 1/2 * u**2 - 3/2 * u**4 + 6/5 * u**5

        def _B(u):
            return u**2 - 2 * u**3 + 3/2 * u**4 - 2/5 * u**5
        
        def _I(u1, u2):
            if u2 <= 1/2:
                return _A(u2) - _A(u1)
            elif u1 > 1/2:
                return _B(u2) - _B(u1)
            else:
                return _A(1/2) - _A(u1) + _B(u2) - _B(1/2)
        _I = np.vectorize(_I)

        _conv_rho_bins = np.zeros_like(r_bins)

        # def _conv_rho_integrand(_r_prime, _r):
        #     _ring_mass = 4*np.pi * _r_prime**2 * self._get_rho_bins(_r_prime)

        #     _u_low = np.abs(_r - _r_prime) / _h
        #     _u_high = min(1, (_r + _r_prime) / _h)

        #     return 4 * _ring_mass / np.pi / _h / _r / _r_prime * _I(_u_low, _u_high)

        for i in tqdm(range(r_bins.shape[0]),desc="Softening density:"):
            _r = r_bins[i]

            if _r < self.halo_edge:
                _shell_r_bin_edges = np.linspace(
                    max(0, _r - _h), _r + _h, self.N_bins + 1
                )
                _shell_r_bins = (_shell_r_bin_edges[1:] + _shell_r_bin_edges[:-1]) / 2
                _shell_mass_bins = 4*np.pi/3 * (_shell_r_bin_edges[1:]**3 - _shell_r_bin_edges[:-1]**3) * self._get_rho_bins(_shell_r_bins)

                _u_low = np.abs(_r - _shell_r_bins) / _h
                _u_high = (_r + _shell_r_bins) / _h
                _u_high[_u_high > 1] = 1

                _conv_rho_bins[i] = np.sum(
                    4*_shell_mass_bins/np.pi/_h/_r/_shell_r_bins * _I(_u_low, _u_high)
                )

                # _conv_rho_bins[i] = quad(
                #     _conv_rho_integrand, max(0, _r - _h), _r + _h, args=(_r,)
                # )[0]

            else:
                _conv_rho_bins[i] = self._get_rho_bins(_r)

        return _conv_rho_bins

    def _get_phi_bins(self, r_bins, mass_bins):
        '''
        Get the potential profile of the halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in kpc.
        rho_bins: array
            Array of density bins in M_sun / kpc^3.
        epsilon: float
            Plummer-equivalent softening length for the gravitational potential in kpc.

        Returns:
        -------
        phi_bins: array
            Array of potential bins in (km/s)^2.
        '''

        _delta_phi_integrand = self.G * mass_bins / r_bins**2
        _delta_phi_bins = cumulative_trapezoid(
            _delta_phi_integrand, r_bins, initial=0
        )

        return _delta_phi_bins - _delta_phi_bins[-1]

    def _get_sigma_r_bins(self, r_bins, rho_bins, phi_bins, beta=0):
        '''
        Get the velocity dispersion profile of the halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in kpc.
        rho_bins: array
            Array of density bins in M_sun / kpc^3.
        phi_bins: array
            Array of potential bins in (km/s)^2.
        beta: float
            Anisotropy parameter. Default is 0, which corresponds to isotropic orbits.

        Returns:
        -------
        sigma_bins: array
            Array of velocity dispersion bins in km/s.
        '''
        # _lin_lin_dlogrho_dlogr_interp = self._get_interp(
        #     (np.log(r_bins[:-1]) + np.log(r_bins[1:])) / 2,
        #     np.diff(np.log(rho_bins)) / np.diff(np.log(r_bins))
        # )

        # _lin_lin_dphi_dlogr_interp = self._get_interp(
        #     (np.log(r_bins[:-1]) + np.log(r_bins[1:])) / 2,
        #     np.diff(phi_bins) / np.diff(np.log(r_bins))
        # )

        # def _sigma_r_sqr_integrand(log_r, sigma_r_sqr):
        #     return - _lin_lin_dphi_dlogr_interp(log_r) \
        #            - sigma_r_sqr * (2*beta + _lin_lin_dlogrho_dlogr_interp(log_r))
        
        # _sigma_r_sqr_sol = solve_ivp(
        #     _sigma_r_sqr_integrand,
        #     t_span = (np.log(r_bins[-1]), np.log(r_bins[0])),
        #     y0 = [0],
        #     t_eval = np.flip(np.log(r_bins)),
        # )

        # # print(_sigma_r_sqr_sol.message)
        # # print(_sigma_r_sqr_sol.t.shape)
        # # print(_sigma_r_sqr_sol.t)
        # # print(_sigma_r_sqr_sol.y[0])

        # return np.flip(
        #     np.sqrt(_sigma_r_sqr_sol.y[0])
        # )

        _delta_rho_sigma_r_sqr_integrand = rho_bins * np.gradient(phi_bins, np.log(r_bins)) / r_bins
        _delta_sigma_rho_bins = cumulative_trapezoid(
            _delta_rho_sigma_r_sqr_integrand, r_bins, initial=0
        )
        return np.sqrt(
            (_delta_sigma_rho_bins[-1] - _delta_sigma_rho_bins) / rho_bins
        )
    
    def _get_simple_profiles(self, r_bins, rho_bins):
        '''
        Get the simple profiles of the halo.

        Returns:
        -------
        mass_bins: array
            Array of mass bins in M_sun.
        phi_bins: array
            Array of potential bins in (km/s)^2.
        sigma_r_bins: array
            Array of radial velocity dispersion bins in km/s.
        '''

        mass_bins = self._get_mass_bins(r_bins, rho_bins)
        phi_bins = self._get_phi_bins(r_bins, mass_bins)
        sigma_r_bins = self._get_sigma_r_bins(r_bins, rho_bins, phi_bins)

        return mass_bins, phi_bins, sigma_r_bins

    def _get_Eddington_bins(self, rho_bins, phi_bins):
        '''
        Get the Eddington inversion bins.

        Parameters:
        ----------
        rho_bins: array
            Array of density bins in M_sun / kpc^3.
        phi_bins: array
            Array of potential bins in (km/s)^2.

        Returns:
        -------
        eta_bins: array
            Array of eta bins in (km/s)^2.
        f_eta_bins: array
            Array of Eddington distribution probability bins in (km/s)^-2.
        '''

        _psi_bins = np.flip(-phi_bins)
        _d2rho_dpsi2_bins = np.flip(
            rho_bins * (
                np.gradient(np.log(rho_bins), phi_bins)**2 + \
                np.gradient(
                    np.gradient(np.log(rho_bins), phi_bins), phi_bins
                )
            )
        )
        
        _lin_log_d2rho_dpsi2_interp = self._get_interp(
            _psi_bins[1:], np.log(_d2rho_dpsi2_bins[1:])
        )

        _eta_bins = _psi_bins
        _f_eta_bins = [0]
        for i in tqdm(range(1,self.N_bins),desc="Eddington's inversion"):
            def _f_eta_integrand(psi):
                return 1/np.sqrt(_psi_bins[i]-psi) * np.exp(
                    _lin_log_d2rho_dpsi2_interp(psi)
                )

            _f_eta_bins.append(
                quad(
                    _f_eta_integrand, 0, _psi_bins[i],
                )[0]
            )

        return _eta_bins, 1/np.sqrt(8)/np.pi**2 * np.array(_f_eta_bins)

    def get_profiles(self):
        '''
        Get the profiles of the halo.

        Returns:
        -------
        r_bins: array
            Array of radius bins in kpc.

        rho_bins: array
            Array of density bins in M_sun / kpc^3.
        mass_bins: array
            Array of mass bins in M_sun.
        phi_bins: array
            Array of potential bins in (km/s)^2.
        sigma_r_bins: array
            Array of radial velocity dispersion bins in km/s.

        conv_rho_bins: array
            Array of convoluted density bins in M_sun / kpc^3.
        conv_mass_bins: array
            Array of convoluted mass bins in M_sun.
        conv_phi_bins: array
            Array of convoluted potential bins in (km/s)^2.
        conv_sigma_r_bins: array
            Array of convoluted radial velocity dispersion bins in km/s.

        eta_bins: array
            Array of eta bins in (km/s)^2.
        f_eta_bins: array
            Array of Eddington distribution probability bins in (km/s)^-2.
        '''

        r_bins = np.logspace(
            np.log10(self.r_bin_min), np.log10(self.r_bin_max), self.N_bins,
            dtype=np.float64
        )
        rho_bins = self._get_rho_bins(r_bins)
        mass_bins, phi_bins, sigma_r_bins = self._get_simple_profiles(r_bins, rho_bins)

        conv_rho_bins = self._get_convoluted_rho_bins(r_bins, self.epsilon)
        conv_mass_bins, conv_phi_bins, conv_sigma_r_bins = self._get_simple_profiles(r_bins, conv_rho_bins)

        # conv_phi_bins = self._get_phi_bins(r_bins, rho_bins, self.epsilon)

        eta_bins, f_eta_bins = self._get_Eddington_bins(rho_bins, conv_phi_bins)

        return r_bins, rho_bins, mass_bins, phi_bins, sigma_r_bins, \
               conv_rho_bins, conv_mass_bins, conv_phi_bins, conv_sigma_r_bins, \
               eta_bins, f_eta_bins
    
    def reconstruct_density(self, conv_phi_bins, eta_bins, f_eta_bins):
        '''
        Reconstruct the density profile from the potential and Eddington distribution.

        Parameters:
        ----------
        conv_phi_bins: array
            Array of convoluted potential bins in (km/s)^2.
        eta_bins: array
            Array of eta bins in (km/s)^2.
        f_eta_bins: array
            Array of Eddington distribution probability bins in (km/s)^-2.
        
        Returns:
        -------
        reconstructed_rho_bins: array
            Array of reconstructed density bins in M_sun / kpc^3.
        '''

        _lin_log_eddington_interp = self._get_interp(
            eta_bins, np.log(f_eta_bins)
        )

        def _rho_integrand(v,psi):
            return 4*np.pi * v**2 * np.exp(
                _lin_log_eddington_interp(psi - v**2/2)
            )
        
        _reconstructed_rho_bins = []
        for phi in tqdm(conv_phi_bins,desc="Reconstructing densities"):
            _reconstructed_rho_bins.append(
                quad(
                    _rho_integrand, 0, np.sqrt(-2*phi), args=(-phi,)
                )[0]
            )

        return np.array(_reconstructed_rho_bins)

    # def reconstruct_velocity_dispersion(self, conv_rho_bins, conv_phi_bins, eta_bins, f_eta_bins):
    #     _lin_log_eddington_interp = self._get_interp(
    #         eta_bins, np.log(f_eta_bins)
    #     )
        
    #     def _velocity_moment_integrand(v, psi, rho, n=1):
    #         return v**n * 4*np.pi * v**2 * np.exp(
    #             _lin_log_eddington_interp(psi - v**2/2)
    #         ) / rho

    #     _reconstructed_conv_sigma_r_bins = []
    #     for i in tqdm(range(conv_rho_bins.shape[0]), desc="Reconstructing velocity dispersion"):
    #         rho = conv_rho_bins[i]
    #         phi = conv_phi_bins[i]

    #         _reconstructed_conv_sigma_r_bins.append(
    #             np.sqrt(
    #                 quad(
    #                     _velocity_moment_integrand, 0, np.sqrt(-2*phi), args=(-phi, rho, 2)
    #                 )[0]
    #             ) / np.sqrt(3)
    #         )

    #     return np.array(_reconstructed_conv_sigma_r_bins)


class BaseEddingtonProfile(BaseProfile):
    pass
    
##########################################################################################

class BaseICs(BaseProfile):
    '''
    Base class for the initial conditions generation of the halo.
    '''

    def __init__(self,
                 r_bin_min, r_bin_max, N_bins,
                 r_sample_min, r_sample_max, 
                 N200, r200,
                 epsilon=0,
                 seed=42):
        super().__init__(r_bin_min=r_bin_min, r_bin_max=r_bin_max, 
                         halo_edge=r200, N_bins=N_bins, epsilon=epsilon)
        '''
        Initialize the base ICs class. 

        Parameters:
        ----------
        r_bin_min: float
            Minimum profile radius of the halo in kpc.
        r_bin_max: float
            Maximum profile radius of the halo in kpc.
        N_bins: int
            Number of bins to use for the profiles.

        r_sample_min: float
            Minimum sampling radius of the halo in kpc.
        r_sample_max: float
            Maximum sampling radius of the halo in kpc.

        r200: float
            Virial radius of the halo in kpc.
        N200: int
            Number of particles within r200 to sample from the halo.

        epsilon: float
            Plummer-equivalent softening length for the gravitational potential in kpc.

        seed: int
            Seed for the random number generator.
        '''
        
        self.r_sample_min = r_sample_min
        self.r_sample_max = r_sample_max

        self._check_sampling_range()

        self.r_bins, self.rho_bins, self.mass_bins, self.phi_bins, self.sigma_r_bins, \
        self.conv_rho_bins, self.conv_mass_bins, self.conv_phi_bins, self.conv_sigma_r_bins, \
        self.eta_bins, self.f_eta_bins = self.get_profiles()

        self.log_log_mass_interp, self.log_log_inverse_mass_interp, self.lin_log_eddington_interp = self._get_profiles_interp()

        self.N_part = self._get_N_part(N200, r200)

        self.seed = seed
        np.random.seed(self.seed)

    def _check_sampling_range(self):
        '''
        Check if the sampling range is within the profile range.
        '''

        if self.r_sample_min < self.r_bin_min:
            raise ValueError(
                "The minimum sampling radius is less than the minimum profile radius."
            )
        if self.r_sample_max > self.r_bin_max:
            raise ValueError(
                "The maximum sampling radius is greater than the maximum profile radius."
            )
        
    def _get_profiles_interp(self):
        '''
        Get the interpolated profiles of the halo.

        Returns:
        -------
        _log_log_mass_interp: CubicSpline
            Interpolated mass profile of the halo in log-log x-y space.
        _log_log_inverse_mass_interp: CubicSpline
            Inverse interpolated mass profile of the halo in log-log x-y space.
        _lin_log_eddington_interp: CubicSpline
            Interpolated Eddington distribution of the halo in linear-log x-y space.
        '''

        _log_log_mass_interp = self._get_interp(
            np.log(self.r_bins), np.log(self.mass_bins)
        )
        _log_log_inverse_mass_interp = self._get_interp(
            np.log(self.mass_bins), np.log(self.r_bins)
        )

        _lin_log_eddington_interp = self._get_interp(
            self.eta_bins, np.log(self.f_eta_bins)
        )

        return _log_log_mass_interp, _log_log_inverse_mass_interp, _lin_log_eddington_interp

    def _get_N_part(self, N200, r200):
        '''
        Get the total number of particles within r_sample_max.

        Parameters:
        ----------
        N200: int
            Number of particles within r200 to sample from the halo.
        r200: float
            Virial radius of the halo in kpc.

        Returns:
        -------
        N_part: int
            The total number of particles to sample from the halo.
        '''

        _m_halo = (
            np.exp(self.log_log_mass_interp(np.log(self.r_sample_max))) - np.exp(self.log_log_mass_interp(np.log(self.r_sample_min)))
        )

        _m_200 = (
            np.exp(self.log_log_mass_interp(np.log(r200))) - np.exp(self.log_log_mass_interp(np.log(self.r_sample_min)))
        )

        return int(N200 * _m_halo / _m_200)

    def _get_particle_mass(self):
        '''
        Get the mass of the particles in the halo.

        Returns:
        -------
        particle_mass: float
            Mass of the particles in M_sun.
        '''
        return (
            np.exp(self.log_log_mass_interp(np.log(self.r_sample_max))) - np.exp(self.log_log_mass_interp(np.log(self.r_sample_min)))
        ) / self.N_part
        
    def _sample_particle_positions(self):
        '''
        Sample the positions of the particles in the halo using the mass profile and the inversion sampling method.

        Returns:
        -------
        part_r: array
            Array of particle radii in kpc.
        part_x: array
            Array of particle x-coordinates in kpc.
        part_y: array
            Array of particle y-coordinates in kpc.
        part_z: array
            Array of particle z-coordinates in kpc.
        '''

        part_phi = 2*np.pi * np.random.uniform(0, 1, self.N_part).astype(np.float64)
        part_theta = np.arcsin(2*np.random.uniform(0, 1, self.N_part).astype(np.float64) - 1)

        part_r = np.exp(self.log_log_inverse_mass_interp(
            np.log(np.random.uniform(
                np.exp(self.log_log_mass_interp(np.log(self.r_sample_min))),
                np.exp(self.log_log_mass_interp(np.log(self.r_sample_max))),
                self.N_part,
            ).astype(np.float64))
        ))

        return part_r, \
               part_r * np.cos(part_theta) * np.cos(part_phi), \
               part_r * np.cos(part_theta) * np.sin(part_phi), \
               part_r * np.sin(part_theta)
    
    def _Gaussian_distribution(self, v, v0, sigma):
        '''
        Get the Gaussian distribution for the velocities.

        Parameters:
        ----------
        v: array
            Array of velocities in km/s.
        v0: float/array
            Mean velocity in km/s.
        sigma: float/array
            Velocity dispersion in km/s.

        Returns:
        -------
        p_v: array
            Array of Gaussian distribution probabilities for the velocities.
        '''

        return 1/np.sqrt(2*np.pi)/sigma * np.exp(- (v-v0)**2/sigma**2/2)
    
    def _Eddington_velocity_distribution(self, v, psi, rho):
        '''
        Get the Eddington distribution for the velocities.

        Parameters:
        ----------
        v: array
            Array of velocities in km/s.
        psi: float/array
            Negative potential in (km/s)^2.
        rho: float/array
            Density in M_sun / kpc^3.

        Returns:
        -------
        p_v: array
            Array of Eddington distribution probabilities for the velocities.
        '''

        return 4*np.pi * v**2 * np.exp(
            self.lin_log_eddington_interp(psi - v**2/2)
        ) / rho
    
    def _sample_particle_velocities(self,part_r):
        '''
        Sample the velocities of the particles in the halo using inversion sampling.

        Parameters:
        ----------
        part_r: array
            Array of particle radii in kpc.

        Returns:
        -------
        part_vx: array
            Array of particle x-velocities in km/s.
        part_vy: array
            Array of particle y-velocities in km/s.
        part_vz: array
            Array of particle z-velocities in km/s.
        '''
        _interp_rho_bins = np.exp((np.log(self.rho_bins[:-1]) + np.log(self.rho_bins[1:]))/2)
        _interp_conv_psi_bins = - (self.conv_phi_bins[:-1] + self.conv_phi_bins[1:]) / 2
        _interp_vmax_bins = np.sqrt(2*_interp_conv_psi_bins)

        _bin_indices = np.digitize(part_r, self.r_bins) - 1

        part_v = np.zeros(self.N_part, dtype=np.float64)

        for i in tqdm(range(self.N_bins-1),desc="Sampling velocities"):
            _bin_mask = _bin_indices == i

            if np.sum(_bin_mask) == 0:
                continue

            _v_bins = np.linspace(0, _interp_vmax_bins[i], self.N_bins, dtype=np.float64)

            _p_v_bins = self._Eddington_velocity_distribution(
                _v_bins, _interp_conv_psi_bins[i], _interp_rho_bins[i]
            )

            _P_v_bins = cumulative_trapezoid(
                _p_v_bins, _v_bins, initial=0
            )
            _P_v_mask = np.logical_and(np.isfinite(_P_v_bins), _P_v_bins <= 1)

            _lin_lin_inverse_P_v_interp = self._get_interp(
                _P_v_bins[_P_v_mask] / _P_v_bins[_P_v_mask].max(), _v_bins[_P_v_mask]
            )

            part_v[_bin_mask] = _lin_lin_inverse_P_v_interp(np.random.uniform(0, 1, np.sum(_bin_mask)).astype(np.float64))

        part_v_phi = 2*np.pi * np.random.uniform(0, 1, self.N_part).astype(np.float64)
        part_v_theta = np.arcsin(2*np.random.uniform(0, 1, self.N_part).astype(np.float64) - 1)

        return part_v * np.cos(part_v_theta) * np.cos(part_v_phi), \
               part_v * np.cos(part_v_theta) * np.sin(part_v_phi), \
               part_v * np.sin(part_v_theta)
    
    def generate_ICs(self,output=None):
        '''
        Generate the initial conditions of the halo and write the snapshot of the halo to a HDF5 file.

        Parameters:
        ----------
        output: str
            Path to the output HDF5 file.
        '''

        part_mass = self._get_particle_mass() / 1e10
        part_r, part_x, part_y, part_z = self._sample_particle_positions()
        part_vx, part_vy, part_vz = self._sample_particle_velocities(part_r)

        if output:
            f = ws.openfile(output)

            massarr=np.array([0,part_mass,0,0,0,0], dtype=np.float64)
            npart=np.array([0,self.N_part,0,0,0,0], dtype=np.int32)
            ws.write_block(f, "POS ", 1, np.array([part_x,part_y,part_z], dtype=np.float64).T)
            ws.write_block(f, "VEL ", 1, np.array([part_vx,part_vy,part_vz], dtype=np.float64).T)
            ws.write_block(f, "MASS", 1, np.repeat(part_mass, self.N_part).astype(np.float64))
            ws.write_block(f, "ID  ", 1, np.arange(1,self.N_part+2, dtype=np.int32))

            header=ws.snapshot_header(npart=npart, nall=npart, massarr=massarr)
            ws.writeheader(f, header)
            ws.closefile(f)

        return part_mass, \
               part_x, part_y, part_z,\
               part_vx, part_vy, part_vz