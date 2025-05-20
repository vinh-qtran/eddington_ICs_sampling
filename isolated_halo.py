import numpy as np

from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid, quad
from scipy.optimize import minimize

import snapHDF5 as ws

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
                 G=4.301e-6):
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
        G: float
            Gravitational constant in kpc M_sun^-1 (km/s)^2.
        '''

        self.r_bin_min = r_bin_min
        self.r_bin_max = r_bin_max

        self.N_bins = N_bins

        self.G = G

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
    
    def _get_phi_bins(self, r_bins, mass_bins):
        '''
        Get the potential profile of the halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in kpc.
        mass_bins: array
            Array of mass bins in M_sun.

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
    
    def _get_sigma_bins(self, r_bins, rho_bins, mass_bins):
        '''
        Get the velocity dispersion profile of the halo.

        Parameters:
        ----------
        r_bins: array
            Array of radius bins in kpc.
        rho_bins: array
            Array of density bins in M_sun / kpc^3.
        mass_bins: array
            Array of mass bins in M_sun.

        Returns:
        -------
        sigma_bins: array
            Array of velocity dispersion bins in km/s.
        '''

        _sigma_integrand = self.G * mass_bins * rho_bins / r_bins**2
        _delta_sigma_bins = cumulative_trapezoid(
            _sigma_integrand, r_bins, initial=0
        )
        return np.sqrt(
            (_delta_sigma_bins[-1] - _delta_sigma_bins) / rho_bins
        )
    
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
        for i in tqdm(range(1,self.N_bins),desc="Eddington's inversion:"):
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
        sigma_bins: array
            Array of velocity dispersion bins in km/s.
        eta_bins: array
            Array of eta bins in (km/s)^2.
        f_eta_bins: array
            Array of Eddington distribution probability bins in (km/s)^-2.
        '''

        r_bins = np.logspace(
            np.log10(self.r_bin_min), np.log10(self.r_bin_max), self.N_bins
        )
        rho_bins = self._get_rho_bins(r_bins)
        mass_bins = self._get_mass_bins(r_bins, rho_bins)
        phi_bins = self._get_phi_bins(r_bins, mass_bins)
        sigma_bins = self._get_sigma_bins(r_bins, rho_bins, mass_bins)
        eta_bins, f_eta_bins = self._get_Eddington_bins(rho_bins, phi_bins)

        return r_bins, rho_bins, mass_bins, phi_bins, sigma_bins, eta_bins, f_eta_bins
    
    def reconstruct_density(self, phi_bins, eta_bins, f_eta_bins):
        '''
        Reconstruct the density profile from the potential and Eddington distribution.

        Parameters:
        ----------
        phi_bins: array
            Array of potential bins in (km/s)^2.
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
        for phi in tqdm(phi_bins):
            _reconstructed_rho_bins.append(
                quad(
                    _rho_integrand, 0, np.sqrt(-2*phi), args=(-phi,)
                )[0]
            )
        
        return np.array(_reconstructed_rho_bins)
    
##########################################################################################

class BaseICs(BaseProfile):
    '''
    Base class for the initial conditions generation of the halo.
    '''

    def __init__(self,
                 r_bin_min, r_bin_max, N_bins,
                 r_sample_min, r_sample_max, N_part,
                 seed=42):
        super().__init__(r_bin_min, r_bin_max, N_bins)
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
        N_part: int
            Number of particles to sample from the halo.

        seed: int
            Seed for the random number generator.
        '''
        
        self.r_sample_min = r_sample_min
        self.r_sample_max = r_sample_max

        self.N_part = N_part

        self._check_sampling_range()

        self.r_bins, self.rho_bins, self.mass_bins, self.phi_bins, self.sigma_bins, self.eta_bins, self.f_eta_bins = self.get_profiles()
        self.log_log_mass_interp, self.log_log_inverse_mass_interp, self.lin_log_eddington_interp = self._get_profiles_interp()

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

        part_phi = 2*np.pi * np.random.uniform(0, 1, self.N_part)
        part_theta = np.arcsin(2*np.random.uniform(0, 1, self.N_part) - 1)
        
        part_r = np.exp(self.log_log_inverse_mass_interp(
            np.log(np.random.uniform(
                np.exp(self.log_log_mass_interp(np.log(self.r_sample_min))),
                np.exp(self.log_log_mass_interp(np.log(self.r_sample_max))),
                self.N_part
            ))
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
            Potential in (km/s)^2.
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
    
    def _get_proposal_distributions(self,min_scale=2,max_scale=5):
        '''
        Get the proposal distributions for the rejection sampling of the velocities.

        Parameters:
        ----------
        N_v_bins: int
            Number of velocity bins to use for the proposal distributions analysis.
        min_scale: float
            Minimum scale for the proposed distribution.
        max_scale: float
            Maximum scale for the proposed distribution.
        
        Returns:
        -------
        interp_rho_bins: array
            Array of interpolated density bins in M_sun / kpc^3.
        interp_psi_bins: array
            Array of interpolated negative potential bins in (km/s)^2.
        interp_vmax_bins: array
            Array of interpolated maximum velocity bins in km/s.

        propose_v0_bins: array
            Array of proposed mean velocity bins in km/s.
        propose_sigma_bins: array
            Array of proposed velocity dispersion bins in km/s.
        propose_scale_bins: array
            Array of proposed distribution scale bins.
        '''
        interp_rho_bins = np.exp((np.log(self.rho_bins[:-1]) + np.log(self.rho_bins[1:]))/2)
        interp_psi_bins = - (self.phi_bins[:-1] + self.phi_bins[1:]) / 2
        interp_vmax_bins = np.sqrt(2*interp_psi_bins)

        interp_sigma_bins = (self.sigma_bins[:-1] + self.sigma_bins[1:]) / 2

        propose_v0_bins = np.zeros(self.N_bins-1)
        propose_sigma_bins = np.zeros(self.N_bins-1)
        propose_scale_bins = np.zeros(self.N_bins-1)

        for i in tqdm(range(self.N_bins-1),desc="Proposal distributions:"):
            Eddington_peak = minimize(
                lambda v : -self._Eddington_velocity_distribution(
                    v, interp_psi_bins[i], interp_rho_bins[i]
                ),
                x0=interp_sigma_bins[i], bounds=[(0, interp_vmax_bins[i])]
            )

            propose_v0_bins[i] = Eddington_peak.x[0]
            propose_sigma_bins[i] = - 1/np.sqrt(2*np.pi)/Eddington_peak.fun

            def _inverted_equivalent_scaling_factor(v):
                return propose_sigma_bins[i] / (v - propose_v0_bins[i]) * np.sqrt(
                    np.log(
                        1 / (
                            2*np.pi * propose_sigma_bins[i]**2 * self._Eddington_velocity_distribution(
                                v, interp_psi_bins[i], interp_rho_bins[i]
                            )**2
                        )
                    )
                )

            propose_scale_bins[i] = 1 / max(1/max_scale, min(1/min_scale,
                minimize(
                    _inverted_equivalent_scaling_factor,
                    x0 = min(interp_vmax_bins[i], propose_v0_bins[i] + propose_sigma_bins[i]),
                    bounds=[(propose_v0_bins[i], interp_vmax_bins[i])],
                ).fun
            ))

        return interp_rho_bins, interp_psi_bins, interp_vmax_bins, \
               propose_v0_bins, propose_sigma_bins, propose_scale_bins
    
    def _sample_particle_velocities(self,part_r):
        '''
        Sample the velocities of the particles in the halo using rejection sampling.

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
        interp_rho_bins, interp_psi_bins, interp_vmax_bins, \
        propose_v0_bins, propose_sigma_bins, propose_scale_bins = self._get_proposal_distributions()            

        proposal_indices = np.digitize(part_r, self.r_bins) - 1

        part_v = np.zeros(self.N_part)
        rejected_indinces = np.arange(self.N_part)

        i = 0
        while True:
            n_resample = len(rejected_indinces)

            if n_resample == 0:
                break
            i += 1

            sampling_indices = proposal_indices[rejected_indinces]
            v_sampled = np.random.normal(
                propose_v0_bins[sampling_indices],
                propose_scale_bins[sampling_indices] * propose_sigma_bins[sampling_indices]
            )

            p_v_sampled_proposal = 1.1 * propose_scale_bins[sampling_indices] * self._Gaussian_distribution(
                v_sampled, propose_v0_bins[sampling_indices], 
                propose_scale_bins[sampling_indices] * propose_sigma_bins[sampling_indices]
            )

            p_v_sampled = self._Eddington_velocity_distribution(
                v_sampled, interp_psi_bins[sampling_indices], interp_rho_bins[sampling_indices]
            )

            part_v[rejected_indinces] = v_sampled
            rejected_indinces = rejected_indinces[np.logical_or(
                np.random.uniform(0, 1, n_resample) > p_v_sampled / p_v_sampled_proposal,
                np.logical_or(
                    v_sampled > interp_vmax_bins[sampling_indices],
                    v_sampled < 0
                )
            )]

            if i % 10 == 0:
                print(f"  inter {i}: {n_resample}")

        part_v_phi = 2*np.pi * np.random.uniform(0, 1, self.N_part)
        part_v_theta = np.arcsin(2*np.random.uniform(0, 1, self.N_part) - 1)

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

        part_mass = self._get_particle_mass()
        part_r, part_x, part_y, part_z = self._sample_particle_positions()
        part_vx, part_vy, part_vz = self._sample_particle_velocities(part_r)

        if output:
            f = ws.openfile(output)

            massarr=np.array([0,part_mass,0,0,0,0], dtype="float64")
            npart=np.array([0,self.N_part,0,0,0,0], dtype="uint32")
            ws.write_block(f, "POS ", 1, np.array([part_x,part_y,part_z]).T)
            ws.write_block(f, "VEL ", 1, np.array([part_vx,part_vy,part_vz]).T)
            ws.write_block(f, "MASS", 1, np.repeat(part_mass, self.N_part))
            ws.write_block(f, "ID  ", 1, np.arange(1,self.N_part+2))

            header=ws.snapshot_header(npart=npart, nall=npart, massarr=massarr)
            ws.writeheader(f, header)
            ws.closefile(f)

        return part_mass, \
               part_x, part_y, part_z,\
               part_vx, part_vy, part_vz