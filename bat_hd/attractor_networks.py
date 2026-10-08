"""Periodic bump-attractor network models.

The classes in this module simulate grid-cell activity on a two-dimensional
toroidal lattice.
"""

import numpy as np
from typing import Union, Tuple, Callable, Dict
from tqdm import tqdm
import warnings


class AttractorNetworkBase:
    """Base class for continuous attractor networks on a periodic lattice.

    Parameters
    ----------
    n : int or tuple of int
        Number of cells along each lattice dimension.
    tau : float
        Network time constant, in seconds.
    dt : float
        Euler integration timestep, in seconds.
    intrinsic_noise, input_noise : float
        Standard noise scales for activity and velocity input, respectively.
    size : float or sequence of float
        Spatial scale of the interaction kernel in each dimension.
    use_single_bump : bool
        Select the single-bump kernel variant when true.
    rng : numpy.random.Generator, optional
        Random number generator used to initialize activity and sample noise.

    Notes
    -----
    Subclasses implement :meth:`_recurrent_input` and
    :meth:`_feedforward_input`.  The current activity is exposed as ``s``.
    """

    def __init__(
        self,
        n=64,
        tau=10e-3,
        dt=0.5e-3,
        intrinsic_noise=0,
        input_noise=0,
        size=1,
        use_single_bump=False,
        rng: np.random.Generator = None,
        **kwargs,
    ):

        self.shape = (int(n),) if np.isscalar(n) else tuple(map(int, n))
        if not self.shape or any(size < 1 for size in self.shape):
            raise ValueError("n must be a positive integer or a sequence of integers")
        self.ndim = len(self.shape)
        self.tau = tau
        self.dt = dt
        self.rng = rng or np.random.default_rng(42)
        self.intrinsic_noise = intrinsic_noise
        self.input_noise = input_noise

        if isinstance(size, (float, int)):
            size = np.ones((self.ndim,), dtype=float) * size
        elif isinstance(size, (tuple, list)):
            size = np.array(size)
        else:
            raise ValueError("WTF?")

        if not use_single_bump:
            beta = 0.01 * size
            gamma = 1.05 * beta
            a_weight = 1
            a_weight = 1
            self.kernel_func = lambda dx: a_weight * np.exp(
                -np.dot(dx, gamma)
            ) - np.exp(-np.dot(dx, beta))
            self.kernel_deriv_func = (
                lambda dx: a_weight
                * gamma.reshape((1,) * self.ndim + (-1,))
                * np.exp(-np.dot(dx, gamma)[..., np.newaxis])
                - beta.reshape((1,) * self.ndim + (-1,))
                * np.exp(-np.dot(dx, beta))[..., np.newaxis]
            )

        else:
            gamma = 0.01 * size / n
            a_weight = 1
            inhibition = 1
            self.kernel_func = (
                lambda dx: a_weight * np.exp(-np.dot(dx, gamma)) - inhibition
            )
            self.kernel_deriv_func = (
                lambda dx: a_weight
                * gamma.reshape((1,) * self.ndim + (-1,))
                * np.exp(-np.dot(dx, gamma))[..., np.newaxis]
            )

        self.s = self.rng.uniform(size=self.shape) * 0.1
        self.anchor_points = []
        self._setup_attractor(**kwargs)

    def add_anchor_point(
        self,
        weight_func: Callable = lambda position: np.exp(
            np.sum(np.square(position - [1, 0, 0])) / 0.5
        ),
        mask=None,
        cell_index=(0, 0),
        strength=0.1,
    ):
        """Add a localized external input tied to a spatial position.

        Parameters
        ----------
        weight_func : callable
            Function mapping a position to the anchor's scalar weight.
        mask : ndarray, optional
            Cell mask receiving the input. If omitted, only ``cell_index`` is
            selected.
        cell_index : tuple of int
            Cell selected when ``mask`` is not supplied.
        strength : float
            Multiplicative input strength.
        """
        if len(cell_index) != self.ndim:
            raise ValueError(f"cell_index must contain {self.ndim} indices")
        if any(not (0 <= i < size) for i, size in zip(cell_index, self.shape)):
            raise ValueError("indices must be smaller than network size per dimension")
        if mask is None:
            mask = np.zeros_like(self.s)
            mask[tuple(cell_index)] = 1

        self.anchor_points.append(lambda pos: strength * weight_func(pos) * mask)

    def step(self, v=0, pos=None, intrinsic_noise=None, input_noise=None):
        """Advance the activity state by one Euler integration step.

        Parameters
        ----------
        v : scalar or array-like
            Velocity vector. A scalar is broadcast to every dimension.
        pos : array-like, optional
            Position passed to registered anchor points.
        intrinsic_noise, input_noise : float, optional
            Per-step noise scales overriding the values configured at init.
        """

        eff_input_noise = input_noise if input_noise is not None else self.input_noise
        eff_intrinsic_noise = (
            intrinsic_noise if intrinsic_noise is not None else self.intrinsic_noise
        )

        if np.isscalar(v):
            v = np.ones((self.ndim)) * v
        else:
            v = np.asarray(v)

        if eff_input_noise:
            v += (
                eff_input_noise
                * np.sqrt(self.dt / self.tau)
                * self.rng.normal(size=v.shape)
                * np.abs(v)
            )

        total_input = self._recurrent_input(self.s, v) + self._feedforward_input(
            self.s, v
        )
        anchor_input = self._compute_anchor_input(pos) if pos is not None else 0.0

        rate_derivatives = -self.s + np.maximum(total_input, 0.0) + anchor_input

        intrinsic_noise_term = 0.0
        if eff_intrinsic_noise:
            noise_amplitude = eff_intrinsic_noise * np.sqrt(self.dt / self.tau)
            intrinsic_noise_term = noise_amplitude * self.rng.normal(size=self.s.shape)

        self.s = self.s + (self.dt / self.tau) * rate_derivatives + intrinsic_noise_term

    def _compute_anchor_input(self, pos):
        anchor_input = np.zeros_like(self.s)
        for anchor in self.anchor_points:
            anchor_input += anchor(pos)
        return anchor_input

    def warm_up(self, tol=1e-5, max_iter=100000, pos=None):
        """Relax the network until consecutive states differ by less than ``tol``."""
        prev_net_state = self.s.copy()
        self.step(pos=pos, intrinsic_noise=0, input_noise=0)
        step = 0
        while np.max(np.abs(prev_net_state - self.s)) > tol:
            prev_net_state = self.s.copy()
            self.step(pos=pos, intrinsic_noise=0, input_noise=0)
            if step >= max_iter:
                raise RuntimeError("Exceed the prescribed recursion depth")
            step += 1
        print(f"Ran warm up for {step} steps")

    def _add_variables(self, output_dict, n_steps):
        pass

    def _record_variables(self, output_dict, step_iter):
        pass

    def run_simulation(
        self, v: np.ndarray, pos=None, rec_cells=None, n_snapshots=1000
    ) -> Dict[str, np.ndarray]:
        """Run the network simulation with a sequence of velocity inputs.

        Parameters
        ----------
        v : np.ndarray
            Velocity input array of shape (n_steps, 2) with vx and vy components.
        pos : np.ndarray, optional
            Position array of shape (n_steps, 2). If provided, passed to step().
        rec_cells : list, optional
            List of cell indices to record. Defaults to 9 random cells if None.
        n_snapshots : int, optional
            Number of population snapshots to record (default: 1000).

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'cell_recording': recorded activity of selected cells
            - 'cell_indices': indices of recorded cells
            - 'popuplation_snapshots': snapshots of full network state
            - 'snapshot_indices': time indices of snapshots
        """
        rec_cells = rec_cells or list(
            self.rng.integers(0, self.shape, size=(9, self.ndim))
        )

        if v.ndim < 2:
            v = v[..., np.newaxis]
        elif v.ndim > 2:
            raise ValueError("WTF?!")

        if v.shape[1] != self.ndim:
            raise ValueError("Velocity input must match dimension of attractor network")

        n_steps = v.shape[0]
        n_cell_records = len(rec_cells)

        output_dict = {}
        output_dict["cell_recording"] = np.zeros((n_steps, n_cell_records))
        output_dict["cell_indices"] = rec_cells
        output_dict["popuplation_snapshots"] = np.zeros((n_snapshots, *self.shape))
        output_dict["snapshot_indices"] = np.linspace(
            0, n_steps - 1, n_snapshots, dtype=int
        )
        self._add_variables(output_dict, n_steps)
        snapshot_iter = 0

        for step_iter in tqdm(range(n_steps), desc="Running Simulation Steps"):
            (
                self.step(*v[step_iter], pos=pos[step_iter])
                if pos is not None
                else self.step(*v[step_iter])
            )
            output_dict["cell_recording"][step_iter] = np.array(
                [
                    self.s[tuple(cell_index)]
                    for cell_index in output_dict["cell_indices"]
                ]
            )
            if step_iter == output_dict["snapshot_indices"][snapshot_iter]:
                output_dict["popuplation_snapshots"][snapshot_iter] = self.s
                snapshot_iter += 1
            self._record_variables(output_dict, step_iter)
        return output_dict

    def _recurrent_input(self, s, v):
        """Compute recurrent input from network state and velocity.

        Parameters
        ----------
        s : np.ndarray
            Network state (activity).
        v : np.ndarray
            velocity.

        Returns
        -------
        np.ndarray
            Recurrent input to the network.
        """
        raise NotImplementedError("Has to be implemented by subclasses")

    def _feedforward_input(self, s, v):
        """Compute feedforward input from velocity.

        Parameters
        ----------
        s : np.ndarray
            Network state (activity).
        v : np.ndarray
            velocity.

        Returns
        -------
        np.ndarray
            Feedforward input to the network.
        """
        raise NotImplementedError("Has to be implemented by subclasses")

    def _setup_attractor(self, **kwargs):
        """Set up attractor-specific parameters.

        This method should be overridden by subclasses to initialize
        any additional attractor-specific parameters or kernels.
        """
        warnings.warn("No additional setup for attractor was implemented")
        pass


class ToroidBurakFiete2009(AttractorNetworkBase):
    """Burak--Fiete (2009) attractor network with directional kernels.

    Parameters control the lattice size, integration timestep, kernel
    periodicity, and whether a single bump should be supported.  The public
    state is stored in :attr:`s` and can be advanced with :meth:`step`.
    To establish the grid pattern, a warm up with :meth:`warmup` is recommended.
    """

    def _setup_attractor(self, **kwargs):
        self.l_shift = 2.0
        self.alpha = 0.10315

        self.B0 = 1
        self.dir_vectors = {
            "E": np.array([1.0, 0.0]),
            "W": np.array([-1.0, 0.0]),
            "N": np.array([0.0, 1.0]),
            "S": np.array([0.0, -1.0]),
        }
        dir_pattern = np.array([["E", "N"], ["S", "W"]])
        theta_dir = np.tile(dir_pattern, (self.shape[0] // 2, self.shape[1] // 2))

        self.directed_kernels = {}
        self.directed_masks = {}

        for preferred_direction in self.dir_vectors:
            self.directed_kernels[preferred_direction] = np.fft.fft2(
                self._periodic_kernel(
                    self.shape,
                    e_theta=self.dir_vectors[preferred_direction],
                )
            )
            self.directed_masks[preferred_direction] = theta_dir == preferred_direction
        e_theta_x = np.vectorize(lambda d: self.dir_vectors[d][0])(theta_dir)
        e_theta_y = np.vectorize(lambda d: self.dir_vectors[d][1])(theta_dir)
        self.e_theta = np.stack([e_theta_x, e_theta_y], axis=-1)

    def _periodic_kernel(self, shape, e_theta=(0.0, 0.0)):
        """Return a periodic kernel shifted in direction ``e_theta``."""
        d_list = []
        for n, iter in enumerate(shape):
            idx = np.arange(n)
            d = idx - n // 2
            d = np.where(d > n / 2, d - n, d)
            d = np.where(d < -n / 2, d + n, d)
            d_list.append(d)
        dx = np.stack(np.meshgrid(*d, indexing="ij"), axis=-1)

        sx = dx - self.l_shift * np.asarray(e_theta).reshape((1,) * self.ndim + (-1,))
        r2 = np.sum(sx**2, axis=-1)

        K = self.kernel_func(r2)
        K = np.fft.ifftshift(K)
        return K

    def _recurrent_input(self, s, v):
        """Calculate recurrent input for activity array ``s``."""
        rec_input = np.zeros_like(s)
        for dir in self.dir_vectors:
            rec_input += np.real(
                np.fft.ifft2(
                    np.fft.fft2(self.directed_masks[dir] * s)
                    * self.directed_kernels[dir]
                )
            )
        return rec_input

    def _feedforward_input(self, s, v):
        """Calculate velocity-dependent input for velocity ``(vx, vy)``."""
        return self.B0 * (1.0 + self.alpha * np.dot(self.e_theta, v))


class ToroidZhang1996(AttractorNetworkBase):
    """Zhang (1996) attractor network with symmetric/asymmetric kernels.

    Velocity input is coupled to spatial derivatives of the recurrent kernel;
    the resulting activity state is available in :attr:`s` and can be advanced with :meth:`step`.
    To establish the grid pattern, a warm up with :meth:`warmup` is recommended.
    """

    def _setup_attractor(self, revolutions: Union[Tuple, float] = 1):
        self.speed_modulation = self._compute_speed_modulation(revolutions)
        self.B0 = 1.0

        K_sym, K_asym = self._build_kernels(self.shape)
        self.K_sym_fft = np.fft.fftn(K_sym)
        self.K_asym_fft = np.fft.fftn(K_asym, axes=tuple(np.arange(K_asym.ndim - 1)))

    def _distance_grid(self, shape):
        """Build wrapped x and y displacement grids for an ``n``-cell lattice."""
        dist_list = []
        for n in shape:
            idx = np.arange(n)
            d = idx - n // 2
            d = np.where(d > n / 2, d - n, d)
            d = np.where(d < -n / 2, d + n, d)
            dist_list.append(d)
        dx = np.meshgrid(*dist_list, indexing="ij")
        dx = np.stack(dx, axis=-1)
        return dx

    def _compute_speed_modulation(self, revolutions=1):
        """Compute the asymmetric-kernel scale for ``revolutions`` per cycle."""
        if isinstance(revolutions, (float, int)):
            revolutions = np.ones((self.ndim,), dtype=float) * revolutions
        elif isinstance(revolutions, tuple):
            revolutions = np.array(revolutions)
        else:
            raise ValueError("WTF?")

        target_gain = np.asarray(self.shape) / (2 * np.pi) * revolutions
        dist_list = []
        for n in self.shape:
            idx = np.arange(n)
            d = idx - n // 2
            d = np.where(d > n / 2, d - n, d)
            d = np.where(d < -n / 2, d + n, d)
            dist_list.append(d)

        dx = np.meshgrid(*dist_list, indexing="ij")
        dx = np.stack(dx, axis=-1)

        K_sym = self.kernel_func(dx**2)
        common = self.kernel_deriv_func(dx**2)
        dK_dx = -2 * dx * common

        norm = np.max(np.abs(K_sym))
        dnorm = np.max(np.abs(dK_dx), axis=tuple(np.arange(self.ndim)))
        return target_gain * self.tau * dnorm / norm

    def _build_kernels(self, shape):
        """Construct centered symmetric and velocity-dependent kernels."""
        dx = self._distance_grid(shape)

        K_sym = self.kernel_func(dx**2)
        common = self.kernel_deriv_func(dx**2)

        dK_dx = -2 * dx * common

        norm = np.max(np.abs(K_sym))
        dnorm = np.max(np.abs(dK_dx), axis=tuple(np.arange(self.ndim)))
        K_asym = self.speed_modulation * dK_dx * (norm / dnorm)

        K_sym = np.fft.ifftshift(K_sym)
        K_asym = np.fft.ifftshift(K_asym, axes=tuple(np.arange(self.ndim)))
        return K_sym, K_asym

    def _recurrent_input(self, s, v: np.ndarray):
        """Calculate recurrent input for state ``s`` and velocity ``(vx, vy)``."""
        s_fft = np.fft.fftn(s)
        rec = np.real(np.fft.ifftn(s_fft * self.K_sym_fft))
        if np.any(v != 0):
            rec += np.real(np.fft.ifftn(s_fft * np.dot(self.K_asym_fft, v)))
        return rec

    def _feedforward_input(self, s, v):
        return self.B0


class HeadDirection(ToroidZhang1996):
    """Zhang network specialized for representing head direction.

    The population activity forms a bump on the periodic sheet.  Use
    :meth:`decode_orientation` to read its position and
    :meth:`encode_orientation` to construct an activity pattern for target
    angles.
    """

    def _add_variables(self, output_dict, n_steps):
        output_dict["decoded_angle"] = np.zeros((n_steps, 2))
        output_dict["anchor_input"] = np.zeros(n_steps)

    def _record_variables(self, output_dict, step_iter):
        output_dict["decoded_angle"][step_iter] = self.decode_orientation()

    def decode_orientation(self):
        """Decode the activity bump position into one angle per axis.

        Returns:
        - np.array([angle_x, angle_y]): Estimated angles in radians [0, 2*pi).
        """
        s_2d = self.s
        shape = s_2d.shape

        angle_list = []
        for axis_iter, nx in enumerate(shape):
            x_phases = 2 * np.pi * np.arange(nx) / nx

            profile_x = np.sum(
                s_2d, axis=tuple([i for i in range(len(shape)) if i != axis_iter])
            )

            mean_x_angle = np.angle(np.sum(profile_x * np.exp(1j * x_phases)))

            angle_1 = (-mean_x_angle + 2 * np.pi) % (2 * np.pi)
            angle_list.append(angle_1)

        return np.array(angle_list)

    def encode_orientation(self, target_angles, width=0.3):
        """Generate a periodic Gaussian bump for target orientation angles.

        Parameters:
        - target_angles: np.array or list of [angle_x, angle_y] in radians.
        - width: Controls the spatial width (spread) of the activity bump.

        Returns:
        - s_2d: np.array of shape (nx, ny) representing the network activity state.
        """
        shape = self.s.shape
        target_angles = np.asarray(target_angles)[*((np.newaxis,) * (len(shape) + 1))]

        phases_list = []
        for nx in shape:
            x_phases = 2 * np.pi * np.arange(nx) / nx
            phases_list.append(x_phases)
        X = np.meshgrid(*phases_list, indexing="ij")
        X = np.stack(X, axis=-1)

        dx = np.arctan2(np.sin(-(X + target_angles)), np.cos(-(X + target_angles)))

        s_2d = np.exp(-(np.sum(dx**2, axis=-1)) / (2 * width**2))

        return s_2d
