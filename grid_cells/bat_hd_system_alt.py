import numpy as np
import numpy as np
from typing import Union, Tuple, Dict
from tqdm import tqdm
from .plot_tools import angular_error


def sphere_to_toroid(yaw, pitch):
    pair_1 = np.array([yaw, pitch])
    pair_2 = np.array([(yaw + 180) % (2 * 180), (180 - pitch) % (2 * 180)])
    return pair_1, pair_2


class HeadDirectionNetwork:
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
            size = np.asarray(size)
        else:
            raise ValueError("WTF?")

        if not use_single_bump:
            beta = 0.01 * size
            gamma = 1.05 * beta
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
            gamma = 0.01 / np.asarray(n) / (size * 6)
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
        self._setup_attractor(**kwargs)

    def step(self, v=0, intrinsic_noise=None, input_noise=None, anchor_input=None):
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
            v_array = np.ones((self.ndim)) * v
        else:
            v_array = np.asarray(v)

        if eff_input_noise:
            v_array += (
                eff_input_noise
                * np.sqrt(self.dt / self.tau)
                * self.rng.normal(size=v_array.shape)
                * np.abs(v_array)
            )

        total_input = self._recurrent_input(self.s, v_array) + self._feedforward_input(
            self.s, v_array
        )
        if anchor_input is not None:
            total_input += anchor_input

        rate_derivatives = -self.s + np.maximum(total_input, 0.0)

        intrinsic_noise_term = 0.0
        if eff_intrinsic_noise:
            noise_amplitude = eff_intrinsic_noise * np.sqrt(self.dt / self.tau)
            intrinsic_noise_term = noise_amplitude * self.rng.normal(size=self.s.shape)

        self.s = self.s + (self.dt / self.tau) * rate_derivatives + intrinsic_noise_term

    def warm_up(self, tol=1e-5, max_iter=100000):
        """Relax the network until consecutive states differ by less than ``tol``."""
        prev_net_state = self.s.copy()
        self.step(intrinsic_noise=0, input_noise=0)
        step = 0
        while np.max(np.abs(prev_net_state - self.s)) > tol:
            prev_net_state = self.s.copy()
            self.step(intrinsic_noise=0, input_noise=0)
            if step >= max_iter:
                raise RuntimeError("Exceed the prescribed recursion depth")
            step += 1
        print(f"Ran warm up for {step} steps")

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

        target_gain = np.asarray(self.shape) / (2 * 180) * revolutions
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

    def _add_variables(self, output_dict, n_steps):
        output_dict["decoded_angle"] = np.zeros((n_steps, 2))
        output_dict["anchor_input"] = np.zeros(n_steps)

    def _record_variables(self, output_dict, step_iter):
        output_dict["decoded_angle"][step_iter] = self.decode_orientation()

    def decode_orientation(self, s=None):
        """Decode the activity bump position into one angle per axis.

        Returns:
        - np.array([angle_x, angle_y]): Estimated angles in radians [0, 2*pi).
        """
        s = self.s if s is None else s
        shape = s.shape

        angle_list = []
        for axis_iter, nx in enumerate(shape):
            x_phases = 2 * np.pi * np.arange(nx) / nx

            profile_x = np.sum(
                s, axis=tuple([i for i in range(len(shape)) if i != axis_iter])
            )

            mean_x_angle = np.rad2deg(np.angle(np.sum(profile_x * np.exp(1j * x_phases))))

            angle_1 = (-mean_x_angle + 2 * 180) % (2 * 180)
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
            x_phases = 2 * 180 * np.arange(nx) / nx
            phases_list.append(x_phases)
        X = np.meshgrid(*phases_list, indexing="ij")
        X = np.stack(X, axis=-1)

        dx = np.rad2deg(np.arctan2(np.sin(np.deg2rad(-(X + target_angles))), np.cos(np.deg2rad(-(X + target_angles))))) / 180

        s_2d = np.exp(-(np.sum(dx**2, axis=-1)) / (2 * width**2))

        return s_2d / np.sum(s_2d)


class BatHeadDirectionSystem:

    def __init__(
        self,
        n_conjunctive=(10,5),
        n_yaw=256,
        n_pitch=256,
        tau=10e-3,
        dt=0.5e-3,
        intrinsic_noise=0,
        input_noise=0.1,
        size=1 / 3,
        tau_visual=None,
        eps=1e-8,
        forward_strength=0.7,
        anchor_strength=0.8,
        feedback_strength=1,
        connect_vc_directly=False,
        gravity_gated = False,
        rng: np.random.Generator = None,
        **kwargs,
    ):
        rng = rng or np.random.default_rng(seed=0)
        self.yaw_ring = HeadDirectionNetwork(
            (n_yaw,),
            tau,
            dt,
            intrinsic_noise,
            input_noise,
            size,
            use_single_bump=True,
            rng=rng,
            **kwargs,
        )
        self.pitch_ring = HeadDirectionNetwork(
            (n_pitch,),
            tau,
            dt,
            intrinsic_noise,
            input_noise,
            size,
            use_single_bump=True,
            rng=rng,
            **kwargs,
        )
        self.tau = tau
        self.dt = dt
        self.tau_visual = tau_visual if tau_visual is not None else 2 * tau
        self.eps = eps
        self.learn_rate = 5e-2

        self.n_conjunctive_total = np.prod(n_conjunctive)
        self.n_conj_yaw = n_conjunctive[0]
        self.n_conj_pitch = n_conjunctive[1]
        self.n_yaw = n_yaw
        self.n_pitch = n_pitch
        self.rng = rng
        self.connect_vc_directly = connect_vc_directly
        self.gravity_gated = gravity_gated

        self.forward_strength = forward_strength
        self.anchor_strength = anchor_strength
        self.feedback_strength = feedback_strength
        self.inhibition = 1

        self.visual_trace = np.zeros(self.n_conjunctive_total, dtype=float)
        self.conjunctive_neurons = np.zeros(self.n_conjunctive_total, dtype=float)

        raw_yaw_conj_w = np.zeros((self.n_conjunctive_total, n_yaw), dtype=float)
        raw_pitch_conj_w = np.zeros((self.n_conjunctive_total, n_pitch), dtype=float)
        yaw_range = (0, 2 * 180)
        pitch_range = (-180 / 2, 180 / 2)
        yaw_linspace = np.linspace(yaw_range[0], yaw_range[1], self.n_conj_yaw, endpoint=False)
        pitch_linspace = np.linspace(pitch_range[0], pitch_range[1], self.n_conj_pitch, endpoint=False)
        self.conjunctive_angels = np.array(np.meshgrid(yaw_linspace, pitch_linspace)).T.reshape(-1, 2)
        self.activation_sigma = np.array([2 /self.n_conj_yaw, 1 / self.n_conj_pitch])
        self.connectivity_sigma = np.array([2 / self.n_conj_yaw, 1 /self.n_conj_pitch])



        for neuron_index, [azimuth, polar] in enumerate(self.conjunctive_angels):
            raw_yaw_conj_w[neuron_index] = self.yaw_ring.encode_orientation(
                azimuth, self.connectivity_sigma[0]
            )
            raw_pitch_conj_w[neuron_index] = self.pitch_ring.encode_orientation(
                polar, self.connectivity_sigma[1]
            )

        raw_yaw_anchor_w = np.zeros((n_yaw, self.n_conjunctive_total), dtype=float)
        raw_pitch_anchor_w = np.zeros((n_pitch, self.n_conjunctive_total), dtype=float)

        for neuron_index, [azimuth, polar] in enumerate(self.conjunctive_angels):
            raw_yaw_anchor_w[..., neuron_index] = self.yaw_ring.encode_orientation(
                azimuth, self.connectivity_sigma[0]
            )
            raw_pitch_anchor_w[..., neuron_index] = self.pitch_ring.encode_orientation(
                polar, self.connectivity_sigma[1]
            )

        self._yaw_fwd = raw_yaw_conj_w
        self._pitch_fwd = raw_pitch_conj_w

        self._yaw_anchor_w = raw_yaw_anchor_w
        self._pitch_anchor_w = raw_pitch_anchor_w

        self.intrinsic_noise = intrinsic_noise
        self.input_noise = input_noise

        self.speed_gate_k = 1 / 50
        self.speed_gate_thr = 50

    def normalisze_anchors(self):
        self._yaw_anchor_w /= np.clip(
            np.sum(self._yaw_anchor_w, axis=0, keepdims=True), 1, None
        )
        self._pitch_anchor_w /= np.clip(
            np.sum(self._pitch_anchor_w, axis=0, keepdims=True), 1, None
        )

    def activation_weight_func(self, position, anchor):
        err = angular_error(position, anchor)
        d2 = err**2
        return np.exp(np.sum(-d2 / (2 * self.activation_sigma**2),axis=-1))

    def step(
        self,
        v,
        dir: float = None,
        inverted: bool = None,
    ):
        if dir is not None:
            raw_visual = self.activation_weight_func(
                self.conjunctive_angels,
                dir[np.newaxis, :],
            )
        else:
            raw_visual = np.zeros_like(self.visual_trace)


        if self.gravity_gated and inverted is not None:
            upright = np.array([True], dtype=bool)[np.newaxis, :] ^ (
                inverted if inverted is not None else False
            )
            upright = upright.astype(float)
        else:
            upright = 1


        speed = np.linalg.norm(v)
        anchor_modulation = 1.0 / (
            1.0 + np.exp(self.speed_gate_k * (speed - self.speed_gate_thr))
        )

        raw_visual = self.feedback_strength * raw_visual * anchor_modulation * upright

        # yaw_ring = self.yaw_ring.s.copy()
        # pitch_ring = self.pitch_ring.s.copy()

        # yaw_anchor_weight_update = (
        #     self.learn_rate
        #     * self.visual_trace[np.newaxis, ...]
        #     * yaw_ring[..., np.newaxis, np.newaxis]
        # )
        # pitch_anchor_weight_update = (
        #     self.learn_rate
        #     * self.visual_trace[np.newaxis, ...]
        #     * pitch_ring[..., np.newaxis, np.newaxis]
        # )

        conj_noise_term = 0.0
        visual_noise_term = 0.0
        if self.intrinsic_noise:
            noise_amp = self.intrinsic_noise * np.sqrt(self.dt / self.tau)
            conj_noise_term = noise_amp * self.rng.normal(
                size=self.conjunctive_neurons.shape
            )

        yaw_overlap = np.dot(self._yaw_fwd, self.yaw_ring.s)
        pitch_overlap = np.dot(self._pitch_fwd, self.pitch_ring.s)
        forward_input = self.forward_strength * (yaw_overlap + pitch_overlap) + (
            self.visual_trace if not self.connect_vc_directly else 0
        )
        total_input = forward_input - self.inhibition

        if self.connect_vc_directly:
            feedback = self.visual_trace
        else:
            feedback = self.conjunctive_neurons

        yw_anchor_input = self.anchor_strength * np.sum(
            self._yaw_anchor_w * feedback, axis=(-1)
        )
        pi_anchor_input = self.anchor_strength * np.sum(
            self._pitch_anchor_w * feedback, axis=(-1)
        )
        # self._yaw_anchor_w += yaw_anchor_weight_update * self.dt
        # self._pitch_anchor_w += pitch_anchor_weight_update * self.dt
        # self.normalisze_anchors()

        self.conjunctive_neurons = (
            self.conjunctive_neurons
            + (self.dt / self.tau)
            * (np.maximum(total_input, 0.0) - self.conjunctive_neurons)
            + conj_noise_term
        )

        self.visual_trace = (
            self.visual_trace
            + (self.dt / self.tau_visual) * (raw_visual - self.visual_trace)
            + visual_noise_term
        )

        self.yaw_ring.step(v[0], anchor_input=yw_anchor_input)
        self.pitch_ring.step(v[1], anchor_input=pi_anchor_input)

    def warm_up(
        self, tol=1e-5, max_iter=100000, initial_dir: np.ndarray = np.array([0, 0])
    ):
        """Relax the network until consecutive states differ by less than ``tol``."""
        yaw_pos, pitch_pos = initial_dir[[0, 1]]
        yaw_init = self.yaw_ring.encode_orientation(yaw_pos)
        self.yaw_ring.s = yaw_init / np.max(yaw_init)

        pitch_init = self.pitch_ring.encode_orientation(pitch_pos)
        self.pitch_ring.s = pitch_init / np.max(pitch_init)

        prev_yaw_state = self.yaw_ring.s.copy()
        self.yaw_ring.step(intrinsic_noise=0, input_noise=0)

        prev_pitch_state = self.pitch_ring.s.copy()
        self.pitch_ring.step(intrinsic_noise=0, input_noise=0)

        step = 0
        while (
            max(
                np.max(np.abs(prev_yaw_state - self.yaw_ring.s)),
                np.max(np.abs(prev_pitch_state - self.pitch_ring.s)),
            )
            > tol
        ):
            prev_yaw_state = self.yaw_ring.s.copy()
            prev_pitch_state = self.pitch_ring.s.copy()
            self.yaw_ring.step(intrinsic_noise=0, input_noise=0)
            self.pitch_ring.step(intrinsic_noise=0, input_noise=0)
            if step >= max_iter:
                raise RuntimeError("Exceed the prescribed recursion depth")
            step += 1
        print(f"Ran warm up for {step} steps")

    def run_simulation(
        self,
        v: np.ndarray,
        dir: np.ndarray = None,
        inverted: np.ndarray = None,
        save_weights=False,
        interval=1,
        verbose=True,
    ) -> Dict[str, np.ndarray]:

        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError("Velocity input must match dimension of attractor network")

        if dir is not None:
            if dir.ndim != 2 or dir.shape[1] != 2:
                raise ValueError(
                    "Velocity input must match dimension of attractor network"
                )

        n_steps = v.shape[0]
        record_steps = (n_steps + interval - 1) // interval

        output_dict = {}
        output_dict["anchor_angles"] = self.conjunctive_angels

        output_dict["conj_cells"] = np.zeros(
            (record_steps, self.n_conjunctive_total), dtype=float
        )
        output_dict["yaw_cells"] = np.zeros((record_steps, self.n_yaw), dtype=float)
        output_dict["pitch_cells"] = np.zeros((record_steps, self.n_pitch), dtype=float)
        output_dict["visual_trace"] = np.zeros(
            (record_steps, *self.visual_trace.shape), dtype=float
        )
        output_dict["decoded_angle"] = np.zeros((record_steps, 2), dtype=float)
        if save_weights:
            output_dict["yaw_anchor_weights"] = np.zeros(
                (record_steps, *self._yaw_anchor_w.shape), dtype=float
            )
            output_dict["pitch_anchor_weights"] = np.zeros(
                (record_steps, *self._pitch_anchor_w.shape), dtype=float
            )

        range_generator = (
            tqdm(range(n_steps), desc="Running Simulation Steps")
            if verbose
            else range(n_steps)
        )

        for step_iter in range_generator:
            kwargs = {}
            if dir is not None:
                kwargs["dir"] = dir[step_iter]
            if inverted is not None:
                kwargs["inverted"] = inverted[step_iter]

            self.step(v[step_iter], **kwargs)

            if step_iter % interval == 0:
                record_iter = step_iter // interval
                output_dict["conj_cells"][record_iter] = self.conjunctive_neurons.copy()
                output_dict["yaw_cells"][record_iter] = self.yaw_ring.s.copy()
                output_dict["pitch_cells"][record_iter] = self.pitch_ring.s.copy()
                output_dict["decoded_angle"][record_iter] = np.stack(
                    [
                        self.yaw_ring.decode_orientation(),
                        self.pitch_ring.decode_orientation(),
                    ]
                ).flatten()
                output_dict["visual_trace"][record_iter] = self.visual_trace
                if save_weights:
                    output_dict["yaw_anchor_weights"][record_iter] = self._yaw_anchor_w
                    output_dict["pitch_anchor_weights"][
                        record_iter
                    ] = self._pitch_anchor_w

        return output_dict
