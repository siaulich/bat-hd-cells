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

            mean_x_angle = np.rad2deg(
                np.angle(np.sum(profile_x * np.exp(1j * x_phases)))
            )

            angle_1 = (-mean_x_angle + 2 * 180) % (2 * 180)
            angle_list.append(angle_1)

        return np.array(angle_list)

    def encode_orientation(self, target_angles, width=0.1):
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

        dx = np.rad2deg(
            np.arctan2(
                np.sin(np.deg2rad(-(X + target_angles))),
                np.cos(np.deg2rad(-(X + target_angles))),
            )
        )
        width = width * 2 * 180

        s_2d = np.exp(-(np.sum(dx**2, axis=-1)) / (2 * width**2))

        return s_2d / np.sum(s_2d)


class BatHeadDirectionSystem:
    YAW_RANGE = (0, 2 * 180)
    PITCH_RANGE = (-50, 50)
    FEEDBACK_MODES = ("none", "direct", "conjunctive")
    C2R_INITS = ("transpose", "random", "zero")

    def __init__(
        self,
        n_ring=(256, 128),
        n_conjunctive=(10, 5),
        n_landmark=10,
        tau=10e-3,
        dt=0.5e-3,
        intrinsic_noise=0,
        input_noise=0.1,
        size=1 / 3,
        tau_visual=None,
        eps=1e-8,
        ring_conj_projection_strength=1,
        ring_input_strength=(0.4,0.2),
        conj_feedback_strength=1,
        visual_drive=1,
        conj_gain=2,
        gravity_gated=False,
        vision_gated=False,
        feedback_mode: str = "conjunctive",
        learn_weights: bool = True,
        learn_interval: int = 1,
        direct_learn_rate: float = 5e-2,
        direct_learn_gate_thr=0.3,
        c2r_init: str = "transpose",
        c2r_learn_rate: float = 0,
        inhibition: float = 1.0,
        rng: np.random.Generator = None,
        **kwargs,
    ):
        if feedback_mode not in self.FEEDBACK_MODES:
            raise ValueError(f"feedback_mode must be one of {self.FEEDBACK_MODES}")
        if np.isscalar(n_conjunctive) or len(n_conjunctive) != 2:
            raise ValueError(
                "n_conjunctive must be a (n_yaw_cells, n_pitch_cells) grid"
            )

        self.n_conj_yaw, self.n_conj_pitch = n_conjunctive
        self.n_landmark = n_landmark
        self.n_yaw = n_ring[0]
        self.n_pitch = n_ring[1]

        rng = rng or np.random.default_rng(seed=0)
        self.yaw_ring = HeadDirectionNetwork(
            (self.n_yaw,),
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
            (self.n_pitch,),
            tau,
            dt,
            intrinsic_noise,
            input_noise,
            size,
            use_single_bump=True,
            rng=rng,
            **{"revolutions": 1, **kwargs},
        )
        self.dt = dt
        self.tau = tau
        self.tau_visual = tau_visual if tau_visual is not None else 2 * tau
        self.tau_conj = tau / 3
        self.eps = eps
        self.conj_learn_rate = 5e-2

        self.gravity_gated = gravity_gated
        self.vision_gated = vision_gated

        self.rng = rng
        self.intrinsic_noise = intrinsic_noise
        self.input_noise = input_noise

        self.ring_conj_projection_strength = np.asarray((ring_conj_projection_strength,) * 2) if np.isscalar(ring_conj_projection_strength) else np.asarray(ring_conj_projection_strength)
        self.ring_input_strength = np.asarray((ring_input_strength,) * 2) if np.isscalar(ring_input_strength) else np.asarray(ring_input_strength)
        self.visual_drive = visual_drive
        self.conj_feedback_strength = conj_feedback_strength
        self.conj_gain = conj_gain
        self.inhibition = inhibition

        self.feedback_mode = feedback_mode
        self.learn_weights = learn_weights
        self.direct_learn_rate = direct_learn_rate
        self.direct_learn_gate_thr = direct_learn_gate_thr
        self.c2r_learn_rate = c2r_learn_rate
        self.learn_interval = learn_interval

        self.landmark_angles = np.stack(
            [
                rng.uniform(*self.YAW_RANGE, size=n_landmark),
                rng.uniform(*self.PITCH_RANGE, size=n_landmark),
            ],
            axis=-1,
        )

        yaw_linspace = np.linspace(0, 360, self.n_conj_yaw, endpoint=False)
        pitch_linspace = np.linspace(-90, 90, self.n_conj_pitch, endpoint=True)
        self.conjunctive_angels = np.array(np.meshgrid(yaw_linspace, pitch_linspace)).T
        self.visual_field = np.array((0.1, 0.1))
        self.connectivity_sigma = np.array([1 / self.n_yaw, 0.5 / self.n_pitch]) * 2

        self.visual_trace = rng.uniform(0, 0.1, size=n_landmark)

        self.conjunctive_neurons = rng.uniform(
            0, 0.1, size=(self.n_conj_yaw, self.n_conj_pitch)
        )

        raw_yaw_conj_w = np.zeros(
            (self.n_conj_yaw, self.n_conj_pitch, self.n_yaw), dtype=float
        )
        raw_pitch_conj_w = np.zeros(
            (self.n_conj_yaw, self.n_conj_pitch, self.n_pitch), dtype=float
        )
        for i, yaw in enumerate(yaw_linspace):
            for j, pitch in enumerate(pitch_linspace):
                raw_yaw_conj_w[i, j] = self.yaw_ring.encode_orientation(
                    yaw, self.connectivity_sigma[0]
                )
                raw_pitch_conj_w[i, j] = self.pitch_ring.encode_orientation(
                    pitch, self.connectivity_sigma[1]
                )
        self._yaw_conj_w = raw_yaw_conj_w.copy()
        self._pitch_conj_w = raw_pitch_conj_w.copy()

        self.conj_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))

        self._c2r_yaw = None
        self._c2r_pitch = None
        self._direct_w_yaw = None
        self._direct_w_pitch = None

        if feedback_mode == "direct":
            self._init_direct_weights()
        elif feedback_mode == "conjunctive":
            self._init_c2r_weights(c2r_init)

        self.last_feedback_norm = np.zeros(2)
        self.speed_gate_k = 1 / 50
        self.speed_gate_thr = 50

    def _init_c2r_weights(self, c2r_init):
        if c2r_init == "transpose":
            self._c2r_yaw = self._yaw_conj_w.copy()
            self._c2r_pitch = self._pitch_conj_w.copy()
        elif c2r_init == "random":
            self._c2r_yaw = self.conj_rng.uniform(size=self._yaw_conj_w.shape)
            self._c2r_pitch = self.conj_rng.uniform(size=self._pitch_conj_w.shape)
            self._c2r_yaw /= self._c2r_yaw.sum(axis=-1, keepdims=True)
            self._c2r_pitch /= self._c2r_pitch.sum(axis=-1, keepdims=True)
        else:  # "zero"
            self._c2r_yaw = np.zeros_like(self._yaw_conj_w)
            self._c2r_pitch = np.zeros_like(self._pitch_conj_w)
        self._conj_feedback_w = self.conj_rng.uniform(
            size=(self.n_conj_yaw, self.n_conj_pitch, self.n_landmark)
        )
        self._conj_feedback_w /= self._conj_feedback_w.sum(axis=-1, keepdims=True)
        self._conj_feedback_w *= 0.1

    def _init_direct_weights(self):
        self._direct_w_yaw = self.conj_rng.uniform(
            size=(self.n_landmark, self.n_yaw)
        )
        self._direct_w_pitch = self.conj_rng.uniform(
            size=(self.n_landmark, self.n_pitch)
        )
        self._direct_w_yaw /= np.sum(self._direct_w_yaw, axis=1, keepdims=True)
        self._direct_w_pitch /= np.sum(self._direct_w_pitch, axis=1, keepdims=True)
        self._direct_w_yaw *= 0.1
        self._direct_w_pitch *= 0.1

    def set_noise(self, intrinsic_noise=None, input_noise=None):
        """Set the per-step noise scales."""
        if intrinsic_noise is not None:
            self.intrinsic_noise = intrinsic_noise
        if input_noise is not None:
            self.input_noise = input_noise
        self.yaw_ring.intrinsic_noise = self.intrinsic_noise
        self.pitch_ring.intrinsic_noise = self.intrinsic_noise
        self.yaw_ring.input_noise = self.input_noise
        self.pitch_ring.input_noise = self.input_noise

    def activation_weight_func(self, position, anchor):
        err = angular_error(position, anchor)
        d2 = err**2
        return np.exp(np.sum(-d2 / (2 * self.visual_field**2), axis=-1))

    def _ring_feedback(self, innovation):
        """Return (yaw_input, pitch_input) for the rings, or (None, None)."""
        if self.feedback_mode == "none":
            return None, None
        elif self.feedback_mode == "direct":
            yaw_fb = self.ring_input_strength[0] * (self.visual_trace @ self._direct_w_yaw)
            pitch_fb = self.ring_input_strength[1] * (
                self.visual_trace @ self._direct_w_pitch
            )
            return yaw_fb, pitch_fb
        elif self.feedback_mode == "conjunctive":
            gate = np.tanh(np.sum(self.visual_trace)) if self.vision_gated else 1.0
            bs = self.ring_input_strength * gate
            yaw_fb = bs[0] * np.einsum("ijk,ij->k", self._c2r_yaw, innovation)
            pitch_fb = bs[1] * np.einsum("ijk,ij->k", self._c2r_pitch, innovation)
            return yaw_fb, pitch_fb
        else:
            raise ValueError(f"Unknown feedback mode: {self.feedback_mode}")

    def _learn_direct(self):
        learning_trace = self.visual_trace * (
            self.visual_trace > self.direct_learn_gate_thr
        )
        for w, ring in (
            (self._direct_w_yaw, self.yaw_ring),
            (self._direct_w_pitch, self.pitch_ring),
        ):
            ring_weight_update = (
                self.dt
                * self.direct_learn_rate
                * self.learn_interval
                * learning_trace[..., np.newaxis]
                * ring.s[np.newaxis, ...]
            )
            w += ring_weight_update
            w /= np.clip(
                np.sum(w, axis=tuple(range(1, w.ndim)), keepdims=True),
                self.eps,
                None,
            )

    def _learn_conj(self):
        for w, ring in (
            (self._c2r_yaw, self.yaw_ring),
            (self._c2r_pitch, self.pitch_ring),
        ):
            ring_weight_update = (
                self.dt
                * self.c2r_learn_rate
                * self.learn_interval
                * self.conjunctive_neurons[..., np.newaxis]
                * ring.s[np.newaxis, np.newaxis, ...]
            )
            w += ring_weight_update
            w /= np.clip(
                np.sum(w, axis=tuple(range(1, w.ndim)), keepdims=True),
                1,
                None,
            )

        conj_feedback_weight_update = (
            self.dt
            * self.learn_interval
            * self.conj_learn_rate
            * self.conjunctive_neurons[..., np.newaxis]
            * self.visual_trace[np.newaxis, np.newaxis, ...]
        )
        self._conj_feedback_w += conj_feedback_weight_update
        self._conj_feedback_w /= np.clip(
            np.sum(
                self._conj_feedback_w,
                axis=tuple(range(self._conj_feedback_w.ndim - 1)),
                keepdims=True,
            ),
            1,
            None,
        )

    def step(
        self,
        v,
        dir: float = None,
        inverted: bool = None,
        iteration: int = 0,
    ):
        v = np.asarray(v, dtype=float)
        if dir is not None:
            raw_visual = self.activation_weight_func(
                self.landmark_angles,
                dir[np.newaxis, :],
            )
        else:
            raw_visual = np.zeros_like(self.visual_trace)

        if self.gravity_gated and inverted is not None:
            upright = 1.0 - float(bool(inverted))
        else:
            upright = 1.0

        speed = np.linalg.norm(v)
        anchor_modulation = 1.0 / (
            1.0 + np.exp(self.speed_gate_k * (speed - self.speed_gate_thr))
        )

        raw_visual = self.visual_drive * raw_visual * anchor_modulation * upright

        conj_noise_term = 0.0
        if self.intrinsic_noise:
            noise_amp = self.intrinsic_noise * np.sqrt(self.dt / self.tau_conj)
            conj_noise_term = noise_amp * self.conj_rng.normal(
                size=self.conjunctive_neurons.shape
            )

        yaw_overlap = np.dot(self._yaw_conj_w, self.yaw_ring.s)
        pitch_overlap = np.dot(self._pitch_conj_w, self.pitch_ring.s)
        forward_input = self.ring_conj_projection_strength[0] * yaw_overlap + self.ring_conj_projection_strength[1] *pitch_overlap

        if self.feedback_mode == "conjunctive":
            visual_input = self.conj_feedback_strength * np.dot(
                self._conj_feedback_w, self.visual_trace
            )
            innovation_input = forward_input + visual_input
        else:
            innovation_input = forward_input

        innovation = (
            np.maximum(innovation_input - self.inhibition, 0.0) * self.conj_gain
        )

        yaw_fb, pitch_fb = self._ring_feedback(innovation)
        self.last_feedback_norm = np.array(
            [
                0.0 if yaw_fb is None else np.linalg.norm(yaw_fb),
                0.0 if pitch_fb is None else np.linalg.norm(pitch_fb),
            ]
        )

        if self.learn_weights and iteration % self.learn_interval == 0:
            if self.feedback_mode == "direct":
                self._learn_direct()
            elif self.feedback_mode == "conjunctive":
                self._learn_conj()
            else:
                pass

        self.conjunctive_neurons = (
            self.conjunctive_neurons
            + (self.dt / self.tau_conj) * (innovation - self.conjunctive_neurons)
            + conj_noise_term
        )

        self.visual_trace = self.visual_trace + (self.dt / self.tau_visual) * (
            raw_visual - self.visual_trace
        )

        self.yaw_ring.step(v[0], anchor_input=yaw_fb)
        self.pitch_ring.step(v[1], anchor_input=pitch_fb)

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
        output_dict["conj_angles"] = self.conjunctive_angels.copy()
        output_dict["landmark_angles"] = self.landmark_angles.copy()

        output_dict["conj_cells"] = np.zeros(
            (record_steps, self.n_conj_yaw, self.n_conj_pitch), dtype=float
        )
        output_dict["yaw_cells"] = np.zeros((record_steps, self.n_yaw), dtype=float)
        output_dict["pitch_cells"] = np.zeros((record_steps, self.n_pitch), dtype=float)
        output_dict["visual_trace"] = np.zeros(
            (record_steps, *self.visual_trace.shape), dtype=float
        )
        output_dict["decoded_angle"] = np.zeros((record_steps, 2), dtype=float)
        output_dict["feedback_norm"] = np.zeros((record_steps, 2), dtype=float)
        if save_weights:
            if self.feedback_mode == "direct":
                output_dict["direct_yaw_w"] = np.zeros(
                    (record_steps, *self._direct_w_yaw.shape), dtype=float
                )
                output_dict["direct_pitch_w"] = np.zeros(
                    (record_steps, *self._direct_w_pitch.shape), dtype=float
                )
            elif self.feedback_mode == "conjunctive":
                output_dict["c2r_yaw_w"] = np.zeros(
                    (record_steps, *self._c2r_yaw.shape), dtype=float
                )
                output_dict["c2r_pitch_w"] = np.zeros(
                    (record_steps, *self._c2r_pitch.shape), dtype=float
                )
                output_dict["conj_feedback_w"] = np.zeros(
                    (record_steps, *self._conj_feedback_w.shape), dtype=float
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
                output_dict["feedback_norm"][record_iter] = self.last_feedback_norm

                if save_weights:
                    if self.feedback_mode == "direct":
                        output_dict["direct_yaw_w"][
                            record_iter
                        ] = self._direct_w_yaw.copy()
                        output_dict["direct_pitch_w"][
                            record_iter
                        ] = self._direct_w_pitch.copy()
                    elif self.feedback_mode == "conjunctive":
                        output_dict["c2r_yaw_w"][record_iter] = self._c2r_yaw.copy()
                        output_dict["c2r_pitch_w"][record_iter] = self._c2r_pitch.copy()
                        output_dict["conj_feedback_w"][
                            record_iter
                        ] = self._conj_feedback_w.copy()

        return output_dict

