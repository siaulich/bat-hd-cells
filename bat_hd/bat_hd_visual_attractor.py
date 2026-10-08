from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from tqdm import tqdm

import numpy as np



def wrap_deg(x: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to [-180, 180)."""
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def wrap360(x: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to [0, 360)."""
    return np.asarray(x) % 360.0


def circular_error_deg(a, b):
    """Smallest signed circular angular difference a-b in degrees."""
    return wrap_deg(np.asarray(a) - np.asarray(b))


def yaw_pitch_to_vector(yaw_deg, pitch_deg):
    """Convert toroidal yaw/pitch coordinates to a 3-D unit vector.

    The vector convention is
        x = cos(pitch) cos(yaw)
        y = cos(pitch) sin(yaw)
        z = sin(pitch)
    with yaw and pitch in degrees.
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cp = np.cos(pitch)
    return np.stack((cp * np.cos(yaw), cp * np.sin(yaw), np.sin(pitch)), axis=-1)


def vector_to_yaw_pitch(v):
    """Convert 3-D vectors to yaw in [0,360) and pitch in [-180,180)."""
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True).clip(min=1e-12)
    yaw = np.rad2deg(np.arctan2(v[..., 1], v[..., 0])) % 360.0
    pitch = np.rad2deg(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    return np.stack((yaw, pitch), axis=-1)


def orientation_matrix(yaw_deg, pitch_deg):
    """Body-to-world rotation for roll=0.

    R = Rz(yaw) Ry(-pitch).

    The sign convention makes positive pitch correspond to an upward pointing
    forward axis.  Only yaw and pitch are represented; roll is identically 0.
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    # Rz(yaw) @ Ry(-pitch)
    R = np.empty(np.broadcast(yaw, pitch).shape + (3, 3), dtype=float)
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = -sy
    R[..., 0, 2] = -cy * sp
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = cy
    R[..., 1, 2] = -sy * sp
    R[..., 2, 0] = sp
    R[..., 2, 1] = 0.0
    R[..., 2, 2] = cp
    return R


def world_to_retinal(landmark_vectors, yaw_deg, pitch_deg):
    """Transform fixed world landmark directions into egocentric directions."""
    R = orientation_matrix(yaw_deg, pitch_deg)
    # Body-frame vector = R^T world-frame vector.
    return np.einsum("...ij,li->l...j", R, landmark_vectors)


def retinal_to_world(retinal_vectors, yaw_deg, pitch_deg):
    """Transform egocentric directions to world directions."""
    R = orientation_matrix(yaw_deg, pitch_deg)
    return np.einsum("...ij,lj->l...i", R, retinal_vectors)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class AttractorConfig:
    n: int = 128
    tau: float = 10e-3
    dt: float = 0.5e-3
    sigma_exc: float = 8.0
    sigma_inh: float = 40.0
    exc_gain: float = 1.6
    inh_gain: float = 0.65
    tonic_drive: float = 0.12
    velocity_gain: float = 0.0146
    noise: float = 0.0
    activation_threshold: float = 0.1
    activation_gain: float = 8.0


@dataclass
class BatToroidalHDConfig:
    n_yaw: int = 128
    n_pitch: int = 128
    n_conj_yaw: int = 64
    n_conj_pitch: int = 64

    tau_ring: float = 10e-3
    tau_conj: float = 10e-3
    dt: float = 0.5e-3

    sigma_exc: float = 8.0
    sigma_inh: float = 45.0
    exc_gain: float = 1.65
    inh_gain: float = 0.70
    tonic_drive: float = 0.10
    velocity_gain_yaw: float = 0.0146
    velocity_gain_pitch: float = 0.0146

    conj_sigma_exc: Tuple[float, float] = (12.0, 12.0)
    conj_sigma_inh: Tuple[float, float] = (55.0, 55.0)
    conj_exc_gain: float = 1.2
    conj_inh_gain: float = 0.6
    conj_tonic_drive: float = 0.08

    ring_to_conj_gain: float = 0.5
    conj_to_ring_gain: Tuple[float, float] = (1.0, 1.0)
    conj_inhibition: float = 0.0

    n_landmarks: int = 8
    visual_sigma_deg: Tuple[float, float] = (12.0, 12.0)
    visual_tau: float = 30e-3
    visual_drive: float = 2.0
    visual_to_conj_gain: float = 5.0
    visual_kappa: float = 20.0

    visual_speed_gate_threshold: float = 80.0
    visual_speed_gate_width: float = 20.0

    ring_noise: float = 0.0
    conj_noise: float = 0.0
    visual_noise: float = 0.0

    seed: int = 42
    activation_threshold: float = 0.1
    activation_gain: float = 8.0



class ZhangCircularAttractor:
    """One-dimensional continuous attractor with angular-velocity integration."""

    def __init__(self, config: AttractorConfig, rng: np.random.Generator):
        self.n = int(config.n)
        self.tau = float(config.tau)
        self.dt = float(config.dt)
        self.noise = float(config.noise)
        self.rng = rng

        self.preferred = np.linspace(0.0, 360.0, self.n, endpoint=False)
        d = circular_error_deg(self.preferred, 0.0)

        exc = np.exp(-(d**2) / (2 * config.sigma_exc**2))
        exc /= exc.sum() + 1e-12
        self.exc_fft = np.fft.fft(exc)
        self.inh_gain = float(config.inh_gain)
        self.exc_gain = float(config.exc_gain)
        self.velocity_gain = float(config.velocity_gain)
        self.tonic_drive = float(config.tonic_drive)
        self.angular_spacing = 360.0 / self.n
        self.activation_threshold = float(config.activation_threshold)
        self.activation_gain = float(config.activation_gain)
        self.s = 0.05 * self.rng.random(self.n)

    def reset(self, angle_deg: float = 0.0, amplitude: float = 1.0):
        d = circular_error_deg(self.preferred, angle_deg)
        self.s = amplitude * np.exp(-(d**2) / (2 * 10.0**2))

    def recurrent_input(self):
        excitation = np.real(np.fft.ifft(np.fft.fft(self.s) * self.exc_fft))
        inhibition = self.inh_gain * np.mean(self.s)
        return self.exc_gain * excitation - inhibition

    def step(self, angular_velocity_deg_s: float = 0.0, external_input=None):
        rec = self.recurrent_input()
        gradient = (
            np.roll(self.s, -1) - np.roll(self.s, 1)
        ) / (2.0 * self.angular_spacing)
        shift = -self.velocity_gain * float(angular_velocity_deg_s) * gradient
        inp = rec + self.tonic_drive + shift
        if external_input is not None:
            inp = inp + external_input

        rate = np.maximum(inp - self.activation_threshold, 0.0)
        rate /= rate.max() + 1e-12
        noise = 0.0
        if self.noise:
            noise = self.noise * np.sqrt(self.dt / self.tau) * self.rng.normal(
                size=self.n
            )
        self.s += (self.dt / self.tau) * (-self.s + rate) + noise
        self.s = np.maximum(self.s, 0.0)

    def decode(self) -> float:
        z = np.sum(self.s * np.exp(1j * np.deg2rad(self.preferred)))
        return float(np.rad2deg(np.angle(z)) % 360.0)

    def bump(self, width_deg: float = 12.0):
        d = circular_error_deg(self.preferred, self.decode())
        return np.exp(-(d**2) / (2 * width_deg**2))



class ZhangToroidalAttractor:
    """Two-dimensional continuous attractor on yaw x pitch torus."""

    def __init__(
        self,
        n_yaw: int,
        n_pitch: int,
        tau: float,
        dt: float,
        tonic_drive: float,
        rng: np.random.Generator,
        noise: float = 0.0,
    ):
        self.n_yaw = int(n_yaw)
        self.n_pitch = int(n_pitch)
        self.shape = (self.n_yaw, self.n_pitch)
        self.tau = tau
        self.dt = dt
        self.noise = noise
        self.rng = rng

        self.yaw = np.linspace(0.0, 360.0, self.n_yaw, endpoint=False)
        self.pitch = np.linspace(0.0, 360.0, self.n_pitch, endpoint=False)
        Y, P = np.meshgrid(self.yaw, self.pitch, indexing="ij")
        self.grid = np.stack((Y, P), axis=-1)

        self.tonic_drive = tonic_drive
        self.activation_threshold = 0.1
        self.activation_gain = 8.0
        self.s = 0.05 * self.rng.random(self.shape)

    def reset(self, yaw_deg: float, pitch_deg: float, amplitude: float = 1.0):
        dy = circular_error_deg(self.yaw[:, None], yaw_deg)
        dp = circular_error_deg(self.pitch[None, :], pitch_deg)
        self.s = amplitude * np.exp(-(dy**2) / (2 * 12.0**2)) * np.exp(
            -(dp**2) / (2 * 12.0**2)
        )

    def recurrent_input(self):
        dy = circular_error_deg(
            np.arange(self.n_yaw)[:, None] * 360.0 / self.n_yaw,
            np.arange(self.n_yaw)[None, :] * 360.0 / self.n_yaw,
        )
        dp = circular_error_deg(
            np.arange(self.n_pitch)[:, None] * 360.0 / self.n_pitch,
            np.arange(self.n_pitch)[None, :] * 360.0 / self.n_pitch,
        )
        return np.real(np.fft.ifftn(np.fft.fftn(self.s) * self._kernel_fft))

    def set_kernel(self, sigma_exc, sigma_inh, exc_gain, inh_gain):
        dy = circular_error_deg(
            np.arange(self.n_yaw)[:, None] * 360.0 / self.n_yaw,
            np.arange(self.n_yaw)[None, :] * 360.0 / self.n_yaw,
        )
        dp = circular_error_deg(
            np.arange(self.n_pitch)[:, None] * 360.0 / self.n_pitch,
            np.arange(self.n_pitch)[None, :] * 360.0 / self.n_pitch,
        )
        W = np.zeros(self.shape)
        Y, P = np.meshgrid(dy[:, 0], dp[:, 0], indexing="ij")
        W = exc_gain * np.exp(
            -(Y**2 / (2 * sigma_exc[0] ** 2) + P**2 / (2 * sigma_exc[1] ** 2))
        ) - inh_gain * np.exp(
            -(Y**2 / (2 * sigma_inh[0] ** 2) + P**2 / (2 * sigma_inh[1] ** 2))
        )
        self._kernel_fft = np.fft.fftn(np.fft.ifftshift(W))

    def step(self, external_input=None):
        rec = self.recurrent_input()
        inp = rec + self.tonic_drive
        if external_input is not None:
            inp = inp + external_input
        rate = np.maximum(inp - self.activation_threshold, 0.0)
        rate /= rate.max() + 1e-12
        noise = 0.0
        if self.noise:
            noise = self.noise * np.sqrt(self.dt / self.tau) * self.rng.normal(
                size=self.shape
            )
        self.s += (self.dt / self.tau) * (-self.s + rate) + noise
        self.s = np.maximum(self.s, 0.0)

    def decode(self) -> np.ndarray:
        z_y = np.sum(self.s * np.exp(1j * np.deg2rad(self.yaw))[:, None])
        z_p = np.sum(self.s * np.exp(1j * np.deg2rad(self.pitch))[None, :])
        return np.array(
            [np.rad2deg(np.angle(z_y)) % 360.0, np.rad2deg(np.angle(z_p)) % 360.0]
        )



class BatToroidalHDModel:
    """Complete yaw/pitch bat HD model with visual landmark anchoring.

    Important modeling assumptions
    -------------------------------
    * Roll is absent from the state and all connectivity.
    * Yaw and pitch are both circular variables over 360 degrees.
    * The conjunctive layer is a 2-D toroidal attractor.
    * Visual landmarks are fixed in allocentric/world coordinates.
    * The visual population receives only egocentric landmark directions.
    * Visual anchoring enters the conjunctive layer; the conjunctive layer then
      feeds back to the two HD rings.
    """

    def __init__(self, config: BatToroidalHDConfig = BatToroidalHDConfig()):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        self.yaw_ring = ZhangCircularAttractor(
            AttractorConfig(
                n=config.n_yaw,
                tau=config.tau_ring,
                dt=config.dt,
                sigma_exc=config.sigma_exc,
                sigma_inh=config.sigma_inh,
                exc_gain=config.exc_gain,
                inh_gain=config.inh_gain,
                tonic_drive=config.tonic_drive,
                velocity_gain=config.velocity_gain_yaw,
                noise=config.ring_noise,
            ),
            self.rng,
        )
        self.pitch_ring = ZhangCircularAttractor(
            AttractorConfig(
                n=config.n_pitch,
                tau=config.tau_ring,
                dt=config.dt,
                sigma_exc=config.sigma_exc,
                sigma_inh=config.sigma_inh,
                exc_gain=config.exc_gain,
                inh_gain=config.inh_gain,
                tonic_drive=config.tonic_drive,
                velocity_gain=config.velocity_gain_pitch,
                noise=config.ring_noise,
            ),
            self.rng,
        )

        self.conj = ZhangToroidalAttractor(
            config.n_conj_yaw,
            config.n_conj_pitch,
            config.tau_conj,
            config.dt,
            config.conj_sigma_exc,
            config.conj_sigma_inh,
            config.conj_exc_gain,
            config.conj_inh_gain,
            config.conj_tonic_drive,
            self.rng,
            config.conj_noise,
        )
        self.conj.set_kernel(
            config.conj_sigma_exc,
            config.conj_sigma_inh,
            config.conj_exc_gain,
            config.conj_inh_gain,
        )

        self._build_ring_to_conj_weights()
        self._build_conj_to_ring_weights()
        self._initialize_landmarks()

        self.visual_cells = np.zeros(self.n_landmarks)
        self.visual_candidate = np.zeros(self.conj.shape)
        self.last_visual_drive = 0.0
        self.last_feedback = np.zeros(2)


    @property
    def n_landmarks(self):
        return self.cfg.n_landmarks

    @property
    def conj_angles(self):
        return self.conj.grid


    @staticmethod
    def _tuning(preferred, x, sigma):
        d = circular_error_deg(preferred, x)
        return np.exp(-(d**2) / (2 * sigma**2))

    def _build_ring_to_conj_weights(self):
        yaw_pref = self.conj.yaw
        pitch_pref = self.conj.pitch
        self.yaw_to_conj = self._tuning(
            self.yaw_ring.preferred[:, None], yaw_pref[None, :], 14.0
        )
        self.pitch_to_conj = self._tuning(
            self.pitch_ring.preferred[:, None], pitch_pref[None, :], 14.0
        )

    def _build_conj_to_ring_weights(self):
        self.conj_to_yaw = self._tuning(
            self.yaw_ring.preferred[:, None], self.conj.yaw[None, :], 14.0
        )
        self.conj_to_pitch = self._tuning(
            self.pitch_ring.preferred[:, None], self.conj.pitch[None, :], 14.0
        )

    def _initialize_landmarks(self):
        yaw = self.rng.uniform(0.0, 360.0, self.n_landmarks)
        pitch = self.rng.uniform(-80.0, 80.0, self.n_landmarks)
        self.landmark_world_angles = np.stack((yaw, pitch), axis=-1)
        self.landmark_world_vectors = yaw_pitch_to_vector(yaw, pitch)

    # -------------------------- visual system -----------------------

    def retinal_landmark_angles(self, true_orientation_deg):
        """Return egocentric yaw/pitch of every fixed landmark."""
        true_orientation_deg = np.asarray(true_orientation_deg, dtype=float)
        R = orientation_matrix(*true_orientation_deg[:2])
        retinal_vec = np.einsum("ij,li->lj", R, self.landmark_world_vectors)
        return vector_to_yaw_pitch(retinal_vec)

    def visual_population(self, true_orientation_deg):
        """Encode landmarks in egocentric/retinal coordinates.

        Each landmark has a separate visual channel.  Its firing rate depends
        only on the landmark's retinal direction, not on the HD estimate.
        """
        retinal = self.retinal_landmark_angles(true_orientation_deg)
        self.current_retinal_landmarks = retinal
        return self.visual_cells

    def _visual_candidate_map(self, true_orientation_deg):
        """Construct a visual likelihood over allocentric yaw x pitch.

        This is the structured visual-to-conjunctive mapping.  For a candidate
        HD orientation h, rotate the observed retinal landmark vectors into the
        world frame and compare them with the known directions of the fixed
        landmarks.  The resulting likelihood peaks at the world orientation
        consistent with the visual scene.
        """
        R_true = orientation_matrix(*np.asarray(true_orientation_deg)[:2])
        retinal_vec = np.einsum("ij,li->lj", R_true, self.landmark_world_vectors)

        R = orientation_matrix(
            self.conj.grid[..., 0], self.conj.grid[..., 1]
        )
        predicted_world = np.einsum("ijab,lb->lija", R, retinal_vec)
        similarity = np.einsum(
            "lija,la->lij", predicted_world, self.landmark_world_vectors
        )

        log_like = self.cfg.visual_kappa * np.sum(similarity, axis=0)
        log_like -= np.max(log_like)
        return np.exp(log_like)

    def _update_visual(self, true_orientation_deg, angular_speed):
        if true_orientation_deg is None:
            target = np.zeros(self.n_landmarks)
            self.visual_candidate.fill(0.0)
            return

        retinal = self.retinal_landmark_angles(true_orientation_deg)
        target = np.ones(self.n_landmarks)
        if self.cfg.visual_noise:
            target += self.cfg.visual_noise * self.rng.normal(size=self.n_landmarks)
            target = np.maximum(target, 0.0)

        gate = 1.0 / (
            1.0
            + np.exp(
                (angular_speed - self.cfg.visual_speed_gate_threshold)
                / self.cfg.visual_speed_gate_width
            )
        )
        target *= self.cfg.visual_drive * gate
        self.visual_cells += (
            self.cfg.dt / self.cfg.visual_tau
        ) * (target - self.visual_cells)

        self.current_retinal_landmarks = retinal
        candidate = self._visual_candidate_map(true_orientation_deg)
        candidate *= gate
        candidate /= candidate.max() + 1e-12
        self.visual_candidate = candidate
        self.last_visual_drive = float(gate)


    def _ring_to_conj(self):
        yaw_overlap = self.yaw_to_conj.T @ self.yaw_ring.s
        pitch_overlap = self.pitch_to_conj.T @ self.pitch_ring.s
        forward = np.outer(yaw_overlap, pitch_overlap)
        if np.max(forward) > 0:
            forward /= np.max(forward)
        return self.cfg.ring_to_conj_gain * forward

    def _conj_to_rings(self):
        yaw_drive = self.conj_to_yaw @ self.conj.s.sum(axis=1)
        pitch_drive = self.conj_to_pitch @ self.conj.s.sum(axis=0)

        yaw_drive /= yaw_drive.max() + 1e-12
        pitch_drive /= pitch_drive.max() + 1e-12
        gate = self.last_visual_drive
        return (
            gate * self.cfg.conj_to_ring_gain[0] * yaw_drive,
            gate * self.cfg.conj_to_ring_gain[1] * pitch_drive,
        )

    # -------------------------- simulation ---------------------------

    def reset(self, yaw_deg=0.0, pitch_deg=0.0):
        self.yaw_ring.reset(yaw_deg)
        self.pitch_ring.reset(pitch_deg)
        self.conj.reset(yaw_deg, pitch_deg)
        self.visual_cells.fill(0.0)
        self.visual_candidate.fill(0.0)


    def warm_up(self, yaw_deg=0.0, pitch_deg=0.0, n_steps=500):
        """Run the model for a few steps to stabilize the bump."""
        self.reset(yaw_deg, pitch_deg)
        for _ in range(n_steps):
            self.step([0.0, 0.0], [yaw_deg, pitch_deg])


    def step(
        self,
        angular_velocity_deg_s,
        true_orientation_deg: Optional[np.ndarray] = None,
    ):
        """Advance the full model by one time step.

        Parameters
        ----------
        angular_velocity_deg_s : array-like, shape (2,)
            [yaw velocity, pitch velocity] in degrees/second.  This is the
            inertial/self-motion input to the two ring attractors.
        true_orientation_deg : array-like, shape (2,), optional
            External ground-truth orientation used *only to generate visual
            sensory input*.  It is never directly injected into an HD ring.
            In a real-data simulation this should be the measured head pose.
        """
        v = np.asarray(angular_velocity_deg_s, dtype=float)
        if v.shape != (2,):
            raise ValueError("angular_velocity_deg_s must have shape (2,)")

        speed = float(np.linalg.norm(v))
        self._update_visual(true_orientation_deg, speed)

        forward = self._ring_to_conj()
        visual = self.cfg.visual_to_conj_gain * self.visual_candidate
        conj_input = forward + visual - self.cfg.conj_inhibition
        self.conj.step(conj_input)

        yaw_fb, pitch_fb = self._conj_to_rings()
        self.last_feedback = np.array([yaw_fb.max(), pitch_fb.max()])

        self.yaw_ring.step(v[0], external_input=yaw_fb)
        self.pitch_ring.step(v[1], external_input=pitch_fb)

    def decode(self):
        """Return [yaw, pitch] decoded from the pure HD rings."""
        return np.array([self.yaw_ring.decode(), self.pitch_ring.decode()])

    def decode_conjunctive(self):
        return self.conj.decode()

    def run(
        self,
        angular_velocity_deg_s: np.ndarray,
        true_orientation_deg: Optional[np.ndarray] = None,
        record_every: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Run a simulation and return population activity and decoded states."""
        v = np.asarray(angular_velocity_deg_s, dtype=float)
        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError("angular_velocity_deg_s must have shape (T, 2)")

        if true_orientation_deg is not None:
            true_orientation_deg = np.asarray(true_orientation_deg, dtype=float)
            if true_orientation_deg.shape != v.shape:
                raise ValueError("true_orientation_deg must have shape (T, 2)")

        idx = np.arange(0, len(v), record_every)
        out = {
            "decoded_angle": np.zeros((len(idx), 2)),
            "decoded_conj_angle": np.zeros((len(idx), 2)),
            "yaw_cells": np.zeros((len(idx), self.cfg.n_yaw)),
            "pitch_cells": np.zeros((len(idx), self.cfg.n_pitch)),
            "conj_cells": np.zeros(
                (len(idx), self.cfg.n_conj_yaw, self.cfg.n_conj_pitch)
            ),
            "visual_cells": np.zeros((len(idx), self.n_landmarks)),
            "visual_candidate": np.zeros(
                (len(idx), self.cfg.n_conj_yaw, self.cfg.n_conj_pitch)
            ),
            "retinal_landmark_angles": np.zeros((len(idx), self.n_landmarks, 2)),
        }

        r = 0
        for t in tqdm(range(len(v)), desc="Simulating"):
            true_t = None if true_orientation_deg is None else true_orientation_deg[t]
            self.step(v[t], true_t)
            if t % record_every == 0:
                out["decoded_angle"][r] = self.decode()
                out["decoded_conj_angle"][r] = self.decode_conjunctive()
                out["yaw_cells"][r] = self.yaw_ring.s
                out["pitch_cells"][r] = self.pitch_ring.s
                out["conj_cells"][r] = self.conj.s
                out["visual_cells"][r] = self.visual_cells
                out["visual_candidate"][r] = self.visual_candidate
                if hasattr(self, "current_retinal_landmarks"):
                    out["retinal_landmark_angles"][r] = self.current_retinal_landmarks
                r += 1

        return out


