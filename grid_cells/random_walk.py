import numpy as np
from typing import Tuple, Dict, Optional
import tqdm


def generate_random_walk(
    T,
    dt=0.5e-3,
    boxsize=1.0,
    heading_update_inverall=2e-2,
    speed_update_interval=0.5e-1,
    ndim=2,
    max_speed=None,
    rng=None,
) -> Dict[str, np.ndarray]:
    """
    Generate a random walk trajectory in an n-dimensional box with reflective boundaries.

    The walk updates heading direction and speed at specified intervals, with velocity
    vectors reflected at box boundaries.

    Args:
        T: Total duration of the walk in seconds.
        dt: Time step in seconds. Default: 0.5e-3.
        boxsize: Size of the box in each dimension. Default: 1.0.
        heading_update_inverall: Time interval for updating heading direction in seconds. Default: 2e-2.
        speed_update_interval: Time interval for updating speed in seconds. Default: 0.5e-1.
        ndim: Number of spatial dimensions. Default: 2.
        max_speed: Maximum speed magnitude. If None, defaults to boxsize/2.
        rng: Random number generator. If None, uses np.random.default_rng(42).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing simulation outputs with keys:
        - 'pos': (n_steps, ndim) position trajectory
        - 'vel': (n_steps, ndim) velocity trajectory
        - 'time': (n_steps,) time array
    """
    rng = rng or np.random.default_rng(42)

    n_steps = int(np.ceil(T / dt))
    heading_update_steps = int(heading_update_inverall / dt)
    speed_update_steps = int(speed_update_interval / dt)
    time = np.zeros(n_steps)
    position = np.zeros((n_steps, ndim))
    velocity = np.zeros((n_steps, ndim))

    if max_speed is None:
        max_speed = boxsize / 2

    turn_std = 0.6

    current_position = np.ones(ndim) * boxsize / 2
    current_time = 0
    current_speed = rng.uniform(max_speed / 2, max_speed)

    current_heading = rng.normal(size=ndim)
    current_heading /= np.linalg.norm(current_heading)

    time[0] = current_time
    position[0] = current_position
    velocity[0] = current_heading * current_speed

    for step_iter in tqdm.tqdm(range(1, n_steps)):
        if step_iter % speed_update_steps == 0:
            current_speed = rng.uniform(max_speed / 2, max_speed)

        if step_iter % heading_update_steps == 0:
            current_heading = current_heading + rng.normal(0, turn_std, size=ndim)
            current_heading /= np.linalg.norm(current_heading)

        current_velocity = current_heading * current_speed
        proposed_position = current_position + current_velocity * dt

        below = proposed_position < 0
        above = proposed_position > boxsize
        flip = below | above
        if np.any(flip):
            current_velocity[flip] = -current_velocity[flip]
            current_heading[flip] = -current_heading[flip]
            proposed_position = current_position + current_velocity * dt

        current_position = proposed_position
        current_time += dt

        time[step_iter] = current_time
        position[step_iter] = current_position
        velocity[step_iter] = current_velocity

    return {
        "pos": position,
        "vel": velocity,
        "time": time,
    }


def compute_spherical_coordinates(heading):
    """
    Convert 3D Cartesian heading vector to spherical coordinates (azimuth, pitch).

    Args:
        heading: 3D heading vector or array of heading vectors.

    Returns:
        Array of (azimuth, pitch) angles in radians.
    """
    heading = np.asarray(heading, dtype=float)
    heading_norm = np.linalg.norm(heading, axis=-1, keepdims=True)
    heading = heading / np.clip(heading_norm, 1e-12, None)

    azimuth = (np.rad2deg(np.arctan2(heading[..., 1], heading[..., 0])) + 180.0) % 360.0 - 180.0
    pitch = np.rad2deg(np.arcsin(np.clip(heading[..., 2], -1.0, 1.0)))

    return np.stack((azimuth, pitch), axis=-1)


def compute_heading_toroid(angles):
    h_x = np.cos(np.deg2rad(angles[...,1])) * np.cos(np.deg2rad(angles[...,0]))
    h_y = np.cos(np.deg2rad(angles[...,1])) * np.sin(np.deg2rad(angles[...,0]))
    h_z = np.sin(np.deg2rad(angles[...,1]))
    return np.stack([h_x, h_y, h_z],axis=-1)

def sphere_from_toroid(angles):
    return compute_spherical_coordinates(compute_heading_toroid(angles))




def generate_bat_flight(
    T: float,
    dt: float = 0.1e-3,
    boxsize: float = 5.0,
    turning_std: float = 30.0,
    speed_mean: float = 2.5,
    speed_std: float = 1.5,
    speed_correlation_time: float = 0.5,
    min_speed: float = 1.2,
    max_speed: float = 5.0,
    wall_repulsion_strength: float = 250,
    max_pitch_deg: float = 80.0,
    pitch_clearance_deg: float = 15.0,
    pitch_repulsion_strength: float = 250,
    rng: Optional[np.random.Generator] = None,
    initial_position=None,
    initial_speed=None,
    initial_heading=None,
) -> Dict[str, np.ndarray]:
    """
    Generate a bat flight trajectory in a 3D box with realistic dynamics.

    Parameters
    ----------
    T : float
        Total simulation time in seconds.
    dt : float, optional
        Time step for integration. Default is 0.1 ms.
    boxsize : float, optional
        Size of the cubic domain [0, boxsize]^3. Default is 5.0.
    turning_std : float, optional
        Standard deviation of turning angles in degrees. Default is 0.5.
    speed_mean : float, optional
        Mean speed in m/s. Default is 2.5.
    speed_std : float, optional
        Standard deviation of speed. Default is 1.5.
    speed_correlation_time : float, optional
        Correlation time for speed fluctuations in seconds. Default is 0.5.
    min_speed : float, optional
        Minimum speed in m/s. Default is 1.2.
    max_speed : float, optional
        Maximum speed in m/s. Default is 5.0.
    wall_repulsion_strength : float, optional
        Strength of wall repulsion force. Default is 5.
    rng : np.random.Generator, optional
        Random number generator. If None, uses default with seed 42.
    initial_position : array-like, optional
        Initial position [x, y, z]. If None, starts at box center.
    initial_speed : float, optional
        Initial speed. If None, uses speed_mean.
    initial_heading : array-like, optional
        Initial heading vector [hx, hy, hz]. If None, random unit vector.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing simulation outputs with keys:
        - 'pos': (n_steps, 3) position trajectory
        - 'vel': (n_steps, 3) velocity trajectory
        - 'dir_torus': (n_steps, 2) heading in toroid coordinates
        - 'dir_sphere': (n_steps, 2) heading in spherical coordinates
        - 'dir_vel': (n_steps, 2) heading velocity
        - 'time': (n_steps,) time array
    """

    rng = rng or np.random.default_rng(42)

    turn_clearance = boxsize / 8
    slow_clearance = boxsize / 8
    n_steps = int(np.ceil(T / dt))
    max_pitch = max_pitch_deg
    pitch_clearance = pitch_clearance_deg

    # Arrays to store state
    position = np.zeros((n_steps, 3))
    velocity = np.zeros((n_steps, 3))
    toroid_heading = np.zeros((n_steps, 2))
    sphere_heading = np.zeros((n_steps, 2))
    heading_velocity = np.zeros((n_steps, 2))
    time = np.arange(n_steps) * dt
    invertedness = np.zeros((n_steps))

    # Initialize state
    if initial_position is None:
        current_position = np.ones(3) * boxsize / 2
    else:
        initial_position = np.array(initial_position, dtype=float).flatten()
        if initial_position.shape == (3,):
            current_position = initial_position
        else:
            raise ValueError("initial_position muts be a (3,) array")

    # Initialize state
    if initial_heading is None:
        current_heading = np.array([1, 0, 0], dtype=float)
        current_heading /= np.linalg.norm(current_heading)
    else:
        initial_heading = np.array(initial_heading, dtype=float).flatten()
        if initial_heading.shape == (3,):
            initial_heading /= np.linalg.norm(initial_heading)
            current_heading = initial_heading
        else:
            raise ValueError("initial_heading must be a (3,) array")

    # Initialize state
    if initial_speed is None:
        current_speed = speed_mean
    else:
        if isinstance(initial_speed, float):
            current_speed = initial_speed
        else:
            raise ValueError("initial_speed must be a float")

    # OU parameters
    ou_speed_decay = np.exp(-dt / speed_correlation_time)
    ou_speed_noise_scale = speed_std * np.sqrt(1 - ou_speed_decay**2)

    steering_correlation_time = 0.2
    ou_steer_decay = np.exp(-dt / steering_correlation_time)
    ou_steer_noise_scale = turning_std * np.sqrt(1 - ou_steer_decay**2)

    def wall_repulsion(pos: np.ndarray) -> np.ndarray:
        eps = 1e-3
        dist_lower = np.clip(pos, eps, turn_clearance)
        dist_upper = np.clip(boxsize - pos, eps, turn_clearance)

        force_lower = (1.0 / dist_lower) - (1.0 / turn_clearance)
        force_upper = (1.0 / dist_upper) - (1.0 / turn_clearance)

        return wall_repulsion_strength * (force_lower - force_upper)

    def pitch_repulsion(pitch: float) -> float:
        """Soft restoring torque that keeps pitch within [-max_pitch, max_pitch]."""
        eps = 1e-3
        dist_lower = np.clip(pitch - (-max_pitch), eps, pitch_clearance)
        dist_upper = np.clip(max_pitch - pitch, eps, pitch_clearance)

        force_lower = (1.0 / dist_lower) - (1.0 / pitch_clearance)
        force_upper = (1.0 / dist_upper) - (1.0 / pitch_clearance)

        return pitch_repulsion_strength * (force_lower - force_upper)

    current_azimuth, current_pitch = compute_spherical_coordinates(current_heading)

    # Steering is now natively [d_azimuth, d_pitch]
    steering_angles = np.zeros(2)

    position[0] = current_position
    velocity[0] = current_speed * current_heading
    toroid_heading[0] = compute_spherical_coordinates(current_heading)
    sphere_heading[0] = compute_spherical_coordinates(current_heading)

    # OU params for steering
    steering_correlation_time = 0.2
    ou_steer_decay = np.exp(-dt / steering_correlation_time)
    ou_steer_noise_scale = turning_std * np.sqrt(1 - ou_steer_decay**2)

    max_angular_vel = 2000  # rad/s

    for step in tqdm.tqdm(range(1, n_steps)):
        h_x = np.cos(np.deg2rad(current_pitch)) * np.cos(np.deg2rad(current_azimuth))
        h_y = np.cos(np.deg2rad(current_pitch)) * np.sin(np.deg2rad(current_azimuth))
        h_z = np.sin(np.deg2rad(current_pitch))
        current_heading = np.array([h_x, h_y, h_z])

        facing_mask = (
            (current_position - turn_clearance < 0) & (current_heading < 0)
        ) | ((current_position + turn_clearance > boxsize) & (current_heading > 0))
        repulsion_force = wall_repulsion(current_position)

        u_azimuth = np.array([-np.sin(np.deg2rad(current_azimuth)), np.cos(np.deg2rad(current_azimuth)), 0.0])
        u_pitch = np.array(
            [
                -np.cos(np.deg2rad(current_azimuth)) * np.sin(np.deg2rad(current_pitch)),
                -np.sin(np.deg2rad(current_azimuth)) * np.sin(np.deg2rad(current_pitch)),
                np.cos(np.deg2rad(current_pitch)),
            ]
        )

        repulsion_angles = np.array(
            [np.dot(repulsion_force, u_azimuth), np.dot(repulsion_force, u_pitch)]
        )
        repulsion_angles[1] += pitch_repulsion(current_pitch)

        turn_noise = ou_steer_noise_scale * rng.normal(size=2)
        if np.any(facing_mask):
            turn_noise = np.zeros(2)

        steering_angles = steering_angles * ou_steer_decay + turn_noise
        total_steering = steering_angles + repulsion_angles

        steering_norm = np.linalg.norm(total_steering)
        if steering_norm > max_angular_vel:
            total_steering = total_steering * (max_angular_vel / steering_norm)

        current_azimuth = (current_azimuth + total_steering[0] * dt) % (2 * 180)
        current_pitch = (current_pitch + total_steering[1] * dt + 180) % (
            2 * 180
        ) - 180

        h_x = np.cos(np.deg2rad(current_pitch)) * np.cos(np.deg2rad(current_azimuth))
        h_y = np.cos(np.deg2rad(current_pitch)) * np.sin(np.deg2rad(current_azimuth))
        h_z = np.sin(np.deg2rad(current_pitch))
        current_heading = np.array([h_x, h_y, h_z])

        slow_mask = (
            (current_position - slow_clearance < 0) & (current_heading < 0)
        ) | ((current_position + slow_clearance > boxsize) & (current_heading > 0))

        if np.any(slow_mask):
            boundary_distance = np.min(
                np.minimum(
                    current_position[slow_mask], boxsize - current_position[slow_mask]
                )
            )
            boundary_factor = np.clip(boundary_distance / slow_clearance, 0.0, 1.0)
            braking_rate = 1.0
            current_speed -= braking_rate * (1.0 - boundary_factor) * current_speed * dt
            current_speed = np.clip(current_speed, 0.0, max_speed)
        else:
            speed_noise = ou_speed_noise_scale * rng.normal()
            current_speed = (
                speed_mean + ou_speed_decay * (current_speed - speed_mean) + speed_noise
            )
            current_speed = np.clip(current_speed, min_speed, max_speed)

        velocity[step] = current_heading * current_speed
        current_position = current_position + velocity[step] * dt
        # current_position = np.clip(current_position, 1e-4, boxsize - 1e-4)

        position[step] = current_position
        toroid_heading[step] = [current_azimuth, current_pitch]
        heading_velocity[step] = total_steering
        sphere_heading[step] = compute_spherical_coordinates(current_heading)
        invertedness[step] = np.abs(current_pitch) > 180 / 2

    return {
        "pos": position,
        "vel": velocity,
        "speed": np.linalg.norm(velocity, axis=-1),
        "dir_torus": toroid_heading,
        "dir_sphere": sphere_heading,
        "inverted": invertedness.astype(bool),
        "dir_vel": heading_velocity,
        "time": time,
    }
