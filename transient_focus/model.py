"""Reduced axial model of a uniformly driven baffled circular piston.

Time convention: exp(+j*omega*t). Propagation toward +z therefore contributes
exp(-j*k*R). The functions here map directly to the axial relations used in the
paper: path difference Delta R, differential aperture delay Delta tau, axial
transfer H, de-ramped aperture response D, and signed aperture signature S.

Distances are in metres, frequency in hertz, sound speed in metres per second,
and delays in seconds.
"""

from __future__ import annotations

import numpy as np


def _positive_scalar(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}")


def _positive_array(name: str, values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only finite positive values")
    return array


def path_difference(z: np.ndarray | float, radius: float) -> np.ndarray:
    """Return Delta R = sqrt(z^2 + a^2) - z using a stable equivalent form."""
    _positive_scalar("radius", radius)
    z_array = _positive_array("z", z)
    return radius * radius / (np.sqrt(z_array * z_array + radius * radius) + z_array)


def differential_delay(
    z: np.ndarray | float,
    radius: float,
    sound_speed: float,
) -> np.ndarray:
    """Return the direct-to-edge differential aperture delay Delta tau."""
    _positive_scalar("sound_speed", sound_speed)
    return path_difference(z, radius) / sound_speed


def axial_transfer(
    frequency: np.ndarray | float,
    z: np.ndarray | float,
    radius: float,
    sound_speed: float,
) -> np.ndarray:
    """Return the reduced axial transfer H(z,f), with shape (n_z, n_f).

    H(z,f) = exp(-j*k*z) - exp(-j*k*sqrt(z^2+a^2))
           = exp(-j*k*z) * [1 - exp(-j*2*pi*f*Delta tau)].

    Multiplicative factors common to both contributions are omitted because the
    paper compares normalized responses rather than absolute pressure.
    """
    _positive_scalar("radius", radius)
    _positive_scalar("sound_speed", sound_speed)
    f = np.atleast_1d(_positive_array("frequency", frequency))
    z_array = np.atleast_1d(_positive_array("z", z))

    k = (2.0 * np.pi * f / sound_speed)[None, :]
    delta_r = path_difference(z_array, radius)[:, None]
    return np.exp(-1j * k * z_array[:, None]) * (1.0 - np.exp(-1j * k * delta_r))


def aperture_response(
    frequency: np.ndarray | float,
    z: np.ndarray | float,
    radius: float,
    sound_speed: float,
) -> np.ndarray:
    """Return D(z,f), the axial transfer after removing the direct-path delay."""
    f = np.atleast_1d(_positive_array("frequency", frequency))
    z_array = np.atleast_1d(_positive_array("z", z))
    tau = differential_delay(z_array, radius, sound_speed)[:, None]
    return 1.0 - np.exp(-1j * 2.0 * np.pi * f[None, :] * tau)


def aperture_signature(
    frequency: np.ndarray | float,
    z: np.ndarray | float,
    radius: float,
    sound_speed: float,
) -> np.ndarray:
    """Return S(z,f) = sin(pi*f*Delta tau), the signed interference signature."""
    f = np.atleast_1d(_positive_array("frequency", frequency))
    z_array = np.atleast_1d(_positive_array("z", z))
    tau = differential_delay(z_array, radius, sound_speed)[:, None]
    return np.sin(np.pi * f[None, :] * tau)
