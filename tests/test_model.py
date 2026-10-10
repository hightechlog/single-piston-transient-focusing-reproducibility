import numpy as np

from transient_focus.model import (
    aperture_response,
    aperture_signature,
    axial_transfer,
    differential_delay,
    path_difference,
)


def test_stable_path_difference_matches_direct_expression():
    radius = 0.05
    z = np.array([0.01, 0.05, 0.5, 5.0])
    stable = path_difference(z, radius)
    direct = np.sqrt(z**2 + radius**2) - z
    np.testing.assert_allclose(stable, direct, rtol=1e-10, atol=1e-15)


def test_axial_factorization():
    radius = 0.05
    sound_speed = 1500.0
    frequency = np.array([50e3, 100e3, 150e3])
    z = np.array([0.02, 0.05, 0.10])
    h = axial_transfer(frequency, z, radius, sound_speed)
    d = aperture_response(frequency, z, radius, sound_speed)
    k = 2.0 * np.pi * frequency / sound_speed
    np.testing.assert_allclose(
        h,
        np.exp(-1j * z[:, None] * k[None, :]) * d,
        rtol=1e-13,
        atol=1e-13,
    )


def test_aperture_response_factorization():
    radius = 0.05
    sound_speed = 1500.0
    frequency = np.linspace(50e3, 150e3, 257)
    z = np.array([0.02, 0.05, 0.10, 0.15])
    d = aperture_response(frequency, z, radius, sound_speed)
    s = aperture_signature(frequency, z, radius, sound_speed)
    tau = differential_delay(z, radius, sound_speed)[:, None]
    reconstructed = 2j * np.exp(-1j * np.pi * frequency[None, :] * tau) * s
    np.testing.assert_allclose(d, reconstructed, rtol=2e-13, atol=2e-13)
