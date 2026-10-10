import numpy as np

from transient_focus.design import normalize_spectrum
from transient_focus.model import axial_transfer
from transient_focus.propagation import (
    field_at_time,
    peak_analytic_envelope,
    rayleigh_transfer_matrix,
)


def _small_case():
    f = np.linspace(50e3, 150e3, 48)
    df = float(f[1] - f[0])
    r = np.array([0.0, 0.01])
    z = np.array([0.04, 0.05, 0.06])
    spectrum = normalize_spectrum(np.exp(1j * 2.0 * np.pi * f * 12.3e-6), df)
    return f, r, z, spectrum


def test_rayleigh_axis_agrees_with_analytic_model():
    f, r, z, _ = _small_case()
    h = rayleigh_transfer_matrix(
        f, r[:1], z, n_rho=32, n_phi=64, point_batch=3, workers=1
    )[0]
    expected = axial_transfer(f, z, 0.05, 1500.0)
    relative_error = np.linalg.norm(h - expected) / np.linalg.norm(expected)
    assert relative_error < 6e-4


def test_peak_envelope_contains_t0_envelope():
    f, r, z, spectrum = _small_case()
    h = rayleigh_transfer_matrix(
        f, r, z, n_rho=16, n_phi=32, point_batch=2, workers=1
    )
    at_zero = 2.0 * np.abs(field_at_time(h, spectrum, f, 0.0))
    peak, peak_time, metadata = peak_analytic_envelope(
        h, spectrum, f, oversample=2.0, point_batch=2, workers=1
    )
    assert np.all(peak + 1e-12 >= at_zero)
    assert peak.shape == at_zero.shape
    assert peak_time.shape == at_zero.shape
    assert metadata["period_s"] > 0.0
