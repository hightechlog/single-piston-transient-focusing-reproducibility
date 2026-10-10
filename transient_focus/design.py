"""Excitation spectra used in the published figures.

Only the on-axis target channel H0(f) is used to design each waveform. The
best-scalar-delay baseline is identified on a 4096-point frequency grid, then
frozen and evaluated on the 256-point synthesis grid used for 2-D propagation.
"""

from __future__ import annotations

import numpy as np

from .model import axial_transfer, differential_delay
from .parameters import (
    DELAY_BATCH_SIZE,
    DELAY_GLOBAL_SAMPLES,
    DELAY_LOCAL_SAMPLES,
    DELAY_SEARCH_N_FREQUENCY,
    FREQUENCY_MAX_HZ,
    FREQUENCY_MIN_HZ,
    PISTON_RADIUS_M,
    SOUND_SPEED_M_S,
    SYNTHESIS_N_FREQUENCY,
    TARGET_EQUIVALENT_REL_TOL,
)


def frequency_grid(n_frequency: int) -> np.ndarray:
    if n_frequency < 2:
        raise ValueError("n_frequency must be at least 2")
    return np.linspace(FREQUENCY_MIN_HZ, FREQUENCY_MAX_HZ, n_frequency)


def normalize_spectrum(spectrum: np.ndarray, df: float) -> np.ndarray:
    """Normalize a sampled positive-frequency spectrum to unit integral |V|^2 df."""
    spectrum = np.asarray(spectrum, dtype=complex).reshape(-1)
    energy = float(np.sum(np.abs(spectrum) ** 2) * df)
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("spectrum must have finite positive energy")
    return spectrum / np.sqrt(energy)


def _target_amplitudes_for_delays(
    target_channel: np.ndarray,
    frequency: np.ndarray,
    df: float,
    target_depth: float,
    delays: np.ndarray,
    *,
    batch_size: int = DELAY_BATCH_SIZE,
) -> np.ndarray:
    """Evaluate |p(z0,t=0)| for scalar delays without a large delay-frequency array."""
    h0 = np.asarray(target_channel, dtype=complex).reshape(-1)
    frequency = np.asarray(frequency, dtype=float).reshape(-1)
    delays = np.asarray(delays, dtype=float).reshape(-1)
    if h0.size != frequency.size:
        raise ValueError("target_channel and frequency must have the same length")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    amplitude = np.empty(delays.size, dtype=float)
    for start in range(0, delays.size, batch_size):
        stop = min(start + batch_size, delays.size)
        phase = np.exp(
            1j
            * 2.0
            * np.pi
            * frequency[None, :]
            * (target_depth / SOUND_SPEED_M_S + delays[start:stop, None])
        )
        phase /= np.sqrt(np.sum(np.abs(phase) ** 2, axis=1)[:, None] * df)
        amplitude[start:stop] = np.abs((phase * h0[None, :]).sum(axis=1) * df)
    return amplitude


def scalar_delay_candidates(target_depth: float) -> tuple[list[dict[str, float]], float]:
    """Find all refined local maxima of the scalar-delay target objective."""
    f = frequency_grid(DELAY_SEARCH_N_FREQUENCY)
    df = float(f[1] - f[0])
    h0 = axial_transfer(
        f,
        np.array([target_depth]),
        PISTON_RADIUS_M,
        SOUND_SPEED_M_S,
    )[0]
    aperture_delay = float(
        differential_delay(target_depth, PISTON_RADIUS_M, SOUND_SPEED_M_S)
    )
    delay_min = -aperture_delay
    delay_max = 2.0 * aperture_delay

    delays = np.linspace(delay_min, delay_max, DELAY_GLOBAL_SAMPLES)
    target_amp = _target_amplitudes_for_delays(h0, f, df, target_depth, delays)

    interior = (
        np.flatnonzero(
            (target_amp[1:-1] >= target_amp[:-2])
            & (target_amp[1:-1] >= target_amp[2:])
        )
        + 1
    )
    indices = [int(i) for i in interior]
    if target_amp[0] >= target_amp[1]:
        indices.append(0)
    if target_amp[-1] >= target_amp[-2]:
        indices.append(target_amp.size - 1)
    indices.append(int(np.argmax(target_amp)))
    indices = sorted(set(indices))

    coarse_step = float(delays[1] - delays[0])
    candidates: list[dict[str, float]] = []
    for index in indices:
        lo = max(delay_min, float(delays[index] - coarse_step))
        hi = min(delay_max, float(delays[index] + coarse_step))
        fine = np.linspace(lo, hi, DELAY_LOCAL_SAMPLES)
        fine_amp = _target_amplitudes_for_delays(h0, f, df, target_depth, fine)
        best = int(np.argmax(fine_amp))
        candidates.append(
            {
                "delay_s": float(fine[best]),
                "target_amplitude": float(fine_amp[best]),
            }
        )
    return candidates, aperture_delay


def select_figure_delay(target_depth: float) -> tuple[float, dict]:
    """Select the direct-path-centred target-equivalent scalar-delay branch."""
    candidates, aperture_delay = scalar_delay_candidates(target_depth)
    best_target = max(item["target_amplitude"] for item in candidates)
    equivalent = [
        item
        for item in candidates
        if item["target_amplitude"]
        >= best_target * (1.0 - TARGET_EQUIVALENT_REL_TOL)
    ]
    chosen = min(equivalent, key=lambda item: abs(item["delay_s"]))
    delay = float(chosen["delay_s"])
    metadata = {
        "aperture_delay_s": aperture_delay,
        "chosen_delay_s": delay,
        "chosen_delay_over_aperture_delay": delay / aperture_delay,
        "best_target_amplitude": best_target,
        "relative_target_tolerance": TARGET_EQUIVALENT_REL_TOL,
        "target_equivalent_candidates": equivalent,
        "all_local_maxima": candidates,
        "search_n_frequency": DELAY_SEARCH_N_FREQUENCY,
        "global_search_samples": DELAY_GLOBAL_SAMPLES,
        "local_refinement_samples": DELAY_LOCAL_SAMPLES,
        "branch_rule": "minimum absolute extra delay among target-equivalent maxima",
    }
    return delay, metadata


def design_spectra(target_depth: float) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Return the four excitation spectra used by the paper figures."""
    f = frequency_grid(SYNTHESIS_N_FREQUENCY)
    df = float(f[1] - f[0])
    h0 = axial_transfer(
        f,
        np.array([target_depth]),
        PISTON_RADIUS_M,
        SOUND_SPEED_M_S,
    )[0]

    delay, delay_meta = select_figure_delay(target_depth)
    spectra = {
        "plain": normalize_spectrum(
            np.exp(1j * 2.0 * np.pi * f * target_depth / SOUND_SPEED_M_S),
            df,
        ),
        "best_delay": normalize_spectrum(
            np.exp(
                1j
                * 2.0
                * np.pi
                * f
                * (target_depth / SOUND_SPEED_M_S + delay)
            ),
            df,
        ),
        "phase": normalize_spectrum(np.exp(-1j * np.angle(h0)), df),
        "mf": normalize_spectrum(np.conj(h0), df),
    }
    return f, spectra, {"best_delay": delay_meta}
