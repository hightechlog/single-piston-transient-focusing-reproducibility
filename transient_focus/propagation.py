"""Axisymmetric Rayleigh propagation and temporal reconstruction.

The Rayleigh operator is built once on an r-z-frequency grid and reused for all
waveforms. Linear interpolation on a path-length grid reproduces the original
implementation while requiring only NumPy.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from .parameters import (
    N_PHI,
    N_RHO,
    PATH_SAMPLES_PER_MIN_WAVELENGTH,
    PISTON_RADIUS_M,
    SOUND_SPEED_M_S,
)


def resolve_workers(workers: int | None) -> int:
    if workers is None or workers <= 0:
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(workers))


def disk_quadrature(
    radius: float = PISTON_RADIUS_M,
    n_rho: int = N_RHO,
    n_phi: int = N_PHI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return polar quadrature nodes cos(phi) and area weights rho*d(rho)*d(phi)."""
    if radius <= 0.0 or n_rho < 1 or n_phi < 1:
        raise ValueError("radius, n_rho, and n_phi must be positive")
    nodes, weights = leggauss(n_rho)
    rho = 0.5 * radius * (nodes + 1.0)
    w_rho = 0.5 * radius * weights
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    w_phi = 2.0 * np.pi / n_phi
    area_weights = (rho[:, None] * w_rho[:, None]) * w_phi * np.ones((1, n_phi))
    return rho, np.cos(phi), area_weights


def _uniform_frequency_grid(frequency: np.ndarray) -> tuple[float, float]:
    frequency = np.asarray(frequency, dtype=float).reshape(-1)
    if frequency.size < 2 or np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequency must contain at least two finite positive values")
    df = float(frequency[1] - frequency[0])
    if df <= 0.0 or not np.allclose(np.diff(frequency), df, rtol=1e-11, atol=1e-9):
        raise ValueError("frequency must be uniformly sampled and increasing")
    return float(frequency[0]), df


def rayleigh_transfer_at_frequency(
    frequency: float,
    r: np.ndarray,
    z: np.ndarray,
    *,
    radius: float = PISTON_RADIUS_M,
    sound_speed: float = SOUND_SPEED_M_S,
    n_rho: int = N_RHO,
    n_phi: int = N_PHI,
    point_batch: int = 128,
) -> np.ndarray:
    """Evaluate the reduced Rayleigh transfer H(r,z,f) at one frequency."""
    if frequency <= 0.0 or radius <= 0.0 or sound_speed <= 0.0 or point_batch < 1:
        raise ValueError("frequency, radius, sound_speed, and point_batch must be positive")
    r = np.asarray(r, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    if np.any(r < 0.0) or np.any(z <= 0.0):
        raise ValueError("r must be non-negative and z must be positive")

    rho, cos_phi, area_weights = disk_quadrature(radius, n_rho, n_phi)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    points_r = rr.ravel()
    points_z = zz.ravel()
    result = np.empty(points_r.size, dtype=complex)
    k = 2.0 * np.pi * frequency / sound_speed

    for start in range(0, points_r.size, point_batch):
        stop = min(start + point_batch, points_r.size)
        rp = points_r[start:stop, None, None]
        zp = points_z[start:stop, None, None]
        path = np.sqrt(
            zp * zp
            + rp * rp
            + rho[None, :, None] ** 2
            - 2.0 * rp * rho[None, :, None] * cos_phi[None, None, :]
        )
        integral = np.sum(
            (np.exp(-1j * k * path) / path) * area_weights[None, :, :],
            axis=(1, 2),
        )
        result[start:stop] = 1j * k * integral / (2.0 * np.pi)
    return result.reshape(r.size, z.size)


def rayleigh_transfer_matrix(
    frequency: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    *,
    radius: float = PISTON_RADIUS_M,
    sound_speed: float = SOUND_SPEED_M_S,
    n_rho: int = N_RHO,
    n_phi: int = N_PHI,
    point_batch: int = 32,
    workers: int | None = 1,
    samples_per_min_wavelength: float = PATH_SAMPLES_PER_MIN_WAVELENGTH,
) -> np.ndarray:
    """Build H(r,z,f) for a uniformly driven circular piston."""
    frequency = np.asarray(frequency, dtype=float).reshape(-1)
    r = np.asarray(r, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    _uniform_frequency_grid(frequency)
    if np.any(r < 0.0) or np.any(z <= 0.0):
        raise ValueError("r must be non-negative and z must be positive")
    if radius <= 0.0 or sound_speed <= 0.0 or point_batch < 1:
        raise ValueError("radius, sound_speed, and point_batch must be positive")
    if samples_per_min_wavelength <= 0.0:
        raise ValueError("samples_per_min_wavelength must be positive")

    path_min = max(1e-6, float(np.min(z)))
    path_max = float(np.sqrt(np.max(z) ** 2 + (np.max(r) + radius) ** 2))
    lambda_min = sound_speed / float(np.max(frequency))
    d_path_requested = lambda_min / samples_per_min_wavelength
    n_path = int(np.ceil((path_max - path_min) / d_path_requested)) + 1
    path_grid = np.linspace(path_min, path_max, n_path)
    d_path = float(path_grid[1] - path_grid[0])

    rho, cos_phi, area_weights = disk_quadrature(radius, n_rho, n_phi)
    k = 2.0 * np.pi * frequency / sound_speed
    path_kernel = np.exp(-1j * path_grid[:, None] * k[None, :]) * (1j * k[None, :])

    rr, zz = np.meshgrid(r, z, indexing="ij")
    points_r = rr.ravel()
    points_z = zz.ravel()
    result = np.empty((points_r.size, frequency.size), dtype=np.complex128)
    n_aperture = n_rho * n_phi

    def compute(start: int, stop: int) -> tuple[int, int, np.ndarray]:
        rp = points_r[start:stop, None, None]
        zp = points_z[start:stop, None, None]
        path = np.sqrt(
            zp * zp
            + rp * rp
            + rho[None, :, None] ** 2
            - 2.0 * rp * rho[None, :, None] * cos_phi[None, None, :]
        )
        x = np.clip((path - path_min) / d_path, 0.0, path_grid.size - 1.000000000001)
        i0 = np.floor(x).astype(np.int32)
        fraction = x - i0
        i1 = i0 + 1
        spatial_weight = area_weights[None, :, :] / path

        batch_points = stop - start
        rows = np.repeat(np.arange(batch_points, dtype=np.int32), n_aperture)
        interpolation = np.zeros((batch_points, path_grid.size), dtype=float)
        np.add.at(
            interpolation,
            (rows, i0.reshape(-1)),
            (spatial_weight * (1.0 - fraction)).reshape(-1),
        )
        np.add.at(
            interpolation,
            (rows, i1.reshape(-1)),
            (spatial_weight * fraction).reshape(-1),
        )
        block = (interpolation @ path_kernel) / (2.0 * np.pi)
        return start, stop, block

    batches = [
        (start, min(start + point_batch, points_r.size))
        for start in range(0, points_r.size, point_batch)
    ]
    n_workers = resolve_workers(workers)
    if n_workers == 1:
        for start, stop in batches:
            _, _, block = compute(start, stop)
            result[start:stop] = block
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for start, stop, block in executor.map(lambda p: compute(*p), batches):
                result[start:stop] = block
    return result.reshape(r.size, z.size, frequency.size)


def load_or_build_transfer(
    cache_path: str | Path,
    frequency: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    *,
    radius: float = PISTON_RADIUS_M,
    sound_speed: float = SOUND_SPEED_M_S,
    n_rho: int = N_RHO,
    n_phi: int = N_PHI,
    point_batch: int = 32,
    workers: int | None = 1,
    samples_per_min_wavelength: float = PATH_SAMPLES_PER_MIN_WAVELENGTH,
    rebuild: bool = False,
) -> np.ndarray:
    """Load a compatible Rayleigh operator cache or rebuild it deterministically."""
    cache_path = Path(cache_path)
    frequency = np.asarray(frequency, dtype=float).reshape(-1)
    r = np.asarray(r, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)

    if cache_path.exists() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as data:
            compatible = (
                np.array_equal(data["frequency"], frequency)
                and np.array_equal(data["r"], r)
                and np.array_equal(data["z"], z)
                and int(data["n_rho"]) == n_rho
                and int(data["n_phi"]) == n_phi
                and np.isclose(float(data["radius"]), radius)
                and np.isclose(float(data["sound_speed"]), sound_speed)
                and np.isclose(
                    float(data["samples_per_min_wavelength"]),
                    samples_per_min_wavelength,
                )
            )
            if compatible:
                return np.asarray(data["transfer"])

    transfer = rayleigh_transfer_matrix(
        frequency,
        r,
        z,
        radius=radius,
        sound_speed=sound_speed,
        n_rho=n_rho,
        n_phi=n_phi,
        point_batch=point_batch,
        workers=workers,
        samples_per_min_wavelength=samples_per_min_wavelength,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        frequency=frequency,
        r=r,
        z=z,
        transfer=transfer,
        n_rho=np.asarray(n_rho),
        n_phi=np.asarray(n_phi),
        radius=np.asarray(radius),
        sound_speed=np.asarray(sound_speed),
        samples_per_min_wavelength=np.asarray(samples_per_min_wavelength),
    )
    return transfer


def field_at_time(
    transfer: np.ndarray,
    spectrum: np.ndarray,
    frequency: np.ndarray,
    time_s: float = 0.0,
) -> np.ndarray:
    """Return the positive-frequency complex field p_+(r,z,t)."""
    frequency = np.asarray(frequency, dtype=float).reshape(-1)
    spectrum = np.asarray(spectrum, dtype=complex).reshape(-1)
    _, df = _uniform_frequency_grid(frequency)
    if transfer.shape[-1] != frequency.size or spectrum.size != frequency.size:
        raise ValueError("transfer, spectrum, and frequency dimensions are incompatible")
    timed_spectrum = spectrum * np.exp(1j * 2.0 * np.pi * frequency * float(time_s))
    return np.tensordot(transfer, timed_spectrum, axes=([-1], [0])) * df


def real_waveform(
    spectrum: np.ndarray,
    frequency: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    """Reconstruct the real waveform 2*Re{integral V(f) exp(+j2*pi*f*t) df}."""
    frequency = np.asarray(frequency, dtype=float).reshape(-1)
    spectrum = np.asarray(spectrum, dtype=complex).reshape(-1)
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    _, df = _uniform_frequency_grid(frequency)
    phase = np.exp(1j * 2.0 * np.pi * time_s[:, None] * frequency[None, :])
    return 2.0 * np.real(phase @ (spectrum * df))


def temporal_fft_grid(
    frequency: np.ndarray,
    *,
    oversample: float = 8.0,
) -> tuple[np.ndarray, int, float, float]:
    """Return the complete periodic time grid implied by a uniform frequency grid."""
    frequency = np.asarray(frequency, dtype=float).reshape(-1)
    _, df = _uniform_frequency_grid(frequency)
    if oversample < 1.0:
        raise ValueError("oversample must be at least 1")
    min_sample_rate = 2.0 * float(np.max(frequency)) * float(oversample)
    min_nfft = max(frequency.size, int(np.ceil(min_sample_rate / df)))
    nfft = 1 << (min_nfft - 1).bit_length()
    dt = 1.0 / (nfft * df)
    index = np.arange(nfft, dtype=np.int64)
    signed_index = np.where(index <= nfft // 2, index, index - nfft)
    time_s = signed_index.astype(float) * dt
    return time_s, nfft, dt, 1.0 / df


def peak_analytic_envelope(
    transfer: np.ndarray,
    spectrum: np.ndarray,
    frequency: np.ndarray,
    *,
    oversample: float = 8.0,
    point_batch: int = 64,
    workers: int | None = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Return max_t 2|p_+(r,z,t)| and the maximizing time at every field point."""
    frequency = np.asarray(frequency, dtype=float).reshape(-1)
    spectrum = np.asarray(spectrum, dtype=complex).reshape(-1)
    _, df = _uniform_frequency_grid(frequency)
    if transfer.shape[-1] != frequency.size or spectrum.size != frequency.size:
        raise ValueError("transfer, spectrum, and frequency dimensions are incompatible")
    if point_batch < 1:
        raise ValueError("point_batch must be positive")

    time_s, nfft, dt, period = temporal_fft_grid(frequency, oversample=oversample)
    original_shape = transfer.shape[:-1]
    flat = np.asarray(transfer, dtype=complex).reshape(-1, frequency.size)
    peak = np.empty(flat.shape[0], dtype=float)
    peak_time = np.empty(flat.shape[0], dtype=float)

    def compute(start: int, stop: int) -> tuple[int, int, np.ndarray, np.ndarray]:
        work = np.zeros((stop - start, nfft), dtype=np.complex128)
        work[:, : frequency.size] = flat[start:stop] * spectrum[None, :]
        baseband = np.fft.ifft(work, axis=1) * (nfft * df)
        envelope = 2.0 * np.abs(baseband)
        index = np.argmax(envelope, axis=1)
        rows = np.arange(stop - start)
        return start, stop, envelope[rows, index], time_s[index]

    batches = [
        (start, min(start + point_batch, flat.shape[0]))
        for start in range(0, flat.shape[0], point_batch)
    ]
    n_workers = resolve_workers(workers)
    if n_workers == 1:
        outputs = (compute(start, stop) for start, stop in batches)
    else:
        executor = ThreadPoolExecutor(max_workers=n_workers)
        outputs = executor.map(lambda p: compute(*p), batches)

    try:
        for start, stop, values, times in outputs:
            peak[start:stop] = values
            peak_time[start:stop] = times
    finally:
        if n_workers != 1:
            executor.shutdown()

    metadata: dict[str, float | int | str] = {
        "definition": "max_t 2|p_+(r,z,t)|",
        "nfft": int(nfft),
        "time_step_s": float(dt),
        "period_s": float(period),
        "oversample_requested": float(oversample),
        "effective_sample_rate_hz": float(1.0 / dt),
    }
    return peak.reshape(original_shape), peak_time.reshape(original_shape), metadata
