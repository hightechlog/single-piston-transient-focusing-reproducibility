"""Single entry point for the paper reproducibility package.

Examples
--------
uv run python reproduce.py --quick
uv run python reproduce.py --full
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from transient_focus.design import design_spectra
from transient_focus.figures import (
    METHODS_DESIGN,
    METHODS_TRANSIENT,
    figure_01,
    figure_02,
    figure_03,
    figure_04,
    figure_05,
    figure_06,
    figure_07,
    figure_08,
    set_style,
)
from transient_focus.model import axial_transfer
from transient_focus.parameters import (
    CENTER_FREQUENCY_HZ,
    DR_M,
    DRIVE_N_TIME,
    DRIVE_TIME_WINDOW_US,
    DZ_M,
    NOMINAL_TARGET_M,
    NOMINAL_Z_MAX_M,
    N_PHI,
    N_RHO,
    PEAK_OVERSAMPLE,
    PISTON_RADIUS_M,
    R_MAX_M,
    R_MIN_M,
    SOUND_SPEED_M_S,
    TARGET_DEPTHS_M,
    Z_MAX_M,
    Z_MIN_M,
    ZT_N_TIME,
    ZT_TIME_WINDOW_US,
)
from transient_focus.propagation import (
    field_at_time,
    load_or_build_transfer,
    peak_analytic_envelope,
    rayleigh_transfer_at_frequency,
    rayleigh_transfer_matrix,
    resolve_workers,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = ROOT / ".cache"


def response_over_time(
    transfer: np.ndarray,
    spectrum: np.ndarray,
    frequency: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    """Positive-frequency response used for the displayed on-axis z-t maps."""
    transfer = np.asarray(transfer, dtype=complex)
    spectrum = np.asarray(spectrum, dtype=complex).reshape(-1)
    frequency = np.asarray(frequency, dtype=float).reshape(-1)
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    if transfer.shape[-1] != frequency.size or spectrum.size != frequency.size:
        raise ValueError("incompatible transfer, spectrum, and frequency dimensions")
    df = float(frequency[1] - frequency[0])
    phase = np.exp(1j * 2.0 * np.pi * frequency[:, None] * time_s[None, :])
    original_shape = transfer.shape[:-1]
    flat = transfer.reshape(-1, frequency.size)
    return (((flat * spectrum[None, :]) @ phase) * df).reshape(
        *original_shape, time_s.size
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the numerical results and figures of the single-piston study."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run a small smoke test only.")
    mode.add_argument("--full", action="store_true", help="Reproduce all eight publication figures.")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Worker threads; 0 uses logical CPUs minus one.",
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="Rebuild the full Rayleigh operator cache.",
    )
    return parser.parse_args()


def field_summary(
    amplitude: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    target_depth: float,
) -> dict[str, float]:
    """Return compact location/amplitude metadata for one field."""
    amplitude = np.asarray(amplitude, dtype=float)
    target_index = int(np.argmin(np.abs(z - target_depth)))
    peak = np.unravel_index(int(np.argmax(amplitude)), amplitude.shape)
    target = max(float(amplitude[0, target_index]), np.finfo(float).tiny)
    return {
        "target_amplitude": float(amplitude[0, target_index]),
        "global_peak_amplitude": float(amplitude[peak]),
        "global_peak_r_m": float(r[peak[0]]),
        "global_peak_z_m": float(z[peak[1]]),
        "global_peak_over_target_db": float(
            20.0 * np.log10(float(amplitude[peak]) / target)
        ),
    }


def run_quick(workers: int) -> None:
    """Exercise the real model/design/propagation path on a small grid."""
    print("QUICK smoke test")
    frequency, _, metadata = design_spectra(NOMINAL_TARGET_M)

    # Reduced settings verify the implementation but are not publication data.
    f_small = np.linspace(float(frequency[0]), float(frequency[-1]), 48)
    r = np.array([0.0, 0.010, 0.020])
    z = np.array([0.030, 0.040, 0.050, 0.060, 0.070])
    transfer = rayleigh_transfer_matrix(
        f_small,
        r,
        z,
        n_rho=16,
        n_phi=32,
        point_batch=4,
        workers=resolve_workers(workers),
    )
    analytic_axis = axial_transfer(f_small, z, PISTON_RADIUS_M, SOUND_SPEED_M_S)
    axis_error = float(
        np.linalg.norm(transfer[0] - analytic_axis) / np.linalg.norm(analytic_axis)
    )

    # Matched waveform on the reduced smoke-test frequency grid.
    df = float(f_small[1] - f_small[0])
    target_index = int(np.argmin(np.abs(z - NOMINAL_TARGET_M)))
    matched = np.conj(analytic_axis[target_index])
    matched /= np.sqrt(float(np.sum(np.abs(matched) ** 2) * df))
    snapshot = 2.0 * np.abs(field_at_time(transfer, matched, f_small, 0.0))

    if not np.all(np.isfinite(snapshot)) or axis_error >= 2e-3:
        raise RuntimeError("QUICK smoke test failed")

    summary = {
        "mode": "quick_smoke_test",
        "status": "passed",
        "axis_relative_l2_error": axis_error,
        "best_delay_s": float(metadata["best_delay"]["chosen_delay_s"]),
        "grid": {"n_r": int(r.size), "n_z": int(z.size), "n_f": int(f_small.size)},
        "note": "Reduced smoke-test settings; not publication results.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "quick_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Axis-model relative L2 error: {axis_error:.3e}")
    print("QUICK reproducibility check: PASSED")


def run_full(workers: int, rebuild_cache: bool) -> None:
    """Reproduce all eight figures with the numerical settings used in the paper."""
    set_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n_workers = resolve_workers(workers)

    print("Generating Figures 1-2 (analytical mechanism)", flush=True)
    figure_01(FIGURE_DIR)
    figure_02(FIGURE_DIR)

    r = np.arange(R_MIN_M, R_MAX_M + DR_M / 2.0, DR_M)
    z_full = np.arange(Z_MIN_M, Z_MAX_M + DZ_M / 2.0, DZ_M)

    spectra_by_target: dict[float, dict[str, np.ndarray]] = {}
    control_metadata: dict[float, dict] = {}
    frequency: np.ndarray | None = None
    for target in TARGET_DEPTHS_M:
        f_i, spectra_i, metadata_i = design_spectra(target)
        if frequency is None:
            frequency = f_i
        elif not np.array_equal(frequency, f_i):
            raise RuntimeError("Frequency grids differ across target depths")
        spectra_by_target[target] = spectra_i
        control_metadata[target] = metadata_i
    assert frequency is not None
    spectra_nominal = spectra_by_target[NOMINAL_TARGET_M]

    cache_path = CACHE_DIR / "rayleigh_full.npz"
    print(
        f"Building/loading Rayleigh operator: {r.size} x {z_full.size} x "
        f"{frequency.size}; workers={n_workers}",
        flush=True,
    )
    transfer_full = load_or_build_transfer(
        cache_path,
        frequency,
        r,
        z_full,
        point_batch=32,
        workers=n_workers,
        rebuild=rebuild_cache,
    )

    nominal_mask = (
        (z_full >= Z_MIN_M - 1e-12)
        & (z_full <= NOMINAL_Z_MAX_M + 1e-12)
    )
    z_nominal = z_full[nominal_mask]
    transfer_nominal = transfer_full[:, nominal_mask, :]
    target_index_full = int(np.argmin(np.abs(z_full - NOMINAL_TARGET_M)))

    snapshots = {
        name: 2.0 * np.abs(
            field_at_time(transfer_nominal, spectra_nominal[name], frequency, 0.0)
        )
        for name in METHODS_TRANSIENT
    }

    time_zt = np.linspace(
        -ZT_TIME_WINDOW_US * 1e-6,
        ZT_TIME_WINDOW_US * 1e-6,
        ZT_N_TIME,
    )
    zt: dict[str, np.ndarray] = {}
    for name in METHODS_DESIGN:
        print(f"Computing on-axis z-t envelope: {name}", flush=True)
        zt[name] = 2.0 * np.abs(
            response_over_time(
                transfer_nominal[0], spectra_nominal[name], frequency, time_zt
            )
        )

    print("Computing nominal peak-over-time fields", flush=True)
    peak_mf_full, _, temporal_metadata = peak_analytic_envelope(
        transfer_full,
        spectra_nominal["mf"],
        frequency,
        oversample=PEAK_OVERSAMPLE,
        point_batch=64,
        workers=n_workers,
    )
    peaks_nominal = {"mf": peak_mf_full[:, nominal_mask]}
    for name in ("best_delay", "phase"):
        peak, _, _ = peak_analytic_envelope(
            transfer_nominal,
            spectra_nominal[name],
            frequency,
            oversample=PEAK_OVERSAMPLE,
            point_batch=64,
            workers=n_workers,
        )
        peaks_nominal[name] = peak

    print("Computing exact CW reference", flush=True)
    cw_full = np.abs(
        rayleigh_transfer_at_frequency(
            CENTER_FREQUENCY_HZ, r, z_full, n_rho=N_RHO, n_phi=N_PHI
        )
    )
    cw_nominal = cw_full[:, nominal_mask]

    print("Generating Figures 3-6", flush=True)
    figure_03(FIGURE_DIR, r, z_nominal, time_zt, snapshots, zt)
    figure_04(FIGURE_DIR, r, z_nominal, cw_nominal, peaks_nominal)
    figure_05(FIGURE_DIR, r, z_nominal, snapshots, cw_nominal, peaks_nominal)
    figure_06(
        FIGURE_DIR,
        transfer_full[0, target_index_full, :],
        frequency,
        spectra_nominal,
        DRIVE_TIME_WINDOW_US,
        DRIVE_N_TIME,
    )

    print("Computing retargeting fields", flush=True)
    broadband_snapshot = 2.0 * np.abs(
        field_at_time(transfer_full, spectra_nominal["plain"], frequency, 0.0)
    )
    matched_snapshots = {
        target: 2.0 * np.abs(
            field_at_time(
                transfer_full, spectra_by_target[target]["mf"], frequency, 0.0
            )
        )
        for target in TARGET_DEPTHS_M
    }
    figure_07(
        FIGURE_DIR, r, z_full, cw_full, broadband_snapshot, matched_snapshots
    )

    print("Computing retargeting peak-over-time fields", flush=True)
    broadband_peak, _, _ = peak_analytic_envelope(
        transfer_full,
        spectra_nominal["plain"],
        frequency,
        oversample=PEAK_OVERSAMPLE,
        point_batch=64,
        workers=n_workers,
    )
    matched_peaks: dict[float, np.ndarray] = {NOMINAL_TARGET_M: peak_mf_full}
    for target in TARGET_DEPTHS_M:
        if np.isclose(target, NOMINAL_TARGET_M):
            continue
        matched_peaks[target], _, _ = peak_analytic_envelope(
            transfer_full,
            spectra_by_target[target]["mf"],
            frequency,
            oversample=PEAK_OVERSAMPLE,
            point_batch=64,
            workers=n_workers,
        )
    figure_08(FIGURE_DIR, r, z_full, cw_full, broadband_peak, matched_peaks)

    summary = {
        "mode": "full_paper_reproduction",
        "definitions": {
            "positive_frequency_field": "p_+(r,z,t)=integral H(r,z,f)V(f)exp(+j2pi f t)df",
            "analytic_pressure": "p_a=2p_+",
            "instantaneous_envelope": "A=2|p_+|",
            "peak_envelope": "A_max=max_t 2|p_+|",
            "cw_reference": "|H(r,z,f0)|",
        },
        "frequency_band_hz": [float(frequency[0]), float(frequency[-1])],
        "center_frequency_hz": float(CENTER_FREQUENCY_HZ),
        "targets_m": [float(value) for value in TARGET_DEPTHS_M],
        "quadrature": {"n_rho": N_RHO, "n_phi": N_PHI},
        "full_grid": {
            "n_r": int(r.size), "n_z": int(z_full.size),
            "r_min_m": float(r[0]), "r_max_m": float(r[-1]),
            "z_min_m": float(z_full[0]), "z_max_m": float(z_full[-1]),
        },
        "temporal_peak_reconstruction": temporal_metadata,
        "workers_resolved": n_workers,
        "scalar_delay_metadata": {
            f"{int(round(target*1e3))}mm": control_metadata[target]["best_delay"]
            for target in TARGET_DEPTHS_M
        },
        "nominal_peak_fields": {
            "cw": field_summary(cw_nominal, r, z_nominal, NOMINAL_TARGET_M),
            **{
                name: field_summary(peaks_nominal[name], r, z_nominal, NOMINAL_TARGET_M)
                for name in ("best_delay", "phase", "mf")
            },
        },
        "retargeting_matched_instantaneous_fields": {
            f"{int(round(target*1e3))}mm": field_summary(
                matched_snapshots[target], r, z_full, target
            )
            for target in TARGET_DEPTHS_M
        },
        "retargeting_matched_peak_fields": {
            f"{int(round(target*1e3))}mm": field_summary(
                matched_peaks[target], r, z_full, target
            )
            for target in TARGET_DEPTHS_M
        },
        "outputs": [f"figures/figure_{number:02d}.pdf" for number in range(1, 9)],
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("FULL paper reproduction: COMPLETE")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Summary: {OUTPUT_DIR / 'summary.json'}")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.quick:
        run_quick(args.workers)
    else:
        run_full(args.workers, args.rebuild_cache)


if __name__ == "__main__":
    main()
