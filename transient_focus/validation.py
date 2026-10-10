"""Validation of FULL reproduction results against canonical paper values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


FIELD_SECTIONS = (
    "nominal_peak_fields",
    "retargeting_matched_instantaneous_fields",
    "retargeting_matched_peak_fields",
)
FIELD_METRICS = (
    "target_amplitude",
    "global_peak_amplitude",
    "global_peak_r_m",
    "global_peak_z_m",
    "global_peak_over_target_db",
)


def load_reference(path: str | Path) -> dict[str, Any]:
    """Load the small version-controlled reference summary."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_summary(generated: dict[str, Any], reference: dict[str, Any]) -> dict[str, float | int]:
    """Compare the generated FULL summary with canonical values.

    Floating-point values use the tolerances stored in the reference file.
    Discrete grid sizes and quadrature orders must match exactly.
    """
    rtol = float(reference["comparison"]["rtol"])
    atol = float(reference["comparison"]["atol"])
    mismatches: list[str] = []
    max_abs = 0.0
    max_rel = 0.0
    comparisons = 0

    def compare_float(label: str, actual: float, expected: float) -> None:
        nonlocal max_abs, max_rel, comparisons
        actual = float(actual)
        expected = float(expected)
        abs_error = abs(actual - expected)
        rel_error = abs_error / max(abs(expected), np.finfo(float).tiny)
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)
        comparisons += 1
        if not np.isclose(actual, expected, rtol=rtol, atol=atol):
            mismatches.append(
                f"{label}: expected {expected:.17g}, got {actual:.17g}"
            )

    def compare_exact(label: str, actual: Any, expected: Any) -> None:
        nonlocal comparisons
        comparisons += 1
        if actual != expected:
            mismatches.append(f"{label}: expected {expected!r}, got {actual!r}")

    for index, expected in enumerate(reference["frequency_band_hz"]):
        compare_float(f"frequency_band_hz[{index}]", generated["frequency_band_hz"][index], expected)
    compare_float("center_frequency_hz", generated["center_frequency_hz"], reference["center_frequency_hz"])

    for key in ("n_rho", "n_phi"):
        compare_exact(f"quadrature.{key}", generated["quadrature"][key], reference["quadrature"][key])
    for key, expected in reference["full_grid"].items():
        if key in ("n_r", "n_z"):
            compare_exact(f"full_grid.{key}", generated["full_grid"][key], expected)
        else:
            compare_float(f"full_grid.{key}", generated["full_grid"][key], expected)
    for key, expected in reference["temporal_peak_reconstruction"].items():
        actual = generated["temporal_peak_reconstruction"][key]
        if key == "nfft":
            compare_exact(f"temporal_peak_reconstruction.{key}", actual, expected)
        else:
            compare_float(f"temporal_peak_reconstruction.{key}", actual, expected)

    for target, expected in reference["chosen_scalar_delays_s"].items():
        actual = generated["scalar_delay_metadata"][target]["chosen_delay_s"]
        compare_float(f"chosen_scalar_delays_s.{target}", actual, expected)

    for section in FIELD_SECTIONS:
        for case, expected_metrics in reference[section].items():
            actual_metrics = generated[section][case]
            for metric in FIELD_METRICS:
                compare_float(
                    f"{section}.{case}.{metric}",
                    actual_metrics[metric],
                    expected_metrics[metric],
                )

    if mismatches:
        raise AssertionError(
            "FULL reproduction differs from the canonical reference:\n  "
            + "\n  ".join(mismatches)
        )
    return {
        "comparisons": comparisons,
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
    }
