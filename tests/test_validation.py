from copy import deepcopy
from pathlib import Path

import pytest

from transient_focus.validation import load_reference, validate_summary


REFERENCE = Path(__file__).resolve().parents[1] / "reference" / "reference_summary.json"


def _generated_from_reference(reference: dict) -> dict:
    return {
        "frequency_band_hz": list(reference["frequency_band_hz"]),
        "center_frequency_hz": reference["center_frequency_hz"],
        "quadrature": dict(reference["quadrature"]),
        "full_grid": dict(reference["full_grid"]),
        "temporal_peak_reconstruction": dict(reference["temporal_peak_reconstruction"]),
        "scalar_delay_metadata": {
            target: {"chosen_delay_s": delay}
            for target, delay in reference["chosen_scalar_delays_s"].items()
        },
        "nominal_peak_fields": deepcopy(reference["nominal_peak_fields"]),
        "retargeting_matched_instantaneous_fields": deepcopy(
            reference["retargeting_matched_instantaneous_fields"]
        ),
        "retargeting_matched_peak_fields": deepcopy(
            reference["retargeting_matched_peak_fields"]
        ),
    }


def test_reference_accepts_canonical_values():
    reference = load_reference(REFERENCE)
    generated = _generated_from_reference(reference)
    result = validate_summary(generated, reference)
    assert result["comparisons"] > 70
    assert result["max_absolute_error"] == 0.0


def test_reference_rejects_material_change():
    reference = load_reference(REFERENCE)
    generated = _generated_from_reference(reference)
    generated["nominal_peak_fields"]["mf"]["global_peak_z_m"] += 1e-3
    with pytest.raises(AssertionError, match="global_peak_z_m"):
        validate_summary(generated, reference)
