import numpy as np

from transient_focus.design import design_spectra, select_figure_delay

EXPECTED_DELAYS = {
    0.020: -1.1015890476550126e-07,
    0.050: 3.619881357172189e-07,
    0.100: 2.158054760977254e-07,
    0.150: 2.704627669472989e-06,
}


def test_scalar_delay_branches_match_published_pipeline():
    for target, expected in EXPECTED_DELAYS.items():
        delay, metadata = select_figure_delay(target)
        assert np.isclose(delay, expected, rtol=0.0, atol=2e-10)
        assert metadata["branch_rule"].startswith("minimum absolute extra delay")


def test_all_published_spectra_have_unit_energy():
    f, spectra, _ = design_spectra(0.050)
    df = float(f[1] - f[0])
    assert set(spectra) == {"plain", "best_delay", "phase", "mf"}
    for spectrum in spectra.values():
        assert np.isclose(np.sum(np.abs(spectrum) ** 2) * df, 1.0, atol=2e-14)
