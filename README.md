# Single-piston transient focusing — reproducibility package

This repository contains the **code and small numerical reference data needed to reproduce the numerical results and eight figures associated with the single-piston transient-focusing study**.

The manuscript, third-party literature PDFs, internal research notes, exploratory experiments, and development history are intentionally not included.

## Reproduce the results

Requirements: **Python 3.12 or 3.13** and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/hightechlog/single-piston-transient-focusing-reproducibility.git
cd single-piston-transient-focusing-reproducibility
uv sync
```

First run the smoke test:

```bash
uv run python reproduce.py --quick
```

`--quick` uses reduced numerical settings only to verify the installation and the model/design/propagation pipeline. It is **not** a reproduction of the publication results.

To reproduce the publication calculations and all eight figures:

```bash
uv run python reproduce.py --full
```

Use a specific number of worker threads if desired:

```bash
uv run python reproduce.py --full --workers 4
```

A successful FULL run also compares the generated numerical summary with the version-controlled canonical reference and ends with `REFERENCE VALIDATION: PASSED`. A material numerical difference causes the command to fail and reports the mismatched quantity.

Generated files are written to:

```text
output/
├── figures/
│   ├── figure_01.pdf
│   ├── ...
│   └── figure_08.pdf
└── summary.json
```

The expensive axisymmetric Rayleigh operator is cached under `.cache/` and can always be regenerated. Neither `output/` nor `.cache/` is version-controlled.

A representative clean FULL run used approximately **430 MB peak RAM** and **18 s wall time with four worker threads**. Runtime varies with hardware and numerical libraries.

## Numerical model

The study uses a uniformly driven baffled circular piston with the time convention `exp(+j omega t)`. The reduced on-axis transfer function is

```text
H(z,f) = exp(-j k z) - exp[-j k sqrt(z^2+a^2)].
```

The nominal parameters are defined once in `transient_focus/parameters.py`. The published calculations use a piston radius of 50 mm, sound speed 1500 m/s, center frequency 100 kHz, and a 50–150 kHz band.

Only the on-axis target channel is used to design the excitation waveform. Full axisymmetric fields are then evaluated independently with the Rayleigh integral.

## Figure palette

The reproduction package explicitly pins the colors used in the paper figures so that their appearance does not depend on Matplotlib defaults:

- principal analytical and transient curves: black `#000000`;
- CW reference: 45% gray (approximately `#737373`);
- guide lines in the cut figure (`-3 dB`, `-6 dB`, and the reference position): Matplotlib C0 blue `#1F77B4`;
- operating-band shading: 88% gray (approximately `#E0E0E0`);
- field maps: Matplotlib `gray`, with **white = 0 dB** and **black = -20 dB**;
- grid lines: `#B0B0B0` with the per-axis alpha set in the plotting code.

The repository-level `matplotlibrc` pins the remaining presentation defaults used by these figures. Numerical calculations are unaffected by this file.

## Repository structure

```text
reproduce.py                    # single user entry point
matplotlibrc                    # pinned publication figure colors
transient_focus/
├── parameters.py               # numerical parameters used by the paper
├── model.py                    # reduced axial piston model
├── design.py                   # excitation-waveform construction
├── propagation.py              # axisymmetric Rayleigh propagation
├── figures.py                  # the eight publication figures
└── validation.py               # comparison with canonical numerical values
reference/
└── reference_summary.json      # small canonical numerical reference
tests/                          # physical/numerical regression tests
LICENSE                         # BSD-3-Clause
CITATION.cff                    # software citation metadata
```

The code is intentionally compact: historical `expXX` modules and calculations not used by the final paper are excluded.

## Tests

```bash
uv run pytest -q
```

The tests check the analytical factorization of the piston model, the scalar-delay branches used by the figures, spectrum normalization, consistency between the axisymmetric Rayleigh calculation and the analytical axial model, and the reference-validation mechanism.

## Reproducibility reference

`reference/reference_summary.json` contains a small set of canonical numerical values from the calculation used for the paper. It stores numerical checks only; no manuscript content is included.

## License and citation

The software is released under the **BSD-3-Clause** license. Citation metadata are provided in `CITATION.cff`.

The final `v1.0.0` release tag and archival DOI will be added when the reproducibility package is frozen for submission.
