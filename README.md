# AirDbM-API

This repository provides a highly robust, parallelized Python interface (`airdbm_api.py`) for generating morphed airfoil geometries using Design-by-Morphing (DbM) and evaluating them dynamically via XFOIL. 

This API is reprocessed from the original [AirDbM](https://github.com/UCBCFD/DbMAirfoilOpt) repository and adapted into a standalone Python workflow for robust batch evaluation.

If this repository contributes to your research, publication, or benchmark, we kindly ask that you credit our work by citing our AirDbM research references:
- Lee, S. & Sheikh, H. M. (2026). Airfoil Optimization using Design-by-Morphing with Minimized Design-Space Dimensionality. *Journal of Computational Design and Engineering*, 13(1), 108-124. [![DOI](https://img.shields.io/badge/DOI-10.1093%2Fjcde%2Fqwaf124-blue)](https://doi.org/10.1093/jcde/qwaf124)
- Sheikh, H. M., Lee, S., Wang, J., & Marcus, P. S. (2023). Airfoil Optimization using Design-by-Morphing. *Journal of Computational Design and Engineering*, 10(4), 1443–1459. [![DOI](https://img.shields.io/badge/DOI-10.1093%2Fjcde%2Fqwad059-blue)](https://doi.org/10.1093/jcde/qwad059)

## Setup & Installation

1. Preliminary requirements:
- Python (tested with `3.10.12`)
- Apptainer (https://apptainer.org/docs/)

2. Install the required Python packages:
~~~bash
pip install -r requirements.txt
~~~

3. Build the Apptainer image for isolated XFOIL execution:
~~~bash
make xfoil-apptainer-build
~~~
*(Verify the container functions correctly by running `make xfoil-apptainer-check`)*

By default, `TestAirfoils` uses the Apptainer backend (`xfoil_backend='apptainer'`) and the Ubuntu 22.04-based XFOIL image built from the container definition (`bin/containers/xfoil-ubuntu22.def`) in this repository. This is the recommended mode for reproducibility and consistency across systems.

You may choose native XFOIL execution by setting `xfoil_backend='native'`.

## How to Use the API

Import `TestAirfoils` from `airdbm_api.py` into your optimization loop or script.

### Function Signature

~~~python
TestAirfoils(x: np.ndarray, args: dict | None = None, m: int = 2) -> list
~~~

### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `x` | `np.ndarray` | Required | Candidate matrix of shape `N x D`, where each candidate parameter is typically in `[0.0, 1.0]`. |
| `args` | `dict` | `{}` | Configuration dictionary for DbM generation, parallelism, and XFOIL execution (see table below). |
| `m` | `int` | `2` | Number of objectives to return when XFOIL is enabled. Supported: `1` or `2`. |

### `args` Configuration Reference

**AIRFOIL DESIGN ARGS**

| Key | Type | Default | Description |
|---|---|---|---|
| `airfoil_db_dir` | `str` | `'airfoilDB'` | Path to the airfoil database folder. |
| `dbm_baselines` | `list[str]` | Internal `EXPECTED_BASELINES` list | Baseline airfoil names. First `D` entries are used for a candidate with `D` parameters. The internal list is based on the optimal baseline set reported in our [2026 paper](https://doi.org/10.1093/jcde/qwaf124). |
| `dbm_weight_range` | `list[float]` | `[-1.0, 1.0]` | Maps input `x` from `[0.0, 1.0]` to morphing weights. |
| `dbm_normalization` | `str \| None` | `None` | Weight normalization mode: `None`, `'SUM'`, or `'ABS_SUM'`. |

**AIRFOIL EVALUATION ARGS**

| Key | Type | Default | Description |
|---|---|---|---|
| `xfoil_evaluation` | `bool` | `True` | If `False`, returns generated `Airfoil` objects without aerodynamic evaluation. |
| `xfoil_backend` | `str` | `'apptainer'` | XFOIL execution backend: `'apptainer'`, `'native'`, or `'auto'`. |
| `xfoil_iter` | `int` | `200` | Max XFOIL iterations per alpha step. |
| `xfoil_timeout` | `float` | `60.0` | Timeout (seconds) per run by default. Set `0.0` for no timeout. |
| `xfoil_retry` | `int` | `1` | Number of reattempts if no XFOIL polar points are parsed at all for a candidate. |
| `xfoil_strict` | `bool` | `True` | If `True`, raise on XFOIL errors; otherwise attach errors in the result payload. |
| `alfa_start` | `float` | `0.0` | Start angle of attack (deg) for scans. |
| `alfa_end` | `float` | `45.0` | End angle of attack (deg) for scans. |
| `reynolds` | `float` | `1e6` | Reynolds number for viscous analysis. |
| `mach` | `float` | `0.0` | Mach number; `0.0` indicates a negligible-compressibility assumption. |
| `n_crit` | `float` | `9.0` | e^N transition amplification factor. |

**MULTIPROCESSING ARGS**

| Key | Type | Default | Description |
|---|---|---|---|
| `parallel` | `bool` | `True` | Enables multiprocessing across candidates. |
| `max_workers` | `int` | `os.cpu_count()` | Maximum worker processes (capped by available CPUs). |

### Return Value

`TestAirfoils` always returns a list of `AirfoilEvaluationResult` objects (length `N`), where each element contains:

| Field | Type | Description |
|---|---|---|
| `.airfoil` | `Airfoil` | Generated morphed airfoil object (always available). |
| `.xfoil_result` | `dict \| None` | Raw XFOIL metrics dictionary when `xfoil_evaluation=True`; otherwise `None`. |
| `.objectives` | `None \| float \| list[float]` | `None` if `xfoil_evaluation=False`; with evaluation enabled: `m=1 -> Cl/Cd_max`, `m=2 -> [Cl/Cd_max, delta_alpha]`. |

The `.airfoil` object exposes both geometry data and helper methods, for example:

- metadata: `.airfoil.airfoil_id`, `.airfoil.name`
- geometry arrays: `.airfoil.x_raw`, `.airfoil.y_raw`
- access helpers: `.airfoil.get_raw_coordinates()`
- quick visualization: `.airfoil.plot(save_path=...)`

As for the objectives,

$$
\left(\frac{C_l}{C_d}\right)_{\max}
= \max_{\alpha}\left(\frac{C_l(\alpha)}{C_d(\alpha)}\right)
$$

This is the maximum lift-to-drag ratio over the evaluated angle-of-attack sweep.

$$
\Delta\alpha = \alpha_{\mathrm{stall}} - \alpha_{\left(\frac{C_l}{C_d}\right)_{\max}}
$$

Here, $\alpha_{\mathrm{stall}}$ is defined as the first local maximum of $C_l$ while marching from low angle of attack (starting at $\alpha = 0$) in the computed polar.

### Python Script Example

~~~python
import numpy as np
from airdbm_api import TestAirfoils

rng = np.random.default_rng(seed=32)

# Example: 4 design candidates (N=4), using 12 input parameters (D=12).
candidate_weights = rng.uniform(0.0, 1.0, size=(4, 12))

# Override some default arguments based on needs
args = {
    'dbm_weight_range': [-1.0, 1.0], # allow both interpolative and extrapolative morphing during DbM
    'dbm_normalization': 'ABS_SUM', # enforce L1-style normalization: sum(|w_i|) = 1 across baselines
    'reynolds': 1e5, # override the Reynolds number (default is 1e6)
}

# Run the API for Bi-Objective evaluation (m=2)
results = TestAirfoils(candidate_weights, args=args, m=2)

for i, res in enumerate(results):
    cl_cd_max, delta_alpha = res.objectives
    print(
        f"Candidate {i+1} ({res.airfoil.name}) -> "
        f"Cl/Cd max: {cl_cd_max:.2f}, Stall Margin (deg): {delta_alpha:.2f}"
    )
~~~

## Customizing Airfoil DbM

You can customize the AirDbM baseline set by providing your own airfoil coordinate files in Selig format (`.dat`).

1. Download or prepare the coordinate files in Selig format.
2. Place the files inside your airfoil database folder, such as `airfoilDB/`, or point `airfoil_db_dir` to a different folder.
3. Explicitly define `dbm_baselines` as an ordered list of airfoil names that matches the database files you want to use.

The order of `dbm_baselines` is important. The first `D` entries are used for a candidate with `D` design parameters, so the baseline ordering must match the intended morphing sequence.

Example:

~~~python
args = {
    'airfoil_db_dir': 'airfoilDB',
    'dbm_baselines': [
        'E195  (11.82%)',
        'FX 79-W-660A',
        'GOE 531 AIRFOIL',
        'EPPLER 864 STRUT AIRFOIL',
    ],
}
~~~

## Parallel Scaling Test

The parallel airfoil design and evaluation performance of `TestAirfoils` was benchmarked using the script in `test/scaling_test.py`.

### Computing Environment

- HPC Platform: RCAC Anvil
- Node type: Two 64-core AMD EPYC Milan processors @ 2.45 GHz (128 cores in total)
- Objective mode: `m=2` (multi-objective)
- Worker counts tested: `1` (serial), `2, 4, 8, 16, 32, 64, 128`

### Weak Scaling

Design candidates per worker: `12`

| workers | candidates | time (sec) | throughput (eval/sec) |
|---:|---:|---:|---:|
| 1 | 12 | 132.796 | 0.09 |
| 2 | 24 | 132.253 | 0.18 |
| 4 | 48 | 131.756 | 0.36 |
| 8 | 96 | 132.335 | 0.73 |
| 16 | 192 | 133.615 | 1.44 |
| 32 | 384 | 135.916 | 2.83 |
| 64 | 768 | 145.104 | 5.29 |
| 128 | 1536 | 182.200 | 8.43 |

### Strong Scaling

Total design candidates: `384`

| workers | candidates | time (sec) | speedup | efficiency |
|---:|---:|---:|---:|---:|
| 1 | 384 | 4022.381 | 1.00 | 1.000 |
| 2 | 384 | 2028.587 | 1.98 | 0.991 |
| 4 | 384 | 1022.814 | 3.93 | 0.983 |
| 8 | 384 | 520.295 | 7.73 | 0.966 |
| 16 | 384 | 264.426 | 15.21 | 0.951 |
| 32 | 384 | 137.485 | 29.26 | 0.914 |
| 64 | 384 | 75.502 | 53.28 | 0.832 |
| 128 | 384 | 53.065 | 75.80 | 0.592 |