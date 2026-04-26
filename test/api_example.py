import sys
from pathlib import Path

import numpy as np

# Ensure repository root is importable whether this script is launched from root or test/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from airdbm_api import TestAirfoils

rng = np.random.default_rng(seed=32)

# Example: 4 design candidates (N=4), using 12 input parameters (D=12).
candidate_weights = rng.uniform(0.0, 1.0, size=(4, 12))

# Override some default arguments based on needs
args = {
    'airfoil_db_dir': str(REPO_ROOT / 'airfoilDB'),
    'dbm_weight_range': [-1.0, 1.0], # allow both interpolative and extrapolative morphing during DbM
    'dbm_normalization': 'ABS_SUM', # enforce L1-style normalization: sum(|w_i|) = 1 across baselines
    'reynolds': 1e5, # override the Reynolds number (default is 1e6)
}

# Run the API for Bi-Objective evaluation (m=2)
results = TestAirfoils(candidate_weights, args=args, m=2)

for i, res in enumerate(results):
    cl_cd_max, delta_alpha = (res.objectives if isinstance(res.objectives, list) else [np.nan, np.nan])
    print(
        f"Candidate {i+1} ({res.airfoil.name}) -> "
        f"Cl/Cd max: {cl_cd_max:.2f}, Stall Margin (deg): {delta_alpha:.2f}"
    )
