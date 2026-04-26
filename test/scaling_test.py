import os
import time
import sys
from pathlib import Path

import numpy as np

# Ensure repository root is importable whether this script is launched from root or test/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from airdbm_api import TestAirfoils

# =============================================================================
# USER CONFIGURATION HEADER
# =============================================================================

# --- Optimization Setup ---
M_OBJECTIVES = 2                  # 1 for single-objective (Cl/Cd max), 2 for bi-objective (+ delta_alpha)
D_DIMENSIONS = 12                 # Number of design parameters (must be <= available baselines)

# --- Scaling Test Workload ---
WEAK_CANDIDATES_PER_WORKER = 12   # Number of evaluations assigned per CPU core for weak scaling
STRONG_TOTAL_CANDIDATES = 384     # Total number of evaluations for strong scaling

# --- Design-by-Morphing (DbM) Setup ---
WEIGHT_RANGE = [0.0, 1.0] # Interpolative morphing weight range for each baseline
NORMALIZATION = 'ABS_SUM' # Options: 'SUM', 'ABS_SUM', or None

# --- XFOIL Evaluation Setup ---
XFOIL_BACKEND = 'auto' # Options: 'apptainer', 'native', or 'auto'
XFOIL_STRICT = True  # True: Hard crash on XFOIL errors. False: Attach error and return 0.0
XFOIL_TIMEOUT = 60.0 # Timeout in seconds for XFOIL evaluation per candidate (0.0 for no timeout)
REYNOLDS = 1e6
MACH = 0.0
N_CRIT = 9
ALFA_START = 0.0
ALFA_END = 45.0
# =============================================================================

def _power_of_two_workers(cpu_total: int) -> list[int]:
    workers = []
    w = 1
    while w <= cpu_total:
        workers.append(w)
        w *= 2
    return workers


def _build_hot_basis_samples(d: int) -> np.ndarray:
    return np.eye(d, dtype=float) * 1.0 / (WEIGHT_RANGE[1] - WEIGHT_RANGE[0]) \
              + (0.0 - WEIGHT_RANGE[0]) / (WEIGHT_RANGE[1] - WEIGHT_RANGE[0])


def _prepend_hot_basis(random_weights: np.ndarray) -> np.ndarray:
    d = random_weights.shape[1]
    hot_basis = _build_hot_basis_samples(d)
    return np.vstack((hot_basis, random_weights))


def _run_case(candidate_weights: np.ndarray, workers: int, base_args: dict, m: int) -> tuple[float, list]:
    args = dict(base_args)
    args['parallel'] = workers > 1
    args['max_workers'] = workers

    t0 = time.perf_counter()
    results = TestAirfoils(candidate_weights, args=args, m=m)
    t1 = time.perf_counter()
    return (t1 - t0), results


def _format_result(value) -> str:
    if hasattr(value, 'objectives'):
        value = value.objectives
    if isinstance(value, (list, tuple, np.ndarray)):
        values = np.asarray(value, dtype=float).tolist()
        return "[" + ", ".join(f"{v:.6g}" for v in values) + "]"
    if value is None:
        return "None"
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def _print_aggregated_basis_results(records: list, title: str) -> None:
    if not records:
        return
        
    # Sort by basis index (ascending), then by worker count (ascending)
    records.sort(key=lambda x: (x[0], x[1]))
    
    print(title)
    print("basis | workers | result")
    print("-" * 50)
    for basis_idx, workers, result in records:
        print(f"{basis_idx:5d} | {workers:7d} | {_format_result(result)}")
    print()


def run_scaling_tests() -> None:
    cpu_total = os.cpu_count() or 1
    workers_list = _power_of_two_workers(cpu_total)

    strong_total_candidates = STRONG_TOTAL_CANDIDATES   

    rng = np.random.default_rng(1)

    base_args = {
        'airfoil_db_dir': str(REPO_ROOT / 'airfoilDB'),
        'dbm_weight_range': WEIGHT_RANGE,
        'dbm_normalization': NORMALIZATION,
        'xfoil_evaluation': True,
        'xfoil_backend': XFOIL_BACKEND,
        'xfoil_strict': XFOIL_STRICT,
        'xfoil_timeout': XFOIL_TIMEOUT,
        'reynolds': REYNOLDS,
        'mach': MACH,
        'n_crit': N_CRIT,
        'alfa_start': ALFA_START,
        'alfa_end': ALFA_END,
    }

    print(f"CPU cores available: {cpu_total}")
    print(f"Worker counts (powers of 2): {workers_list}")
    print(f"Objective Mode (m): {M_OBJECTIVES}")
    print()

    # Lists to accumulate (basis_idx, worker_count, result) tuples
    weak_hot_basis_records = []
    strong_hot_basis_records = []

    print("=== Weak Scaling (fixed candidates per worker) ===")
    print(f"Candidates per worker: {WEAK_CANDIDATES_PER_WORKER}")
    print("workers | candidates | time_seconds | throughput_eval_per_sec")
    print("-" * 61)
    
    for workers in workers_list:
        n_random_candidates = max(0, workers * WEAK_CANDIDATES_PER_WORKER - D_DIMENSIONS)
        random_weights = rng.uniform(0.0, 1.0, size=(n_random_candidates, D_DIMENSIONS))
        candidate_weights = _prepend_hot_basis(random_weights)
        
        elapsed, results = _run_case(candidate_weights, workers, base_args, m=M_OBJECTIVES)
        throughput = len(results) / elapsed if elapsed > 1e-12 else np.inf
        print(f"{workers:7d} | {len(results):10d} | {elapsed:11.3f} | {throughput:21.2f}")
        
        # Accumulate hot basis results for this worker count
        if len(results) >= D_DIMENSIONS:
            for i in range(D_DIMENSIONS):
                weak_hot_basis_records.append((i, workers, results[i].objectives))

    print()
    print("=== Strong Scaling (fixed total candidates) ===")
    print(f"Total candidates: {strong_total_candidates}")
    print("workers | candidates | time_seconds | speedup_vs_1 | efficiency")
    print("-" * 63)

    n_random_strong = max(0, strong_total_candidates - D_DIMENSIONS)
    candidate_weights_strong = _prepend_hot_basis(rng.uniform(0.0, 1.0, size=(n_random_strong, D_DIMENSIONS)))
    
    strong_baseline_time = None

    for workers in workers_list:
        elapsed, results = _run_case(candidate_weights_strong, workers, base_args, m=M_OBJECTIVES)
        
        if workers == 1:
            strong_baseline_time = elapsed
            speedup = 1.0
            efficiency = 1.0
        else:
            speedup = (strong_baseline_time / elapsed) if strong_baseline_time and elapsed > 1e-12 else np.inf
            efficiency = speedup / workers if np.isfinite(speedup) else np.inf

        print(f"{workers:7d} | {len(results):10d} | {elapsed:11.3f} | {speedup:12.2f} | {efficiency:10.3f}")
        
        # Accumulate hot basis results for this worker count
        if len(results) >= D_DIMENSIONS:
            for i in range(D_DIMENSIONS):
                strong_hot_basis_records.append((i, workers, results[i].objectives))

    # --- PRINT STORED HOT BASIS RESULTS AT THE END ---
    print("\n" + "="*60)
    print("HOT BASIS EVALUATION SUMMARY")
    print("="*60 + "\n")
    
    _print_aggregated_basis_results(weak_hot_basis_records, title="Hot basis results (Weak Scaling)")
    _print_aggregated_basis_results(strong_hot_basis_records, title="Hot basis results (Strong Scaling)")

if __name__ == '__main__':
    run_scaling_tests()