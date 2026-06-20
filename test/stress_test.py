import os
import sys
import time
import json
from pathlib import Path
import numpy as np

# Ensure repository root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from airdbm_api import TestAirfoils

def run_stress_test():
    # Setup parameters
    N_TOTAL = 1000
    D_DIM = 12
    CHUNK_SIZE = 100
    NUM_WORKERS = 64
    
    print("==================================================")
    print("              AIRDBM API STRESS TEST              ")
    print("==================================================")
    print(f"Target: {N_TOTAL} random design candidates ({D_DIM} dimensions)")
    print(f"XFOIL evaluation: Enabled (strict=False)")

    rng = np.random.default_rng(seed=42) # Set seed for reproducibility
    all_weights = rng.uniform(0.0, 1.0, size=(N_TOTAL, D_DIM))
    
    args = {
        'airfoil_db_dir': str(REPO_ROOT / 'airfoilDB'),
        'dbm_weight_range': [0.0, 1.0],
        'dbm_normalization': 'ABS_SUM',
        'reynolds': 1e6,
        'xfoil_evaluation': True,
        'xfoil_backend': 'auto',
        'xfoil_strict': False, # capture errors without crashing
        'xfoil_timeout': 60.0,  # 60s timeout per candidate
        'xfoil_retry': 1,
        'parallel': True,
        'max_workers': NUM_WORKERS,
    }
    
    output_file = REPO_ROOT / 'test' / 'stress_test_results.jsonl'
    
    # Load existing progress if available (allows resuming)
    completed_candidates = []
    if output_file.exists():
        print(f"Found existing results file: {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    completed_candidates.append(json.loads(line))
        print(f"Loaded {len(completed_candidates)} already completed candidates.")
    
    start_idx = len(completed_candidates)
    
    if start_idx >= N_TOTAL:
        print("Stress test already fully completed!")
        print_summary(completed_candidates)
        return
        
    print(f"Starting stress test from candidate index {start_idx}...")
    
    t_start = time.time()
    
    for i in range(start_idx, N_TOTAL, CHUNK_SIZE):
        chunk_end = min(i + CHUNK_SIZE, N_TOTAL)
        chunk_weights = all_weights[i:chunk_end]
        
        print(f"\nProcessing chunk {i//CHUNK_SIZE + 1}/{(N_TOTAL-1)//CHUNK_SIZE + 1} (Candidates {i} to {chunk_end-1})...")
        t0 = time.time()
        
        try:
            results = TestAirfoils(chunk_weights, args=args, m=2)
        except Exception as e:
            print(f"CRITICAL ERROR in TestAirfoils call for chunk {i}-{chunk_end}: {e}")
            sys.exit(1)
            
        t1 = time.time()
        chunk_elapsed = t1 - t0
        print(f"Chunk completed in {chunk_elapsed:.2f} seconds ({chunk_elapsed / len(results):.2f}s per candidate).")
        
        # Append results to file
        with open(output_file, 'a') as f:
            for idx_in_chunk, res in enumerate(results):
                global_idx = i + idx_in_chunk
                
                # Check for objectives
                cl_cd_max = np.nan
                delta_alpha = np.nan
                has_error = False
                error_msg = ""
                
                if res.xfoil_result:
                    cl_cd_max = res.xfoil_result.get('cl_cd_max', np.nan)
                    delta_alpha = res.xfoil_result.get('delta_alpha', np.nan)
                    if 'error' in res.xfoil_result:
                        has_error = True
                        error_msg = res.xfoil_result['error']
                else:
                    has_error = True
                    error_msg = "No XFOIL result attached"
                
                is_non_converged = False
                is_catastrophic_error = False
                if has_error:
                    if error_msg == "All XFOIL scans failed to converge.":
                        is_non_converged = True
                    else:
                        is_catastrophic_error = True

                record = {
                    'index': global_idx,
                    'name': res.airfoil.name,
                    'weights': chunk_weights[idx_in_chunk].tolist(),
                    'cl_cd_max': cl_cd_max if np.isfinite(cl_cd_max) else None,
                    'delta_alpha': delta_alpha if np.isfinite(delta_alpha) else None,
                    'has_error': has_error,
                    'error_msg': error_msg,
                    'is_non_converged': is_non_converged,
                    'is_catastrophic_error': is_catastrophic_error
                }
                f.write(json.dumps(record) + "\n")
                completed_candidates.append(record)
                
        # Print miniature progress update
        chunk_errors = sum(1 for r in completed_candidates[-len(results):] if r['has_error'])
        print(f"Chunk stats: Converged: {len(results) - chunk_errors}/{len(results)}, Failed/Error: {chunk_errors}")
        
    t_end = time.time()
    print(f"\nStress test run completed in {t_end - t_start:.2f} seconds.")
    print_summary(completed_candidates)

def print_summary(records):
    total = len(records)
    if total == 0:
        print("No records to summarize.")
        return
        
    non_converged = sum(1 for r in records if r.get('is_non_converged', r.get('error_msg') == "All XFOIL scans failed to converge."))
    catastrophic = sum(1 for r in records if r.get('is_catastrophic_error', r.get('has_error', False) and r.get('error_msg') != "All XFOIL scans failed to converge."))
    success = total - non_converged - catastrophic
    
    print("\n==================================================")
    print("               STRESS TEST SUMMARY                ")
    print("==================================================")
    print(f"Total candidates evaluated:    {total}")
    print(f"Converged:        {success} ({success/total*100:.1f}%)")
    print(f"Non-converged:  {non_converged} ({non_converged/total*100:.1f}%)")
    print(f"Catastrophic errors/crashes:   {catastrophic} ({catastrophic/total*100:.1f}%)")
    
    # Check for negative delta_alpha values
    negative_margins = sum(1 for r in records if r['delta_alpha'] is not None and r['delta_alpha'] < 0.0)
    print(f"Negative stall margins:        {negative_margins}")
    if negative_margins > 0:
        print("WARNING: Some stall margins were negative!")
    else:
        print("Pass: All stall margins are non-negative.")
        
    print("-" * 50)
    if catastrophic == 0:
        print("Pass: No catastrophic errors encountered during the stress test.")
    else:
        print("FAIL: Catastrophic errors were encountered during the stress test!")
        sys.exit(1)
    print("==================================================")

if __name__ == '__main__':
    run_stress_test()
