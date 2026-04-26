"""
dbm_api.py
API script for DbM Airfoil generation and evaluation.

Currently implemented as an interim version that processes NxD candidate arrays,
maps values from [0, 1] to specified weight ranges, applies chosen normalizations, 
and returns the morphed Airfoil geometry objects rather than XFOIL objective values.
"""

import os
import pickle
import shutil
import hashlib
import tempfile
import subprocess
import signal
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

try:
    from shapely.geometry import Polygon, LineString, Point, MultiPoint
    from shapely.validation import make_valid
except ImportError:
    Polygon = None
    print("Warning: Shapely library not available. Geometry correction will be bypassed.")

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
NUM_POINTS_INTERP: int = 161
THETA_HALF: np.ndarray = np.linspace(0.0, np.pi, NUM_POINTS_INTERP // 2 + 1)
X_INTERP_HALF_ASC: np.ndarray = 0.5 * (1.0 - np.cos(THETA_HALF))[1:]
X_INTERP_HALF_DESC: np.ndarray = (0.5 * (1.0 - np.cos(THETA_HALF)))[::-1]
X_INTERP: np.ndarray = np.concatenate((X_INTERP_HALF_DESC, X_INTERP_HALF_ASC))

# Assuming flat directory structure for the minimal package
DATA_FOLDER: str = 'airfoilDB'

# Updated Default Expected Baselines order
EXPECTED_BASELINES: list[str] = [
    "E195  (11.82%)", "FX 79-W-660A", "GOE 531 AIRFOIL", "EPPLER 864 STRUT AIRFOIL",
    "RONCZ R1145MS MAIN ELEMENT", "CHEN AIRFOIL", "Griffith 30% Suction Airfoil", "S9104", 
    "AH 93-W-480B", "AH 81-K-144 W-F KLAPPE", "EPPLER 664 (EXTENDED) AIRFOIL", "SARATOV AIRFOIL"
]

# Cache to avoid reloading the database on repeated function calls
_CACHED_AIRFOIL_DB_DICT = None

# Geometry correction parameters
MIN_INTERIOR_THICKNESS: float = 1e-3  # Chord-normalized interior thickness floor for geometry correction
MIN_TRAILING_EDGE_THICKNESS: float = 1e-4  # Keep upper/lower TE endpoints from crossing

# XFOIL configuration
XFOIL_APP: str = 'bin/xfoil-ubuntu22.sif'  # Default apptainer image path for portable XFOIL execution
REYNOLDS: float = 1e6  # Default chord-based Reynolds number Re_c for XFOIL simulations
MACH: float = 0.0 # Default Mach number for XFOIL simulations (incompressible flow)
N_CRIT: float = 9.0 # Default N_crit for transition prediction in XFOIL
COARSE_ALPHA_STEP: float = 0.5 # Step size for initial coarse alpha scan in XFOIL
REFINED_ALPHA_STEP: float = 0.1 # Step size for refined alpha scan around identified centers in XFOIL
REFINE_ALPHA_WINDOW: float = 1.0 # Window size around identified centers for refined scanning in XFOIL

# =============================================================================
# UTILITIES
# =============================================================================
def moving_average(y_values: np.ndarray, window_size: int = 5) -> np.ndarray:
    if window_size < 1: return y_values
    return np.convolve(y_values, np.ones(window_size)/window_size, mode='same')

def _smooth_surface_preserve_endpoints(y_values: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Smooth a 1D surface while preserving endpoint values (LE/TE anchors)."""
    if window_size <= 1 or y_values.size < 3:
        return y_values.copy()

    if window_size % 2 == 0:
        window_size += 1

    pad = window_size // 2
    padded = np.pad(y_values, (pad, pad), mode='edge')
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded, kernel, mode='valid')

    smoothed[0] = y_values[0]
    smoothed[-1] = y_values[-1]
    return smoothed

def _prepare_surface_for_interp(x_surface: np.ndarray, y_surface: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Sort a surface in ascending x and merge duplicate x values using y-averaging.
    Returns None when insufficient points remain for interpolation.
    """
    if len(x_surface) < 2:
        return None

    order = np.argsort(x_surface)
    x_sorted = x_surface[order]
    y_sorted = y_surface[order]

    x_unique, inv = np.unique(x_sorted, return_inverse=True)
    if x_unique.size < 2:
        return None

    y_acc = np.zeros_like(x_unique, dtype=float)
    counts = np.zeros_like(x_unique, dtype=float)
    for i, grp in enumerate(inv):
        y_acc[grp] += y_sorted[i]
        counts[grp] += 1.0
    y_unique = y_acc / counts

    return x_unique, y_unique

def _resample_surface(x_surface: np.ndarray, y_surface: np.ndarray, x_target: np.ndarray) -> np.ndarray | None:
    prepared = _prepare_surface_for_interp(x_surface, y_surface)
    if prepared is None:
        return None
    x_unique, y_unique = prepared

    try:
        interpolator = PchipInterpolator(x_unique, y_unique)
    except ValueError:
        return None

    return interpolator(x_target)

def _enforce_min_interior_thickness(
    y_upper: np.ndarray,
    y_lower: np.ndarray,
    min_thickness: float = MIN_INTERIOR_THICKNESS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enforce minimum thickness on interior points only (exclude LE/TE anchors).
    This prevents local surface crossing and near-zero-thickness pockets.
    """
    y_u = y_upper.copy()
    y_l = y_lower.copy()

    if y_u.size < 3 or y_l.size < 3:
        return y_u, y_l

    thickness = y_u - y_l
    for i in range(1, len(thickness) - 1):
        if thickness[i] < min_thickness:
            delta = 0.5 * (min_thickness - thickness[i])
            y_u[i] += delta
            y_l[i] -= delta

    return y_u, y_l

def _resolve_xfoil_command(
    backend: str,
    apptainer_image: str,
    run_dir: str,
) -> tuple[list[str], str]:
    """
    Resolve command for XFOIL execution.
    Resolution order for auto mode is fixed: apptainer image, then system PATH 'xfoil'.
    """
    backend = (backend or 'auto').lower()
    if backend not in {'auto', 'native', 'apptainer'}:
        raise ValueError("xfoil_backend must be one of: 'auto', 'native', 'apptainer'.")

    def _apptainer_cmd() -> tuple[list[str], str]:
        apptainer_exe = shutil.which('apptainer')
        if not apptainer_exe:
            raise RuntimeError("Apptainer executable not found in system PATH.")

        image_abs = os.path.abspath(apptainer_image)
        if not os.path.exists(image_abs):
            raise RuntimeError(f"Apptainer image not found: {image_abs}")

        bind_arg = f"{run_dir}:{run_dir}"
        return [
            apptainer_exe,
            'run',
            '--bind', bind_arg,
            image_abs,
        ], 'apptainer'

    def _native_cmd() -> tuple[list[str], str]:
        system_xfoil = shutil.which('xfoil')
        if not system_xfoil:
            raise RuntimeError("XFOIL executable not found in system PATH.")
            
        xvfb_run = shutil.which('xvfb-run')
        if xvfb_run:
            # Safely wrap native execution in an isolated dummy display
            return [
                xvfb_run, 
                '-a', 
                '-s', '-screen 0 640x480x8', 
                system_xfoil
            ], 'native-xvfb'
        else:
            # Ultimate fallback if running natively and xvfb is missing
            return [system_xfoil], 'native-raw'

    if backend == 'apptainer':
        return _apptainer_cmd()

    if backend == 'native':
        return _native_cmd()

    # backend == 'auto': apptainer first, then native executable.
    try:
        return _apptainer_cmd()
    except Exception:
        return _native_cmd()

def _run_xfoil_aseq(
cmd: list[str],
    run_dir: str,
    coord_file: str,
    reynolds: float,
    mach: float,
    n_crit: float,
    repanel_n: int,
    max_iter: int,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    timeout_sec: float,
    warm_start: bool = False,
    anchor_alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    polar_file = os.path.join(run_dir, f"polar_{hashlib.sha256(os.urandom(32)).hexdigest()[:16]}.out")

    def _build_xfoil_input(use_ppar: bool) -> str:
        commands = [f"LOAD {coord_file}"]
        
        if use_ppar and int(repanel_n) > 0:
            commands.extend(["PPAR", f"N {int(repanel_n)}", ""])
            
        commands.extend([
            "PANE",
            "OPER",
            f"VISC {reynolds:.0f}",
            f"MACH {mach:.6f}",
            "VPAR",
            f"N {n_crit:.6f}",
            "",
            f"ITER {max_iter}"
        ])

        if warm_start:
            # Use the known-converged anchor, fallback to 0.0 if not provided
            start_ang = anchor_alpha if anchor_alpha is not None else 0.0
            commands.append(f"ALFA {start_ang:.3f}")
            
            # March toward alpha_start using a larger step (0.5) to build the BL quickly
            gap = alpha_start - start_ang
            if abs(gap) > 0.5:
                step = 0.5 if gap > 0 else -0.5
                # March up to just before the refined start to ensure a smooth handoff
                commands.append(f"ASEQ {start_ang:.3f} {alpha_start - step:.3f} {step:.3f}")

        commands.extend([
            "PACC",
            f"{polar_file}",
            "",
            f"ASEQ {alpha_start:.6f} {alpha_end:.6f} {alpha_step:.6f}",
            "PACC",
            "",
            "QUIT",
            ""
        ])
        
        return "\n".join(commands)

    def _run_once(exec_cmd: list[str], xfoil_input: str) -> subprocess.CompletedProcess:
        proc = subprocess.Popen(
            exec_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=run_dir,
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(
                input=xfoil_input, 
                timeout=timeout_sec if timeout_sec > 0.0 else None
            )
            return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
            
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            proc.communicate()
            raise
            
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            proc.communicate()
            raise

    xfoil_input = _build_xfoil_input(use_ppar=True)
        
    try:
        proc = _run_once(cmd, xfoil_input)

        # Fallback to PANE-only if PPAR/N geometry repaneling fails
        if proc.returncode != 0 or not os.path.exists(polar_file):
            xfoil_input_fallback = _build_xfoil_input(use_ppar=False)
            proc = _run_once(cmd, xfoil_input_fallback)
            
    except subprocess.TimeoutExpired:
        # If a strict timeout occurs, XFOIL is killed.
        # The parser will safely handle the missing/incomplete polar file.
        pass

    return _parse_xfoil_polar(polar_file)

def _write_airfoil_for_xfoil(airfoil: 'Airfoil', file_path: str) -> None:
    with open(file_path, 'w') as f:
        f.write(f"{airfoil.name}\n")
        for x, y in zip(X_INTERP, airfoil.colloc_vec):
            f.write(f"{x:.10f} {y:.10f}\n")

def _parse_xfoil_polar(polar_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha_vals, cl_vals, cd_vals = [], [], []
    if not os.path.exists(polar_file):
        return np.array([]), np.array([]), np.array([])
        
    with open(polar_file, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                alpha = float(parts[0])
                cl = float(parts[1])
                cd = float(parts[2])
            except ValueError:
                continue
            alpha_vals.append(alpha)
            cl_vals.append(cl)
            cd_vals.append(cd)
    return np.array(alpha_vals), np.array(cl_vals), np.array(cd_vals)

def _compute_polar_metrics(
    alpha: np.ndarray,
    cl: np.ndarray,
    cd: np.ndarray,
    reynolds: float,
) -> dict:
    # 1. Dynamic Physical Drag Floor
    cd_min_physical = 2.656 / np.sqrt(reynolds)
    valid = cd > cd_min_physical

    # 2. Continuity Filter: Detect boundary layer solver collapse
    # Drag rarely drops by more than 20-30% even when entering a deep laminar bucket.
    # A 50% sudden drop indicates XFOIL's transition model has crashed.
    for i in range(1, len(cd)):
        if cd[i] < 0.5 * cd[i-1]:
            valid[i:] = False  # Invalidate this point and all subsequent points
            break

    if not np.any(valid):
        return {
            'alpha': alpha.tolist(),
            'cl': cl.tolist(),
            'cd': cd.tolist(),
            'cl_cd': cl.tolist(),
            'cl_cd_max': 0.0,
            'alpha_at_cl_cd_max': 0.0,
            'cl_max': 0.0,
            'alpha_at_cl_max': 0.0,
            'alpha_stall': 0.0,
            'delta_alpha': 0.0,
            'error': f'All valid Cd values were below physical threshold or solver continuity broke.'
        }

    cl_cd = np.full_like(cl, np.nan, dtype=float)
    cl_cd[valid] = cl[valid] / cd[valid]

    MAX_REALISTIC_CLCD = 350.0
    cl_cd = np.clip(cl_cd, a_min=None, a_max=MAX_REALISTIC_CLCD)

    # Smooth peak-related signals on valid polar entries to reduce zig-zag noise in alpha marching.
    cl_smooth = cl.copy()
    cl_cd_smooth = cl_cd.copy()
    valid_indices = np.where(valid)[0]
    if len(valid_indices) >= 5:
        window_size = min(5, len(valid_indices))
        if window_size % 2 == 0:
            window_size -= 1
        poly_order = min(3, window_size - 1)

        if window_size >= 3 and poly_order >= 1:
            cl_smooth[valid_indices] = savgol_filter(cl[valid_indices], window_size, poly_order)
            cl_cd_smooth[valid_indices] = savgol_filter(cl_cd[valid_indices], window_size, poly_order)

    idx_best = int(np.nanargmax(cl_cd_smooth))
    alpha_best = float(alpha[idx_best])
    idx_cl_max = int(np.argmax(cl_smooth))

    # Conservative stall definition: first local Cl maximum while marching alpha upward.
    idx_stall = None
    if cl_smooth.size >= 3:
        for i in range(1, len(cl_smooth) - 1):
            if cl_smooth[i] > cl_smooth[i - 1] and cl_smooth[i] >= cl_smooth[i + 1]:
                idx_stall = i
                break
    if idx_stall is None:
        idx_stall = idx_cl_max

    alpha_stall = float(alpha[idx_stall])
    raw_delta_alpha = float(alpha_stall - alpha_best)
    delta_alpha = max(0.0, raw_delta_alpha)

    return {
        'alpha': alpha.tolist(),
        'cl': cl.tolist(),
        'cd': cd.tolist(),
        'cl_cd': cl_cd.tolist(),
        'cl_cd_max': float(cl_cd[idx_best]),
        'alpha_at_cl_cd_max': alpha_best,
        'cl_max': float(cl[idx_cl_max]),
        'alpha_at_cl_max': float(alpha[idx_cl_max]),
        'alpha_stall': alpha_stall,
        'delta_alpha': delta_alpha,
    }

def run_xfoil_evaluation(airfoil: 'Airfoil', xfoil_config: dict, m: int = 1) -> dict:
    """
    Execute XFOIL and compute polar metrics without truncating post-stall data.
    """
    apptainer_image = str(xfoil_config.get('apptainer_image', XFOIL_APP))
    backend = str(xfoil_config.get('xfoil_backend', 'auto'))

    reynolds = float(xfoil_config.get('reynolds', REYNOLDS))
    mach = float(xfoil_config.get('mach', MACH))
    n_crit = float(xfoil_config.get('n_crit', N_CRIT))
    
    repanel_n = int(xfoil_config.get('repanel_n', 160))
    max_iter = int(xfoil_config.get('xfoil_iter', 200))
    timeout_sec = float(xfoil_config.get('xfoil_timeout', 60.0))

    alpha_start = float(xfoil_config.get('alfa_start', 0.0))
    alpha_end = float(xfoil_config.get('alfa_end', 45.0))
    
    with tempfile.TemporaryDirectory(prefix='xfoil_run_') as run_dir:
        coord_file = os.path.join(run_dir, 'airfoil.dat')

        _write_airfoil_for_xfoil(airfoil, coord_file)
        cmd, runner = _resolve_xfoil_command(
            backend=backend,
            apptainer_image=apptainer_image,
            run_dir=run_dir,
        )

        # Stage-1 coarse scan: No warm start needed if starting from 0 (or close to it)
        alpha_c, cl_c, cd_c = _run_xfoil_aseq(
            cmd=cmd,
            run_dir=run_dir,
            coord_file=coord_file,
            reynolds=reynolds,
            mach=mach,
            n_crit=n_crit,
            repanel_n=repanel_n,
            max_iter=max_iter,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            alpha_step=COARSE_ALPHA_STEP,
            timeout_sec=timeout_sec,
            warm_start=False
        )

        valid_c = cd_c > 0
        refine_centers = []

        # Attempt to find centers if the coarse scan yielded positive drag data
        if np.any(valid_c):
            cl_cd_c = np.full_like(cl_c, np.nan, dtype=float)
            cl_cd_c[valid_c] = cl_c[valid_c] / cd_c[valid_c]
            
            cl_c_smooth = cl_c.copy()
            cl_cd_c_smooth = cl_cd_c.copy()
            
            valid_indices = np.where(valid_c)[0]
            if len(valid_indices) >= 5: 
                window_size = 5 
                poly_order = 3  
                cl_c_smooth[valid_indices] = savgol_filter(cl_c[valid_indices], window_size, poly_order)
                cl_cd_c_smooth[valid_indices] = savgol_filter(cl_cd_c[valid_indices], window_size, poly_order)

            alpha_center_clcd = float(alpha_c[int(np.nanargmax(cl_cd_c_smooth))])
            if m == 2:
                alpha_center_clmax = float(alpha_c[int(np.argmax(cl_c_smooth))])
                refine_centers = sorted({alpha_center_clcd, alpha_center_clmax})
            else:
                refine_centers = [alpha_center_clcd]

        # Initialize the merge dictionary with whatever coarse data we got (even if empty)
        merged = {
            round(float(a), 6): (float(a), float(c_l), float(c_d))
            for a, c_l, c_d in zip(alpha_c, cl_c, cd_c)
        }

        # Stage-2/3 refined scans (This loop will safely skip if refine_centers is empty)
        for center in refine_centers:
            a0 = max(alpha_start, center - REFINE_ALPHA_WINDOW)
            a1 = min(alpha_end, center + REFINE_ALPHA_WINDOW)
            if a1 <= a0:
                continue

            if len(alpha_c) > 0:
                anchor_alpha = float(np.min(alpha_c))
            else:
                anchor_alpha = 0.0

            alpha_r, cl_r, cd_r = _run_xfoil_aseq(
                cmd=cmd,
                run_dir=run_dir,
                coord_file=coord_file,
                reynolds=reynolds,
                mach=mach,
                n_crit=n_crit,
                repanel_n=repanel_n,
                max_iter=max_iter,
                alpha_start=a0,
                alpha_end=a1,
                alpha_step=REFINED_ALPHA_STEP,
                timeout_sec=timeout_sec,
                warm_start=True,
                anchor_alpha=anchor_alpha
            )

            for a, c_l, c_d in zip(alpha_r, cl_r, cd_r):
                merged[round(float(a), 6)] = (float(a), float(c_l), float(c_d))

        # Convert the merged dictionary back to arrays
        merged_rows = sorted(merged.values(), key=lambda row: row[0])
        alpha = np.array([r[0] for r in merged_rows], dtype=float)
        cl = np.array([r[1] for r in merged_rows], dtype=float)
        cd = np.array([r[2] for r in merged_rows], dtype=float)

        # Only fail if absolutely NO data was gathered across all 3 stages
        if len(alpha) == 0:
            metrics = {
                'alpha': [],
                'cl': [],
                'cd': [],
                'cl_cd': [],
                'cl_cd_max': 0.0,          # Worst possible efficiency
                'alpha_at_cl_cd_max': 0.0,
                'cl_max': 0.0,             # Worst possible lift
                'alpha_at_cl_max': 0.0,
                'alpha_stall': 0.0,
                'delta_alpha': 0.0,        # Worst possible stall margin
                'error': 'All XFOIL scans failed to converge.'
            }
        else:
            metrics = _compute_polar_metrics(alpha, cl, cd, reynolds)

        # Attach execution metadata regardless of success or failure
        metrics['runner'] = runner
        metrics['command'] = cmd
        metrics['scan_scheme'] = {
            'coarse_step': COARSE_ALPHA_STEP,
            'refined_step': REFINED_ALPHA_STEP,
            'refine_window': REFINE_ALPHA_WINDOW,
        }
        
        return metrics

# =============================================================================
# AIRFOIL CORE CLASSES & FILE I/O
# =============================================================================
class Airfoil:
    def __init__(self, airfoil_id: int, name: str, colloc_vec: np.ndarray, x_raw: np.ndarray, y_raw: np.ndarray):
        self.airfoil_id = airfoil_id
        self.name = name
        self.colloc_vec = colloc_vec
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.xfoil_result = None

    def __str__(self) -> str:
        return f"Airfoil(ID={self.airfoil_id}, Name='{self.name}', Raw Data Points={len(self.x_raw)})"

    def get_raw_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x_raw, self.y_raw

    def get_interpolated_data(self) -> np.ndarray:
        return self.colloc_vec

    def plot(self, save_path: str | None = None) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_raw, self.y_raw, 'o', label='Raw Data', alpha=0.7, markersize=4)
        # plt.plot(X_INTERP, self.colloc_vec, '-', label='Interpolated Data', linewidth=2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Airfoil: {self.name}", fontsize=14)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        # plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def read_airfoil_file(file_path: str) -> tuple[str, np.ndarray, np.ndarray]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        name = lines[0].strip()
        coords_list = []
        is_goe451 = name.startswith("GOE 451")

        for line_content in lines[1:]:
            line_content = line_content.strip()
            if line_content.startswith("#") or not line_content:
                continue
            coords_list.append(list(map(float, line_content.split())))
        
        coords_np = np.array(coords_list)
        x_original, y_original = coords_np[:, 0], coords_np[:, 1]

        if is_goe451: # DB has error for this specific airfoil
            y_original[x_original == 0.0249500] = 0.0192620 

        x_min, x_max = x_original.min(), x_original.max()
        if x_max > x_min:
            x_original = (x_original - x_min) / (x_max - x_min)

    leading_edge_indices = np.where(np.isclose(x_original, 0, atol=1e-8))[0]

    if len(leading_edge_indices) == 0:
        for i in range(1, len(x_original)):
            if x_original[i-1] > 0 and x_original[i] < 0:
                y_le = np.interp(0, [x_original[i-1], x_original[i]], [y_original[i-1], y_original[i]])
                x_original = np.insert(x_original, i, 0)
                y_original = np.insert(y_original, i, y_le)
                break
    elif len(leading_edge_indices) == 2:
        upper_le_idx, lower_le_idx = leading_edge_indices
        y_avg_le = (y_original[upper_le_idx] + y_original[lower_le_idx]) / 2.0
        x_original[upper_le_idx] = 1e-6 
        x_original[lower_le_idx] = 1e-6
        insert_idx = min(upper_le_idx, lower_le_idx) + 1
        x_original = np.insert(x_original, insert_idx, 0)
        y_original = np.insert(y_original, insert_idx, y_avg_le)
        
    return name, x_original, y_original

def interp_airfoil(x_coords: np.ndarray, y_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(x_coords) != len(y_coords) or len(x_coords) < 4:
        return None

    min_x = np.min(x_coords)
    le_indices = np.where(np.isclose(x_coords, min_x, atol=1e-8))[0]
    if le_indices.size == 0:
        return None

    le_start, le_end = int(le_indices[0]), int(le_indices[-1])
    x_upper_raw, y_upper_raw = x_coords[:le_start + 1].copy(), y_coords[:le_start + 1].copy()
    x_lower_raw, y_lower_raw = x_coords[le_end:].copy(), y_coords[le_end:].copy()

    if len(x_lower_raw) < 2 or len(x_upper_raw) < 2:
        return None

    y_le = np.mean(y_coords[le_indices])
    x_upper_raw[-1], y_upper_raw[-1] = min_x, y_le
    x_lower_raw[0], y_lower_raw[0] = min_x, y_le

    prepared_upper = _prepare_surface_for_interp(x_upper_raw, y_upper_raw)
    prepared_lower = _prepare_surface_for_interp(x_lower_raw, y_lower_raw)
    if prepared_upper is None or prepared_lower is None:
        return None

    x_upper, y_upper = prepared_upper
    x_lower, y_lower = prepared_lower

    try:
        f_upper = PchipInterpolator(x_upper, y_upper)
        f_lower = PchipInterpolator(x_lower, y_lower)
    except ValueError:
        return None

    y_interp_upper = f_upper(X_INTERP[:NUM_POINTS_INTERP // 2 + 1])
    y_interp_lower = f_lower(X_INTERP[NUM_POINTS_INTERP // 2:])

    # Ensure a single consistent LE point where upper and lower branches meet.
    y_le_interp = 0.5 * (y_interp_upper[-1] + y_interp_lower[0])
    y_interp_upper[-1] = y_le_interp
    y_interp_lower[0] = y_le_interp

    y_interp_combined = np.concatenate((y_interp_upper, y_interp_lower[1:]))
    return X_INTERP, y_interp_combined

def load_airfoil_database_from_files(data_folder: str) -> list[Airfoil]:
    airfoils_db: list[Airfoil] = []
    pickle_file_path = os.path.join(data_folder, f'_db.pkl')

    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            airfoils_db = pickle.load(f)
    else:
        if os.path.isdir(data_folder):
            affile_list_raw = os.listdir(data_folder)
            incompatible_files = {'30p-30n.dat', 'naca1.dat'} 
            affile_list_filtered = [f for f in affile_list_raw if f not in incompatible_files and f.endswith('.dat')]

            for idx, filename in enumerate(sorted(affile_list_filtered)):
                file_full_path = os.path.join(data_folder, filename)
                af_name, x_o, y_o = read_airfoil_file(file_full_path)
                
                interp_result = interp_airfoil(x_o, y_o) 
                if interp_result is None: continue
                
                _, y_interp = interp_result
                airfoil_obj = Airfoil(airfoil_id=idx, name=af_name, colloc_vec=y_interp, x_raw=x_o, y_raw=y_o)
                airfoils_db.append(airfoil_obj)
                
        if airfoils_db and not os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(airfoils_db, f)
    
    return airfoils_db

def get_baselines(data_folder: str, expected_names: list[str]) -> list[Airfoil]:
    """Loads exact specified baselines from the database."""
    global _CACHED_AIRFOIL_DB_DICT
    
    if _CACHED_AIRFOIL_DB_DICT is None:
        full_db = load_airfoil_database_from_files(data_folder)
        _CACHED_AIRFOIL_DB_DICT = {af.name: af for af in full_db}
        
    baselines = []
    for name in expected_names:
        if name in _CACHED_AIRFOIL_DB_DICT:
            baselines.append(_CACHED_AIRFOIL_DB_DICT[name])
        else:
            raise RuntimeError(f"Requested baseline '{name}' not found in the airfoil database.")
            
    return baselines

# =============================================================================
# GEOMETRY CORRECTION & MORPHING
# =============================================================================
def correct_airfoil_geometry(airfoil_to_correct: Airfoil) -> Airfoil:
    if Polygon is None:
        return airfoil_to_correct

    points = list(zip(X_INTERP, airfoil_to_correct.colloc_vec))
    if tuple(points[0]) != tuple(points[-1]):
        points.append(points[0])

    try:
        airfoil_shape_current = Polygon(points).simplify(1e-5, preserve_topology=True)
    except Exception:
        return airfoil_to_correct

    try:
        airfoil_shape_current = make_valid(airfoil_shape_current)
        if not isinstance(airfoil_shape_current, Polygon):
            if hasattr(airfoil_shape_current, 'geoms'):
                polygons = [g for g in airfoil_shape_current.geoms if isinstance(g, Polygon)]
                if polygons: airfoil_shape_current = max(polygons, key=lambda p: p.area)
                else: return airfoil_to_correct
            else: return airfoil_to_correct

        # Must match the active collocation distribution (cosine) to avoid shape distortion.
        x_slices = X_INTERP[NUM_POINTS_INTERP // 2:]
        upper_surface_pts, lower_surface_pts = [], []
        final_boundary = LineString(airfoil_shape_current.exterior.coords)

        for x_val in x_slices:
            vertical_line = LineString([(x_val, -1), (x_val, 1)])
            intersection = final_boundary.intersection(vertical_line)
            if intersection.is_empty: continue
            
            y_at_x = []
            if isinstance(intersection, Point): y_at_x.append(intersection.y)
            elif hasattr(intersection, 'geoms'):
                for geom_item in intersection.geoms:
                    if isinstance(geom_item, Point): y_at_x.append(geom_item.y)

            if not y_at_x: continue
            y_at_x.sort(reverse=True)
            upper_surface_pts.append((x_val, y_at_x[0]))
            lower_surface_pts.append((x_val, y_at_x[-1]))
        
        if not upper_surface_pts or not lower_surface_pts: return airfoil_to_correct

        x_upper_hit = np.array([p[0] for p in upper_surface_pts])
        y_upper_hit = np.array([p[1] for p in upper_surface_pts])
        x_lower_hit = np.array([p[0] for p in lower_surface_pts])
        y_lower_hit = np.array([p[1] for p in lower_surface_pts])

        y_upper_resampled = _resample_surface(x_upper_hit, y_upper_hit, x_slices)
        y_lower_resampled = _resample_surface(x_lower_hit, y_lower_hit, x_slices)
        if y_upper_resampled is None or y_lower_resampled is None:
            return airfoil_to_correct

        y_upper_smoothed = _smooth_surface_preserve_endpoints(y_upper_resampled, window_size=5)
        y_lower_smoothed = _smooth_surface_preserve_endpoints(y_lower_resampled, window_size=5)

        # Enforce a unique LE and two explicit TE points (upper/lower branch endpoints).
        y_le = 0.5 * (y_upper_smoothed[0] + y_lower_smoothed[0])
        y_upper_smoothed[0] = y_le
        y_lower_smoothed[0] = y_le

        # Keep a small positive interior thickness to avoid local crossings.
        y_upper_smoothed, y_lower_smoothed = _enforce_min_interior_thickness(
            y_upper_smoothed,
            y_lower_smoothed,
            min_thickness=MIN_INTERIOR_THICKNESS,
        )

        # Prevent self-intersection at trailing edge by enforcing TE ordering.
        te_thickness = y_upper_smoothed[-1] - y_lower_smoothed[-1]
        if te_thickness < MIN_TRAILING_EDGE_THICKNESS:
            te_delta = 0.5 * (MIN_TRAILING_EDGE_THICKNESS - te_thickness)
            y_upper_smoothed[-1] += te_delta
            y_lower_smoothed[-1] -= te_delta

        y_selig_final = np.concatenate((y_upper_smoothed[::-1], y_lower_smoothed[1:]))

        return Airfoil(
            airfoil_id=airfoil_to_correct.airfoil_id,
            name=f"{airfoil_to_correct.name}_corrected",
            colloc_vec=y_selig_final,
            x_raw=X_INTERP,
            y_raw=y_selig_final.copy()
        )
    except Exception:
        return airfoil_to_correct

def create_morphed_airfoil(weights: np.ndarray, baseline_airfoils: list[Airfoil], correct_geometry: bool = True) -> Airfoil:
    if len(weights) != len(baseline_airfoils):
        raise ValueError("Length of weights must match the number of baseline airfoils.")

    weights_np = np.array(weights)
    
    # Intentionally bypass original DbM L1 normalization here. 
    # Normalization should be explicitly requested via TestAirfoils API args.
    
    if np.allclose(weights_np, 0):
        return Airfoil(airfoil_id=-1, name='Morphed_FlatLine', colloc_vec=np.zeros_like(X_INTERP),
                       x_raw=X_INTERP.copy(), y_raw=np.zeros_like(X_INTERP))

    y_morphed_sum = np.zeros_like(X_INTERP)
    for airfoil, weight in zip(baseline_airfoils, weights_np):
        y_morphed_sum += weight * airfoil.get_interpolated_data()
    
    weight_str = "_".join(f"{w:.4f}" for w in weights_np)
    morphed_name = f"Morphed_W[{weight_str[:50]}]"

    morphed_obj = Airfoil(airfoil_id=-1, name=morphed_name, colloc_vec=y_morphed_sum,
                          x_raw=X_INTERP.copy(), y_raw=y_morphed_sum.copy())
    
    if correct_geometry:
        return correct_airfoil_geometry(morphed_obj)
    return morphed_obj

def _evaluate_single_candidate(
    phi: np.ndarray,
    weight_range: tuple[float, float],
    normalization: str | None,
    data_folder: str,
    selected_baseline_names: list[str],
    xfoil_config: dict,
    m : int,
) -> Airfoil:
    """Worker-safe single-candidate evaluator for parallel TestAirfoils execution."""
    baselines = get_baselines(data_folder, selected_baseline_names)
    w_lower, w_upper = weight_range

    weights = w_lower + phi * (w_upper - w_lower)

    if normalization == 'SUM':
        norm = np.sum(weights)
        if norm > 1e-8:
            weights = weights / norm
        else:
            raise ValueError("Sum of weights is too close to zero for normalization.")
    elif normalization == 'ABS_SUM':
        norm = np.sum(np.abs(weights))
        if norm > 1e-8:
            weights = weights / norm
        else:
            raise ValueError("Sum of absolute weights is too close to zero for normalization.")

    morphed_airfoil = create_morphed_airfoil(weights, baselines, correct_geometry=True)
    morphed_airfoil.name = "_".join(f"{w:.4f}" for w in weights)

    if xfoil_config.get('xfoil_evaluation', False):
        strict = bool(xfoil_config.get('xfoil_strict', True))
        try:
            morphed_airfoil.xfoil_result = run_xfoil_evaluation(morphed_airfoil, xfoil_config, m)
        except Exception as exc:
            if strict:
                raise
            morphed_airfoil.xfoil_result = {
                'error': str(exc),
                'runner': None,
            }

    return morphed_airfoil

def _format_testairfoils_output(airfoils: list[Airfoil], xfoil_evaluation: bool, m: int) -> list:
    """
    Return airfoil objects when XFOIL evaluation is disabled.
    Return objective values derived from attached XFOIL results when enabled.
    """
    if not xfoil_evaluation:
        return airfoils

    if m not in (1, 2):
        raise ValueError("When xfoil_evaluation=True, only m=1 or m=2 are currently supported.")

    outputs = []
    for airfoil in airfoils:
        xr = airfoil.xfoil_result or {}
        cl_cd_max_raw = xr.get('cl_cd_max', np.nan)
        delta_alpha_raw = xr.get('delta_alpha', np.nan)

        cl_cd_max = np.nan if cl_cd_max_raw is None else float(cl_cd_max_raw)
        delta_alpha = np.nan if delta_alpha_raw is None else float(delta_alpha_raw)

        if m == 1:
            outputs.append(cl_cd_max)
        else:
            outputs.append([cl_cd_max, delta_alpha])

    return outputs

# =============================================================================
# MAIN API: TestAirfoils
# =============================================================================
def TestAirfoils(x: np.ndarray, args: dict = None, m: int = 2) -> list:
    """
    Test function for generating DbM airfoils based on an N x D candidate matrix.
    
        Parameters:
        - x: N x D array, where N is candidates and D is design parameters.
                 Input values must range from 0 to 1.
        - args: Configuration dictionary supporting:
            AIRFOIL DESIGN ARGS:
                - 'airfoil_db_dir': Path to airfoil database (defaults to DATA_FOLDER).
                - 'dbm_baselines': List of airfoil names to use as baselines (defaults to EXPECTED_BASELINES).
                - 'dbm_weight_range': [lower, upper] limits (default: [-1.0, 1.0]). Allows negative weights for extrapolative morphing.
                - 'dbm_normalization': None (default), 'SUM', or 'ABS_SUM'.
            AIRFOIL EVALUATION ARGS:
                - 'xfoil_evaluation': Run XFOIL evaluation after geometry generation (default: True).
                - 'xfoil_backend': 'apptainer' (default), 'auto', or 'native'.
                    ('auto': apptainer image first, then system 'xfoil').
                - 'apptainer_image': Path to apptainer image containing xfoil (default: XFOIL_APP).
                - 'xfoil_iter': Max XFOIL iterations per alpha (default: 200).
                - 'xfoil_timeout': Timeout in seconds per candidate (default: 60.0).
                - 'xfoil_strict': If True, fail on XFOIL errors; if False, attach error in airfoil.xfoil_result.
                - 'alfa_start', 'alfa_end': Polar angle range (defaults: 0, 45).
                - 'reynolds': Reynolds number (defaults to REYNOLDS).
                - 'mach': Mach number for XFOIL (default: MACH=0).
                - 'n_crit': e^N critical amplification factor (default: N_CRIT=9).
                - 'repanel_n': Internal XFOIL repanel node count via PPAR/N (default: 160).
            MULTIPROCESSING ARGS:
                - 'parallel': Enable multiprocessing across candidates (default: True).
                - 'max_workers': Max worker processes (default: total CPU count in environment).
    - m: Number of objectives (1 for Cl/Cd <single objective>, 2 for Cl/Cd and delta alpha <bi-objective>).
    
    Returns:
    - If xfoil_evaluation is False:
        airfoils: A list of N generated Airfoil objects containing interpolated coordinates.
                  The airfoil names strictly map to w1_w2_..._wn format.
    - If xfoil_evaluation is True:
        m=1 -> list[float] of Cl/Cd_max
        m=2 -> list[[Cl/Cd_max, delta_alpha]]
    """
    if args is None:
        args = {}
        
    # Extract configuration with safe defaults
    data_folder = args.get('airfoil_db_dir', DATA_FOLDER)
    weight_range = args.get('dbm_weight_range', [-1.0, 1.0])
    normalization = args.get('dbm_normalization', None)
    expected_baselines = args.get('dbm_baselines', EXPECTED_BASELINES)
    parallel = args.get('parallel', True)
    cpu_total = os.cpu_count() or 1
    max_workers_cfg = args.get('max_workers', cpu_total)
    max_workers = min(max(1, int(max_workers_cfg)), cpu_total)
    xfoil_config = {
        'xfoil_evaluation': args.get('xfoil_evaluation', True),
        'xfoil_backend': args.get('xfoil_backend', 'apptainer'),
        'apptainer_image': args.get('apptainer_image', 'bin/xfoil-ubuntu22.sif'),
        'reynolds': args.get('reynolds', REYNOLDS),
        'mach': args.get('mach', MACH),
        'n_crit': args.get('n_crit', N_CRIT),
        'alfa_start': args.get('alfa_start', 0.0),
        'alfa_end': args.get('alfa_end', 45.0),
        'repanel_n': args.get('repanel_n', 160),
        'xfoil_iter': args.get('xfoil_iter', 200),
        'xfoil_timeout': args.get('xfoil_timeout', 60.0),
        'xfoil_strict': args.get('xfoil_strict', True),
    }
    
    # Ensure standard 2D array matrix behavior
    x = np.atleast_2d(x)
    N, D = x.shape
    
    # Validate D against the length of expected baselines
    if D > len(expected_baselines):
        raise ValueError(f"Number of design parameters (D={D}) exceeds the number of provided baselines ({len(expected_baselines)}).")
    
    # Slice exactly D baselines from the provided expected baselines list
    selected_baseline_names = expected_baselines[:D]

    # WEIGHT RANGE VALIDATION ---
    w_lower, w_upper = float(weight_range[0]), float(weight_range[1])
    if w_lower > 0.0 or w_upper < 1.0:
        raise ValueError(
            f"Invalid dbm_weight_range: [{w_lower}, {w_upper}]. "
            "The range must fully encompass [0.0, 1.0] to guarantee that "
            "pure baseline reconstruction remains mathematically possible."
        )
    weight_range_tuple = (w_lower, w_upper)
    # -------------------------------------------
    xfoil_evaluation = bool(xfoil_config.get('xfoil_evaluation', True))

    if parallel and N > 1 and max_workers > 1:
        workers = min(max_workers, N)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            airfoils = list(
                executor.map(
                    _evaluate_single_candidate,
                    x,
                    repeat(weight_range_tuple),
                    repeat(normalization),
                    repeat(data_folder),
                    repeat(selected_baseline_names),
                    repeat(xfoil_config),
                    repeat(m),
                )
            )
        return _format_testairfoils_output(airfoils, xfoil_evaluation=xfoil_evaluation, m=m)

    # Serial fallback (also useful for very small N).
    airfoils = [
        _evaluate_single_candidate(
            phi=x[i, :],
            weight_range=weight_range_tuple,
            normalization=normalization,
            data_folder=data_folder,
            selected_baseline_names=selected_baseline_names,
            xfoil_config=xfoil_config,
            m=m,
        )
        for i in range(N)
    ]
    return _format_testairfoils_output(airfoils, xfoil_evaluation=xfoil_evaluation, m=m)