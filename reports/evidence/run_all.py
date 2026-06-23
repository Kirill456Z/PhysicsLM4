"""
Entrypoint: runs all evidence scripts in order.

Usage (from repo root):
    uv run python reports/evidence/run_all.py

Each subfolder's run_data.py fetches wandb data, prints a verification summary,
and saves plots to its own plots/ directory.
"""
import subprocess, sys, os, time

SCRIPTS = [
    "01_seed_variance/run_data.py",
    "02_canon_performance/run_data.py",
    "03_canon_ablations/run_data.py",
    "04_training_dynamics/run_data.py",
    "05_task_mixing_variance/run_data.py",
    "06_scaling_laws/run_data.py",
    "07_dynamics_on_synthetic/run_data.py",
    "08_canon_stability/run_data.py",
    "09_seed_flip/run_data.py",
    "10_variance_recompute/run_data.py",
    "11_scale_task_robustness/run_data.py",
    "13_bfs_sp_hard/run_data.py",
    "14_depth_ablation/run_data.py",
    "17_depo_task_correlation/run_data.py",
]

BASE = os.path.dirname(os.path.abspath(__file__))


def run_script(rel_path):
    abs_path = os.path.join(BASE, rel_path)
    name = rel_path.split("/")[0]
    print(f"\n{'#'*70}")
    print(f"# {name}")
    print(f"{'#'*70}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, abs_path],
        cwd=BASE,
        capture_output=False,
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n[{name}] {status}  ({elapsed:.1f}s)")
    return result.returncode == 0


if __name__ == "__main__":
    # Allow running a single script: python run_all.py 03
    target = sys.argv[1] if len(sys.argv) > 1 else None
    scripts = [s for s in SCRIPTS if target is None or s.startswith(target)]
    if not scripts:
        print(f"No scripts match '{target}'. Available: {SCRIPTS}")
        sys.exit(1)

    results = {}
    for script in scripts:
        ok = run_script(script)
        results[script] = ok

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    all_ok = True
    for script, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status}  {script}")
        if not ok:
            all_ok = False
    sys.exit(0 if all_ok else 1)
