#!/usr/bin/env python
"""SI EIT in 2D/3D: desktop configurator or reproducible JSON batch runs."""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gui", action="store_true", help="Open the desktop configuration panel")
    ap.add_argument("--config", type=Path, help="Run a saved SI JSON configuration")
    ap.add_argument("--example", type=Path, help="Write an example SI JSON configuration")
    ap.add_argument("--dimension", type=int, choices=(2, 3), default=3)
    ap.add_argument("--validate", action="store_true", help="Validate --config without FEniCSx")
    ap.add_argument("--worker-pid-file", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    raw_config=json.loads(args.config.read_text(encoding='utf-8')) if args.config else None
    if raw_config is not None:
        threads=str(raw_config.get('reconstruction',{}).get('threads',1))
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            os.environ[key]=threads
    from eitcore.physical import default_config, normalize_config
    if args.example:
        args.example.parent.mkdir(parents=True, exist_ok=True)
        args.example.write_text(json.dumps(default_config(args.dimension), indent=2), encoding="utf-8")
        print(f"Wrote {args.example}")
        return
    if args.config:
        cfg = normalize_config(raw_config)
        if args.validate:
            print(f"Valid {cfg['mesh']['dimension']}D SI configuration: {cfg['mesh']['n_electrodes']} electrodes, {cfg['frequencies_hz']} Hz")
            return
        if args.worker_pid_file:
            args.worker_pid_file.write_text(str(os.getpid()), encoding="ascii")
            print(f"Solver PID: {os.getpid()}", flush=True)
        threads = str(cfg['reconstruction'].get('threads', 1))
        os.environ.setdefault("OMP_NUM_THREADS", threads)
        os.environ.setdefault("MKL_NUM_THREADS", threads)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", threads)
        from eitcore.physical_workflow import run
        run(cfg, log=lambda message: print(message, flush=True))
        return
    from eit_gui import launch
    launch(args.dimension)


if __name__ == "__main__":
    main()
