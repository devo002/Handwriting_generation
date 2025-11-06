# main_launcher.py
import os
import sys
import argparse
import subprocess
import yaml

def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def main():
    parser = argparse.ArgumentParser(description="Launch the correct main script based on config.run_mode")
    parser.add_argument("--id", type=int, default=None,
                        help="Epoch/ID to start from (e.g., 0 for scratch, 1200 to resume).")
    parser.add_argument("--config", type=str, default="config.yaml")
    args, passthrough = parser.parse_known_args()

    cfg = load_cfg(args.config)
    run_mode = (cfg.get("run_mode") or "default").strip().lower()

    # Resolve script path relative to this file
    here = os.path.dirname(os.path.abspath(__file__))
    if run_mode == "teacher":
        script = os.path.join(here, "main_run2.py")
    else:
        script = os.path.join(here, "main_run.py")
        
    print(f"[config] run_mode={cfg.get('run_mode','default')}, image_encoder={cfg.get('generator',{}).get('image_encoder','<unset>')}")


    # Build command: python3 <script> <ID> [any extra args passed through]
    cmd = ["python3", script]
    if args.id is not None:
        cmd.append(str(args.id))
    cmd += passthrough

    print(f"🔹 Launching: {os.path.basename(script)}  (run_mode: {run_mode}, id: {args.id})")
    sys.stdout.flush()

    # Run as a subprocess so behavior matches your current job scripts
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
