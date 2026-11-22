#!/usr/bin/env python3
"""Project launcher.

    python run.py setup        # .env + dataset + model training + database + demo data (first run)
    python run.py start        # API (8000) + dashboard (8501) together
    python run.py api          # backend only
    python run.py dashboard    # frontend only
    python run.py generate-data | train | seed | test
"""
from __future__ import annotations

import argparse
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
ENV_FILE, ENV_EXAMPLE = ROOT / ".env", ROOT / ".env.example"


def sh(cmd, **kw) -> int:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.call(cmd, cwd=ROOT, env=env, **kw)


def ensure_env() -> None:
    if ENV_FILE.exists():
        return
    text = ENV_EXAMPLE.read_text()
    text = text.replace("SECRET_KEY=\n", f"SECRET_KEY={secrets.token_urlsafe(48)}\n")
    ENV_FILE.write_text(text)
    print(f"Created .env with a freshly generated SECRET_KEY. Review the demo credentials inside it.")


def cmd_generate(_=None) -> int:
    return sh([sys.executable, "dataset/generate_sample_data.py"])


def cmd_train(args=None) -> int:
    extra = ["--source", args.source] if args is not None and getattr(args, "source", None) else []
    if args is not None and getattr(args, "quick", False):
        extra.append("--quick")
    return sh([sys.executable, "-m", "ml.train_model", *extra])


def cmd_seed(args=None) -> int:
    ensure_env()
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=True)
    from backend.database import SessionLocal, init_db
    from backend.services import bootstrap_users, seed_demo_data, sync_model_performance
    from ml.predict import get_predictor

    init_db()
    predictor = get_predictor()
    n = getattr(args, "rows", 1500) if args is not None else 1500
    with SessionLocal() as db:
        created = bootstrap_users(db)
        if created:
            print("Created users:", ", ".join(created))
        sync_model_performance(db, predictor.metadata)
        added = seed_demo_data(db, predictor, n=n, force=bool(getattr(args, "force", False)))
    print(f"Seeded {added} demo transactions." if added else "Demo data already present (use --force to add more).")
    return 0


def cmd_setup(args=None) -> int:
    ensure_env()
    from ml.config import MODEL_PATH, SYNTHETIC_DATASET_PATH

    if not SYNTHETIC_DATASET_PATH.exists() and cmd_generate():
        return 1
    if (not MODEL_PATH.exists() or getattr(args, "retrain", False)) and cmd_train():
        return 1
    return cmd_seed()


def cmd_api(_=None) -> int:
    ensure_env()
    host, port = os.getenv("API_HOST", "127.0.0.1"), os.getenv("API_PORT", "8000")
    return sh([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", host, "--port", port,
               *(["--reload"] if os.getenv("API_RELOAD") == "1" else [])])


def cmd_dashboard(_=None) -> int:
    port = os.getenv("DASHBOARD_PORT", "8501")
    return sh([sys.executable, "-m", "streamlit", "run", "frontend/dashboard.py", "--server.port", port])


def cmd_start(_=None) -> int:
    ensure_env()
    procs = [subprocess.Popen([sys.executable, "run.py", "api"], cwd=ROOT),
             subprocess.Popen([sys.executable, "run.py", "dashboard"], cwd=ROOT)]
    print("\nAPI docs  : http://127.0.0.1:8000/docs\nDashboard : http://localhost:8501\nPress Ctrl+C to stop.\n")
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    return 0


def cmd_test(_=None) -> int:
    return sh([sys.executable, "-m", "pytest", "-v"])


def main() -> int:
    ap = argparse.ArgumentParser(description="UPI Sentinel AI launcher")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="first-time setup"); s.add_argument("--retrain", action="store_true"); s.set_defaults(fn=cmd_setup)
    sub.add_parser("generate-data").set_defaults(fn=cmd_generate)
    t = sub.add_parser("train"); t.add_argument("--source", choices=["synthetic", "paysim"]); t.add_argument("--quick", action="store_true"); t.set_defaults(fn=cmd_train)
    d = sub.add_parser("seed"); d.add_argument("--rows", type=int, default=1500); d.add_argument("--force", action="store_true"); d.set_defaults(fn=cmd_seed)
    sub.add_parser("api").set_defaults(fn=cmd_api)
    sub.add_parser("dashboard").set_defaults(fn=cmd_dashboard)
    sub.add_parser("start").set_defaults(fn=cmd_start)
    sub.add_parser("test").set_defaults(fn=cmd_test)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
