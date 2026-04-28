"""
Entry point.
  python run.py          → starts Flask (charts must already exist)
  python run.py --full   → regenerates all charts, then starts Flask
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent


def generate_charts() -> None:
    scripts = [
        ROOT / "data"     / "fetch_prices.py",
        ROOT / "analysis" / "technical.py",
        ROOT / "analysis" / "quantitative.py",
        ROOT / "analysis" / "fundamental.py",
    ]
    for script in scripts:
        print(f"Running {script.name} ...")
        result = subprocess.run([sys.executable, str(script)], capture_output=False)
        if result.returncode != 0:
            print(f"ERROR in {script.name} — aborting.")
            sys.exit(1)
    print("All charts generated.")


def start_server() -> None:
    sys.path.insert(0, str(ROOT))
    from web.app import app
    print("\nStarting Flask on http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)


if __name__ == "__main__":
    if "--full" in sys.argv:
        generate_charts()
    start_server()
