import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, redirect, url_for, send_file, jsonify
import subprocess

from config import TICKERS, CHARTS_DIR, ROOT

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _company_list() -> list[dict]:
    """Return the company list in a JS-friendly format."""
    return [
        {"ticker": t, "name": n, "slug": n.lower().replace(" ", "_")}
        for t, n in TICKERS.items()
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", companies=_company_list())


@app.route("/run-analysis", methods=["POST"])
def run_analysis():
    """Execute all analysis scripts then return success JSON."""
    scripts = [
        ROOT / "data"     / "fetch_prices.py",
        ROOT / "analysis" / "technical.py",
        ROOT / "analysis" / "quantitative.py",
        ROOT / "analysis" / "fundamental.py",
    ]
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return jsonify({"ok": False, "error": result.stderr}), 500

    return jsonify({"ok": True})


@app.route("/run-optimisation", methods=["POST"])
def run_optimisation():
    """Execute only the optimisation script then return success JSON."""
    script = ROOT / "analysis" / "optimisation.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return jsonify({"ok": False, "error": result.stderr}), 500
    return jsonify({"ok": True})


@app.route("/charts/<category>/<filename>")
def serve_chart(category: str, filename: str):
    path = CHARTS_DIR / category / filename
    if not path.exists():
        return "Chart not found", 404
    return send_file(str(path))
