"""
PAIMANA-AI :: static snapshot exporter
======================================
Calls the FastAPI route handlers in-process and writes their responses to
frontend/data/*.json, so the dashboard also works with no Python backend at all - served by
any static web server (nginx, GitHub Pages, NIC static hosting, or `python -m http.server`).

Note: it must be served over HTTP. Browsers block fetch() on `file://` origins, so opening
index.html by double-clicking it will not load the snapshots.

Run AFTER ml/train.py:   python ml/export_static.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from backend import app as A  # noqa: E402

OUT = ROOT / "frontend" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def w(name, obj):
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, separators=(",", ":"), default=str))
    print(f"  {name+'.json':<26} {p.stat().st_size/1024:>8.1f} KB")


def main():
    print("Exporting static snapshots -> frontend/data/")
    w("kpis", A.kpis())
    w("filters", A.filters())
    w("model_performance", A.model_performance())
    w("breakdown", {d: A.breakdown(by=d) for d in ("ministry", "sector", "state",
                                                   "risk_band", "project_status")})
    w("benchmark", {d: A.benchmark(dimension=d) for d in ("sector", "ministry", "state")})
    w("alerts", A.alerts(limit=3000))

    cols = ["project_id", "project_name", "ministry", "sector", "state", "original_cost_cr",
            "revised_cost_cr", "expenditure_cr", "physical_progress_pct", "elapsed_schedule_pct",
            "project_status", "pred_cost_overrun_pct", "pred_time_overrun_months",
            "risk_score", "risk_band", "alert_count", "risk_rank"]
    w("projects", {"total": len(A.DF), "rows": A._clean(A.DF[cols].to_dict("records"))})

    details = {pid: A.project(pid) for pid in A.DF.project_id}
    w("project_details", details)

    print("\nStatic build ready. Serve it with:  cd frontend && python -m http.server 8080")


if __name__ == "__main__":
    main()
