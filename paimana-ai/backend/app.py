"""
PAIMANA-AI :: Backend API
=========================
FastAPI service that (1) serves the PAIMANA-styled frontend and (2) exposes the
predictive / early-warning endpoints.

Run:   uvicorn backend.app:app --reload --port 8000
Open:  http://localhost:8000
Docs:  http://localhost:8000/api/docs
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
import features as F  # noqa: E402
from backend import database as DB  # noqa: E402

DATA, MODELS, REPORTS, FRONTEND = ROOT / "data", ROOT / "models", ROOT / "reports", ROOT / "frontend"

app = FastAPI(title="PAIMANA-AI Predictive Analytics & Early Warning API",
              version="1.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --------------------------------------------------------------------------
def _read_project_data():
    """Prefer the imported PAIMANA CSV when present, but keep the generated
    synthetic dataset as the fallback for this repo's default demo build.

    The repo ships a demo export file named PAIMANA_early_warning_output.csv,
    but it is not a full PAIMANA CUF extract and is missing several fields used
    by the dashboard and static export (for example agency_type). Fall back to
    the generated scored portfolio whenever the imported file does not satisfy
    the required schema.
    """
    required = {
        "project_id", "project_name", "ministry", "sector", "state",
        "agency_type", "funding_mode", "contract_type", "date_of_sanction",
        "original_cost_cr", "revised_cost_cr", "expenditure_cr",
        "original_duration_m", "physical_progress_pct", "financial_progress_pct",
        "elapsed_schedule_pct", "land_acquired_pct", "forest_clearance",
        "env_clearance", "award_lag_months", "num_cost_revisions",
        "snapshot_date", "num_packages", "monitoring_frequency",
        "reason_for_delay", "project_status", "cost_overrun_pct",
        "time_overrun_months", "risk_score", "risk_band", "alerts",
        "alert_count", "pred_cost_overrun_pct", "pred_time_overrun_months"
    }

    real = DATA / "PAIMANA_early_warning_output.csv"
    if real.exists():
        d = pd.read_csv(real)
        d = d.copy()
        if required.issubset(d.columns):
            # Built for a full PAIMANA export; keep it.
            pass
        else:
            d = pd.read_csv(DATA / "scored_projects.csv")
            d["risk_rank"] = d.risk_score.rank(ascending=False, method="min").astype(int)
            return d

        d["project_id"] = d["project_code"].fillna(d["slno"]).astype(str)
        d["project_name"] = d["project_name"].fillna("Unnamed project")
        d["ministry"] = d["ministry"].fillna("Unknown")
        d["sector"] = d["sector"].fillna("Unknown")
        d["state"] = d["state"].fillna("Unknown")

        numeric = {
            "original_cost_cr": ["original_cost_cr", "cost_original_cr"],
            "revised_cost_cr": ["revised_cost_cr", "cost_revised_cr"],
            "expenditure_cr": ["cumulative_expenditure", "cost_anticipated_cr"],
            "physical_progress_pct": ["physical_progress_pct"],
            "pred_cost_overrun_pct": ["cost_overrun_ratio"],
            "pred_time_overrun_months": ["time_slippage_months"],
            "time_overrun_months": ["time_slippage_months"],
            "cost_overrun_pct": ["cost_overrun_ratio"],
            "elapsed_schedule_pct": ["schedule_pct_elapsed"],
            "original_duration_m": ["planned_duration_months"],
            "land_acquired_pct": ["land_acquired_pct"],
            "award_lag_months": ["award_lag_months"],
            "num_cost_revisions": ["num_cost_revisions"],
            "alert_count": ["alert_count"],
        }
        for target, cols in numeric.items():
            vals = pd.Series(np.nan, index=d.index, dtype=float)
            for c in cols:
                if c in d.columns:
                    vals = pd.to_numeric(d[c], errors="coerce").combine_first(vals)
            if target in {"pred_cost_overrun_pct", "pred_time_overrun_months", "time_overrun_months", "cost_overrun_pct"}:
                vals = vals.fillna(0.0)
                if target in {"pred_cost_overrun_pct", "cost_overrun_pct"}:
                    vals = vals * 100
            if target == "original_duration_m":
                vals = vals.fillna(48.0)
            elif target == "land_acquired_pct":
                vals = vals.fillna(80.0)
            elif target == "award_lag_months":
                vals = vals.fillna(3.0)
            elif target == "num_cost_revisions":
                vals = vals.fillna(0.0)
            elif target == "alert_count":
                vals = vals.fillna(0.0)
            d[target] = vals

        d["financial_progress_pct"] = pd.to_numeric(d.get("cumulative_expenditure", pd.Series(np.nan, index=d.index)), errors="coerce")
        d["financial_progress_pct"] = d["financial_progress_pct"] / d["revised_cost_cr"].replace(0, np.nan) * 100
        d["financial_progress_pct"] = d["financial_progress_pct"].fillna(d["physical_progress_pct"]).fillna(0.0)

        if "date_of_approval_dt" in d.columns:
            d["date_of_sanction"] = pd.to_datetime(d["date_of_approval_dt"], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            d["date_of_sanction"] = pd.to_datetime(d["date_of_approval"], errors="coerce").dt.strftime("%Y-%m-%d")
        d["date_of_sanction"] = d["date_of_sanction"].fillna("2024-01-01")

        d["original_completion"] = pd.to_datetime(d.get("original_target_doc_dt", d.get("original_target_doc")), errors="coerce").dt.strftime("%Y-%m-%d")
        d["anticipated_completion"] = pd.to_datetime(d.get("date_of_commissioning_anticipated", d.get("revised_doc_dt", d.get("revised_doc", d.get("original_target_doc")))), errors="coerce").dt.strftime("%Y-%m-%d")
        d["original_completion"] = d["original_completion"].fillna(d["date_of_sanction"])
        d["anticipated_completion"] = d["anticipated_completion"].fillna(d["date_of_sanction"])

        d["forest_clearance"] = d.get("forest_clearance", pd.Series("Obtained", index=d.index))
        d["env_clearance"] = d.get("env_clearance", pd.Series("Obtained", index=d.index))
        d["monitoring_frequency"] = d.get("monitoring_frequency", pd.Series("Monthly", index=d.index))
        d["reason_for_delay"] = d.get("reason_for_delay", pd.Series("None reported", index=d.index))
        d["project_status"] = d.get("project_status", pd.Series("On Schedule", index=d.index))
        d["project_status"] = d["project_status"].fillna("On Schedule")

        d["pred_cost_overrun_pct"] = d["pred_cost_overrun_pct"].fillna(np.clip((d["revised_cost_cr"] / d["original_cost_cr"].replace(0, np.nan) - 1) * 100, 0, 100))
        d["pred_time_overrun_months"] = d["pred_time_overrun_months"].fillna(d["time_overrun_months"].fillna(0.0))
        d["cost_overrun_pct"] = d["cost_overrun_pct"].fillna(d["pred_cost_overrun_pct"])
        d["time_overrun_months"] = d["time_overrun_months"].fillna(d["pred_time_overrun_months"])

        d["risk_score"] = (100 * (
            0.35 * np.clip(d["pred_cost_overrun_pct"] / 60, 0, 1) +
            0.35 * np.clip(d["pred_time_overrun_months"] / 36, 0, 1) +
            0.20 * np.clip((d["elapsed_schedule_pct"].fillna(0) - d["physical_progress_pct"].fillna(0)).clip(lower=0) / 45, 0, 1) +
            0.10 * np.clip((100 - d["land_acquired_pct"].fillna(80)) / 60, 0, 1)
        )).round(1)
        d["risk_band"] = pd.cut(d["risk_score"], bins=[-0.1, 25, 50, 72, 100], labels=["Low", "Medium", "High", "Critical"])
        d["risk_band"] = d["risk_band"].astype(str)

        alerts = []
        for _, r in d.iterrows():
            a = []
            if float(r["pred_cost_overrun_pct"]) > 15:
                a.append(f"Forecast cost escalation {r['pred_cost_overrun_pct']:.0f}% above sanctioned cost")
            if float(r["pred_time_overrun_months"]) > 6:
                a.append(f"Forecast schedule slippage {r['pred_time_overrun_months']:.0f} months")
            gap = float(r["elapsed_schedule_pct"]) - float(r["physical_progress_pct"])
            if gap > 10:
                a.append(f"Physical progress {r['physical_progress_pct']:.0f}% vs {r['elapsed_schedule_pct']:.0f}% schedule elapsed")
            if float(r["land_acquired_pct"]) < 70:
                a.append(f"Land acquisition at {r['land_acquired_pct']:.0f}%")
            if float(r["pred_cost_overrun_pct"]) > 0 or float(r["pred_time_overrun_months"]) > 0:
                a.append("Risk flagged in early-warning review")
            alerts.append(" | ".join(a) if a else "No active alert")
        d["alerts"] = alerts
        d["alert_count"] = [0 if a == "No active alert" else a.count("|") + 1 for a in alerts]
        d["risk_rank"] = d["risk_score"].rank(ascending=False, method="min").astype(int)
        return d

    d = pd.read_csv(DATA / "scored_projects.csv")
    d["risk_rank"] = d.risk_score.rank(ascending=False, method="min").astype(int)
    return d


DF = _read_project_data()
DB.initialize()

BUNDLES = {
    "cost": joblib.load(MODELS / "lgbm_cost_overrun_pct_extended.pkl"),
    "time": joblib.load(MODELS / "lgbm_time_overrun_months_extended.pkl"),
    "risk": joblib.load(MODELS / "lgbm_risk_extended.pkl"),
}
SUMMARY = json.loads((REPORTS / "training_summary.json").read_text())
DRIVERS = json.loads((REPORTS / "top_drivers.json").read_text())
ABLATION = json.loads(pd.read_csv(REPORTS / "cuf_ablation.csv").to_json(orient="records"))
COMPARISON = json.loads(pd.read_csv(REPORTS / "model_comparison.csv").to_json(orient="records"))

LAKH = 1e5  # 1 lakh crore = 100,000 crore


def _clean(recs):
    return json.loads(pd.DataFrame(recs).to_json(orient="records"))


def _safe_float(val, default=0.0):
    try:
        x = float(val)
        return default if (pd.isna(x) or not np.isfinite(x)) else x
    except Exception:
        return default


# ==========================================================================
# KPIs / portfolio
# ==========================================================================
@app.get("/api/kpis")
def kpis():
    d = DF
    delayed = d[d.project_status.str.startswith("Delayed")]
    avg_time = _safe_float(delayed.time_overrun_months.mean(), 0.0)
    total_cost = _safe_float(d.original_cost_cr.sum(), 0.0)
    total_revised = _safe_float(d.revised_cost_cr.sum(), 0.0)
    total_exp = _safe_float(d.expenditure_cr.sum(), 0.0)
    risk_bands = d.risk_band.value_counts().reindex(F.RISK_ORDER).fillna(0).astype(int).to_dict()
    return {
        "as_of": "April 2026",
        "total_projects": int(len(d)),
        "ministries": int(d.ministry.nunique()),
        "sectors": int(d.sector.nunique()),
        "states": int(d.state.nunique()),
        "original_cost_lakh_cr": round(total_cost / LAKH, 2),
        "revised_cost_lakh_cr": round(total_revised / LAKH, 2),
        "expenditure_lakh_cr": round(total_exp / LAKH, 2),
        "aggregate_cost_overrun_pct": round(((total_revised / total_cost) - 1) * 100 if total_cost else 0.0, 2),
        "expenditure_ratio_pct": round((total_exp / total_revised) * 100 if total_revised else 0.0, 1),
        "delayed_projects": int(len(delayed)),
        "delayed_pct": round((len(delayed) / len(d)) * 100 if len(d) else 0.0, 1),
        "avg_time_overrun_months": round(avg_time, 1),
        "risk_bands": {k: int(v) for k, v in risk_bands.items()},
        "projects_with_alerts": int((d.alert_count > 0).sum()),
        "forecast_additional_escalation_lakh_cr": round(
            _safe_float((d.original_cost_cr * d.pred_cost_overrun_pct / 100).sum() / LAKH, 0.0) -
            _safe_float((total_revised - total_cost) / LAKH, 0.0),
            2,
        ),
        "value_at_risk_lakh_cr": round(
            _safe_float(d.loc[d.risk_band.isin(["High", "Critical"]), "revised_cost_cr"].sum() / LAKH, 0.0),
            2,
        ),
    }


@app.get("/api/breakdown")
def breakdown(by: str = Query("ministry", pattern="^(ministry|sector|state|risk_band|project_status)$")):
    g = DF.groupby(by).agg(
        projects=("project_id", "count"),
        original_cost_cr=("original_cost_cr", "sum"),
        revised_cost_cr=("revised_cost_cr", "sum"),
        expenditure_cr=("expenditure_cr", "sum"),
        avg_risk_score=("risk_score", "mean"),
        avg_pred_cost_overrun=("pred_cost_overrun_pct", "mean"),
        avg_pred_time_overrun=("pred_time_overrun_months", "mean"),
        delayed=("project_status", lambda s: int(s.str.startswith("Delayed").sum())),
        critical=("risk_band", lambda s: int((s == "Critical").sum())),
    ).reset_index()
    g["cost_overrun_pct"] = ((g.revised_cost_cr / g.original_cost_cr - 1) * 100).round(2)
    g["delayed_pct"] = (g.delayed / g.projects * 100).round(1)
    for c in ("original_cost_cr", "revised_cost_cr", "expenditure_cr"):
        g[c] = g[c].round(2)
    for c in ("avg_risk_score", "avg_pred_cost_overrun", "avg_pred_time_overrun"):
        g[c] = g[c].round(2)
    return _clean(g.sort_values("revised_cost_cr", ascending=False).to_dict("records"))


# ==========================================================================
# Projects
# ==========================================================================
@app.get("/api/projects")
def projects(ministry: str = "", sector: str = "", state: str = "", risk_band: str = "",
             status: str = "", q: str = "", sort: str = "risk_score", desc: bool = True,
             page: int = 1, size: int = 25):
    d = DF
    for col, val in [("ministry", ministry), ("sector", sector), ("state", state),
                     ("risk_band", risk_band), ("project_status", status)]:
        if val:
            d = d[d[col] == val]
    if q:
        d = d[d.project_name.str.contains(q, case=False, na=False) |
              d.project_id.str.contains(q, case=False, na=False)]
    if sort not in d.columns:
        sort = "risk_score"
    d = d.sort_values(sort, ascending=not desc)
    total = len(d)
    page = max(1, page)
    sl = d.iloc[(page - 1) * size: page * size]
    cols = ["project_id", "project_name", "ministry", "sector", "state", "original_cost_cr",
            "revised_cost_cr", "expenditure_cr", "physical_progress_pct", "elapsed_schedule_pct",
            "project_status", "pred_cost_overrun_pct", "pred_time_overrun_months",
            "risk_score", "risk_band", "alert_count", "risk_rank"]
    return {"total": total, "page": page, "size": size,
            "pages": int(np.ceil(total / size)) if total else 0,
            "rows": _clean(sl[cols].to_dict("records"))}


@app.get("/api/project/{pid}")
def project(pid: str):
    r = DF[DF.project_id == pid]
    if r.empty:
        raise HTTPException(404, f"Project {pid} not found")
    rec = json.loads(r.iloc[0:1].to_json(orient="records"))[0]
    rec["alert_list"] = [] if rec["alerts"] == "No active alert" else \
        [a.strip() for a in rec["alerts"].split("|")]
    rec["drivers"] = _project_drivers(r.iloc[0])
    peers = DF[(DF.sector == rec["sector"])]
    rec["peer_benchmark"] = {
        "sector": rec["sector"], "peer_count": int(len(peers)),
        "peer_median_cost_overrun": round(float(peers.pred_cost_overrun_pct.median()), 2),
        "peer_median_time_overrun": round(float(peers.pred_time_overrun_months.median()), 2),
        "peer_median_risk_score": round(float(peers.risk_score.median()), 1),
        "percentile_risk": round(float((peers.risk_score < rec["risk_score"]).mean() * 100), 1),
    }
    return rec


class LoginIn(BaseModel):
    username: str
    password: str
    role: str


@app.post("/api/auth/login")
def login(credentials: LoginIn):
    if credentials.role not in {"mospi", "reporting_manager"}:
        raise HTTPException(400, "Unsupported login role")
    if not DB.verify_user(credentials.username, credentials.password, credentials.role):
        raise HTTPException(401, "Invalid username, password, or role")
    return {"authenticated": True, "username": credentials.username.strip().lower(), "role": credentials.role}


class ProgressReportIn(BaseModel):
    project_id: str
    project_name: str
    submitted_by: str
    role: str
    month: str
    physical_progress_pct: float = Field(..., ge=0, le=100)
    financial_progress_pct: float = Field(..., ge=0, le=100)
    notes: str = ""


@app.post("/api/progress-reports")
def create_progress_report(report: ProgressReportIn):
    if report.role not in {"mospi", "reporting_manager"}:
        raise HTTPException(403, "Only MOSPI and Reporting Manager users can submit reports")
    if report.project_id not in set(DF.project_id.astype(str)):
        raise HTTPException(404, "Project not found")
    saved = DB.create_progress_report(report.model_dump())
    return {"saved": True, "report": {**report.model_dump(), **saved}}


@app.get("/api/progress-reports/{project_id}")
def progress_reports(project_id: str):
    return {"project_id": project_id, "reports": DB.list_progress_reports(project_id)}


def _project_drivers(row):
    """Transparent, additive contribution breakdown behind the composite risk
    score. Kept rule-based on purpose: a monitoring officer must be able to see
    exactly why a project was flagged."""
    gap = max(0.0, row.elapsed_schedule_pct - row.physical_progress_pct)
    items = [
        ("Forecast cost escalation", min(row.pred_cost_overrun_pct / 60, 1) * 30,
         f"{row.pred_cost_overrun_pct:.1f}% predicted over sanctioned cost"),
        ("Forecast schedule slippage", min(max(row.pred_time_overrun_months, 0) / 36, 1) * 30,
         f"{row.pred_time_overrun_months:.1f} months predicted slippage"),
        ("Physical progress shortfall", min(gap / 45, 1) * 16,
         f"{gap:.0f} pp behind elapsed schedule"),
        ("Land acquisition gap", min((100 - row.land_acquired_pct) / 60, 1) * 10,
         f"{row.land_acquired_pct:.0f}% land acquired"),
        ("Pending statutory clearances", (0.6 * (row.forest_clearance == "Applied - Pending") +
                                          0.4 * (row.env_clearance == "Applied - Pending")) * 8,
         f"Forest: {row.forest_clearance} / Environment: {row.env_clearance}"),
        ("Expenditure vs progress", min(max(row.financial_progress_pct /
                                            max(row.physical_progress_pct, 0.5) - 1, 0), 1) * 6,
         f"{row.financial_progress_pct:.0f}% financial vs {row.physical_progress_pct:.0f}% physical"),
    ]
    return [{"factor": k, "contribution": round(float(v), 1), "detail": d}
            for k, v, d in sorted(items, key=lambda x: -x[1])]


# ==========================================================================
# Early warning
# ==========================================================================
@app.get("/api/alerts")
def alerts(severity: str = "", ministry: str = "", limit: int = 100):
    d = DF[DF.alert_count > 0]
    if severity:
        d = d[d.risk_band == severity]
    if ministry:
        d = d[d.ministry == ministry]
    d = d.sort_values(["risk_score", "alert_count"], ascending=False).head(limit)
    out = []
    for _, r in d.iterrows():
        out.append({
            "project_id": r.project_id, "project_name": r.project_name,
            "ministry": r.ministry, "sector": r.sector, "state": r.state,
            "risk_score": float(r.risk_score), "risk_band": r.risk_band,
            "revised_cost_cr": float(r.revised_cost_cr),
            "pred_cost_overrun_pct": float(r.pred_cost_overrun_pct),
            "pred_time_overrun_months": float(r.pred_time_overrun_months),
            "signals": [a.strip() for a in r.alerts.split("|")],
            "recommended_action": _recommend(r),
        })
    return {"count": len(out), "alerts": out}


def _recommend(r):
    """Prescriptive layer: map the dominant signal to a concrete intervention."""
    if r.land_acquired_pct < 70:
        return ("Escalate to the District Land Acquisition Committee; seek 3(D)/award "
                "notification timeline from the State Revenue Department before releasing "
                "the next tranche.")
    if r.forest_clearance == "Applied - Pending" or r.env_clearance == "Applied - Pending":
        return ("Place the pending clearance on the PRAGATI / MoEFCC bi-weekly follow-up list "
                "and freeze mobilisation advance until Stage-II clearance is received.")
    if r.financial_progress_pct > r.physical_progress_pct + 12:
        return ("Order a third-party quantity audit; expenditure is running ahead of verified "
                "physical work, which is a leading indicator of front-loaded billing.")
    if r.pred_time_overrun_months > 18:
        return ("Convene a Standing Committee review; re-baseline the schedule and evaluate "
                "de-scoping or splitting into additional packages to restore critical path.")
    if r.num_cost_revisions >= 2:
        return ("Refer to the competent financial authority for a Revised Cost Estimate review; "
                "repeat revisions indicate the original DPR under-estimated scope.")
    return ("Continue monthly monitoring; escalate to the Ministry nodal officer if physical "
            "progress does not recover within two reporting cycles.")


# ==========================================================================
# What-if prediction
# ==========================================================================
class PredictIn(BaseModel):
    ministry: str = "Ministry of Road Transport & Highways"
    sector: str = "Road Transport & Highways"
    state: str = "Uttar Pradesh"
    agency_type: str = "Central PSU"
    funding_mode: str = "Budgetary Support"
    contract_type: str = "EPC"
    forest_clearance: str = "Obtained"
    env_clearance: str = "Obtained"
    monitoring_frequency: str = "Monthly"
    reason_for_delay: str = "None reported"
    date_of_sanction: str = "2022-06-15"
    original_cost_cr: float = Field(1200.0, gt=0)
    original_duration_m: float = Field(48.0, gt=0)
    physical_progress_pct: float = 40.0
    financial_progress_pct: float = 45.0
    elapsed_schedule_pct: float = 55.0
    land_acquired_pct: float = 80.0
    award_lag_months: float = 7.0
    num_cost_revisions: int = 0
    num_packages: int = 3
    ext_monsoon_stoppage_days: float = 45
    ext_steel_price_index: float = 100
    ext_cement_price_index: float = 100
    ext_litigation_cases: int = 0
    ext_contractor_stress: float = 30
    ext_row_availability_pct: float = 75
    ext_local_unrest_days: int = 2


@app.post("/api/predict")
def predict(p: PredictIn):
    row = pd.DataFrame([p.model_dump()])
    out = {}
    for key, tgt in [("cost", "pred_cost_overrun_pct"), ("time", "pred_time_overrun_months")]:
        b = BUNDLES[key]
        X, _ = F.make_matrix(row, "extended")
        X = F.align_categories(X, b["categories"])[b["features"]]
        out[tgt] = round(float(b["model"].predict(X)[0]), 2)

    b = BUNDLES["risk"]
    X, _ = F.make_matrix(row, "extended")
    X = F.align_categories(X, b["categories"])[b["features"]]
    proba = b["model"].predict_proba(X)[0]
    out["risk_class"] = F.RISK_ORDER[int(proba.argmax())]
    out["risk_probabilities"] = {c: round(float(proba[i]), 4) for i, c in enumerate(F.RISK_ORDER)}

    gap = max(0.0, p.elapsed_schedule_pct - p.physical_progress_pct)
    score = 100 * (
        0.30 * min(max(out["pred_cost_overrun_pct"], 0) / 60, 1) +
        0.30 * min(max(out["pred_time_overrun_months"], 0) / 36, 1) +
        0.16 * min(gap / 45, 1) +
        0.10 * min((100 - p.land_acquired_pct) / 60, 1) +
        0.08 * (0.6 * (p.forest_clearance == "Applied - Pending") +
                0.4 * (p.env_clearance == "Applied - Pending")) +
        0.06 * min(max(p.financial_progress_pct / max(p.physical_progress_pct, 0.5) - 1, 0), 1))
    out["risk_score"] = round(min(max(score, 0), 100), 1)
    out["risk_band"] = ("Critical" if score > 72 else "High" if score > 50
                        else "Medium" if score > 25 else "Low")
    out["projected_revised_cost_cr"] = round(p.original_cost_cr * (1 + out["pred_cost_overrun_pct"] / 100), 2)
    out["projected_escalation_cr"] = round(p.original_cost_cr * out["pred_cost_overrun_pct"] / 100, 2)
    return out


# ==========================================================================
# Model transparency (technical dimensions a / b / c)
# ==========================================================================
@app.get("/api/model/performance")
def model_performance():
    return {"summary": {k: SUMMARY[k] for k in ("trained_at", "train_rows", "test_rows", "split")},
            "comparison": COMPARISON, "ablation": ABLATION, "drivers": DRIVERS}


@app.get("/api/benchmark")
def benchmark(dimension: str = "sector"):
    """Comparative analytics: how a cohort performs against the portfolio."""
    g = DF.groupby(dimension).agg(
        projects=("project_id", "count"),
        median_cost_overrun=("pred_cost_overrun_pct", "median"),
        median_time_overrun=("pred_time_overrun_months", "median"),
        p90_cost_overrun=("pred_cost_overrun_pct", lambda s: s.quantile(0.9)),
        median_risk=("risk_score", "median"),
        value_at_risk_cr=("revised_cost_cr", lambda s: 0),
    ).reset_index()
    var = DF[DF.risk_band.isin(["High", "Critical"])].groupby(dimension).revised_cost_cr.sum()
    g["value_at_risk_cr"] = g[dimension].map(var).fillna(0).round(2)
    national = {
        "median_cost_overrun": round(float(DF.pred_cost_overrun_pct.median()), 2),
        "median_time_overrun": round(float(DF.pred_time_overrun_months.median()), 2),
        "median_risk": round(float(DF.risk_score.median()), 1),
    }
    for c in ("median_cost_overrun", "median_time_overrun", "p90_cost_overrun", "median_risk"):
        g[c] = g[c].round(2)
    return {"dimension": dimension, "national": national,
            "rows": _clean(g.sort_values("median_risk", ascending=False).to_dict("records"))}


@app.get("/api/filters")
def filters():
    return {c: sorted(DF[c].dropna().unique().tolist())
            for c in ("ministry", "sector", "state", "risk_band", "project_status",
                      "agency_type", "funding_mode", "contract_type", "forest_clearance",
                      "env_clearance", "monitoring_frequency")}


# ==========================================================================
# Project Intelligence Assistant (grounded, retrieval-based)
# ==========================================================================
class AskIn(BaseModel):
    question: str


@app.post("/api/assistant")
def assistant(a: AskIn):
    """
    Retrieval-grounded assistant. It resolves the question against the LIVE
    scored portfolio and returns an answer plus the exact rows it used, so every
    statement is auditable. Set PAIMANA_LLM_ENDPOINT to route the grounded
    context through a local open-source LLM (Ollama / llama.cpp) for free-form
    phrasing - see docs/LLM_ASSISTANT.md.
    """
    q = a.question.lower().strip()
    d = DF

    def fmt(rows, cols=("project_id", "project_name", "ministry", "risk_score", "risk_band",
                        "pred_cost_overrun_pct", "pred_time_overrun_months", "revised_cost_cr")):
        return _clean(rows[list(cols)].to_dict("records"))

    # entity resolution over ministries / sectors / states
    ent, entcol = None, None
    for col in ("ministry", "sector", "state"):
        for v in d[col].unique():
            toks = [t for t in v.lower().replace("&", " ").replace(",", " ").split() if len(t) > 4]
            if toks and sum(t in q for t in toks) >= max(1, len(toks) // 2):
                ent, entcol = v, col
                break
        if ent:
            break
    scope = d[d[entcol] == ent] if ent else d
    scope_label = f"{ent}" if ent else "the national portfolio"

    if any(k in q for k in ("riskiest", "highest risk", "top risk", "most at risk", "critical")):
        r = scope.nlargest(10, "risk_score")
        ans = (f"The 10 highest-risk projects in {scope_label} carry risk scores from "
               f"{r.risk_score.max():.0f} down to {r.risk_score.min():.0f}. Together they account for "
               f"₹{r.revised_cost_cr.sum():,.0f} crore of revised cost. The dominant signal is "
               f"'{r.reason_for_delay.mode().iloc[0]}'.")
        return {"answer": ans, "evidence": fmt(r), "scope": scope_label}

    if any(k in q for k in ("cost overrun", "escalation", "over budget", "cost")):
        agg = (scope.revised_cost_cr.sum() / scope.original_cost_cr.sum() - 1) * 100
        fwd = (scope.original_cost_cr * scope.pred_cost_overrun_pct / 100).sum()
        ans = (f"{scope_label} covers {len(scope):,} projects with a sanctioned cost of "
               f"₹{scope.original_cost_cr.sum()/LAKH:,.2f} lakh crore. Cost overrun booked so far is "
               f"{agg:.1f}%. The model forecasts total escalation of ₹{fwd/LAKH:,.2f} lakh crore by "
               f"completion, i.e. a further ₹{(fwd - (scope.revised_cost_cr.sum()-scope.original_cost_cr.sum()))/LAKH:,.2f} "
               f"lakh crore beyond what has already been revised.")
        return {"answer": ans, "evidence": fmt(scope.nlargest(8, "pred_cost_overrun_pct")), "scope": scope_label}

    if any(k in q for k in ("delay", "time overrun", "slippage", "behind schedule", "late")):
        dl = scope[scope.project_status.str.startswith("Delayed")]
        ans = (f"{len(dl):,} of {len(scope):,} projects ({len(dl)/max(len(scope),1)*100:.0f}%) in "
               f"{scope_label} are running behind schedule, with a mean slippage of "
               f"{dl.time_overrun_months.mean():.1f} months. The most frequently cited reason is "
               f"'{dl.reason_for_delay.mode().iloc[0]}'. The model predicts a further "
               f"{scope.pred_time_overrun_months.median():.1f} months of median slippage portfolio-wide.")
        return {"answer": ans, "evidence": fmt(dl.nlargest(8, "pred_time_overrun_months")), "scope": scope_label}

    if any(k in q for k in ("land", "acquisition")):
        la = scope[scope.land_acquired_pct < 75]
        ans = (f"{len(la):,} projects in {scope_label} report land acquisition below 75%. They hold "
               f"₹{la.revised_cost_cr.sum()/LAKH:,.2f} lakh crore of revised cost and carry a mean risk "
               f"score of {la.risk_score.mean():.0f} against a portfolio mean of {d.risk_score.mean():.0f}.")
        return {"answer": ans, "evidence": fmt(la.nlargest(8, "risk_score")), "scope": scope_label}

    if any(k in q for k in ("clearance", "forest", "environment")):
        cl = scope[(scope.forest_clearance == "Applied - Pending") |
                   (scope.env_clearance == "Applied - Pending")]
        ans = (f"{len(cl):,} projects in {scope_label} are held up on a pending statutory clearance, "
               f"blocking ₹{cl.revised_cost_cr.sum()/LAKH:,.2f} lakh crore. Their predicted slippage "
               f"averages {cl.pred_time_overrun_months.mean():.1f} months versus "
               f"{scope.pred_time_overrun_months.mean():.1f} months for the cohort.")
        return {"answer": ans, "evidence": fmt(cl.nlargest(8, "risk_score")), "scope": scope_label}

    if any(k in q for k in ("driver", "why", "cause", "reason")):
        drv = DRIVERS.get("cost_overrun_pct|extended", [])[:6]
        rs = scope.reason_for_delay.value_counts().head(5)
        ans = ("Across the training corpus the strongest predictors of cost escalation are: " +
               ", ".join(f"{x['feature']} ({x['gain_pct']}% of model gain)" for x in drv) +
               f". Within {scope_label} the most frequently reported delay reasons are: " +
               ", ".join(f"{k} ({v})" for k, v in rs.items()) + ".")
        return {"answer": ans, "evidence": fmt(scope.nlargest(6, "risk_score")), "scope": scope_label}

    if ent:
        ans = (f"{ent} has {len(scope):,} ongoing projects worth ₹{scope.revised_cost_cr.sum()/LAKH:,.2f} "
               f"lakh crore (revised). {(scope.risk_band.isin(['High','Critical'])).sum()} are High or "
               f"Critical risk. Mean risk score {scope.risk_score.mean():.0f} vs national "
               f"{d.risk_score.mean():.0f}. Predicted median slippage "
               f"{scope.pred_time_overrun_months.median():.1f} months.")
        return {"answer": ans, "evidence": fmt(scope.nlargest(8, "risk_score")), "scope": scope_label}

    k = kpis()
    return {"answer": (f"PAIMANA is monitoring {k['total_projects']:,} ongoing projects across "
                       f"{k['ministries']} Ministries and {k['sectors']} sectors, with a revised cost of "
                       f"₹{k['revised_cost_lakh_cr']} lakh crore against a sanctioned "
                       f"₹{k['original_cost_lakh_cr']} lakh crore ({k['aggregate_cost_overrun_pct']}% overrun). "
                       f"{k['risk_bands'].get('Critical',0)} projects are Critical risk and "
                       f"{k['projects_with_alerts']:,} have at least one active early-warning signal. "
                       f"Ask me about a Ministry, sector or state, about delays, cost overruns, land "
                       f"acquisition or clearances."),
            "evidence": fmt(DF.nlargest(6, "risk_score")), "scope": "national"}


# ==========================================================================
# Static frontend
# ==========================================================================
@app.get("/", include_in_schema=False)
def frontend_login():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "projects": len(DF), "models": list(BUNDLES)}


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
