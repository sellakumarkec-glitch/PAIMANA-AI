"""
PAIMANA-AI :: Synthetic-but-calibrated CUF dataset builder
==========================================================

WHY THIS FILE EXISTS
--------------------
The PAIMANA portal (https://paimana-proj.mospi.gov.in) does not expose a public
bulk-download / open API for project-level rows. The MoSPI Flash Reports publish
*aggregates* only. So this script builds a project-level corpus that is
STRUCTURALLY IDENTICAL to the PAIMANA Common Upload Form (CUF) schema and is
NUMERICALLY CALIBRATED to the published MoSPI aggregates:

    As of April 2026 (problem statement + MoSPI flash report):
      * 1,981 ongoing projects
      * 17 Central Ministries / Departments
      * 22 infrastructure sectors
      * Original cost      ~ Rs 37.13 lakh crore
      * Revised cost       ~ Rs 42.78 lakh crore   (=> ~15.2% aggregate cost overrun)
      * Cumulative expend. ~ Rs 20.36 lakh crore   (=> ~47.6% of revised cost)

    Historical OCMS era (2006-2025): ~9,000 completed projects used as the
    TRAINING corpus. The 1,981 ongoing projects are the INFERENCE corpus.

IMPORTANT: swap this file out for the real extract when you get PAIMANA
credentials. `ml/train.py` and `backend/app.py` only depend on the COLUMN NAMES,
not on this generator. See docs/DATA_SWAP.md.

The generator uses a *causal* data-generating process (land acquisition ->
delay -> escalation -> cost overrun) so that the ML models learn genuine,
explainable structure rather than noise.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(20260417)
OUT = Path(__file__).parent

# --------------------------------------------------------------------------
# 1. Reference masters (mirrors PAIMANA ministry / sector masters)
# --------------------------------------------------------------------------

# ministry -> (sector, share of portfolio, base cost scale in Rs crore)
MINISTRY_SECTOR = [
    ("Ministry of Road Transport & Highways", "Road Transport & Highways", 0.298, 1250),
    ("Ministry of Railways",                  "Railways",                  0.186, 2400),
    ("Ministry of Petroleum & Natural Gas",   "Petroleum",                 0.121, 3100),
    ("Ministry of Coal",                      "Coal",                      0.057, 1050),
    ("Ministry of Power",                     "Power",                     0.052, 4200),
    ("Ministry of Ports, Shipping & Waterways", "Ports & Shipping",        0.041,  900),
    ("Ministry of Jal Shakti",                "Water Resources",           0.036, 1400),
    ("Ministry of Housing & Urban Affairs",   "Urban Development",         0.035, 2900),
    ("Department of Telecommunications",      "Telecommunication",         0.027, 1800),
    ("Ministry of Civil Aviation",            "Civil Aviation",            0.024, 1150),
    ("Ministry of Steel",                     "Steel",                     0.022, 2600),
    ("Department of Atomic Energy",           "Atomic Energy",             0.018, 9500),
    ("Ministry of Mines",                     "Mining",                    0.017,  620),
    ("Ministry of New & Renewable Energy",    "Renewable Energy",          0.016, 1600),
    ("Department of Fertilizers",             "Fertilizers",               0.014, 3400),
    ("Ministry of Health & Family Welfare",   "Social Infrastructure",     0.021,  700),
    ("Department of Heavy Industry",          "Heavy Industry",            0.015,  850),
]
# Secondary sectors so that we reach the 22-sector coverage of PAIMANA
EXTRA_SECTORS = {
    "Ministry of Road Transport & Highways": ["Bridges & Tunnels"],
    "Ministry of Railways": ["Railway Electrification", "Dedicated Freight Corridor"],
    "Ministry of Petroleum & Natural Gas": ["Natural Gas Pipeline"],
    "Ministry of Housing & Urban Affairs": ["Urban Metro Rail"],
}

STATES = ["Uttar Pradesh","Maharashtra","Gujarat","Rajasthan","Madhya Pradesh","Tamil Nadu",
          "Karnataka","Andhra Pradesh","Telangana","West Bengal","Bihar","Odisha","Jharkhand",
          "Chhattisgarh","Kerala","Assam","Punjab","Haryana","Uttarakhand","Himachal Pradesh",
          "Jammu & Kashmir","Goa","Delhi","Multi-State"]
# Terrain / governance difficulty multiplier (drives delay hazard)
STATE_FRICTION = {s: v for s, v in zip(STATES, [
    1.15,1.05,0.85,0.95,1.00,0.90,0.92,1.02,0.98,1.28,1.32,1.10,1.22,1.18,1.08,1.35,
    0.94,0.90,1.30,1.33,1.45,0.88,1.12,1.20])}

AGENCY_TYPE   = ["Central PSU", "Ministry / Department", "SPV / Joint Venture",
                 "Autonomous Body", "State Implementing Agency"]
FUNDING_MODE  = ["Budgetary Support", "Internal & Extra Budgetary Resources (IEBR)",
                 "EAP / Multilateral Loan", "PPP - BOT", "PPP - HAM", "Mixed"]
CONTRACT_TYPE = ["EPC", "Item Rate", "Turnkey / LSTK", "BOT (Toll)", "HAM", "Departmental"]
CLEARANCE     = ["Obtained", "Applied - Pending", "Not Required", "Partially Obtained"]

# CUF "reason for delay" master (matches MoSPI flash report reason codes)
DELAY_REASONS = [
    "Land acquisition problems", "Forest / Environment clearance delay",
    "Delay in tendering / award of work", "Contractual issues with agency",
    "Fund constraints", "Law & order / local agitation",
    "Shortage of construction material / labour", "Geological surprises",
    "Change in scope / design", "Utility shifting & Right of Way",
    "Statutory / regulatory approvals", "Adverse weather / monsoon",
    "Rehabilitation & resettlement issues", "None reported",
]

N_HIST    = 9000   # completed OCMS-era projects (2006-2025)  -> training corpus
N_ONGOING = 1981   # PAIMANA ongoing projects as of Apr-2026  -> inference corpus


# --------------------------------------------------------------------------
# 2. Helpers
# --------------------------------------------------------------------------
def _pick_ministry(n):
    names = [m[0] for m in MINISTRY_SECTOR]
    p = np.array([m[2] for m in MINISTRY_SECTOR], dtype=float)
    p = p / p.sum()
    return RNG.choice(names, size=n, p=p)


def _sector_for(ministry):
    base = {m[0]: m[1] for m in MINISTRY_SECTOR}[ministry]
    pool = [base] * 3 + EXTRA_SECTORS.get(ministry, [])
    return RNG.choice(pool)


def _cost_scale(ministry):
    return {m[0]: m[3] for m in MINISTRY_SECTOR}[ministry]


def _project_name(sector, state, i):
    verbs = {
        "Road Transport & Highways": ["4-Laning of NH-{n} ({a}-{b}) Section", "6-Laning of NH-{n} {a} Bypass",
                                      "Greenfield Access Controlled Corridor Pkg-{p}"],
        "Bridges & Tunnels": ["Construction of Major Bridge over {a} River", "{a} Tunnel Project Pkg-{p}"],
        "Railways": ["New BG Line {a}-{b} ({k} km)", "Doubling of {a}-{b} Section", "Gauge Conversion {a}-{b}"],
        "Railway Electrification": ["Electrification of {a}-{b} Section"],
        "Dedicated Freight Corridor": ["DFC Pkg-{p} {a}-{b} Section"],
        "Petroleum": ["{a} Refinery Expansion Unit-{p}", "Crude Oil Terminal at {a}"],
        "Refinery": ["Residue Upgradation Facility at {a} Refinery"],
        "Natural Gas Pipeline": ["{a}-{b} Natural Gas Pipeline ({k} km)"],
        "Coal": ["{a} Opencast Expansion ({k} MTPA)", "{a} Underground Mine Project"],
        "Mining": ["{a} Mineral Block Development", "Beneficiation Plant at {a}"],
        "Power": ["{a} Super Thermal Power Project Stage-{p}", "{a} Hydro Electric Project ({k} MW)"],
        "Power Transmission": ["{a}-{b} 765 kV D/C Transmission Line"],
        "Renewable Energy": ["{a} Ultra Mega Solar Park ({k} MW)", "{a} Wind-Solar Hybrid Park"],
        "Ports & Shipping": ["Development of Berth No.{p} at {a} Port", "Capital Dredging at {a} Port"],
        "Civil Aviation": ["New Integrated Terminal Building at {a} Airport", "Runway Extension at {a} Airport"],
        "Water Resources": ["{a} Major Irrigation Project Phase-{p}", "{a} Barrage & Canal System"],
        "Irrigation": ["{a} Lift Irrigation Scheme Phase-{p}"],
        "Urban Development": ["{a} Smart City Infrastructure Pkg-{p}"],
        "Urban Metro Rail": ["{a} Metro Rail Phase-{p} ({k} km)"],
        "Water & Sanitation": ["{a} Sewerage & STP Project ({k} MLD)"],
        "Telecommunication": ["BharatNet Phase-{p} in {a}", "Optical Fibre Backbone for {a}"],
        "Steel": ["Expansion of {a} Steel Plant to {k} MTPA", "New Blast Furnace at {a} Works"],
        "Atomic Energy": ["{a} Atomic Power Project Unit-{p}"],
        "Fertilizers": ["Revival of {a} Fertilizer Plant", "Ammonia-Urea Complex at {a}"],
        "Social Infrastructure": ["New AIIMS at {a}", "Upgradation of Govt. Medical College, {a}"],
        "Heavy Industry": ["Modernisation of {a} Heavy Engineering Unit"],
    }
    towns = ["Barauni","Talcher","Raichur","Vizag","Haldia","Kota","Dhule","Nashik","Guwahati",
             "Tuticorin","Salem","Rourkela","Bhilai","Kandla","Paradip","Jhansi","Gorakhpur",
             "Silchar","Bilaspur","Mangaluru","Nagpur","Patna","Indore","Jaipur","Kochi","Surat"]
    tpl = RNG.choice(verbs.get(sector, ["{a} Infrastructure Project Pkg-{p}"]))
    return tpl.format(n=int(RNG.integers(2, 766)), a=RNG.choice(towns), b=RNG.choice(towns),
                      p=int(RNG.integers(1, 9)), k=int(RNG.integers(8, 1400)))


# --------------------------------------------------------------------------
# 3. The causal data generating process
# --------------------------------------------------------------------------
def build(n, completed: bool, id_prefix: str, start_year_lo: int, start_year_hi: int):
    ministry = _pick_ministry(n)
    sector   = np.array([_sector_for(m) for m in ministry])
    state    = RNG.choice(STATES, size=n,
                          p=np.array([1.4,1.2,1.0,0.9,0.9,0.9,0.9,0.8,0.7,0.8,0.8,0.8,0.5,0.5,
                                      0.5,0.5,0.5,0.5,0.35,0.3,0.3,0.15,0.4,0.6]) /
                            np.sum([1.4,1.2,1.0,0.9,0.9,0.9,0.9,0.8,0.7,0.8,0.8,0.8,0.5,0.5,
                                    0.5,0.5,0.5,0.5,0.35,0.3,0.3,0.15,0.4,0.6]))

    # ---- Sanction / cost -------------------------------------------------
    scale = np.array([_cost_scale(m) for m in ministry])
    # log-normal cost, floor at Rs 150 crore (PAIMANA monitoring threshold)
    original_cost = np.clip(scale * RNG.lognormal(-0.35, 0.95, n), 150, 260000).round(2)

    sanction_year  = RNG.integers(start_year_lo, start_year_hi + 1, n)
    sanction_month = RNG.integers(1, 13, n)
    sanction = pd.to_datetime(dict(year=sanction_year, month=sanction_month, day=15))

    # Original schedule: bigger & heavier sectors take longer
    heavy = np.isin(sector, ["Atomic Energy","Power","Steel","Refinery","Urban Metro Rail",
                             "Water Resources","Irrigation","Dedicated Freight Corridor"])
    orig_dur = (24 + 14 * np.log1p(original_cost / 300) + heavy * 18
                + RNG.normal(0, 7, n)).clip(12, 132).round()

    # ---- CUF governance / readiness fields -------------------------------
    agency_type   = RNG.choice(AGENCY_TYPE, n, p=[.40,.18,.16,.10,.16])
    funding_mode  = RNG.choice(FUNDING_MODE, n, p=[.34,.24,.12,.08,.12,.10])
    contract_type = RNG.choice(CONTRACT_TYPE, n, p=[.38,.17,.13,.10,.12,.10])

    land_pct = np.clip(RNG.beta(5.2, 1.9, n) * 100, 5, 100).round(1)
    forest_cl = RNG.choice(CLEARANCE, n, p=[.52,.24,.16,.08])
    env_cl    = RNG.choice(CLEARANCE, n, p=[.58,.20,.14,.08])
    award_lag = np.clip(RNG.gamma(2.2, 3.1, n), 0, 48).round(1)   # sanction -> award, months
    n_revisions_final = RNG.poisson(0.62, n)
    monitoring_freq = RNG.choice(["Monthly","Quarterly","Fortnightly"], n, p=[.72,.20,.08])
    pkg_count = np.clip(RNG.poisson(2.4, n) + 1, 1, 22)

    # ---- EXTENDED (non-CUF) variables, for the CUF-vs-Extended ablation ----
    monsoon_days      = np.clip(RNG.normal(46, 22, n) * np.array([STATE_FRICTION[s] for s in state]), 0, 160).round()
    steel_index       = np.clip(RNG.normal(100, 17, n), 55, 185).round(1)   # input-cost index at award
    cement_index      = np.clip(RNG.normal(100, 12, n), 62, 170).round(1)
    litigation_cases  = RNG.poisson(0.9 + 1.8 * (land_pct < 70), n)
    contractor_stress = np.clip(RNG.beta(2.3, 4.2, n) * 100, 0, 100).round(1)  # 0=healthy,100=distressed
    local_unrest_days = RNG.poisson(3.0 * np.array([STATE_FRICTION[s] for s in state]), n)

    friction = np.array([STATE_FRICTION[s] for s in state])
    row_latent = np.clip(land_pct - RNG.gamma(2, 4, n), 0, 100)   # latent right-of-way

    # ---- LATENT DELAY HAZARD (the causal core) ---------------------------
    z = (
        -3.62
        + 0.026 * (100 - land_pct)                       # land not acquired -> delay
        + 0.036 * award_lag                              # slow award -> delay
        + 0.62  * (forest_cl == "Applied - Pending")
        + 0.38  * (env_cl == "Applied - Pending")
        + 0.30  * np.log1p(original_cost / 1000)
        + 0.75  * (friction - 1.0)
        + 0.48  * n_revisions_final
        + 0.18  * (orig_dur / 24)
        + 0.34  * (contract_type == "Item Rate")
        + 0.32  * (funding_mode == "PPP - BOT")
        - 0.26  * (agency_type == "Central PSU")
        + 0.020 * local_unrest_days
        # ---- EXTENDED (non-CUF) signals: materially predictive ----
        + 0.058 * (monsoon_days - 46) / 10
        + 0.085 * contractor_stress / 10
        + 0.42  * litigation_cases
        - 0.048 * (row_latent - 60) / 10
        + RNG.normal(0, 0.46, n)                         # irreducible noise
    )
    p_delay = 1 / (1 + np.exp(-z))
    is_delayed = RNG.random(n) < p_delay

    # Time overrun (months) - heavy right tail, as in MoSPI flash reports
    time_overrun = np.where(
        is_delayed,
        np.clip(RNG.gamma(2.6, 1.0, n) * (4.0 + 11.0 * p_delay), 1, 192),
        np.clip(RNG.normal(-0.4, 1.4, n), -6, 1)
    ).round(1)

    # Cost overrun % is CAUSED by time overrun + input-price drift + scope change
    esc = (
        1.4
        + 0.74 * np.maximum(time_overrun, 0)                  # escalation with time
        + 0.15 * (steel_index - 100)
        + 0.08 * (cement_index - 100)
        + 5.6  * n_revisions_final                            # scope revisions
        + 2.3  * (contract_type == "Item Rate")
        + 0.055 * (monsoon_days - 46)
        + 0.052 * contractor_stress
        + 1.4  * (friction - 1.0) * 10
        + RNG.normal(0, 4.4, n)
    )
    cost_overrun_pct = np.clip(esc, -8, 320).round(2)
    cost_overrun_pct = np.where(RNG.random(n) < 0.20, np.clip(RNG.normal(0.0, 0.8, n), -2, 2).round(2),
                                cost_overrun_pct)

    revised_cost = (original_cost * (1 + cost_overrun_pct / 100)).round(2)

    # ---- Dates -----------------------------------------------------------
    orig_completion = sanction + pd.to_timedelta(orig_dur * 30.44, unit="D")
    anticipated     = orig_completion + pd.to_timedelta(np.maximum(time_overrun, 0) * 30.44, unit="D")

    # ======================================================================
    # MONITORING SNAPSHOT
    # ----------------------------------------------------------------------
    # An early-warning system must learn from what a project LOOKED LIKE
    # mid-flight, not from its finished state. So every historical project is
    # frozen at a random monitoring month (25-85% of its sanctioned schedule)
    # and labelled with its FINAL outcome. Ongoing projects are frozen at the
    # PAIMANA as-of date. Both corpora therefore share the same feature
    # distribution -> no train/serve skew.
    # ======================================================================
    if completed:
        f = RNG.uniform(0.25, 0.88, n)                     # snapshot fraction of schedule
        elapsed_pct = (f * 100).round(1)
        snapshot_date = sanction + pd.to_timedelta(orig_dur * f * 30.44, unit="D")
        status = np.array(["Completed"] * n)
    else:
        asof = pd.Timestamp("2026-04-30")
        elapsed = (asof - sanction).dt.days / 30.44
        elapsed_pct = np.clip(elapsed / orig_dur * 100, 4, 99).round(1)
        f = elapsed_pct / 100
        snapshot_date = pd.Series([asof] * n)
        status = np.where(time_overrun > 12, "Delayed (>12 months)",
                  np.where(time_overrun > 0.5, "Delayed (<=12 months)",
                   np.where(time_overrun < -2, "Ahead of Schedule", "On Schedule")))

    # Physical progress at the snapshot lags elapsed schedule when the project
    # is on a delayed trajectory - this is the core early-warning signal.
    severity = np.clip(np.maximum(time_overrun, 0) / orig_dur, 0, 0.9)
    lag = np.where(is_delayed, np.clip(1.0 - (0.30 + 0.72 * severity), 0.18, 1.0),
                   RNG.uniform(0.94, 1.07, n))
    phys = np.clip(elapsed_pct * lag + RNG.normal(0, 4.2, n), 0.4, 99.5).round(1)
    fin  = np.clip(phys * RNG.uniform(0.88, 1.15, n) + 1.6 * n_revisions_final * f,
                   0.4, 99.9).round(1)
    expenditure = (revised_cost * fin / 100).round(2)

    # Cost revisions APPROVED SO FAR (not the final count - that is unknown at
    # monitoring time). Revisions accumulate as the project progresses.
    n_revisions = RNG.binomial(n_revisions_final, np.clip(f * 1.05, 0, 1))

    # Land acquisition reported at snapshot: it improves over the project life.
    land_reported = np.clip(land_pct * (0.52 + 0.48 * f) + RNG.normal(0, 3.0, n), 2, 100).round(1)
    row_availability = np.clip(land_reported - RNG.gamma(2, 3.4, n), 0, 100).round(1)

    # ---- Reason for delay (CUF coded field) ------------------------------
    reason = np.full(n, "None reported", dtype=object)
    di = np.where(is_delayed)[0]
    w = np.array([.19,.11,.13,.10,.07,.05,.07,.04,.09,.06,.04,.03,.02,.0]); w = w / w.sum()
    reason[di] = RNG.choice(DELAY_REASONS, size=len(di), p=w)
    # keep the coded reason consistent with the dominant causal driver
    reason[np.where(is_delayed & (land_reported < 55))[0]] = "Land acquisition problems"
    reason[np.where(is_delayed & (forest_cl == "Applied - Pending") & (land_reported >= 55))[0]] = \
        "Forest / Environment clearance delay"
    reason[np.where(is_delayed & (litigation_cases >= 3))[0]] = "Land acquisition problems"

    df = pd.DataFrame({
        # ---------- CUF CORE FIELDS ----------
        "project_id":            [f"{id_prefix}{i:06d}" for i in range(1, n + 1)],
        "project_name":          [_project_name(s, st, i) for i, (s, st) in enumerate(zip(sector, state))],
        "ministry":              ministry,
        "sector":                sector,
        "state":                 state,
        "agency_type":           agency_type,
        "funding_mode":          funding_mode,
        "contract_type":         contract_type,
        "date_of_sanction":      sanction.dt.strftime("%Y-%m-%d"),
        "original_cost_cr":      original_cost,
        "revised_cost_cr":       revised_cost,
        "expenditure_cr":        expenditure,
        "original_completion":   orig_completion.dt.strftime("%Y-%m-%d"),
        "anticipated_completion": anticipated.dt.strftime("%Y-%m-%d"),
        "original_duration_m":   orig_dur,
        "physical_progress_pct": phys,
        "financial_progress_pct": fin,
        "elapsed_schedule_pct":  elapsed_pct,
        "land_acquired_pct":     land_reported,
        "forest_clearance":      forest_cl,
        "env_clearance":         env_cl,
        "award_lag_months":      award_lag,
        "num_cost_revisions":    n_revisions,
        "snapshot_date":         pd.Series(snapshot_date).dt.strftime("%Y-%m-%d").values,
        "num_packages":          pkg_count,
        "monitoring_frequency":  monitoring_freq,
        "reason_for_delay":      reason,
        "project_status":        status,
        # ---------- EXTENDED (NOT IN CUF) ----------
        "ext_monsoon_stoppage_days": monsoon_days,
        "ext_steel_price_index":     steel_index,
        "ext_cement_price_index":    cement_index,
        "ext_litigation_cases":      litigation_cases,
        "ext_contractor_stress":     contractor_stress,
        "ext_row_availability_pct":  row_availability,
        "ext_local_unrest_days":     local_unrest_days,
        # ---------- TARGETS ----------
        "cost_overrun_pct":      cost_overrun_pct,
        "time_overrun_months":   time_overrun,
    })
    return df


# --------------------------------------------------------------------------
# 4. Calibrate ongoing portfolio to the published MoSPI aggregates
# --------------------------------------------------------------------------
def calibrate(df, target_orig_lakh_cr, target_rev_lakh_cr, target_exp_lakh_cr):
    """Scale costs so portfolio totals match the MoSPI Flash Report (Apr-2026)."""
    k = (target_orig_lakh_cr * 1e5) / df["original_cost_cr"].sum()
    df["original_cost_cr"] = (df["original_cost_cr"] * k).round(2)
    # Scale the OVERRUN PERCENTAGES (not the revised cost) so that projects with
    # zero escalation keep exactly zero escalation after calibration.
    need  = target_rev_lakh_cr * 1e5 - df["original_cost_cr"].sum()
    have  = (df["original_cost_cr"] * df["cost_overrun_pct"] / 100).sum()
    alpha = need / have
    df["cost_overrun_pct"] = (df["cost_overrun_pct"] * alpha).round(2)
    df["revised_cost_cr"]  = (df["original_cost_cr"] * (1 + df["cost_overrun_pct"] / 100)).round(2)
    df["expenditure_cr"] = (df["revised_cost_cr"] * df["financial_progress_pct"] / 100).round(2)
    k3 = (target_exp_lakh_cr * 1e5) / df["expenditure_cr"].sum()
    df["expenditure_cr"] = (df["expenditure_cr"] * k3).round(2)
    df["financial_progress_pct"] = (df["expenditure_cr"] / df["revised_cost_cr"] * 100).clip(0.3, 99.9).round(1)
    return df


def main():
    hist = build(N_HIST, completed=True, id_prefix="OCMS", start_year_lo=2006, start_year_hi=2021)
    ong  = build(N_ONGOING, completed=False, id_prefix="PMN", start_year_lo=2016, start_year_hi=2025)
    ong  = calibrate(ong, 37.13, 42.78, 20.36)

    hist["corpus"] = "historical_completed"
    ong["corpus"]  = "paimana_ongoing"

    hist.to_csv(OUT / "projects_historical.csv", index=False)
    ong.to_csv(OUT / "projects_ongoing.csv", index=False)
    pd.concat([hist, ong], ignore_index=True).to_csv(OUT / "projects_all.csv", index=False)

    print("=" * 66)
    print("PAIMANA-AI dataset built")
    print("=" * 66)
    print(f"Historical (training)  : {len(hist):>6,} completed projects 2006-2021 sanction")
    print(f"Ongoing (inference)    : {len(ong):>6,} projects")
    print(f"Ministries             : {ong.ministry.nunique()}")
    print(f"Sectors                : {ong.sector.nunique()}")
    print(f"Original cost   (L cr) : {ong.original_cost_cr.sum()/1e5:>8.2f}   [target 37.13]")
    print(f"Revised  cost   (L cr) : {ong.revised_cost_cr.sum()/1e5:>8.2f}   [target 42.78]")
    print(f"Expenditure     (L cr) : {ong.expenditure_cr.sum()/1e5:>8.2f}   [target 20.36]")
    print(f"Aggregate cost overrun : {(ong.revised_cost_cr.sum()/ong.original_cost_cr.sum()-1)*100:>7.2f}%")
    print(f"Delayed projects       : {(ong.time_overrun_months>0).mean()*100:>7.1f}%")
    print(f"Mean time overrun (m)  : {ong.loc[ong.time_overrun_months>0,'time_overrun_months'].mean():>7.1f}")


if __name__ == "__main__":
    main()
