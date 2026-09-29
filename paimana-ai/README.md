# PAIMANA-AI — Predictive Analytics & Early Warning System for Infrastructure Project Monitoring

An AI decision-support layer built on top of the **PAIMANA** framework of the Infrastructure &
Project Monitoring Division (IPMD), Ministry of Statistics and Programme Implementation,
Government of India.

PAIMANA today answers *"what has happened"*. This project answers *"what is about to happen,
to which project, and what should be done about it"* — turning a descriptive reporting system
into a predictive and prescriptive one.

Everything here is open source: Python, pandas, scikit-learn, LightGBM, FastAPI, Chart.js.
No paid services, no external API keys, no cloud dependency.

---

## 1. Quick start

```bash
cd paimana-ai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python data/generate_dataset.py     # build the CUF-schema corpus   (~10 s)
python ml/train.py                  # train + evaluate + score      (~90 s)
python ml/export_static.py          # write static JSON snapshots   (~20 s)

uvicorn backend.app:app --port 8000
```

Then open **http://localhost:8000**. API docs live at `/api/docs`.

`./run.sh` (or `run.bat` on Windows) does all four steps in order.

**Static mode — no Python needed.** After `export_static.py`, the whole site runs off any
static web server, because the JavaScript falls back to the pre-computed JSON snapshots when the
API is unreachable:

```bash
cd frontend && python -m http.server 8080      # or nginx, or GitHub Pages, or NIC static hosting
```

All eight pages, all charts and the assistant work in this mode. The only feature that degrades
is what-if prediction, which switches to a transparent distilled surrogate (clearly labelled in
the UI). **It must be a server, not `file://`** — browsers block `fetch()` on file origins, so
double-clicking `index.html` will show empty panels.

---

## 2. What you get

| Problem-statement outcome | Where it lives |
|---|---|
| (a) Cost Overrun Prediction Model | `ml/train.py` → `models/lgbm_cost_overrun_pct_*.pkl` |
| (b) Time Overrun Prediction Model | `ml/train.py` → `models/lgbm_time_overrun_months_*.pkl` |
| (c) Project Risk Scoring Framework | composite 0–100 score, `train.py::score_ongoing` + `/api/predict` |
| (d) Early Warning Alert System | `backend/app.py::alerts` + `frontend/early-warning.html` |
| (e) Benchmarking & Comparative Analytics | `/api/benchmark` + `frontend/benchmarking.html` |
| (f) Cost Escalation Driver Analysis | `reports/importance_*.csv` + `frontend/model.html#drivers` |
| (g) AI-powered Monitoring Dashboard | `frontend/dashboard.html` |
| (h) LLM-enabled Project Intelligence Assistant | `/api/assistant` + `frontend/assistant.html` |
| (i) Documentation & deployment framework | this file + `docs/` + `run.sh` |

Pages: home (`index.html`), monitoring dashboard, early-warning queue, project deep-dive,
benchmarking, what-if prediction, assistant, and a model-transparency page.

---

## 3. The data question — read this first

**PAIMANA publishes aggregates, not project rows.** There is no public bulk download or open
API, and the portal disallows automated access. So this repository ships
`data/generate_dataset.py`, which builds a corpus that is:

* **structurally identical** to the PAIMANA Common Upload Form (CUF) schema, and
* **numerically calibrated** to the published MoSPI Flash Report aggregates for April 2026:

| Published figure | Target | This dataset |
|---|---|---|
| Ongoing projects | 1,981 | 1,981 |
| Ministries / Departments | 17 | 17 |
| Infrastructure sectors | 22 | 22 |
| Original cost | ₹37.13 lakh cr | ₹37.13 lakh cr |
| Revised cost | ₹42.78 lakh cr | ₹42.78 lakh cr |
| Cumulative expenditure | ₹20.36 lakh cr | ₹20.36 lakh cr |
| Aggregate cost overrun | ~15.2% | 15.22% |

Plus **9,000 historical OCMS-era completed projects (2006–2021 sanction)** as the training
corpus, representing the "nearly two decades of project histories" the problem statement
describes.

The generator is **causal, not random**: land acquisition and clearance status drive a latent
delay hazard; the delay hazard drives time overrun; time overrun plus commodity prices plus
scope revisions drive cost overrun. That means the models learn genuine, explainable structure
instead of memorising noise, and the driver analysis is meaningful.

**When you get real PAIMANA data, swap it in.** `ml/train.py` and `backend/app.py` depend only
on column *names*, not on the generator. See `docs/DATA_SWAP.md` for the 3-step procedure.

---

## 4. The methodological decision that matters most

An early-warning system must learn from what a project **looked like mid-flight**, not from its
finished state.

The obvious approach — train on completed projects — silently breaks. In a completed project
`physical_progress_pct` is always 100 and `elapsed_schedule_pct` is always 100, so the single
most diagnostic feature (*progress behind schedule*) is **constant during training** and
**variable at inference**. The model never learns to use it. That is textbook train/serve skew,
and the first version of this pipeline had exactly that bug.

The fix: every historical project is frozen at a **random monitoring month** between 25% and 88%
of its sanctioned schedule, with progress, expenditure, land status and cost revisions recorded
*as of that month*, and labelled with its **final** outcome. Ongoing projects are frozen at the
PAIMANA as-of date. Both corpora now share one feature distribution.

Effect on held-out schedule prediction: **R² 0.49 → 0.80**.

---

## 5. Model performance

Trained on 7,200 projects, tested on a **temporal hold-out** of 1,800 (split by date of
sanction — we never train on the future and score the past).

### (a) Predictive models

| Target | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| Cost overrun (pp) | OLS | 7.76 | — | — |
| Cost overrun (pp) | Ridge | 7.76 | — | — |
| Cost overrun (pp) | **LightGBM** | **7.22** | 10.65 | **0.464** |
| Time overrun (months) | OLS | 6.06 | — | — |
| Time overrun (months) | **LightGBM** | **3.36** | 7.62 | **0.796** |
| Risk class (4-band) | Multinomial logit | — | — | — |
| Risk class (4-band) | **LightGBM** | acc 0.650 | — | macro-F1 **0.672** |

### (b) Does AI actually beat conventional statistics?

| Target | OLS MAE | LightGBM MAE | Error reduction |
|---|---|---|---|
| Time overrun | 6.06 months | 3.36 months | **−44.5%** |
| Cost overrun | 7.76 pp | 7.22 pp | **−7.0%** |

**The honest answer is "it depends on the target."** Gradient boosting nearly halves schedule
error, because slippage is driven by threshold effects and interactions (land below ~70% *and*
a pending clearance *and* an item-rate contract) that a linear model structurally cannot
express. For cost escalation the gain is modest, because escalation is dominated by a roughly
linear pass-through from slippage and input prices — OLS is already competitive there.

ML still earns its place for cost, but for a different reason: it also yields calibrated risk
probabilities and per-project attribution, which a single regression coefficient cannot.

### (c) How much predictive power do the current CUF fields deliver?

Ablation: CUF-only vs CUF + 7 external signals (monsoon stoppage days, steel and cement price
indices, land litigation cases, contractor financial stress, right-of-way availability, local
agitation days).

| Target | CUF-only | CUF + external | Change |
|---|---|---|---|
| Cost overrun MAE | 7.65 | 7.22 | **−5.6%** |
| Cost overrun R² | 0.437 | 0.464 | +0.027 |
| Time overrun MAE | 3.359 | 3.363 | −0.1% (noise) |
| Risk class macro-F1 | 0.659 | 0.672 | +0.013 |

**Finding.** The existing CUF already carries essentially all the signal needed for *schedule*
prediction — adding external data changes nothing measurable. For *cost* prediction it does not:
commodity price indices and contractor financial stress explain escalation that no progress
field can, and they buy a real 5.6% error reduction.

**Policy implication for IPMD.** If the CUF is ever extended, the three highest-value additions
are: a **commodity price index at date of award**, a **contractor financial-health flag**, and a
**land-litigation case count**. The other four external variables can be derived from data the
Government already holds elsewhere (IMD, MoEFCC, state revenue records) without burdening the
reporting ministries at all.

### Top drivers of cost escalation (LightGBM gain share)

`reason_for_delay` (10.4%), `state` (9.4%), `schedule_burn_ratio` (8.6%),
`ext_steel_price_index` (7.8%), `ext_monsoon_stoppage_days` (5.4%), `num_cost_revisions` (4.9%),
`ext_contractor_stress` (4.6%).

---

## 6. Risk scoring framework

Two complementary constructs, deliberately kept separate:

1. **Risk class** — a supervised 4-band LightGBM classifier (Low / Medium / High / Critical)
   trained on thresholds that follow MoSPI flash-report conventions.
2. **Composite risk score (0–100)** — a transparent weighted blend of model forecasts and live
   CUF telemetry:

```
30%  forecast cost overrun      (LightGBM)
30%  forecast schedule slippage (LightGBM)
16%  physical progress shortfall vs elapsed schedule
10%  land acquisition gap
 8%  pending statutory clearances
 6%  expenditure running ahead of physical progress
```

The score is rule-based on purpose. A monitoring officer must be able to see *exactly* why a
project was flagged, and the project page shows the point contribution and the evidence for
every component. A pure black-box score would not survive a review meeting.

Current portfolio: 1,204 Low · 345 Medium · 338 High · 94 Critical.

---

## 7. Early warning and prescriptive layer

Seven alert rules fire from live CUF telemetry plus model forecasts — progress shortfall,
forecast escalation above 20%, forecast slippage above 12 months, land below 70%, pending
clearance, expenditure outpacing physical work, and repeat cost revisions.

Each alert carries the **evidence that triggered it** and a **prescriptive next step** mapped to
the dominant signal — escalate to the District Land Acquisition Committee, place the clearance
on the MoEFCC follow-up list, order a third-party quantity audit, convene a Standing Committee
re-baselining, or refer for a Revised Cost Estimate review.

---

## 8. Project Intelligence Assistant

Retrieval-grounded rather than free-generating. It resolves the question to a cohort (ministry,
sector, state) and an intent (risk, cost, schedule, land, clearances, drivers), **computes** the
answer from the scored portfolio, and returns the exact rows used. Because figures are computed
rather than generated, **it cannot hallucinate a number.**

To add free-form phrasing on top of the grounded context, point `PAIMANA_LLM_ENDPOINT` at a
local open-source model (Ollama, llama.cpp). See `docs/LLM_ASSISTANT.md`.

---

## 9. Repository layout

```
paimana-ai/
├── data/
│   ├── generate_dataset.py      CUF-schema corpus builder (calibrated)
│   ├── projects_historical.csv  9,000 completed projects  -> training
│   ├── projects_ongoing.csv     1,981 live projects       -> inference
│   └── scored_projects.csv      ongoing + predictions + risk + alerts
├── ml/
│   ├── features.py              single source of truth for feature engineering
│   ├── train.py                 baselines, LightGBM, ablation, scoring
│   └── export_static.py         static JSON snapshot builder
├── models/                      trained .pkl bundles (model + feature + category spec)
├── reports/                     metrics, ablation, feature importances
├── backend/app.py               FastAPI: 11 endpoints + static hosting
├── frontend/                    7 pages, PAIMANA design system, vendored Chart.js
└── docs/                        data swap, LLM, branding, architecture, model card
```

---

## 10. Honest limitations

* **The data is calibrated, not real.** Aggregates match published MoSPI figures; individual
  project rows are generated. Accuracy figures will change on real data — probably downward for
  schedule, since a real DGP is messier than a modelled one. Do not quote these numbers as
  results on government data.
* **Cost is intrinsically hard.** R² 0.46 with MAE ±7.2 percentage points. A large share of
  escalation comes from scope changes and price shocks that are simply not knowable from a
  monitoring record. Treat the cost forecast as a prioritisation signal, not a revised cost
  estimate.
* **Sector imbalance.** The portfolio skews to transport and energy, mirroring reality. Sectors
  with few historical analogues carry wider uncertainty than the headline metrics suggest.
* **Not an official Government of India website.** This is a prototype built against a published
  problem statement. See `docs/BRANDING.md` on emblem and logo usage.

---

## 11. Attribution

Design language follows the public PAIMANA portal (https://paimana-proj.mospi.gov.in). All
artwork in `frontend/js/common.js` — the crest, the Data-for-Development mark, the PAIMANA
wordmark and the four hero illustrations — is **original SVG drawn for this project**, not
copied from the portal. Aggregate figures are from MoSPI Flash Reports and PIB releases.
