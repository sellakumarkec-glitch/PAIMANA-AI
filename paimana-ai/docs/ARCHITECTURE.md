# Architecture

```
                 +------------------------------------------+
  PAIMANA / OCMS |  CUF monthly returns . DPIIT IPMP API    |
  (Government)   +--------------------+---------------------+
                                      | project-level rows
                                      v
  +-------------------------------------------------------------------+
  | DATA LAYER            data/generate_dataset.py                    |
  |  . historical corpus  9,000 completed, frozen mid-flight          |
  |  . ongoing corpus     1,981 live, frozen at as-of date            |
  +--------------------------------+----------------------------------+
                                   v
  +-------------------------------------------------------------------+
  | FEATURE LAYER         ml/features.py   (fit and serve share this) |
  |  . engineered: progress_gap, schedule_burn_ratio, spend_efficiency|
  |  . two feature sets: cuf | extended  -> enables the ablation      |
  +--------------------------------+----------------------------------+
                                   v
  +-------------------------------------------------------------------+
  | MODEL LAYER           ml/train.py                                 |
  |  baselines: OLS, Ridge, multinomial logit   (dimension b)         |
  |  models:    LightGBM x3   cost . time . risk class                |
  |  split:     temporal, by date_of_sanction                         |
  |  outputs:   models/*.pkl  reports/*.csv  data/scored_projects.csv |
  +--------------------------------+----------------------------------+
                                   v
  +-------------------------------------------------------------------+
  | SERVICE LAYER         backend/app.py (FastAPI)                    |
  |  /api/kpis /breakdown /projects /project/{id} /alerts             |
  |  /predict /benchmark /model/performance /assistant /filters       |
  |              ml/export_static.py --> frontend/data/*.json         |
  +--------------------------------+----------------------------------+
                                   v
  +-------------------------------------------------------------------+
  | PRESENTATION          frontend/  (PAIMANA design system)          |
  |  index . dashboard . early-warning . project . benchmarking       |
  |  predict . assistant . model                                      |
  |  js/common.js  API client with automatic static fallback          |
  +-------------------------------------------------------------------+
```

## Design decisions worth knowing

**One feature module, two consumers.** `ml/features.py` is imported by both `train.py` and
`app.py`. Feature engineering can therefore never drift between fit and serve, which is the
commonest way an ML service silently degrades in production.

**Category levels are pinned in the model bundle.** Each `.pkl` stores
`{model, features, categories}`. At serve time `align_categories()` forces incoming
categoricals to exactly the training levels, so an unseen ministry name becomes `NaN` (which
LightGBM handles) rather than shifting every category code by one.

**Static fallback is a first-class path, not a hack.** `js/common.js` routes to
`offlineRoute()` when the backend is unreachable, and `offlineRoute` mirrors the FastAPI route
semantics including filtering, sorting and pagination. That is what lets the same build run on
NIC static hosting or on a laptop with no Python - anything that can serve files over HTTP. Note
that it must be an HTTP server, not a `file://` open: browsers block `fetch()` on file origins.

**The composite risk score is rule-based on purpose.** Two components are model output; the
other four are live telemetry with fixed weights. A black-box score cannot be defended in a
review meeting. The project page renders the point contribution and the evidence for every
component.

## Deployment notes

* **Single process.** FastAPI serves both the API and the static frontend, so there is no CORS
  configuration and no second web server.
* **Production.** `uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 4` behind nginx.
  Models load once at import, so workers share nothing and scale linearly.
* **Scheduled retraining.** Add a monthly cron after the CUF cycle closes:
  `python ml/train.py && python ml/export_static.py && systemctl reload paimana-ai`.
* **Air-gapped.** Vendor Poppins locally (see `docs/BRANDING.md`); Chart.js is already vendored
  in `frontend/vendor/`. Nothing else reaches the network.
