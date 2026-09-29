# Replacing the calibrated dataset with real PAIMANA data

`ml/train.py` and `backend/app.py` depend only on **column names**, never on the generator.
Swapping in a real extract is three steps.

## Step 1 — map your extract to the CUF column names

Produce two CSVs with these columns (see `frontend/model.html#dictionary` for the full
dictionary):

**Required for training** (`data/projects_historical.csv`) — completed projects, one row per
project, with progress fields recorded at a **mid-flight monitoring month**, not at completion
(this is essential; see README section 4):

```
project_id, project_name, ministry, sector, state, agency_type, funding_mode,
contract_type, date_of_sanction, original_cost_cr, revised_cost_cr, expenditure_cr,
original_completion, anticipated_completion, original_duration_m,
physical_progress_pct, financial_progress_pct, elapsed_schedule_pct,
land_acquired_pct, forest_clearance, env_clearance, award_lag_months,
num_cost_revisions, num_packages, monitoring_frequency, reason_for_delay,
project_status, cost_overrun_pct, time_overrun_months
```

**Required for inference** (`data/projects_ongoing.csv`) — same columns; the two target columns
may hold the values booked so far, and are never used as features.

If you have monthly CUF history per project, the best extract is **one row per project-month**
for historical projects — that multiplies your training corpus and teaches the model how risk
evolves. Edit `temporal_split` to group-split by `project_id` (use `GroupShuffleSplit`) so that
snapshots of the same project cannot straddle train and test.

## Step 2 — handle the external variables

The seven `ext_*` columns are not in the CUF. Either:

* **Drop them.** Change `for fs in ("cuf", "extended")` in `train.py::main` to `for fs in ("cuf",)`
  and point `score_ongoing` at `lgbm_*_cuf.pkl`. You lose about 5.6% of cost accuracy and
  nothing on schedule.
* **Source them.** Monsoon days from IMD gridded rainfall; price indices from the Office of the
  Economic Adviser WPI series; litigation counts from eCourts; contractor stress from MCA
  filings or GST return regularity; right-of-way from the implementing agency MIS.

## Step 3 — rerun

```bash
python ml/train.py && python ml/export_static.py
```

Re-read `reports/model_comparison.csv` and `reports/cuf_ablation.csv`. **Expect the headline
numbers to change.** Real data is messier than a modelled data-generating process; schedule R2
in particular will very likely fall below 0.80. Report what you measure, not what is in the
README.
