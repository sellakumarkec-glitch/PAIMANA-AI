# Model card - PAIMANA-AI

## Intended use

**Intended.** Prioritising monitoring attention across a large portfolio of Central Sector
infrastructure projects; flagging projects whose cost or schedule trajectory warrants
intervention earlier than a monthly status report would reveal; supporting cohort comparison
across ministries, sectors and states.

**Not intended.** Revised Cost Estimate preparation, contractual claims, penalty determination,
contractor blacklisting, or any decision affecting an identified party without independent
verification. The forecasts are prioritisation signals, not audited estimates.

## Models

| Model | Type | Target | Test performance |
|---|---|---|---|
| Cost overrun regressor | LightGBM, L1 objective | `cost_overrun_pct` | MAE 7.22 pp, R2 0.464 |
| Time overrun regressor | LightGBM, L1 objective | `time_overrun_months` | MAE 3.36 mo, R2 0.796 |
| Risk classifier | LightGBM, 4-class | Low/Medium/High/Critical | acc 0.650, macro-F1 0.672 |
| Baselines | OLS, Ridge, multinomial logit | same | `reports/model_comparison.csv` |

The L1 (MAE) objective is chosen deliberately over L2: overrun distributions have a heavy right
tail, and squared loss would let a handful of 200%-overrun projects dominate the fit.

## Training data

7,200 completed projects, temporal split with the newest 20% (1,800) held out. Each frozen at a
random monitoring month between 25% and 88% of sanctioned schedule and labelled with its final
outcome. The portfolio is calibrated to MoSPI April 2026 aggregates but **generated, not
observed** - see README sections 3 and 10.

## Evaluation

Temporal hold-out by `date_of_sanction`. No cross-validation over shuffled rows, which would
leak future information into past predictions and inflate every metric.

## Known limitations and failure modes

* **Cost R2 is 0.46.** Roughly half the variance in escalation is unexplained. Much of it comes
  from scope changes and price shocks unknowable from a monitoring record. A project predicted
  at 10% overrun can plausibly land anywhere from 3% to 17%.
* **Sector imbalance.** Road, rail and petroleum dominate the corpus. Atomic Energy, Fertilizers
  and Heavy Industry have few analogues, so their predictions carry wider real uncertainty than
  the aggregate MAE implies.
* **Reason-for-delay leakage risk.** `reason_for_delay` is the single strongest feature (10.4% of
  cost gain, 48.6% of schedule gain). On real data, check whether it is recorded *before* or
  *after* the delay is booked. If it is recorded retrospectively it is a leak - drop it and
  re-measure. This is the first thing to audit on a real extract.
* **State as a feature.** `state` carries 9.4% of cost gain. That encodes real differences in
  land, terrain and clearance throughput, but it also risks entrenching a self-fulfilling
  expectation that projects in certain states run late. Monitor for that in use, and consider
  reporting state-adjusted as well as raw risk scores.
* **No uncertainty intervals.** Point estimates only. For production, wrap with quantile
  regression (LightGBM `objective="quantile"` at 0.1/0.5/0.9) and show the band.

## Ethical and governance notes

Outputs concern projects, not individuals, so there is no personal-data processing. But risk
scores can influence fund release and contractor standing. Every flagged project therefore
carries its full evidence trail, and the composite score is rule-based and inspectable rather
than a black box. Any deployment should keep a human decision-maker between the score and any
consequence.

## Maintenance

Retrain monthly after the CUF cycle closes. Track MAE drift against the previous vintage; a
sustained rise signals either a data-schema change or a genuine regime shift (a new contracting
model, a commodity shock) that warrants re-specification rather than more trees.
