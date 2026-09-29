/* ==========================================================================
   PAIMANA-AI :: static-mode surrogates
   --------------------------------------------------------------------------
   When the FastAPI backend is unavailable (site opened from the filesystem or
   deployed as a static page), these two functions stand in for /api/predict
   and /api/assistant. They are DISTILLED, transparent approximations of the
   trained models - coefficients were fitted to the LightGBM predictions on the
   held-out test set (see ml/distil_surrogate.py). They are deliberately simple
   so that nothing about the fallback is hidden from the user.
   ========================================================================== */

async function offlinePredict(p) {
  const gap = Math.max(0, p.elapsed_schedule_pct - p.physical_progress_pct);
  const burn = Math.max(0, p.financial_progress_pct / Math.max(p.physical_progress_pct, 0.5) - 1);
  const pend = (p.forest_clearance === 'Applied - Pending' ? 0.6 : 0) +
               (p.env_clearance === 'Applied - Pending' ? 0.4 : 0);

  // ---- distilled schedule model (months) --------------------------------
  let t = -1.2
    + 0.42 * gap
    + 0.10 * (100 - p.land_acquired_pct)
    + 0.22 * p.award_lag_months
    + 3.9 * p.num_cost_revisions
    + 4.4 * pend
    + 0.035 * (p.ext_monsoon_stoppage_days - 45)
    + 0.9 * p.ext_litigation_cases
    + 0.05 * p.ext_contractor_stress
    + 0.02 * p.original_duration_m
    + (p.reason_for_delay !== 'None reported' ? 6.5 : 0);
  t = Math.max(-4, Math.min(120, t));

  // ---- distilled cost model (%) -----------------------------------------
  let c = 1.4
    + 0.62 * Math.max(t, 0)
    + 0.14 * (p.ext_steel_price_index - 100)
    + 0.07 * (p.ext_cement_price_index - 100)
    + 5.2 * p.num_cost_revisions
    + 0.045 * p.ext_contractor_stress
    + 6 * burn
    + (p.contract_type === 'Item Rate' ? 2.3 : 0);
  c = Math.max(-6, Math.min(300, c));

  const score = Math.max(0, Math.min(100, 100 * (
    0.30 * Math.min(Math.max(c, 0) / 60, 1) +
    0.30 * Math.min(Math.max(t, 0) / 36, 1) +
    0.16 * Math.min(gap / 45, 1) +
    0.10 * Math.min((100 - p.land_acquired_pct) / 60, 1) +
    0.08 * pend + 0.06 * Math.min(burn, 1))));

  const band = score > 72 ? 'Critical' : score > 50 ? 'High' : score > 25 ? 'Medium' : 'Low';
  // soft probabilities around the banded score
  const centres = { Low: 12, Medium: 38, High: 61, Critical: 85 };
  const raw = Object.fromEntries(Object.entries(centres)
    .map(([k, v]) => [k, Math.exp(-Math.pow(score - v, 2) / 520)]));
  const z = Object.values(raw).reduce((a, b) => a + b, 0);

  return {
    pred_cost_overrun_pct: +c.toFixed(2),
    pred_time_overrun_months: +t.toFixed(2),
    risk_score: +score.toFixed(1),
    risk_band: band,
    risk_class: band,
    risk_probabilities: Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, +(v / z).toFixed(4)])),
    projected_revised_cost_cr: +(p.original_cost_cr * (1 + c / 100)).toFixed(2),
    projected_escalation_cr: +(p.original_cost_cr * c / 100).toFixed(2),
    _surrogate: true,
  };
}

async function offlineAssistant(question) {
  const q = question.toLowerCase();
  const k = await snapshot('kpis');
  const rows = (await snapshot('projects')).rows;
  const bm = await snapshot('benchmark');
  const dr = await snapshot('model_performance');

  let scope = rows, label = 'the national portfolio';
  for (const col of ['ministry', 'sector', 'state']) {
    const vals = [...new Set(rows.map(r => r[col]))];
    const hit = vals.find(v => {
      const toks = v.toLowerCase().replace(/[&,]/g, ' ').split(/\s+/).filter(t => t.length > 4);
      return toks.length && toks.filter(t => q.includes(t)).length >= Math.max(1, Math.floor(toks.length / 2));
    });
    if (hit) { scope = rows.filter(r => r[col] === hit); label = hit; break; }
  }
  const top = [...scope].sort((a, b) => b.risk_score - a.risk_score);
  const ev = top.slice(0, 8);
  const sum = (a, f) => a.reduce((s, r) => s + f(r), 0);
  const med = a => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; };

  if (/riskiest|highest risk|top risk|most at risk|critical/.test(q))
    return { answer: `The 10 highest-risk projects in ${label} score between ${top[0].risk_score.toFixed(0)} and ` +
      `${top[9] ? top[9].risk_score.toFixed(0) : top[top.length-1].risk_score.toFixed(0)} on the composite risk index, ` +
      `and together hold ₹${sum(top.slice(0,10), r=>r.revised_cost_cr).toLocaleString('en-IN',{maximumFractionDigits:0})} crore of revised cost.`,
      evidence: ev, scope: label };

  if (/cost overrun|escalation|over budget|cost/.test(q))
    return { answer: `${label} covers ${scope.length.toLocaleString('en-IN')} projects with ₹${(sum(scope,r=>r.original_cost_cr)/1e5).toFixed(2)} ` +
      `lakh crore sanctioned and ₹${(sum(scope,r=>r.revised_cost_cr)/1e5).toFixed(2)} lakh crore revised. ` +
      `The models forecast a median further escalation of ${med(scope.map(r=>r.pred_cost_overrun_pct)).toFixed(1)}% on sanctioned cost.`,
      evidence: [...scope].sort((a,b)=>b.pred_cost_overrun_pct-a.pred_cost_overrun_pct).slice(0,8), scope: label };

  if (/delay|time overrun|slippage|behind schedule|late/.test(q)) {
    const dl = scope.filter(r => r.project_status.startsWith('Delayed'));
    return { answer: `${dl.length.toLocaleString('en-IN')} of ${scope.length.toLocaleString('en-IN')} projects ` +
      `(${(dl.length/scope.length*100).toFixed(0)}%) in ${label} are behind schedule. Median forecast slippage across ` +
      `the cohort is ${med(scope.map(r=>r.pred_time_overrun_months)).toFixed(1)} months.`,
      evidence: [...scope].sort((a,b)=>b.pred_time_overrun_months-a.pred_time_overrun_months).slice(0,8), scope: label };
  }

  if (/driver|why|cause|reason/.test(q)) {
    const d = (dr.drivers['cost_overrun_pct|extended'] || []).slice(0, 6);
    return { answer: 'The strongest predictors of cost escalation across the training corpus are: ' +
      d.map(x => `${x.feature} (${x.gain_pct}% of model gain)`).join(', ') + '.', evidence: ev, scope: label };
  }

  if (/sector|compare|benchmark/.test(q)) {
    const s = bm.sector.rows.slice(0, 5);
    return { answer: 'Highest-risk sectors by median composite score: ' +
      s.map(r => `${r.sector} (${r.median_risk})`).join(', ') +
      `. National median is ${bm.sector.national.median_risk}.`, evidence: ev, scope: label };
  }

  if (label !== 'the national portfolio')
    return { answer: `${label} has ${scope.length.toLocaleString('en-IN')} ongoing projects worth ` +
      `₹${(sum(scope,r=>r.revised_cost_cr)/1e5).toFixed(2)} lakh crore (revised). ` +
      `${scope.filter(r=>['High','Critical'].includes(r.risk_band)).length} are High or Critical risk.`,
      evidence: ev, scope: label };

  return { answer: `PAIMANA is monitoring ${k.total_projects.toLocaleString('en-IN')} ongoing projects across ` +
    `${k.ministries} Ministries and ${k.sectors} sectors, with ₹${k.revised_cost_lakh_cr} lakh crore revised cost ` +
    `against ₹${k.original_cost_lakh_cr} lakh crore sanctioned (${k.aggregate_cost_overrun_pct}% overrun). ` +
    `${k.risk_bands.Critical || 0} projects are Critical risk. Ask me about a Ministry, sector or state, or about ` +
    `delays, cost overruns, land acquisition or clearances.`, evidence: ev, scope: 'national' };
}
