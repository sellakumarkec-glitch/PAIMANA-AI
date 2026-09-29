/* ==========================================================================
   PAIMANA-AI :: shared chrome + API client
   ========================================================================== */

/* ---- API client ---------------------------------------------------------
   Talks to the FastAPI backend. If the backend is unreachable (e.g. the site
   is served as a static build with no Python behind it), it transparently falls
   back to the pre-computed JSON snapshots in /data.
   Note: the snapshots are fetched over HTTP, so the static build needs a web
   server - browsers block fetch() on file:// origins.
   ------------------------------------------------------------------------ */
const API_BASE = (location.protocol === 'file:' || (location.port && location.port !== '8000'))
  ? 'http://127.0.0.1:8000' : '';
const SNAPSHOT = {};
let OFFLINE = API_BASE === null;

async function snapshot(name) {
  if (SNAPSHOT[name]) return SNAPSHOT[name];
  const r = await fetch(`data/${name}.json`);
  SNAPSHOT[name] = await r.json();
  return SNAPSHOT[name];
}

async function api(path, opts) {
  if (!OFFLINE) {
    const controller = new AbortController();
    const requestOpts = { ...(opts || {}), signal: controller.signal };
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const r = await fetch(API_BASE + path, requestOpts);
      if (r.ok) return await r.json();
      if (r.status === 404 && path.startsWith('/api/project/')) throw new Error('404');
      OFFLINE = true;
    } catch (e) { OFFLINE = true; }
    finally { clearTimeout(timeout); }
  }
  return offlineRoute(path, opts);
}

/* Static fallback router — mirrors the FastAPI route semantics. */
async function offlineRoute(path, opts) {
  const [p, qs] = path.split('?');
  const q = new URLSearchParams(qs || '');
  if (p === '/api/kpis') return snapshot('kpis');
  if (p === '/api/filters') return snapshot('filters');
  if (p === '/api/model/performance') return snapshot('model_performance');
  if (p === '/api/breakdown') return (await snapshot('breakdown'))[q.get('by') || 'ministry'];
  if (p === '/api/benchmark') {
    const b = await snapshot('benchmark');
    return b[q.get('dimension') || 'sector'];
  }
  if (p === '/api/alerts') {
    const a = await snapshot('alerts');
    let rows = a.alerts;
    if (q.get('severity')) rows = rows.filter(x => x.risk_band === q.get('severity'));
    if (q.get('ministry')) rows = rows.filter(x => x.ministry === q.get('ministry'));
    return { count: rows.length, alerts: rows.slice(0, +(q.get('limit') || 100)) };
  }
  if (p === '/api/projects') {
    const all = (await snapshot('projects')).rows;
    let rows = all.filter(r =>
      (!q.get('ministry') || r.ministry === q.get('ministry')) &&
      (!q.get('sector') || r.sector === q.get('sector')) &&
      (!q.get('state') || r.state === q.get('state')) &&
      (!q.get('risk_band') || r.risk_band === q.get('risk_band')) &&
      (!q.get('status') || r.project_status === q.get('status')) &&
      (!q.get('q') || (r.project_name + r.project_id).toLowerCase().includes(q.get('q').toLowerCase()))
    );
    const sort = q.get('sort') || 'risk_score', desc = q.get('desc') !== 'false';
    rows.sort((a, b) => (a[sort] > b[sort] ? 1 : -1) * (desc ? -1 : 1));
    const size = +(q.get('size') || 25), page = +(q.get('page') || 1);
    return { total: rows.length, page, size, pages: Math.ceil(rows.length / size),
             rows: rows.slice((page - 1) * size, page * size) };
  }
  if (p.startsWith('/api/project/')) {
    const id = p.split('/').pop();
    const d = await snapshot('project_details');
    if (!d[id]) throw new Error('404');
    return d[id];
  }
  if (p === '/api/assistant') return offlineAssistant(JSON.parse(opts.body).question);
  if (p === '/api/predict') return offlinePredict(JSON.parse(opts.body));
  throw new Error('offline route not available: ' + path);
}

/* ---- formatters --------------------------------------------------------- */
const fmt = {
  cr: v => '₹' + Number(v).toLocaleString('en-IN', { maximumFractionDigits: 0 }) + ' cr',
  lakh: v => '₹' + Number(v).toFixed(2) + ' L cr',
  n: (v, d = 0) => Number(v).toLocaleString('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d }),
  pct: (v, d = 1) => (v >= 0 ? '' : '') + Number(v).toFixed(d) + '%',
  mon: v => Number(v).toFixed(1) + ' mo',
};
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* ---- SVG assets (drawn in-house; no third-party artwork) ---------------- */
const SVG = {
  /* Neutral institutional crest placeholder. The official State Emblem of
     India is protected under the State Emblem of India (Prohibition of
     Improper Use) Act, 2005 and must be inserted only by an authorised
     Government deployment — see docs/BRANDING.md. */
  emblem: `<svg class="emblem" viewBox="0 0 60 84" role="img" aria-label="Government of India crest placeholder">
    <g fill="#1b2a5e">
      <circle cx="30" cy="16" r="9"/><circle cx="14" cy="21" r="7"/><circle cx="46" cy="21" r="7"/>
      <path d="M6 30h48v5H6z"/><path d="M12 35h36l-3 9H15z"/>
      <rect x="27" y="44" width="6" height="26" rx="1"/>
      <path d="M17 70h26l4 8H13z"/>
      <circle cx="30" cy="55" r="6" fill="none" stroke="#1b2a5e" stroke-width="2"/>
      <circle cx="30" cy="55" r="1.6"/>
    </g>
    <text x="30" y="83" text-anchor="middle" font-size="6" fill="#5a6379" font-family="Poppins,sans-serif">सत्यमेव जयते</text>
  </svg>`,

  dfd: `<svg class="dfd" viewBox="0 0 64 64" role="img" aria-label="Data for Development">
    <circle cx="32" cy="32" r="30" fill="#fff" stroke="#1b2a5e" stroke-width="2"/>
    <circle cx="32" cy="28" r="18" fill="#12296e"/>
    <g fill="#e9a63c"><rect x="24" y="24" width="4" height="12" rx="1"/><rect x="30" y="19" width="4" height="17" rx="1"/><rect x="36" y="27" width="4" height="9" rx="1"/></g>
    <text x="32" y="55" text-anchor="middle" font-size="6.4" fill="#12296e" font-weight="700" font-family="Poppins,sans-serif">DATA FOR</text>
    <text x="32" y="62" text-anchor="middle" font-size="6.4" fill="#12296e" font-weight="700" font-family="Poppins,sans-serif">DEVELOPMENT</text>
  </svg>`,

  /* PAIMANA-style infrastructure wordmark (original artwork) */
  logo: `<svg class="paimana-logo" viewBox="0 0 430 78" role="img" aria-label="PAIMANA">
    <g fill="#1d4ed8" opacity=".9">
      <rect x="4" y="30" width="13" height="42"/><rect x="20" y="18" width="9" height="54"/>
      <rect x="32" y="38" width="15" height="34"/>
      <path d="M52 72 60 20 68 72z"/>
      <rect x="74" y="26" width="12" height="46"/>
    </g>
    <circle cx="60" cy="13" r="7" fill="#e9a63c"/>
    <text x="96" y="62" font-family="Poppins,sans-serif" font-size="50" font-weight="700" fill="#12296e" letter-spacing="1">PAIMANA</text>
    <path d="M306 62 332 34 348 48 380 14" fill="none" stroke="#2f7fd4" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M366 12h16v16" fill="none" stroke="#2f7fd4" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
    <g stroke="#12296e" stroke-width="4" fill="none">
      <path d="M392 72V26M416 72V26M392 34h24M392 46h24M392 58h24"/>
    </g>
  </svg>`,
};

/* Hero scenes — original vector illustrations (energy, transport, water, AI) */
function scene(kind) {
  const sky = { energy: ['#1b3a6b', '#4d7fb5'], transport: ['#163256', '#3f6f9e'],
                water: ['#14456b', '#3d87a8'], ai: ['#101f4d', '#2c4a8f'] }[kind];
  const common = `<defs><linearGradient id="g${kind}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="${sky[0]}"/><stop offset="1" stop-color="${sky[1]}"/></linearGradient></defs>
    <rect width="1600" height="700" fill="url(#g${kind})"/>`;
  const art = {
    energy: `<g fill="#0f2547" opacity=".85">
        <path d="M980 700V300c0-40 30-60 70-60s70 20 70 60v400z"/>
        <path d="M1180 700V340c0-34 26-52 60-52s60 18 60 52v360z"/></g>
      <g stroke="#173059" stroke-width="10" fill="none"><path d="M300 700V220M300 220l-90 60M300 220l90 60M300 330l-70 46M300 330l70 46"/></g>
      <circle cx="300" cy="212" r="26" fill="#1d3a66"/>
      <g fill="#14406e"><rect x="520" y="560" width="300" height="14" rx="4"/>
        <g transform="translate(520,470)"><rect width="300" height="86" rx="6" fill="#1a5288" transform="skewY(-8)"/></g></g>`,
    transport: `<g fill="#0e2748"><rect y="560" width="1600" height="140"/></g>
      <g stroke="#e9a63c" stroke-width="8" stroke-dasharray="70 52"><path d="M0 640h1600"/></g>
      <g fill="#123a63"><rect x="120" y="300" width="26" height="260"/><rect x="1440" y="300" width="26" height="260"/>
        <path d="M133 300 C 400 120, 1180 120, 1453 300" fill="none" stroke="#1a4e85" stroke-width="12"/></g>
      <g stroke="#1a4e85" stroke-width="5">${Array.from({length:14},(_,i)=>`<path d="M${170+i*95} ${300+Math.abs(i-6.5)*4} V560"/>`).join('')}</g>`,
    water: `<g fill="#0d3a5c"><path d="M0 520h1600v180H0z"/></g>
      <g fill="#12507c" opacity=".7"><path d="M0 520q200-40 400 0t400 0 400 0 400 0v40H0z"/></g>
      <g fill="#123a63"><rect x="500" y="250" width="600" height="270" rx="8"/></g>
      <g fill="#1a5288">${Array.from({length:5},(_,i)=>`<rect x="${540+i*118}" y="300" width="70" height="220" rx="5"/>`).join('')}</g>
      <rect x="470" y="230" width="660" height="26" rx="6" fill="#1d4f80"/>`,
    ai: `<g stroke="#3f6fbf" stroke-width="2" fill="none" opacity=".55">
        ${Array.from({length:22},(_,i)=>{const x=80+((i*137)%1450),y=90+((i*211)%520);
          return `<circle cx="${x}" cy="${y}" r="5" fill="#7fb2ee"/><path d="M${x} ${y} L${80+(((i+3)*137)%1450)} ${90+(((i+5)*211)%520)}"/>`}).join('')}</g>
      <g fill="#1b3f7d" opacity=".7"><rect x="560" y="420" width="90" height="180" rx="5"/>
        <rect x="680" y="350" width="90" height="250" rx="5"/><rect x="800" y="270" width="90" height="330" rx="5"/>
        <rect x="920" y="400" width="90" height="200" rx="5"/></g>`,
  }[kind];
  return `<svg class="scene" viewBox="0 0 1600 700" preserveAspectRatio="xMidYMid slice" aria-hidden="true">${common}${art}</svg>`;
}

const SLIDES = [
  { k: 'transport', h: 'Project Monitoring', p: 'Central Sector Infrastructure Projects Costing Rs. 150 crore &amp; above' },
  { k: 'ai', h: 'Predictive Analytics', p: 'Forecasting cost and time overruns before they materialise' },
  { k: 'energy', h: 'Early Warning', p: 'Risk-scored alerts across 1,981 ongoing projects in 22 sectors' },
  { k: 'water', h: 'Evidence Based Action', p: 'From descriptive reporting to prescriptive decision support' },
];

/* ---- chrome ------------------------------------------------------------- */
const NAV = [
  { label: 'Home', href: 'index.html' },
  { label: 'Publications', items: [
      ['Flash Report (Apr 2026)', 'model.html'],
      ['Model Performance & Ablation', 'model.html'],
      ['Data Dictionary (CUF)', 'model.html#dictionary'],
  ]},
  { label: 'Dashboard', items: [
      ['AI Monitoring Dashboard', 'dashboard.html'],
      ['Early Warning Alerts', 'early-warning.html'],
      ['Benchmarking & Comparatives', 'benchmarking.html'],
  ]},
  { label: 'Risk Analytics', items: [
      ['Project Risk Register', 'dashboard.html#register'],
      ['Cost Escalation Drivers', 'model.html#drivers'],
      ['What-if Prediction', 'predict.html'],
  ]},
  { label: 'Assistant', href: 'assistant.html' },
];

function chrome(active) {
  const nav = NAV.map(n => {
    if (n.href) return `<a class="nav-item ${active === n.label ? 'active' : ''}" href="${n.href}">${n.label}</a>`;
    return `<div class="nav-item ${active === n.label ? 'active' : ''}"><button aria-haspopup="true">${n.label}<span class="caret"></span></button>
      <div class="dropdown">${n.items.map(([t, h]) => `<a href="${h}">${t}</a>`).join('')}</div></div>`;
  }).join('');

  document.body.insertAdjacentHTML('afterbegin', `
  <div class="utility"><div class="wrap">
    <div class="u-left">Government of India · Ministry of Statistics and Programme Implementation</div>
    <div class="u-right">
      <button title="Language">अ<sub>A</sub></button>
      <button title="Accessibility">♿</button>
    </div>
  </div></div>

  <header class="masthead"><div class="wrap">
    <div class="brand">
      ${SVG.emblem}
      <div class="brand-text">
        <div class="l1">Ministry of Statistics and</div>
        <div class="l2">Programme Implementation</div>
        <div class="l3">Government of India</div>
      </div>
      ${SVG.dfd}
    </div>
    <div class="mast-actions">
      <div id="authSlot"></div>
      <a class="btn-pill btn-amber" href="predict.html">Add Project / Update</a>
      <a class="btn-pill btn-navy" href="model.html">Reports</a>
    </div>
    ${SVG.logo}
  </div></header>

  <nav class="navband" aria-label="Primary"><div class="wrap">${nav}</div></nav>`);

  applyRoleUi();
  renderAuthControls();

  document.body.insertAdjacentHTML('beforeend', `
  <footer class="site"><div class="wrap">
    <div class="cols">
      <div><h5>PAIMANA</h5>
        <p style="margin:0">Project Assessment, Infrastructure Monitoring and Analytics for Nation-building — extended with an AI predictive and early-warning layer.</p></div>
      <div><h5>Monitoring</h5>
        <a href="dashboard.html">AI Monitoring Dashboard</a><a href="early-warning.html">Early Warning Alerts</a>
        <a href="benchmarking.html">Benchmarking</a><a href="predict.html">What-if Prediction</a></div>
      <div><h5>Transparency</h5>
        <a href="model.html">Model Performance</a><a href="model.html#drivers">Cost Escalation Drivers</a>
        <a href="model.html#dictionary">CUF Data Dictionary</a><a href="/api/docs">API Documentation</a></div>
      <div><h5>Reference</h5>
        <a href="https://paimana-proj.mospi.gov.in/" target="_blank" rel="noopener">Official PAIMANA Portal</a>
        <a href="https://mospi.gov.in/" target="_blank" rel="noopener">MoSPI</a>
        <a href="assistant.html">Project Intelligence Assistant</a></div>
    </div>
    <div class="legal">
      <span>Prototype built for the "AI for Infrastructure Monitoring" problem statement. Not an official Government of India website.</span>
      <span>Data as of April 2026 · Open-source stack</span>
    </div>
  </div></footer>`);
}

function hero(container) {
  container.innerHTML = SLIDES.map((s, i) => `
    <div class="slide ${i === 0 ? 'on' : ''}" data-i="${i}">
      ${scene(s.k)}
      <div class="slide-caption"><h2>${s.h}</h2><p>${s.p}</p></div>
    </div>`).join('') +
    `<button class="hero-arrow prev" aria-label="Previous slide">‹</button>
     <button class="hero-arrow next" aria-label="Next slide">›</button>
     <div class="hero-dots">${SLIDES.map((_, i) => `<span class="${i === 0 ? 'on' : ''}" data-i="${i}" role="button" tabindex="0" aria-label="Slide ${i + 1}"></span>`).join('')}</div>`;

  let cur = 0, timer;
  const go = i => {
    cur = (i + SLIDES.length) % SLIDES.length;
    container.querySelectorAll('.slide').forEach(el => el.classList.toggle('on', +el.dataset.i === cur));
    container.querySelectorAll('.hero-dots span').forEach(el => el.classList.toggle('on', +el.dataset.i === cur));
  };
  const restart = () => { clearInterval(timer); timer = setInterval(() => go(cur + 1), 6000); };
  container.querySelector('.next').onclick = () => { go(cur + 1); restart(); };
  container.querySelector('.prev').onclick = () => { go(cur - 1); restart(); };
  container.querySelectorAll('.hero-dots span').forEach(el => {
    el.onclick = () => { go(+el.dataset.i); restart(); };
    el.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { go(+el.dataset.i); restart(); } };
  });
  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) restart();
}

/* ---- Chart.js defaults -------------------------------------------------- */
function chartDefaults() {
  if (!window.Chart) return;
  Chart.defaults.font.family = "Poppins, Segoe UI, sans-serif";
  Chart.defaults.font.size = 12;
  Chart.defaults.color = '#5d6880';
  Chart.defaults.plugins.legend.labels.boxWidth = 12;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
}
const PALETTE = ['#133a9e', '#e9a63c', '#2f7fd4', '#1f9d55', '#c62828', '#7b5ea7',
                 '#e2703a', '#0f766e', '#9c6b1f', '#4a5a86', '#b23c6e', '#347d9e'];
const RISKCOLOR = { Low: '#1f9d55', Medium: '#e9a63c', High: '#e2703a', Critical: '#c62828' };

const AUTH_KEY = 'paimana_auth_role';
const AUTH_USER_KEY = 'paimana_auth_user';
const ROLE_USERNAMES = {
  mospi: 'mospi',
  reporting_manager: 'reporting_manager',
};
const ROLE_PASSWORDS = {
  mospi: 'MOSPI@2026',
  reporting_manager: 'REPORT@2026',
};
const ROLE_META = {
  guest: { label: 'Guest dashboard only', pages: ['dashboard.html'] },
  reporting_manager: { label: 'REPORTING MANAGER', pages: ['dashboard.html', 'project.html'] },
  mospi: { label: 'MOSPI', pages: ['index.html', 'dashboard.html', 'early-warning.html', 'benchmarking.html', 'project.html', 'model.html', 'predict.html', 'assistant.html'] },
};

function getRole() {
  return localStorage.getItem(AUTH_KEY) || 'guest';
}

function getUsername() {
  return localStorage.getItem(AUTH_USER_KEY) || '';
}

function setRole(role, username = '') {
  localStorage.setItem(AUTH_KEY, role);
  if (role === 'guest') localStorage.removeItem(AUTH_USER_KEY);
  else if (username) localStorage.setItem(AUTH_USER_KEY, username.trim().toLowerCase());
}

function authenticateRole(role) {
  setRole(role === 'guest' ? 'guest' : 'guest');
  return true;
}

function requireRoleAccess() {
  return;
}

function renderAuthControls() {
  const authSlot = document.getElementById('authSlot');
  if (authSlot) authSlot.innerHTML = '';
}

function applyRoleUi() {
  requireRoleAccess();
  const primaryActions = document.querySelector('.mast-actions');
  if (primaryActions) {
    primaryActions.innerHTML = `
      <div id="authSlot"></div>
      <a class="btn-pill btn-amber" href="predict.html">Add Project / Update</a>
      <a class="btn-pill btn-navy" href="model.html">Reports</a>
    `;
  }
}
