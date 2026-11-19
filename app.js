/**
 * Finsistant Version 2 — Live Data & Multi-Agent Frontend Controller
 *
 * ZERO SIMULATION POLICY:
 * - No hardcoded fallback prices, no seededRandom candles, no synthetic holdings.
 * - Loads live Yahoo Finance OHLCV candles, fundamentals, and 3-Agent Orchestrator
 *   outputs either from the Python backend (/api/state, /api/scan) or from the
 *   cloud-generated ./live_scan.json snapshot (GitHub Actions).
 */

let appState = {
  backendConnected: false,
  dataSource: "",
  kiteStatus: "Checking...",
  kiteAuth: null,
  generatedAt: "",
  activeSymbol: "RELIANCE",
  cashBalance: 0,
  assets: {},       // keyed by symbol -> live analysis payload
  pipelines: {},    // keyed by symbol -> AgentRunner PipelineResult payload
  holdingsMap: {},  // keyed by symbol -> { qty, avg_price }
  queue: { buys: [], sells: [], executed: [] },
  rules: [],
  health: null,
  config: null,
  errorMsg: null,
  chatHistory: []
};

const fmtCurrency = (n, currency = "INR", d = 0) => {
  const cur = String(currency || "INR").toUpperCase();
  const num = Number(n || 0);
  if (cur === "USD") {
    return "$" + num.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  return "₹" + num.toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });
};

function drawUnifiedChart(asset) {
  const canvas = document.getElementById("unifiedChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(300, rect.width) * dpr;
  canvas.height = Math.max(200, rect.height) * dpr;
  ctx.scale(dpr, dpr);

  const w = Math.max(300, rect.width);
  const h = Math.max(200, rect.height);
  ctx.clearRect(0, 0, w, h);

  if (!asset || !Array.isArray(asset.candles) || asset.candles.length === 0) {
    ctx.font = "13px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsl(350, 84%, 65%)";
    ctx.fillText(appState.errorMsg || "LIVE DATA UNAVAILABLE — SIMULATION DISABLED", 20, h / 2);
    return;
  }

  const candles = asset.candles;
  const currency = asset.currency || "INR";
  const symPrefix = currency === "USD" ? "$" : "₹";
  const m = asset.moreshwar_levels || {};

  let minP = Math.min(m.stop_loss ? m.stop_loss * 0.992 : Infinity, ...candles.map((c) => c.low));
  let maxP = Math.max(m.target ? m.target * 1.008 : -Infinity, asset.projected_30d_target || -Infinity, ...candles.map((c) => c.high));

  const padTop = 18, padBot = 24, padLeft = 10, padRight = 76;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBot;
  const yFor = (val) => padTop + plotH - ((val - minP) / Math.max(1e-6, maxP - minP)) * plotH;
  const stepX = plotW / candles.length;

  // Grid lines
  ctx.strokeStyle = "hsla(222, 15%, 24%, 0.45)";
  ctx.lineWidth = 1;
  ctx.font = "10.5px 'JetBrains Mono', monospace";
  ctx.fillStyle = "hsl(215, 15%, 65%)";
  for (let i = 0; i <= 4; i++) {
    const val = minP + ((maxP - minP) * i) / 4;
    const y = yFor(val);
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();
    ctx.fillText(`${symPrefix}${val.toFixed(val < 500 ? 1 : 0)}`, w - padRight + 6, y + 3);
  }

  // Real Date labels on X-axis (start, mid, end)
  const firstDate = candles[0].date || "";
  const midDate = candles[Math.floor(candles.length / 2)].date || "";
  const lastDate = candles[candles.length - 1].date || "";
  ctx.fillStyle = "hsl(215, 15%, 55%)";
  ctx.fillText(firstDate, padLeft, h - 6);
  ctx.fillText(midDate, padLeft + plotW * 0.45, h - 6);
  ctx.fillText(lastDate, w - padRight - 72, h - 6);

  // Degree-2 Polynomial Regression Curve (Amber)
  ctx.strokeStyle = "hsl(38, 92%, 55%)";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  candles.forEach((c, idx) => {
    const x = padLeft + idx * stepX + stepX * 0.5;
    const y = yFor(c.reg_value ?? c.close);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Real OHLC Candlesticks
  candles.forEach((c, idx) => {
    const x = padLeft + idx * stepX + stepX * 0.5;
    const isBull = c.close >= c.open;
    const col = isBull ? "hsl(158, 74%, 44%)" : "hsl(350, 84%, 60%)";
    ctx.strokeStyle = col;
    ctx.fillStyle = col;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(x, yFor(c.high));
    ctx.lineTo(x, yFor(c.low));
    ctx.stroke();

    const bTop = yFor(Math.max(c.open, c.close));
    const bBot = yFor(Math.min(c.open, c.close));
    const bH = Math.max(2, bBot - bTop);
    const bW = Math.max(4, stepX * 0.56);
    ctx.fillRect(x - bW / 2, bTop, bW, bH);
  });

  // Moreshwar Levels
  const drawLevelLine = (price, color, label) => {
    if (typeof price !== "number" || Number.isNaN(price)) return;
    const y = yFor(price);
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.fillText(label, padLeft + 6, Math.max(12, y - 5));
    ctx.restore();
  };

  drawLevelLine(m.target, "hsl(158, 74%, 44%)", `TP ${symPrefix}${m.target}`);
  drawLevelLine(asset.price, "hsl(196, 86%, 54%)", `CP ${symPrefix}${asset.price}`);
  drawLevelLine(m.stop_loss, "hsl(350, 84%, 60%)", `SL ${symPrefix}${m.stop_loss}`);
}

function applyPayload(payload) {
  appState.dataSource = payload.data_source || "Live Market Data";
  appState.kiteStatus = payload.kite_status || "0 holdings";
  appState.kiteAuth = payload.kite_auth || null;
  appState.generatedAt = payload.generated_at || new Date().toISOString();
  appState.cashBalance = Number((payload.portfolio && payload.portfolio.stats && payload.portfolio.stats.balance) || 0);
  appState.assets = payload.assets || {};
  appState.pipelines = payload.pipelines || {};
  appState.queue = payload.queue || { buys: [], sells: [], executed: [] };
  appState.rules = payload.rules || [];
  appState.health = payload.health || null;
  appState.config = payload.config || null;
  appState.errorMsg = null;

  appState.holdingsMap = {};
  const allPortfolioItems = [
    ...((payload.portfolio && payload.portfolio.holdings) || []),
    ...((payload.portfolio && payload.portfolio.watchlist) || [])
  ];
  allPortfolioItems.forEach((item) => {
    appState.holdingsMap[item.symbol] = {
      qty: Number(item.qty || 0),
      avg_price: Number(item.avg_price || item.ltp || 0)
    };
  });

  if (!appState.assets[appState.activeSymbol]) {
    const firstSym = Object.keys(appState.assets)[0];
    if (firstSym) appState.activeSymbol = firstSym;
  }

  if (appState.chatHistory.length === 0) {
    const syms = Object.keys(appState.assets).length;
    appState.chatHistory.push({
      role: "bot",
      text: `🤖 Connected to LIVE DATA ONLY (${appState.dataSource}).\n• Scanned ${syms} live symbols via 3-Agent Orchestrator (TechBot_1, FundBot_1, ExecBot_1).\n• Portfolio: ${appState.kiteStatus}.\n\nClick any quick chip or ask me to analyze or compare any scanned stock.`
    });
  }
}

function renderAllViews() {
  const assetsList = Object.values(appState.assets);

  // Subtitle & Kite Status
  const subEl = document.getElementById("liveSourceSubtitle");
  if (subEl) {
    if (appState.errorMsg) {
      subEl.textContent = `❌ ${appState.errorMsg}`;
      subEl.style.color = "var(--accent-crimson)";
    } else {
      const dtText = appState.generatedAt ? new Date(appState.generatedAt).toLocaleString() : "";
      subEl.textContent = `${appState.backendConnected ? "Live Backend API" : "GitHub Cloud Scan"} · ${appState.dataSource} · Updated ${dtText}`;
      subEl.style.color = "";
    }
  }
  const kiteEl = document.getElementById("kiteStatusSubtitle");
  if (kiteEl) kiteEl.textContent = appState.kiteStatus;

  // Holdings from live state (0 holdings when none held)
  const holdings = assetsList.filter((a) => (appState.holdingsMap[a.symbol]?.qty || 0) > 0);
  const invested = holdings.reduce((s, a) => s + (appState.holdingsMap[a.symbol].qty * appState.holdingsMap[a.symbol].avg_price), 0);
  const current = holdings.reduce((s, a) => s + (appState.holdingsMap[a.symbol].qty * a.price), 0);
  const pnl = current - invested;
  const pnlPct = invested > 0 ? (pnl / invested) * 100 : 0;
  const healthScore = holdings.length
    ? `${(holdings.reduce((s, a) => s + a.fundamental_score.total * (appState.holdingsMap[a.symbol].qty * a.price), 0) / Math.max(1, current)).toFixed(1)} / 100`
    : "0 Holdings";

  document.getElementById("hdrInvested").textContent = fmtCurrency(invested, "INR");
  document.getElementById("hdrCurrent").textContent = fmtCurrency(current, "INR");
  const pnlEl = document.getElementById("hdrPnl");
  pnlEl.textContent = `${pnl >= 0 && invested > 0 ? "+" : ""}${fmtCurrency(pnl, "INR")} (${pnlPct.toFixed(2)}%)`;
  pnlEl.style.color = pnl > 0 ? "var(--accent-emerald)" : (pnl < 0 ? "var(--accent-crimson)" : "var(--text-primary)");
  document.getElementById("hdrBalance").textContent = fmtCurrency(appState.cashBalance, "INR");
  document.getElementById("hdrHealth").textContent = healthScore;

  // Top Picks Queue (strictly from live orchestrator queue.buys)
  const picksContainer = document.getElementById("picksQueueContainer");
  picksContainer.innerHTML = "";
  const buysQueue = appState.queue.buys || [];
  if (buysQueue.length === 0) {
    picksContainer.innerHTML = `<div class="terminal-box" style="color: var(--text-secondary);">No active orchestrator picks loaded.</div>`;
  } else {
    buysQueue.forEach((p) => {
      const cur = p.currency || appState.assets[p.symbol]?.currency || "INR";
      const row = document.createElement("div");
      row.className = `pick-row ${p.symbol === appState.activeSymbol ? "selected" : ""}`;
      row.innerHTML = `
        <div>
          <span class="pick-sym">${p.symbol}</span>
          <span class="pick-sub">${p.setup_type} · ${p.decision_badge}</span>
        </div>
        <span class="tag-pill tag-entry">${fmtCurrency(p.entry, cur, p.entry < 500 ? 2 : 0)}</span>
        <span class="tag-pill tag-target">${fmtCurrency(p.target, cur, p.target < 500 ? 2 : 0)}</span>
        <span class="tag-pill tag-stop">${fmtCurrency(p.stop_loss, cur, p.stop_loss < 500 ? 2 : 0)}</span>
        <span class="tag-pill tag-score">${p.fundamental_score}</span>
      `;
      row.addEventListener("click", () => {
        appState.activeSymbol = p.symbol;
        renderAllViews();
      });
      picksContainer.appendChild(row);
    });
  }

  // Sell Warning Box (only real held assets that drop below safety floor)
  const sellContainer = document.getElementById("sellWarningContainer");
  const sellsQueue = appState.queue.sells || [];
  if (sellsQueue.length > 0) {
    const s = sellsQueue[0];
    sellContainer.innerHTML = `
      <div class="sell-warning-box">
        <div>
          <strong style="color: var(--accent-crimson); font-family: var(--font-mono);">⚠️ SELL WARNING: ${s.symbol}</strong>
          <div style="font-size: 11px; color: var(--text-secondary);">Fundamental Score ${s.fundamental_score}/100 · Decision: ${s.decision_badge} (${s.qty} sh)</div>
        </div>
        <button type="button" class="btn-danger" id="sellWarnBtn">Exit Now</button>
      </div>
    `;
    document.getElementById("sellWarnBtn").addEventListener("click", () => executeQueueTrades([s.id]));
  } else {
    sellContainer.innerHTML = `
      <div class="terminal-box" style="color: var(--text-secondary); text-align: center;">
        ${holdings.length === 0 ? "✅ 0 Holdings — No Active Sell Warnings" : "✅ Zero Active Sell Warnings — All Holdings Above Safety Floor"}
      </div>
    `;
  }

  // Right Column Live Kite Holdings
  const miniBody = document.getElementById("miniHoldingsTableBody");
  miniBody.innerHTML = "";
  if (holdings.length === 0) {
    miniBody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--text-secondary);padding:14px;">0 Live Holdings (${appState.kiteStatus})</td></tr>`;
  } else {
    holdings.forEach((h) => {
      const qty = appState.holdingsMap[h.symbol].qty;
      const avg = appState.holdingsMap[h.symbol].avg_price;
      const hPnl = (h.price - avg) * qty;
      const cur = h.currency || "INR";
      const tr = document.createElement("tr");
      tr.style.cursor = "pointer";
      tr.innerHTML = `
        <td style="font-family: var(--font-mono); font-weight: 600;">${h.symbol}</td>
        <td style="font-family: var(--font-mono);">${qty}</td>
        <td style="font-family: var(--font-mono);">${fmtCurrency(h.price, cur, 2)}</td>
        <td style="font-family: var(--font-mono); color: ${hPnl >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)"};">${hPnl >= 0 ? "+" : ""}${fmtCurrency(hPnl, cur)}</td>
        <td><span class="tag-pill ${h.fundamental_score.total >= 65 ? "tag-target" : "tag-stop"}">${h.fundamental_score.total}</span></td>
      `;
      tr.addEventListener("click", () => {
        appState.activeSymbol = h.symbol;
        renderAllViews();
      });
      miniBody.appendChild(tr);
    });
  }

  // Active Asset Center Panel & 3-Agent Pipeline
  const active = appState.assets[appState.activeSymbol] || assetsList[0];
  if (!active) {
    drawUnifiedChart(null);
    return;
  }

  const cur = active.currency || "INR";
  const f = active.fundamental_score;
  const p = active.porters_score;
  const m = active.moreshwar_levels;
  const d = active.decision_matrix;
  const pipe = appState.pipelines[active.symbol] || {};
  const msgs = pipe.messages || [];

  document.getElementById("centerChartTitle").textContent = `${active.symbol} (${active.yahoo_symbol || active.symbol}) — ${active.name} (${active.exchange} · 1D)`;
  document.getElementById("centerChartSub").textContent = `Real OHLCV (${active.candles[0]?.date || ""} → ${active.candles[active.candles.length - 1]?.date || ""}) · Degree-2 Polynomial Regression (${active.regression_curve})`;
  document.getElementById("centerSpotPrice").textContent = fmtCurrency(active.price, cur, 2);
  const chgEl = document.getElementById("centerSpotChg");
  chgEl.textContent = `${active.change_pct >= 0 ? "+" : ""}${active.change_pct}%`;
  chgEl.style.color = active.change_pct >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)";

  const decBadge = document.getElementById("centerDecisionBadge");
  decBadge.textContent = d.action;
  decBadge.className = `tag-pill ${f.total >= 65 ? "tag-target" : f.total <= 45 ? "tag-stop" : "tag-score"}`;

  document.getElementById("legTarget").textContent = fmtCurrency(m.target, cur, 2);
  document.getElementById("legEntry").textContent = fmtCurrency(active.price, cur, 2);
  document.getElementById("legStop").textContent = fmtCurrency(m.stop_loss, cur, 2);
  document.getElementById("legReg").textContent = fmtCurrency(active.projected_30d_target, cur, 2);

  const execQty = pipe.quantity || 0;
  const convPct = ((pipe.conviction_score || 0) * 100).toFixed(1);
  document.getElementById("analysisTerminalBox").innerHTML = `
    <div style="color: var(--accent-cyan); font-weight: 600;">${active.symbol} Live Orchestrator Verdict</div>
    <div>• StockSight V2 Score: <strong>${f.total}/100</strong> | Porter Moat: <strong>${p.total}/100</strong></div>
    <div>• Moreshwar Levels: Entry <strong>${fmtCurrency(m.entry_price, cur, 2)}</strong> · Target <strong>${fmtCurrency(m.target, cur, 2)}</strong> · SL <strong>${fmtCurrency(m.stop_loss, cur, 2)}</strong> (R:R 1:${m.rr_ratio})</div>
    <div>• Auto-Risk Sized Qty: <strong>${execQty} Shares</strong> (${fmtCurrency(execQty * active.price, cur, 2)})</div>
    <div>• Multi-Agent Verdict: <strong style="color: var(--accent-emerald);">${d.action}</strong> (${convPct}% Conviction)</div>
  `;

  // 3-Agent Cards from real pipeline messages
  const mTech = msgs[0] || {};
  const mFund = msgs[1] || {};
  const mExec = msgs[2] || {};

  const stamp = appState.generatedAt ? new Date(appState.generatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "";
  document.getElementById("pipelineMetaText").textContent = `Live Orchestration Scan (${stamp}) · ${(appState.queue.buys || []).length} Queued Picks · ${(appState.queue.sells || []).length} Sell Warnings`;

  const techConfPct = Math.round((mTech.confidence || 0) * 100);
  document.getElementById("agTechBadge").textContent = mTech.signal || "NEUTRAL";
  document.getElementById("agTechBadge").className = `tag-pill ${mTech.signal === "BULLISH" ? "tag-target" : mTech.signal === "BEARISH" ? "tag-stop" : "tag-score"}`;
  document.getElementById("agTechConf").textContent = `${techConfPct}%`;
  document.getElementById("agTechBar").style.width = `${techConfPct}%`;
  document.getElementById("agTechDesc").textContent = mTech.summary || `RSI(14)=${active.rsi_14}, 20D SMA=${fmtCurrency(active.sma_20, cur, 2)}`;

  document.getElementById("agFundBadge").textContent = mFund.signal || "NEUTRAL";
  document.getElementById("agFundBadge").className = `tag-pill ${f.total >= 65 ? "tag-target" : f.total <= 45 ? "tag-stop" : "tag-score"}`;
  document.getElementById("agFundConf").textContent = `${f.total}/100`;
  document.getElementById("agFundBar").style.width = `${f.total}%`;
  document.getElementById("agFundDesc").textContent = mFund.summary || "";

  document.getElementById("agExecBadge").textContent = mExec.signal || "STANDBY";
  document.getElementById("agExecConf").textContent = `${convPct}%`;
  document.getElementById("agExecBar").style.width = `${Math.min(100, Number(convPct))}%`;
  document.getElementById("agExecDesc").textContent = mExec.summary || "";

  // TAB 2: StockSight V2 Deep-Dive & Portfolio Manager
  const r = f.ratios || {};
  document.getElementById("ssDeepDiveTitle").textContent = `StockSight V2 Dual-Scoring Engine — ${active.symbol} (${active.name})`;
  document.getElementById("ssIndustryBadge").textContent = `Industry Profile: ${f.industry} · P/E ${r.pe}x · ROE ${r.roe}% · ROCE ${r.roce}%`;
  document.getElementById("ssFundTotalBadge").textContent = `Fundamental: ${f.total} / 100`;
  document.getElementById("ssPorterTotalBadge").textContent = `Porter Moat: ${p.total} / 100`;

  document.getElementById("ssFourPillarsContainer").innerHTML = `
    <div class="pillar-card">
      <div style="display:flex;justify-content:space-between;"><span>1. Business Pillar</span><strong>${f.business} / 40</strong></div>
      <div class="progress-track"><div class="progress-fill" style="width:${(f.business / 40) * 100}%;"></div></div>
      <span style="font-size:11px;color:var(--text-secondary);">Sales Growth ${r.sales_growth}% · Profit Growth ${r.profit_growth}% · OPM ${r.opm}%</span>
    </div>
    <div class="pillar-card">
      <div style="display:flex;justify-content:space-between;"><span>2. Economic Moat Pillar</span><strong>${f.moat} / 20</strong></div>
      <div class="progress-track"><div class="progress-fill" style="width:${(f.moat / 20) * 100}%;"></div></div>
      <span style="font-size:11px;color:var(--text-secondary);">ROE ${r.roe}% · ROCE ${r.roce}% · MCap ${Number(r.mcap_cr || 0).toLocaleString()} ${cur === "USD" ? "M" : "Cr"}</span>
    </div>
    <div class="pillar-card">
      <div style="display:flex;justify-content:space-between;"><span>3. Management &amp; Valuation</span><strong>${f.management} / 20</strong></div>
      <div class="progress-track"><div class="progress-fill" style="width:${(f.management / 20) * 100}%;"></div></div>
      <span style="font-size:11px;color:var(--text-secondary);">Trailing P/E ${r.pe}x · Beta ${r.beta}</span>
    </div>
    <div class="pillar-card">
      <div style="display:flex;justify-content:space-between;"><span>4. Risk &amp; Stability</span><strong>${f.risk} / 20</strong></div>
      <div class="progress-track"><div class="progress-fill" style="width:${(f.risk / 20) * 100}%;"></div></div>
      <span style="font-size:11px;color:var(--text-secondary);">${f.explanation}</span>
    </div>
  `;

  document.getElementById("ssPortersGrid").innerHTML = `
    <div class="pillar-card"><span>Barriers to Entry (MCap/ROCE)</span><strong>${p.entrants} / 20</strong></div>
    <div class="pillar-card"><span>Supplier Power (OPM Proxy)</span><strong>${p.suppliers} / 20</strong></div>
    <div class="pillar-card"><span>Buyer Pricing Power (ROE)</span><strong>${p.buyers} / 20</strong></div>
    <div class="pillar-card"><span>Threat of Substitutes &amp; Rivalry</span><strong>${p.substitutes + p.rivalry} / 40</strong></div>
  `;

  document.getElementById("ssDecisionMatrixBox").innerHTML = `
    <div style="color:var(--accent-emerald);font-weight:600;">Moreshwar Mathematical Model &amp; Final Decision Matrix</div>
    <div>• Composite Quality (X): <strong>${m.x_composite_score} / 100</strong> | Spot Price (Y): <strong>${fmtCurrency(active.price, cur, 2)}</strong></div>
    <div>• Dynamic Entry: <strong>${fmtCurrency(m.entry_price, cur, 2)}</strong> | Dynamic Target: <strong>${fmtCurrency(m.target, cur, 2)}</strong> | Dynamic Stop-Loss: <strong>${fmtCurrency(m.stop_loss, cur, 2)}</strong></div>
    <div>• Final Matrix Signal: <strong>${d.action}</strong> (${d.rationale})</div>
  `;

  const mgrBody = document.getElementById("fullPortfolioManagerBody");
  mgrBody.innerHTML = "";
  assetsList.forEach((a) => {
    const aCur = a.currency || "INR";
    const heldQty = appState.holdingsMap[a.symbol]?.qty || 0;
    const heldAvg = appState.holdingsMap[a.symbol]?.avg_price || a.price;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>
        <strong style="font-family:var(--font-mono);cursor:pointer;color:var(--accent-cyan);" class="sym-link">${a.symbol}</strong>
        <span style="display:block;font-size:10.5px;color:var(--text-secondary);">${heldQty > 0 ? "PORTFOLIO HOLDING" : "WATCHLIST (0 SH)"}</span>
      </td>
      <td style="font-family:var(--font-mono);">${a.fundamental_score.total} / ${a.porters_score.total}</td>
      <td><span class="tag-pill ${a.fundamental_score.total >= 65 ? "tag-target" : a.fundamental_score.total <= 45 ? "tag-stop" : "tag-score"}">${a.decision_matrix.action}</span></td>
      <td><input type="number" class="input-field qty-in" value="${heldQty}" min="0" style="width:74px;" /></td>
      <td><input type="number" class="input-field avg-in" value="${heldAvg}" min="0" step="0.01" style="width:94px;" /></td>
      <td style="font-family:var(--font-mono);">${fmtCurrency(a.price, aCur, 2)} → <span style="color:var(--accent-emerald);">${fmtCurrency(a.moreshwar_levels.target, aCur, 2)}</span></td>
      <td><button type="button" class="btn-secondary save-row-btn">Update</button></td>
    `;
    tr.querySelector(".sym-link").addEventListener("click", () => {
      appState.activeSymbol = a.symbol;
      renderAllViews();
    });
    tr.querySelector(".save-row-btn").addEventListener("click", async () => {
      const newQty = Math.max(0, parseInt(tr.querySelector(".qty-in").value, 10) || 0);
      const newAvg = parseFloat(tr.querySelector(".avg-in").value) || a.price;
      if (appState.backendConnected) {
        const res = await fetch("/api/portfolio/update", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ symbol: a.symbol, qty: newQty, avg_price: newAvg })
        });
        if (res.ok) applyPayload(await res.json());
      } else {
        appState.holdingsMap[a.symbol] = { qty: newQty, avg_price: newAvg };
      }
      renderAllViews();
    });
    mgrBody.appendChild(tr);
  });

  // TAB 3: Rules & Verification Console
  const rulesBox = document.getElementById("autoRulesList");
  rulesBox.innerHTML = "";
  appState.rules.forEach((rule) => {
    const div = document.createElement("div");
    div.className = "pillar-card";
    div.style.flexDirection = "row";
    div.style.alignItems = "center";
    div.style.justifyContent = "space-between";
    div.innerHTML = `
      <span style="font-size:12px;">${rule.name}</span>
      <button type="button" class="btn-secondary">${rule.active ? "ACTIVE ✓" : "PAUSED"}</button>
    `;
    div.querySelector("button").addEventListener("click", async () => {
      if (appState.backendConnected) {
        const res = await fetch("/api/rules/toggle", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rule_id: rule.rule_id })
        });
        if (res.ok) appState.rules = (await res.json()).rules || appState.rules;
      } else {
        rule.active = !rule.active;
      }
      renderAllViews();
    });
    rulesBox.appendChild(div);
  });

  const kiteEngineEl = document.getElementById("kiteEngineStatusText");
  if (kiteEngineEl) {
    const ka = appState.kiteAuth;
    if (ka) {
      kiteEngineEl.textContent = ka.access_token_active
        ? `Session Active (${ka.trading_day_ist}) · API Key: ${ka.api_key_configured ? "Ready" : "Missing"} · Auto-TOTP: ${ka.auto_login_ready ? "Enabled" : "Needs Secrets"}`
        : `Daily Token Idle · API Key: ${ka.api_key_configured ? "Ready" : "Not Set"} · Auto-TOTP: ${ka.auto_login_ready ? "Enabled" : "Set KITE_* Secrets"}`;
    } else {
      kiteEngineEl.textContent = appState.kiteStatus;
    }
  }

  document.getElementById("unifiedVerifyOutput").textContent =
    `$ python scripts/verify_setup.py\n` +
    `Live Telemetry (${appState.dataSource})\n` +
    JSON.stringify({
      pipeline_status: appState.health?.pipeline_status || "HEALTHY",
      data_mode: "LIVE_ONLY (ZERO SIMULATION)",
      kite_status: appState.kiteStatus,
      kite_auth: appState.kiteAuth,
      generated_at: appState.generatedAt,
      active_symbol: active.symbol,
      live_price: active.price,
      currency: cur,
      fundamental_score: `${f.total}/100`,
      porters_5_forces: `${p.total}/100`,
      moreshwar_target: m.target,
      moreshwar_stop_loss: m.stop_loss,
      projected_30d_target: active.projected_30d_target,
      regression_curve: active.regression_curve,
      decision_matrix: d.action
    }, null, 2);

  drawUnifiedChart(active);
}

function renderChat() {
  const box = document.getElementById("stockyChatMessages");
  box.innerHTML = "";
  appState.chatHistory.forEach((m) => {
    const div = document.createElement("div");
    div.className = `chat-bubble ${m.role}`;
    div.textContent = m.text;
    box.appendChild(div);
  });
  box.scrollTop = box.scrollHeight;
}

async function handleStockyQuery(rawText) {
  const q = rawText.trim();
  if (!q) return;
  appState.chatHistory.push({ role: "user", text: q });
  renderChat();

  if (appState.backendConnected) {
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: q })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.action_symbol && appState.assets[data.action_symbol]) {
          appState.activeSymbol = data.action_symbol;
          renderAllViews();
        }
        appState.chatHistory.push({ role: "bot", text: data.reply || "No reply." });
        renderChat();
        return;
      }
    } catch (_) {
      // fall through to live snapshot query evaluator
    }
  }

  const ql = q.toLowerCase();
  let reply = "";
  const cmp = ql.match(/(?:compare|vs)\s+([a-z0-9_-]+)\s+(?:and|with|vs)\s+([a-z0-9_-]+)/);
  const inv = ql.match(/(?:invest|allocate)\s+([\d,.]+)\s*(k|l|lakh|cr)?/);

  if (cmp) {
    const s1 = cmp[1].toUpperCase(), s2 = cmp[2].toUpperCase();
    const a1 = appState.assets[s1];
    const a2 = appState.assets[s2];
    if (!a1 || !a2) {
      reply = `❌ Both ${s1} and ${s2} must be present in the live market scan (simulation disabled).`;
    } else {
      const win = (a1.fundamental_score.total + a1.porters_score.total) >= (a2.fundamental_score.total + a2.porters_score.total) ? a1 : a2;
      reply =
        `📊 Live Head-to-Head: ${a1.symbol} vs ${a2.symbol}\n\n` +
        `• ${a1.symbol} (${fmtCurrency(a1.price, a1.currency, 2)}): Fund ${a1.fundamental_score.total}/100 · Porter ${a1.porters_score.total}/100 · ${a1.decision_matrix.action} · Target ${fmtCurrency(a1.moreshwar_levels.target, a1.currency, 2)}\n` +
        `• ${a2.symbol} (${fmtCurrency(a2.price, a2.currency, 2)}): Fund ${a2.fundamental_score.total}/100 · Porter ${a2.porters_score.total}/100 · ${a2.decision_matrix.action} · Target ${fmtCurrency(a2.moreshwar_levels.target, a2.currency, 2)}\n\n` +
        `🏆 Verdict: ${win.symbol} leads on risk-adjusted composite moat & growth quality.`;
    }
  } else if (inv) {
    let amt = parseFloat(inv[1].replace(/,/g, "")) || 100000;
    const u = (inv[2] || "").toLowerCase();
    if (u.startsWith("k")) amt *= 1000;
    else if (u.startsWith("l")) amt *= 100000;
    else if (u.startsWith("c")) amt *= 10000000;

    const top = Object.values(appState.assets)
      .filter((a) => (a.currency || "INR") === "INR" && a.fundamental_score.total >= 55)
      .sort((a, b) => b.moreshwar_levels.x_composite_score - a.moreshwar_levels.x_composite_score)
      .slice(0, 4);
    const sumS = top.reduce((s, a) => s + a.moreshwar_levels.x_composite_score, 0) || 1;
    reply = `💰 Score-Weighted Live Allocation Plan for ${fmtCurrency(amt, "INR")}:\n\n`;
    top.forEach((a) => {
      const w = a.moreshwar_levels.x_composite_score / sumS;
      const sh = Math.floor((amt * w) / a.price);
      reply += `• ${a.symbol} (Score ${a.moreshwar_levels.x_composite_score} · ${a.decision_matrix.action}): ${sh} sh @ ${fmtCurrency(a.price, "INR", 2)} = ${fmtCurrency(sh * a.price, "INR")}\n`;
    });
  } else if (ql.includes("efficiency") || ql.includes("portfolio") || ql.includes("trap")) {
    const held = Object.values(appState.assets).filter((a) => (appState.holdingsMap[a.symbol]?.qty || 0) > 0);
    if (held.length === 0) {
      reply = `🛡️ Live Portfolio Audit:\n• Currently 0 holdings (${appState.kiteStatus}).\n• Connect Zerodha Kite or add a live position to audit capital traps.`;
    } else {
      const traps = held.filter((a) => a.fundamental_score.total <= 45);
      reply = traps.length
        ? `⚠️ Capital Trap Detected: ${traps.map((t) => `${t.symbol} (${t.fundamental_score.total}/100)`).join(", ")}`
        : `✅ Zero Capital Traps across ${held.length} holdings.`;
    }
  } else {
    const found = Object.keys(appState.assets).find((s) => ql.includes(s.toLowerCase()));
    const a = appState.assets[found || appState.activeSymbol];
    if (!a) {
      reply = "❌ Symbol not found in live market scan (simulation disabled).";
    } else {
      appState.activeSymbol = a.symbol;
      renderAllViews();
      reply =
        `🔍 Live Analysis: ${a.symbol} (${a.name})\n` +
        `• Live Spot: ${fmtCurrency(a.price, a.currency, 2)} (${a.change_pct >= 0 ? "+" : ""}${a.change_pct}%)\n` +
        `• Fundamental Score: ${a.fundamental_score.total}/100 | Porter Moat: ${a.porters_score.total}/100\n` +
        `• Moreshwar Levels: Entry ${fmtCurrency(a.moreshwar_levels.entry_price, a.currency, 2)} | Target ${fmtCurrency(a.moreshwar_levels.target, a.currency, 2)} | SL ${fmtCurrency(a.moreshwar_levels.stop_loss, a.currency, 2)}\n` +
        `• 30D Polynomial Regression: ${fmtCurrency(a.projected_30d_target, a.currency, 2)} (${a.regression_curve}) | Decision: ${a.decision_matrix.action}`;
    }
  }

  appState.chatHistory.push({ role: "bot", text: reply });
  renderChat();
}

async function executeQueueTrades(tradeIds) {
  if (!appState.backendConnected) {
    appState.chatHistory.push({
      role: "bot",
      text: "ℹ️ Order execution requires the live Python backend (`server.py`) with Zerodha Kite Connect. Simulated local fills are disabled."
    });
    renderChat();
    return;
  }
  const res = await fetch("/api/approve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trade_ids: tradeIds, active_symbol: appState.activeSymbol })
  });
  if (res.ok) {
    applyPayload(await res.json());
    renderAllViews();
  }
}

const GH_OWNER = "Parth2004124";
const GH_REPO = "Finsistant-version-2";
const GH_WORKFLOW = "live-scan.yml";
const GH_PAT_KEY = "finsistant_gh_pat";

function getStoredGhPat() {
  try {
    return (window.localStorage.getItem(GH_PAT_KEY) || "").trim();
  } catch (_) {
    return "";
  }
}

function updateCloudRunnerStatusUI(customMsg = null) {
  const el = document.getElementById("ghCloudStatusText");
  if (!el) return;
  if (customMsg) {
    el.textContent = customMsg;
    return;
  }
  const hasPat = Boolean(getStoredGhPat());
  el.textContent = hasPat
    ? "GitHub Cloud Control Ready (Browser PAT active) · Scan Now & Add Asset run directly on GitHub Actions"
    : "Automated 15m Mon–Fri GitHub Cloud Scans active · Save optional browser PAT below for 1-click on-demand scans";
}

async function fetchLatestCloudSnapshot(preferRaw = false) {
  const candidates = preferRaw
    ? [
        `https://raw.githubusercontent.com/${GH_OWNER}/${GH_REPO}/main/live_scan.json?ts=${Date.now()}`,
        `live_scan.json?ts=${Date.now()}`
      ]
    : [`live_scan.json?ts=${Date.now()}`];

  for (const url of candidates) {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) continue;
      const payload = await res.json();
      if (payload && payload.assets && Object.keys(payload.assets).length > 0) {
        return payload;
      }
    } catch (_) {
      continue;
    }
  }
  throw new Error("live_scan.json unavailable");
}

async function triggerGitHubCloudWorkflow(inputs = {}, targetSymbol = null) {
  const pat = getStoredGhPat();
  if (!pat) return { dispatched: false, reason: "no_pat" };

  const metaEl = document.getElementById("pipelineMetaText");
  const prevGeneratedAt = appState.generatedAt;
  const setStatus = (msg) => {
    if (metaEl) metaEl.textContent = msg;
    updateCloudRunnerStatusUI(msg);
  };

  setStatus("Dispatching Live Orchestration Scan on GitHub Cloud...");
  const dispatchAt = Date.now();
  const headers = {
    Accept: "application/vnd.github+json",
    Authorization: `Bearer ${pat}`,
    "Content-Type": "application/json",
    "X-GitHub-Api-Version": "2022-11-28"
  };

  const dispatchRes = await fetch(
    `https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/actions/workflows/${GH_WORKFLOW}/dispatches`,
    {
      method: "POST",
      headers,
      body: JSON.stringify({ ref: "main", inputs })
    }
  );
  if (!dispatchRes.ok) {
    const errBody = await dispatchRes.json().catch(() => ({}));
    throw new Error(errBody.message || `GitHub dispatch HTTP ${dispatchRes.status}`);
  }

  // Poll GitHub Actions run status for up to ~75 seconds
  let runId = null;
  for (let attempt = 0; attempt < 20; attempt++) {
    await new Promise((r) => setTimeout(r, 3500));
    const runsRes = await fetch(
      `https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/actions/workflows/${GH_WORKFLOW}/runs?per_page=3`,
      { headers, cache: "no-store" }
    );
    if (!runsRes.ok) continue;
    const runsJson = await runsRes.json();
    const recent = (runsJson.workflow_runs || []).find(
      (r) => new Date(r.created_at).getTime() >= dispatchAt - 15000
    ) || (runsJson.workflow_runs || [])[0];
    if (!recent) continue;
    runId = recent.id;

    if (recent.status !== "completed") {
      setStatus(`GitHub Cloud Scan #${runId}: ${recent.status.replace("_", " ")}...`);
      continue;
    }
    if (recent.conclusion !== "success") {
      throw new Error(`GitHub Cloud Scan #${runId} finished with status: ${recent.conclusion}`);
    }

    setStatus(`GitHub Cloud Scan #${runId} succeeded — loading fresh live data...`);
    for (let p = 0; p < 4; p++) {
      const fresh = await fetchLatestCloudSnapshot(true);
      if (fresh.generated_at !== prevGeneratedAt || !targetSymbol || fresh.assets[targetSymbol]) {
        appState.backendConnected = false;
        applyPayload(fresh);
        if (targetSymbol && fresh.assets[targetSymbol]) {
          appState.activeSymbol = targetSymbol;
        }
        renderAllViews();
        renderChat();
        updateCloudRunnerStatusUI(`GitHub Cloud Scan #${runId} completed (${new Date(fresh.generated_at).toLocaleTimeString()})`);
        return { dispatched: true, runId };
      }
      await new Promise((r) => setTimeout(r, 2000));
    }
    break;
  }

  // Reload whatever latest snapshot is available
  const latest = await fetchLatestCloudSnapshot(true);
  applyPayload(latest);
  renderAllViews();
  return { dispatched: true, runId };
}

async function runLiveScan(isAuto = false) {
  const metaEl = document.getElementById("pipelineMetaText");
  if (metaEl) metaEl.textContent = isAuto ? "Running auto-scan on live market feed..." : "Refreshing live orchestration scan...";

  // 1. Try live Python backend first when not hosted on static GitHub Pages
  const isStaticPages = window.location.hostname.endsWith("github.io") || window.location.protocol === "file:";
  if (!isStaticPages) {
    try {
      const endpoint = isAuto ? "/api/state" : "/api/scan";
      const res = await fetch(endpoint, {
        method: isAuto ? "GET" : "POST",
        headers: isAuto ? undefined : { "Content-Type": "application/json" },
        body: isAuto ? undefined : JSON.stringify({ symbol: appState.activeSymbol })
      });
      const ct = res.headers.get("content-type") || "";
      if (res.ok && ct.includes("application/json")) {
        appState.backendConnected = true;
        applyPayload(await res.json());
        renderAllViews();
        renderChat();
        return;
      }
    } catch (_) {
      // Not running under server.py; proceed to cloud-generated live_scan.json
    }
  }

  // 2. If user clicked Scan Now manually and has a browser GitHub PAT, trigger live cloud scan on GitHub Actions
  if (!isAuto && getStoredGhPat()) {
    try {
      const res = await triggerGitHubCloudWorkflow({}, appState.activeSymbol);
      if (res.dispatched) return;
    } catch (err) {
      updateCloudRunnerStatusUI(`Cloud dispatch error: ${err.message}`);
    }
  }

  // 3. Load cloud-generated live_scan.json (GitHub Actions + Yahoo Finance + Zerodha Kite + 3-Agent Orchestrator)
  try {
    const payload = await fetchLatestCloudSnapshot(!isAuto);
    appState.backendConnected = false;
    applyPayload(payload);
    renderAllViews();
    renderChat();
  } catch (err) {
    appState.errorMsg = `Live market data unavailable (${err.message}). Simulated fallback is disabled.`;
    renderAllViews();
    renderChat();
  }
}

function bootFinsistantApp() {
  updateCloudRunnerStatusUI();
  runLiveScan(true);

  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`panel-${btn.dataset.tab}`).classList.add("active");
      if (btn.dataset.tab === "mission") {
        drawUnifiedChart(appState.assets[appState.activeSymbol]);
      }
    });
  });

  document.getElementById("switchToStockSightBtn").addEventListener("click", () => {
    document.querySelector('[data-tab="stocksight"]').click();
  });
  document.getElementById("manageHoldingsBtn").addEventListener("click", () => {
    document.querySelector('[data-tab="stocksight"]').click();
  });

  document.getElementById("executeSelectedBtn").addEventListener("click", () => {
    const pick = (appState.queue.buys || []).find((b) => b.symbol === appState.activeSymbol);
    if (pick) executeQueueTrades([pick.id]);
  });

  document.getElementById("executeAllBtn").addEventListener("click", () => {
    const ids = (appState.queue.buys || []).map((b) => b.id);
    executeQueueTrades(ids);
  });

  document.getElementById("scanNowBtn").addEventListener("click", () => {
    runLiveScan(false);
  });

  document.getElementById("addSymbolBtn").addEventListener("click", async () => {
    const inp = document.getElementById("addSymbolInput");
    const sym = inp.value.trim().toUpperCase();
    if (!sym) return;
    if (appState.backendConnected) {
      const res = await fetch("/api/portfolio/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: sym, qty: 0, avg_price: 0 })
      });
      if (res.ok) {
        applyPayload(await res.json());
        appState.activeSymbol = sym;
        inp.value = "";
        renderAllViews();
      } else {
        const err = await res.json().catch(() => ({}));
        alert(`Live lookup failed for ${sym}: ${err.error || res.status}`);
      }
    } else if (appState.assets[sym]) {
      appState.activeSymbol = sym;
      inp.value = "";
      renderAllViews();
    } else if (getStoredGhPat()) {
      try {
        inp.disabled = true;
        await triggerGitHubCloudWorkflow({ add_symbol: sym }, sym);
        inp.value = "";
      } catch (err) {
        alert(`GitHub Cloud ticker scan failed for ${sym}: ${err.message}`);
      } finally {
        inp.disabled = false;
      }
    } else {
      const openActions = confirm(
        `'${sym}' is not in the current cloud snapshot.\n\n` +
        `Click OK to open GitHub Actions and run Live Orchestration Scan with add_symbol=${sym} (or save a fine-grained GitHub Actions PAT in Tab 3 for 1-click in-app scanning).`
      );
      if (openActions) {
        window.open(
          `https://github.com/${GH_OWNER}/${GH_REPO}/actions/workflows/${GH_WORKFLOW}`,
          "_blank",
          "noopener,noreferrer"
        );
      }
    }
  });

  document.querySelectorAll(".prompt-chip").forEach((chip) => {
    chip.addEventListener("click", () => handleStockyQuery(chip.dataset.prompt));
  });

  document.getElementById("stockySendBtn").addEventListener("click", () => {
    const inp = document.getElementById("stockyChatInput");
    handleStockyQuery(inp.value);
    inp.value = "";
  });
  document.getElementById("stockyChatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      handleStockyQuery(e.target.value);
      e.target.value = "";
    }
  });

  async function saveRuntimeConfig() {
    if (!appState.backendConnected) return false;
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        finsistant_environment: document.getElementById("cfgEnv").value,
        agent_mode: document.getElementById("cfgMode").value,
        max_concurrent_trades: parseInt(document.getElementById("cfgMaxTrades").value, 10) || 5,
        kite_api_key: document.getElementById("cfgKiteKey").value,
        kite_api_secret: document.getElementById("cfgKiteSecret").value,
        kite_user_id: document.getElementById("cfgKiteUser").value,
        kite_password: document.getElementById("cfgKitePass").value,
        kite_totp_secret: document.getElementById("cfgKiteTotp").value,
        kite_access_token: document.getElementById("cfgKiteToken").value
      })
    });
    return res.ok;
  }

  document.getElementById("saveUnifiedConfigBtn").addEventListener("click", async () => {
    await saveRuntimeConfig();
    runLiveScan(false);
  });

  const autoLoginBtn = document.getElementById("kiteAutoLoginBtn");
  if (autoLoginBtn) {
    autoLoginBtn.addEventListener("click", async () => {
      const statusEl = document.getElementById("kiteEngineStatusText");
      if (!appState.backendConnected) {
        if (getStoredGhPat()) {
          if (statusEl) statusEl.textContent = "Triggering GitHub Cloud Kite TOTP refresh...";
          try {
            await triggerGitHubCloudWorkflow({ refresh_kite: "true" }, appState.activeSymbol);
          } catch (err) {
            if (statusEl) statusEl.textContent = `Cloud Kite refresh failed: ${err.message}`;
          }
          return;
        }
        if (statusEl) {
          statusEl.textContent = "GitHub Cloud TOTP is active on scheduled runs. Save a browser PAT below or open Actions to force an immediate refresh.";
        }
        return;
      }
      if (statusEl) statusEl.textContent = "Fetching fresh daily Kite access_token via TOTP...";
      await saveRuntimeConfig();
      const res = await fetch("/api/kite/auto-login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active_symbol: appState.activeSymbol })
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.assets) {
        applyPayload(data);
        renderAllViews();
      } else if (statusEl) {
        statusEl.textContent = `Auto-login failed: ${data.error || res.status}`;
      }
    });
  }

  const loginLinkBtn = document.getElementById("kiteLoginLinkBtn");
  if (loginLinkBtn) {
    loginLinkBtn.addEventListener("click", () => {
      const enteredKey = (document.getElementById("cfgKiteKey")?.value || "").trim();
      const url = (appState.kiteAuth && appState.kiteAuth.login_url) ||
        (enteredKey ? `https://kite.zerodha.com/connect/login?v=3&api_key=${encodeURIComponent(enteredKey)}` : "");
      if (url) {
        window.open(url, "_blank", "noopener,noreferrer");
      } else {
        const statusEl = document.getElementById("kiteEngineStatusText");
        if (statusEl) statusEl.textContent = "Enter KITE_API_KEY (or set it in .env / GitHub Secrets) to open Zerodha Kite Login.";
      }
    });
  }

  const exchangeBtn = document.getElementById("kiteExchangeTokenBtn");
  if (exchangeBtn) {
    exchangeBtn.addEventListener("click", async () => {
      const statusEl = document.getElementById("kiteEngineStatusText");
      const reqToken = (document.getElementById("kiteReqTokenInput")?.value || "").trim();
      if (!reqToken) {
        if (statusEl) statusEl.textContent = "Paste the request_token from Zerodha's redirect URL first.";
        return;
      }
      if (!appState.backendConnected) {
        if (statusEl) statusEl.textContent = "GitHub Cloud runs use automatic TOTP login (no manual request_token needed).";
        return;
      }
      const res = await fetch("/api/kite/exchange-token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_token: reqToken,
          api_key: document.getElementById("cfgKiteKey")?.value || "",
          api_secret: document.getElementById("cfgKiteSecret")?.value || "",
          active_symbol: appState.activeSymbol
        })
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.assets) {
        document.getElementById("kiteReqTokenInput").value = "";
        applyPayload(data);
        renderAllViews();
      } else if (statusEl) {
        statusEl.textContent = `Token exchange failed: ${data.error || res.status}`;
      }
    });
  }

  const savePatBtn = document.getElementById("saveGhPatBtn");
  if (savePatBtn) {
    savePatBtn.addEventListener("click", () => {
      const val = (document.getElementById("ghCloudPatInput")?.value || "").trim();
      if (!val) {
        updateCloudRunnerStatusUI("Enter a GitHub fine-grained PAT (Actions: Read & Write) first.");
        return;
      }
      try {
        window.localStorage.setItem(GH_PAT_KEY, val);
        document.getElementById("ghCloudPatInput").value = "";
        updateCloudRunnerStatusUI("Browser GitHub PAT saved locally · 1-click cloud scans & Add Asset enabled");
      } catch (err) {
        updateCloudRunnerStatusUI(`Could not save PAT in localStorage: ${err.message}`);
      }
    });
  }

  const clearPatBtn = document.getElementById("clearGhPatBtn");
  if (clearPatBtn) {
    clearPatBtn.addEventListener("click", () => {
      try {
        window.localStorage.removeItem(GH_PAT_KEY);
      } catch (_) {}
      updateCloudRunnerStatusUI();
    });
  }

  const triggerCloudBtn = document.getElementById("triggerCloudScanBtn");
  if (triggerCloudBtn) {
    triggerCloudBtn.addEventListener("click", async () => {
      if (appState.backendConnected) {
        runLiveScan(false);
        return;
      }
      if (!getStoredGhPat()) {
        window.open(
          `https://github.com/${GH_OWNER}/${GH_REPO}/actions/workflows/${GH_WORKFLOW}`,
          "_blank",
          "noopener,noreferrer"
        );
        updateCloudRunnerStatusUI("Opened GitHub Actions workflow page (save a browser PAT above to run silently in-app).");
        return;
      }
      try {
        await triggerGitHubCloudWorkflow({}, appState.activeSymbol);
      } catch (err) {
        updateCloudRunnerStatusUI(`Cloud scan error: ${err.message}`);
      }
    });
  }

  document.getElementById("runUnifiedVerifyBtn").addEventListener("click", () => runLiveScan(false));

  window.addEventListener("resize", () => {
    drawUnifiedChart(appState.assets[appState.activeSymbol]);
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootFinsistantApp);
} else {
  bootFinsistantApp();
}
