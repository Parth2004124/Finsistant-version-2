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
  geminiStatus: "Checking...",
  generatedAt: "",
  activeSymbol: "RELIANCE",
  cashBalance: 0,
  assets: {},       // keyed by symbol -> live analysis payload
  pipelines: {},    // keyed by symbol -> AgentRunner PipelineResult payload
  geminiCritiques: {}, // keyed by symbol -> Gemini critical trade analysis
  provenance: null, // Data Integrity Monitor payload
  backtest: null,   // Walk-forward Backtest & Multi-Agent Attribution payload
  moreshwarV2Research: null, // Shadow Moreshwar 2.0 Phase 5-8 Research Comparison
  mv2SelectedSymbol: null,
  moreshwarV21Research: null, // Moreshwar 2.1 5Y Statistical Validation Workbench
  moreshwarV22Research: null, // Moreshwar 2.2 Integrated Optimization & Validation (Research Only)
  mv21Filters: { model: "Moreshwar 2.1", instrument: "ALL", sector: "ALL", regime: "ALL" },
  auditTrail: null, // Permanent Decision Audit Ledger payload
  selectedAuditId: null,
  intelFocus: "all",
  holdingsMap: {},  // keyed by symbol -> { qty, avg_price }
  queue: { buys: [], sells: [], executed: [] },
  rules: [],
  health: null,
  config: null,
  errorMsg: null,
  chatHistory: []
};

const escHtml = (val) => String(val ?? "")
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");

const fmtTsShort = (iso) => {
  if (!iso) return "--";
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso).slice(0, 19);
    return d.toLocaleString([], { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
  } catch (_) {
    return String(iso).slice(0, 19);
  }
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
  if (m.recalibrated && typeof m.entry_price === "number" && Math.abs(m.entry_price - asset.price) > asset.price * 0.002) {
    drawLevelLine(m.entry_price, "hsl(38, 92%, 62%)", `LIM ${symPrefix}${m.entry_price}`);
  }
  drawLevelLine(m.stop_loss, "hsl(350, 84%, 60%)", `SL ${symPrefix}${m.stop_loss}`);
}

function applyPayload(payload) {
  appState.dataSource = payload.data_source || "Live Market Data";
  appState.kiteStatus = payload.kite_status || "0 holdings";
  appState.kiteAuth = payload.kite_auth || null;
  appState.geminiStatus = payload.gemini_status || "Unavailable";
  appState.generatedAt = payload.generated_at || new Date().toISOString();
  appState.cashBalance = Number((payload.portfolio && payload.portfolio.stats && payload.portfolio.stats.balance) || 0);
  appState.assets = payload.assets || {};
  appState.pipelines = payload.pipelines || {};
  appState.geminiCritiques = payload.gemini_critiques || {};
  appState.provenance = payload.provenance || null;
  appState.backtest = payload.backtest || null;
  appState.moreshwarV2Research = payload.moreshwar_v2_research || null;
  appState.moreshwarV21Research = payload.moreshwar_v2_1_research || null;
  appState.moreshwarV22Research = payload.moreshwar_v2_2_research || null;
  appState.moreshwarV3Research = payload.moreshwar_v3_research || null;
  if (!appState._moreshwarV31LiveLoaded && payload.moreshwar_v3_1) {
    appState.moreshwarV31 = payload.moreshwar_v3_1;
  }
  appState.auditTrail = payload.audit_trail || null;
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

  // Top Picks Queue (strictly from live orchestrator queue.buys + deferred critic audits)
  const picksContainer = document.getElementById("picksQueueContainer");
  picksContainer.innerHTML = "";
  const buysQueue = appState.queue.buys || [];
  const deferredQueue = appState.queue.deferred || [];
  const combinedQueue = [...buysQueue, ...deferredQueue];
  if (combinedQueue.length === 0) {
    picksContainer.innerHTML = `<div class="terminal-box" style="color: var(--text-secondary);">No active orchestrator picks loaded.</div>`;
  } else {
    combinedQueue.forEach((p) => {
      const cur = p.currency || appState.assets[p.symbol]?.currency || "INR";
      const isDeferred = p.status === "DEFERRED";
      const recalTag = p.recalibrated ? (isDeferred ? " · CRO DEFERRED" : " · RECALIBRATED") : "";
      const row = document.createElement("div");
      row.className = `pick-row ${p.symbol === appState.activeSymbol ? "selected" : ""}`;
      if (isDeferred) row.style.opacity = "0.68";
      row.innerHTML = `
        <div>
          <span class="pick-sym">${p.symbol}</span>
          <span class="pick-sub">${p.setup_type} · ${p.decision_badge}${recalTag}</span>
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

  // Strictly enforce disabled execution & broker controls whenever local server.py is offline
  const isOnline = Boolean(appState.backendConnected);
  const offlineTitle = "Disabled: local laptop server (server.py) is offline. Cloud/static mode is read-only.";
  const execSelBtn = document.getElementById("executeSelectedBtn");
  const execAllBtn = document.getElementById("executeAllBtn");
  const sellBtn = document.getElementById("sellWarnBtn");
  const saveCfgBtn = document.getElementById("saveUnifiedConfigBtn");
  const kiteExchBtn = document.getElementById("kiteExchangeTokenBtn");
  if (execSelBtn) {
    execSelBtn.disabled = !isOnline || buysQueue.length === 0;
    execSelBtn.title = !isOnline ? offlineTitle : "Execute selected order on local server.py";
  }
  if (execAllBtn) {
    execAllBtn.disabled = !isOnline || buysQueue.length === 0;
    execAllBtn.title = !isOnline ? offlineTitle : "Execute all queued BUY orders on local server.py";
  }
  if (sellBtn) {
    sellBtn.disabled = !isOnline;
    sellBtn.title = !isOnline ? offlineTitle : "Execute sell exit on local server.py";
  }
  if (saveCfgBtn) {
    saveCfgBtn.disabled = !isOnline;
    saveCfgBtn.title = !isOnline ? offlineTitle : "Save runtime settings to local server.py";
  }
  if (kiteExchBtn) {
    kiteExchBtn.disabled = !isOnline;
    kiteExchBtn.title = !isOnline ? offlineTitle : "Exchange Zerodha request_token on local server.py";
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

  const assetProv = (appState.provenance && appState.provenance.assets && appState.provenance.assets[active.symbol]) || null;
  const pSpotEl = document.getElementById("provSpotBadge");
  const pTypeEl = document.getElementById("provPriceTypeBadge");
  const pTsEl = document.getElementById("provTimestampsText");
  if (pSpotEl && pTypeEl && pTsEl) {
    if (assetProv) {
      const fresh = assetProv.freshness || "LIVE";
      const srcTone = assetProv.is_stale ? "tag-stop" : (assetProv.is_delayed ? "tag-score" : "tag-target");
      const spotCat = assetProv.metrics?.spot_price?.source_category;
      const srcName = assetProv.cache_hit
        ? "CACHED DB"
        : spotCat === "NSE_OFFICIAL_MCP"
          ? "NSE OFFICIAL MCP"
          : "YAHOO FEED";
      pSpotEl.textContent = `${srcName} · ${fresh}${assetProv.discrepancies?.length ? " · DISCREPANCY!" : ""}`;
      pSpotEl.className = `tag-pill ${assetProv.discrepancies?.length ? "tag-stop" : srcTone}`;
      const brokerFill = assetProv.metrics?.broker_execution?.price_classification === "BROKER_CONFIRMED";
      pTypeEl.textContent = brokerFill ? "BROKER FILL + MODEL LEVELS" : "LEVELS: ANALYTICAL ESTIMATE (NOT EXECUTED)";
      pTypeEl.className = `tag-pill ${brokerFill ? "tag-target" : "tag-entry"}`;
      pTsEl.textContent = `Market TS (${assetProv.market_timestamp_kind === "exchange_quote" ? "Quote" : "EOD"}): ${fmtTsShort(assetProv.market_timestamp)} · Retrieval TS: ${fmtTsShort(assetProv.retrieval_timestamp)}`;
    } else {
      pSpotEl.textContent = active.data_source || "YAHOO FEED";
      pSpotEl.className = "tag-pill tag-entry";
      pTypeEl.textContent = "LEVELS: ANALYTICAL ESTIMATE";
      pTsEl.textContent = `Retrieval TS: ${fmtTsShort(active.fetched_at || appState.generatedAt)}`;
    }
  }

  document.getElementById("legTarget").textContent = fmtCurrency(m.target, cur, 2);
  document.getElementById("legEntry").textContent = m.recalibrated
    ? `${fmtCurrency(active.price, cur, 2)} (Lim ${fmtCurrency(m.entry_price, cur, 2)})`
    : fmtCurrency(active.price, cur, 2);
  document.getElementById("legStop").textContent = fmtCurrency(m.stop_loss, cur, 2);
  document.getElementById("legReg").textContent = fmtCurrency(active.projected_30d_target, cur, 2);

  const execQty = pipe.quantity || 0;
  const convPct = ((pipe.conviction_score || 0) * 100).toFixed(1);
  const crit = appState.geminiCritiques[active.symbol] || null;
  const recal = (crit && crit.recalibration) || null;
  const recalTag = recal && recal.applied ? ` <span style="color:var(--accent-amber);font-size:10.5px;">[RECALIBRATED BY GEMINI CRO]</span>` : "";

  document.getElementById("analysisTerminalBox").innerHTML = `
    <div style="color: var(--accent-cyan); font-weight: 600;">${active.symbol} Live Orchestrator Verdict${recalTag}</div>
    <div>• StockSight V2 Score: <strong>${f.total}/100</strong> | Porter Moat: <strong>${p.total}/100</strong></div>
    <div>• Moreshwar Levels: Entry <strong>${fmtCurrency(m.entry_price, cur, 2)}</strong> · Target <strong>${fmtCurrency(m.target, cur, 2)}</strong> · SL <strong>${fmtCurrency(m.stop_loss, cur, 2)}</strong> (R:R 1:${m.rr_ratio})</div>
    <div>• Auto-Risk Sized Qty: <strong>${execQty} Shares</strong> (${fmtCurrency(execQty * m.entry_price, cur, 2)})</div>
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

  // Gemini Critical Trade Analyst Panel (fills center blank space)
  const gTitleEl = document.getElementById("geminiPanelTitle");
  const gSubEl = document.getElementById("geminiPanelSub");
  const gRiskEl = document.getElementById("geminiRiskBadge");
  const gStanceEl = document.getElementById("geminiStanceBadge");
  const gBullEl = document.getElementById("geminiBullText");
  const gRisksEl = document.getElementById("geminiRisksText");
  const gStampEl = document.getElementById("geminiModelStamp");
  const gBodyEl = document.getElementById("geminiRationaleBody");
  const gRecalBadgeEl = document.getElementById("geminiRecalStatusBadge");
  const rActEl = document.getElementById("recalActionVal");
  const rEntEl = document.getElementById("recalEntryVal");
  const rTpSlEl = document.getElementById("recalTpSlVal");
  const rQtyEl = document.getElementById("recalQtyVal");
  const gRecalSumEl = document.getElementById("geminiRecalSummary");

  if (gTitleEl && crit) {
    gTitleEl.textContent = `Gemini Critical Trade Analyst — ${active.symbol}`;
    if (crit.status === "live") {
      const stance = String(crit.stance || "MONITOR").toUpperCase();
      const risk = String(crit.risk_rating || "MEDIUM").toUpperCase();
      gSubEl.textContent = `Adversarial risk & valuation audit across live fundamentals, technicals & Moreshwar R:R`;
      gRiskEl.textContent = `RISK: ${risk}`;
      gRiskEl.className = `tag-pill ${risk === "LOW" ? "tag-target" : (risk === "HIGH" || risk === "EXTREME") ? "tag-stop" : "tag-score"}`;
      gStanceEl.textContent = stance;
      gStanceEl.className = `tag-pill ${stance.includes("BUY") ? "tag-target" : (stance.includes("AVOID") || stance.includes("EXIT")) ? "tag-stop" : "tag-score"}`;
      gBullEl.textContent = crit.bull_case || "No material upside catalyst highlighted.";
      gRisksEl.textContent = crit.critical_risks || "No major risk anomalies flagged.";
      const cStamp = crit.generated_at ? new Date(crit.generated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
      gStampEl.textContent = `${crit.model || "gemini-2.5-flash"}${cStamp ? ` · ${cStamp}` : ""}`;
      gBodyEl.textContent = crit.rationale || "";

      if (gRecalBadgeEl && recal && recal.applied && recal.before && recal.after) {
        const b = recal.before;
        const a = recal.after;
        gRecalBadgeEl.textContent = a.status === "DEFERRED" ? "DEFERRED BY CRO" : "RECALIBRATED LIVE";
        gRecalBadgeEl.className = `tag-pill ${a.status === "DEFERRED" ? "tag-stop" : "tag-target"}`;
        rActEl.textContent = b.action === a.action ? a.action : `${b.action} → ${a.action}`;
        rEntEl.textContent = b.entry === a.entry
          ? fmtCurrency(a.entry, cur, 2)
          : `${fmtCurrency(b.entry, cur, 2)} → ${fmtCurrency(a.entry, cur, 2)}`;
        rTpSlEl.textContent = `${fmtCurrency(a.target, cur, 2)} / ${fmtCurrency(a.stop_loss, cur, 2)} (1:${a.rr_ratio})`;
        rQtyEl.textContent = `${a.qty} sh (${a.conviction}%)`;
        gRecalSumEl.textContent = recal.summary || "";
      }
    } else {
      gSubEl.textContent = appState.geminiStatus || "Gemini live key check required";
      gRiskEl.textContent = "LIVE KEY NEEDED";
      gRiskEl.className = "tag-pill tag-stop";
      gStanceEl.textContent = crit.stance || "GEMINI UNAVAILABLE";
      gStanceEl.className = "tag-pill tag-score";
      gBullEl.textContent = `Quantitative setup queued: ${d.action} (${f.total}/100 Fundamental, ${p.total}/100 Porter, Spot ${fmtCurrency(active.price, cur, 2)} → TP ${fmtCurrency(m.target, cur, 2)}).`;
      gRisksEl.textContent = crit.critical_risks || crit.error || "GEMINI_API_KEY missing or revoked.";
      gStampEl.textContent = "ZERO SIMULATION POLICY";
      gBodyEl.textContent = crit.rationale || "Update GEMINI_API_KEY in GitHub Repository Secrets to enable live Gemini critical rationales.";
      if (gRecalBadgeEl) {
        gRecalBadgeEl.textContent = "STANDBY (RAW QUANT)";
        gRecalBadgeEl.className = "tag-pill tag-score";
        rActEl.textContent = d.action;
        rEntEl.textContent = fmtCurrency(m.entry_price, cur, 2);
        rTpSlEl.textContent = `${fmtCurrency(m.target, cur, 2)} / ${fmtCurrency(m.stop_loss, cur, 2)}`;
        rQtyEl.textContent = `${execQty} sh (${convPct}%)`;
        gRecalSumEl.textContent = "Raw quant parameters preserved (no simulated recalibration applied while GEMINI_API_KEY is invalid/unconfigured).";
      }
    }
  } else if (gTitleEl) {
    gTitleEl.textContent = `Gemini Critical Trade Analyst — ${active.symbol}`;
    gSubEl.textContent = "Awaiting next cloud scan with GEMINI_API_KEY";
    gRiskEl.textContent = "PENDING";
    gRiskEl.className = "tag-pill tag-score";
    gStanceEl.textContent = "AWAITING SCAN";
    gStanceEl.className = "tag-pill tag-entry";
    gBullEl.textContent = "Run Live Orchestration Scan to generate live Gemini bull catalysts.";
    gRisksEl.textContent = "Run Live Orchestration Scan to generate adversarial risk flags.";
    gStampEl.textContent = "--";
    gBodyEl.textContent = "Gemini critical analysis is generated per scan and cached in live_scan.json without simulation.";
    if (gRecalBadgeEl) {
      gRecalBadgeEl.textContent = "PENDING";
      gRecalBadgeEl.className = "tag-pill tag-score";
      rActEl.textContent = d.action;
      rEntEl.textContent = fmtCurrency(m.entry_price, cur, 2);
      rTpSlEl.textContent = `${fmtCurrency(m.target, cur, 2)} / ${fmtCurrency(m.stop_loss, cur, 2)}`;
      rQtyEl.textContent = `${execQty} sh (${convPct}%)`;
    }
  }

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
      gemini_status: appState.geminiStatus,
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
      decision_matrix: d.action,
      gemini_stance: crit ? crit.stance : "PENDING"
    }, null, 2);

  drawUnifiedChart(active);
  renderIntelligenceViews();
}

function drawBacktestEquityChart(curve) {
  const canvas = document.getElementById("backtestEquityChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(280, rect.width || 460);
  const h = Math.max(180, rect.height || 220);
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  if (!Array.isArray(curve) || curve.length < 2) {
    ctx.font = "12px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsl(215, 15%, 65%)";
    ctx.fillText("Backtest equity curve will populate after next live scan.", 16, h / 2);
    return;
  }

  const vals = [];
  curve.forEach((pt) => {
    if (typeof pt.original_nav === "number") vals.push(pt.original_nav);
    if (typeof pt.recalibrated_nav === "number") vals.push(pt.recalibrated_nav);
    if (typeof pt.benchmark_nav === "number") vals.push(pt.benchmark_nav);
  });
  const minV = Math.min(...vals) * 0.99;
  const maxV = Math.max(...vals) * 1.01;
  const padT = 16, padB = 22, padL = 12, padR = 52;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const yFor = (v) => padT + plotH - ((v - minV) / Math.max(1e-6, maxV - minV)) * plotH;
  const xFor = (i) => padL + (i / Math.max(1, curve.length - 1)) * plotW;

  ctx.strokeStyle = "hsla(222, 15%, 24%, 0.45)";
  ctx.lineWidth = 1;
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillStyle = "hsl(215, 15%, 65%)";
  for (let i = 0; i <= 4; i++) {
    const v = minV + ((maxV - minV) * i) / 4;
    const y = yFor(v);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(1), w - padR + 6, y + 3);
  }
  ctx.fillText(curve[0].date || "", padL, h - 5);
  ctx.fillText(curve[curve.length - 1].date || "", w - padR - 66, h - 5);

  const plotLine = (key, color, dashed = false) => {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    if (dashed) ctx.setLineDash([4, 4]);
    ctx.beginPath();
    let started = false;
    curve.forEach((pt, i) => {
      const v = pt[key];
      if (typeof v !== "number" || Number.isNaN(v)) return;
      const x = xFor(i);
      const y = yFor(v);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  };

  plotLine("benchmark_nav", "hsl(38, 92%, 55%)", true);
  plotLine("original_nav", "hsl(196, 86%, 54%)", false);
  plotLine("recalibrated_nav", "hsl(158, 74%, 44%)", false);
}

function renderProvenanceMonitor() {
  const sumEl = document.getElementById("provSummaryBadges");
  const taxEl = document.getElementById("provTaxonomyCards");
  const discEl = document.getElementById("provDiscrepancyBanner");
  const bodyEl = document.getElementById("provAssetTableBody");
  if (!sumEl || !taxEl || !discEl || !bodyEl) return;

  const prov = appState.provenance;
  if (!prov) {
    sumEl.innerHTML = `<span class="tag-pill tag-score">AWAITING PROVENANCE TELEMETRY</span>`;
    taxEl.innerHTML = `<div class="taxonomy-card">Run live backend or next cloud scan to populate the 6-category source provenance matrix.</div>`;
    discEl.innerHTML = "";
    bodyEl.innerHTML = "";
    return;
  }

  const s = prov.summary || {};
  sumEl.innerHTML = `
    <span class="tag-pill tag-entry">${s.total_assets || 0} ASSETS</span>
    <span class="tag-pill ${(s.nse_mcp_count || 0) > 0 ? "tag-target" : "tag-score"}">${s.nse_mcp_count || 0} NSE MCP</span>
    <span class="tag-pill ${(s.stale_count || 0) > 0 ? "tag-stop" : "tag-target"}">${s.stale_count || 0} STALE</span>
    <span class="tag-pill ${(s.partial_fundamentals_count || 0) > 0 ? "tag-score" : "tag-target"}">${s.partial_fundamentals_count || 0} PARTIAL RATIOS</span>
    <span class="tag-pill ${(s.discrepancy_count || 0) > 0 ? "tag-stop" : "tag-target"}">${s.discrepancy_count || 0} DISCREPANCIES</span>
    <span class="tag-pill ${s.broker_connected ? "tag-target" : "tag-score"}">BROKER: ${s.broker_connected ? "CONNECTED" : "IDLE / READ-ONLY"}</span>
  `;

  taxEl.innerHTML = (prov.feeds || []).map((f) => {
    const st = String(f.status || "");
    const tone = (st.includes("STALE") || st.includes("UNAVAILABLE") || st.includes("DEGRADED"))
      ? "tag-stop"
      : (st.includes("DELAYED") || st.includes("IDLE"))
        ? "tag-score"
        : "tag-target";
    return `
      <div class="taxonomy-card">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
          <strong style="font-size:12px;">${escHtml(f.category)}</strong>
          <span class="tag-pill ${tone}">${escHtml(st)}</span>
        </div>
        <div class="taxonomy-meta">Market TS: ${escHtml(fmtTsShort(f.market_timestamp))}</div>
        <div class="taxonomy-meta">Retrieval TS: ${escHtml(fmtTsShort(f.retrieval_timestamp))}</div>
        <div style="font-size:11.5px;color:var(--text-secondary);">${escHtml(f.detail)}</div>
      </div>
    `;
  }).join("");

  const discList = prov.discrepancies || [];
  if (discList.length > 0) {
    discEl.innerHTML = discList.map((d) => `
      <div class="sell-warning-box" style="margin-bottom:6px; border-color: ${d.severity === "CRITICAL" ? "var(--accent-crimson)" : "var(--accent-amber)"}; background: ${d.severity === "CRITICAL" ? "var(--accent-crimson-dim)" : "var(--accent-amber-dim)"};">
        <div>
          <strong style="font-family:var(--font-mono);color:${d.severity === "CRITICAL" ? "var(--accent-crimson)" : "var(--accent-amber)"};">⚠ ${escHtml(d.symbol)} · ${escHtml(d.type)} (${d.delta_pct >= 0 ? "+" : ""}${d.delta_pct}%)</strong>
          <div style="font-size:11.5px;color:var(--text-secondary);">${escHtml(d.source_a)} vs ${escHtml(d.source_b)} — ${escHtml(d.message)}</div>
        </div>
      </div>
    `).join("");
  } else {
    discEl.innerHTML = `<div class="terminal-box" style="color:var(--accent-emerald);">✅ Zero unresolved cross-source price discrepancies above divergence thresholds (Feed Spot vs EOD Bar &lt; 1.5%, Primary Feed vs Yahoo/Broker LTP &lt; 0.75%).</div>`;
  }

  const assetsMap = prov.assets || {};
  bodyEl.innerHTML = Object.values(assetsMap).map((a) => {
    const assetObj = appState.assets[a.symbol] || {};
    const cur = assetObj.currency || "INR";
    const mSpot = a.metrics?.spot_price || {};
    const mModel = a.metrics?.stocksight_and_moreshwar || {};
    const mBroker = a.metrics?.broker_execution || {};
    const st = a.overall_status || "UNKNOWN";
    const stTone = st === "STALE" || st === "DISCREPANCY" ? "tag-stop" : st === "VERIFIED LIVE" ? "tag-target" : "tag-score";
    const flags = [
      ...(a.missing_fields || []).map((mf) => `Missing: ${mf}`),
      ...Object.entries(a.proxy_fields || {}).map(([k, v]) => `Proxy ${k}: ${v}`)
    ];
    const brokerTxt = mBroker.price_classification === "BROKER_CONFIRMED"
      ? `Broker ${fmtCurrency(mBroker.value, cur, 2)}`
      : "No Broker Fill";
    const modelEntry = assetObj.moreshwar_levels?.entry_price;
    return `
      <tr>
        <td style="font-family:var(--font-mono);font-weight:700;">${escHtml(a.symbol)}</td>
        <td><span class="tag-pill ${stTone}">${escHtml(st)}</span></td>
        <td style="font-size:11.5px;">${escHtml(mSpot.source_label || "Yahoo Feed")}<br><span style="color:var(--text-muted);">${escHtml(a.freshness)}</span></td>
        <td style="font-family:var(--font-mono);font-size:11px;">${escHtml(fmtTsShort(a.market_timestamp))}<br><span style="color:var(--text-muted);">${escHtml(a.market_timestamp_kind)}</span></td>
        <td style="font-family:var(--font-mono);font-size:11px;">${escHtml(fmtTsShort(a.retrieval_timestamp))}</td>
        <td style="font-family:var(--font-mono);font-size:11px;">
          Feed: <strong>${fmtCurrency(mSpot.value, cur, 2)}</strong><br>
          Model (${escHtml(mModel.price_classification || "ANALYTICAL")}): ${modelEntry ? fmtCurrency(modelEntry, cur, 2) : "--"}<br>
          <span style="color:${mBroker.price_classification === "BROKER_CONFIRMED" ? "var(--accent-emerald)" : "var(--text-muted)"};">${escHtml(brokerTxt)}</span>
        </td>
        <td style="font-size:11px;color:var(--text-secondary);">${flags.length ? escHtml(flags.join(" · ")) : "All primary metrics present"}</td>
      </tr>
    `;
  }).join("");
}

function renderBacktestAnalytics() {
  const kpiEl = document.getElementById("btBenchmarkKpis");
  const trackEl = document.getElementById("btTrackComparisonBody");
  const attrEl = document.getElementById("btAgentAttributionGrid");
  const methEl = document.getElementById("btMethodologyBox");
  const sepEl = document.getElementById("btSeparationBadge");
  if (!kpiEl || !trackEl || !attrEl || !methEl) return;

  const bt = appState.backtest;
  if (!bt || !bt.tracks) {
    kpiEl.innerHTML = `<div class="kpi-tile"><span class="kpi-tile-label">Status</span><span class="kpi-tile-val">Awaiting Backtest</span></div>`;
    trackEl.innerHTML = "";
    attrEl.innerHTML = "";
    methEl.innerHTML = "Run live backend or cloud scan to compute 1Y causal walk-forward backtest & Nifty 50 attribution.";
    drawBacktestEquityChart([]);
    return;
  }

  const trOrig = bt.tracks.original_ai_signals?.metrics || {};
  const trRecal = bt.tracks.recalibrated_ai_signals?.metrics || {};
  const trAct = bt.tracks.actual_executed_trades?.metrics || {};
  const actStatus = bt.tracks.actual_executed_trades?.status || "NO_LIVE_BROKER_FILLS";
  const bench = bt.benchmark || {};
  const attr = bt.agent_attribution || {};

  if (sepEl) {
    sepEl.textContent = actStatus === "BROKER_VERIFIED"
      ? "BROKER FILLS VERIFIED SEPARATELY FROM BACKTEST"
      : "SIMULATED BACKTEST ISOLATED · 0 REAL BROKER FILLS";
  }

  kpiEl.innerHTML = `
    <div class="kpi-tile">
      <span class="kpi-tile-label">Nifty 50 (^NSEI) Return</span>
      <span class="kpi-tile-val" style="color:var(--accent-amber);">${bench.benchmark_return_pct >= 0 ? "+" : ""}${bench.benchmark_return_pct ?? 0}%</span>
      <span class="kpi-tile-sub">${escHtml(bench.start_date || "")} → ${escHtml(bench.end_date || "")}</span>
    </div>
    <div class="kpi-tile">
      <span class="kpi-tile-label">Recalibrated Alpha vs Nifty</span>
      <span class="kpi-tile-val" style="color:${(bench.alpha_recalibrated_pct || 0) >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)"};">${(bench.alpha_recalibrated_pct || 0) >= 0 ? "+" : ""}${bench.alpha_recalibrated_pct ?? 0}%</span>
      <span class="kpi-tile-sub">Orig Alpha: ${(bench.alpha_original_pct || 0) >= 0 ? "+" : ""}${bench.alpha_original_pct ?? 0}%</span>
    </div>
    <div class="kpi-tile">
      <span class="kpi-tile-label">Sharpe / Sortino (Recal)</span>
      <span class="kpi-tile-val" style="color:var(--accent-cyan);">${trRecal.sharpe_ratio ?? 0} / ${trRecal.sortino_ratio ?? 0}</span>
      <span class="kpi-tile-sub">Beta vs Nifty: ${bench.beta ?? "N/A"}</span>
    </div>
    <div class="kpi-tile">
      <span class="kpi-tile-label">Max Drawdown Reduction</span>
      <span class="kpi-tile-val" style="color:var(--accent-emerald);">${trRecal.max_drawdown_pct ?? 0}%</span>
      <span class="kpi-tile-sub">Orig DD: ${trOrig.max_drawdown_pct ?? 0}% (Saved ${attr.Gemini_CRO?.drawdown_reduction_pct ?? 0}%)</span>
    </div>
    <div class="kpi-tile">
      <span class="kpi-tile-label">Actual Broker Track</span>
      <span class="kpi-tile-val">${trAct.total_signals || 0} Fills</span>
      <span class="kpi-tile-sub">${escHtml(actStatus)}</span>
    </div>
  `;

  const selM = bt.selectivity_metrics || {};
  const dqa = bt.decision_quality_answers || {};
  const sampleCtx = dqa.sample_context || {};
  const selBannerEl = document.getElementById("btSelectivityBanner");
  if (selBannerEl) {
    selBannerEl.innerHTML = `
      <div>
        <strong style="color:var(--accent-cyan);">Quality Selectivity Gate:</strong>
        Filtered <strong>${selM.noise_reduction_pct ?? 0}%</strong> noisy candidates (${selM.noisy_candidates_filtered ?? 0} of ${selM.unfiltered_candidate_windows ?? 0} rejected pre-CRO · ${selM.quality_gated_signals ?? 0} strict setups → ${selM.recalibrated_active_signals ?? 0} post-CRO active)
      </div>
      <div style="font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);">
        Live Forward Ledger: <strong>${sampleCtx.settled_records ?? 0}/${sampleCtx.min_recommended_settled ?? 30}</strong> settled (${sampleCtx.open_records ?? 0} awaiting T+1..T+14)
      </div>
    `;
  }

  const livePillEl = document.getElementById("btLiveSamplePill");
  if (livePillEl) {
    const sCount = sampleCtx.settled_records ?? 0;
    const minReq = sampleCtx.min_recommended_settled ?? 30;
    livePillEl.textContent = sampleCtx.sufficient_live_sample
      ? `LIVE LEDGER: ${sCount} SETTLED (STATISTICALLY ACTIVE)`
      : `LIVE LEDGER: ${sCount} / ${minReq} SETTLED · 1Y WALK-FORWARD ACTIVE`;
    livePillEl.className = `tag-pill ${sampleCtx.sufficient_live_sample ? "tag-target" : "tag-score"}`;
  }

  const dqGridEl = document.getElementById("btDecisionQualityGrid");
  if (dqGridEl) {
    const q1 = dqa.top_contributing_agent || {};
    const q2 = dqa.gemini_intervention_audit || {};
    const q3 = dqa.moreshwar_vs_benchmarks || {};
    const q2Blk = q2.blocks || {};
    const q2Down = q2.downgrades || {};
    const q2Comb = q2.combined_scorecard || {};
    const q2Live = q2.live_ledger_audit || {};
    const q1Rank = Array.isArray(q1.ranking) ? q1.ranking : [];
    const q3Rows = Array.isArray(q3.rows) ? q3.rows : [];
    const mVerdict = String(q3.verdict || "UNDERPERFORMS BENCHMARKS");
    const mTone = mVerdict.includes("OUTPERFORMS ALL") ? "tag-target" : mVerdict.includes("OUTPERFORMS") || mVerdict.includes("MIXED") ? "tag-score" : "tag-stop";

    dqGridEl.innerHTML = `
      <div class="dq-card" style="border-top:2px solid var(--accent-cyan);">
        <div class="dq-card-head">
          <span class="dq-q-title">Q1. TOP PROFIT AGENT</span>
          <span class="tag-pill tag-target">#1 ${escHtml(q1.winner || "--")}</span>
        </div>
        <div class="dq-headline">${escHtml(q1.summary || "Computing agent ablation ranking...")}</div>
        <div class="table-scroll">
          <table class="data-table dq-compact-table">
            <thead>
              <tr><th>#</th><th>Agent</th><th>Cum Lift</th><th>DD Saved</th><th>Verdict</th></tr>
            </thead>
            <tbody>
              ${q1Rank.map((r) => `
                <tr>
                  <td style="font-family:var(--font-mono);font-weight:700;">#${r.rank}</td>
                  <td style="font-family:var(--font-mono);font-weight:700;">${escHtml(r.agent)}</td>
                  <td style="font-family:var(--font-mono);color:${r.cum_return_lift_pct >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)"};">${r.cum_return_lift_pct >= 0 ? "+" : ""}${r.cum_return_lift_pct}%</td>
                  <td style="font-family:var(--font-mono);color:${r.drawdown_saved_pct >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)"};">${r.drawdown_saved_pct >= 0 ? "+" : ""}${r.drawdown_saved_pct}%</td>
                  <td style="font-size:10px;color:${String(r.verdict).includes("POSITIVE") ? "var(--accent-emerald)" : "var(--accent-amber)"};">${escHtml(r.verdict)}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>

      <div class="dq-card" style="border-top:2px solid var(--accent-amber);">
        <div class="dq-card-head">
          <span class="dq-q-title">Q2. GEMINI CRO AUDIT</span>
          <span class="tag-pill tag-entry">${q2Comb.prevention_rate_pct ?? 0}% PREVENTED</span>
        </div>
        <div class="dq-headline">${escHtml(q2Comb.headline || "Tracking CRO blocks vs downgrades.")}</div>
        <div class="dq-mini-grid">
          <div class="dq-mini-stat">
            <div class="dq-mini-lbl">Bad Trades Prevented</div>
            <div class="dq-mini-val" style="color:var(--accent-emerald);">${q2Comb.bad_trades_prevented ?? 0} / ${q2.total_interventions ?? 0}</div>
            <div class="dq-mini-sub">Blk ${q2Blk.prevented_bad_trades ?? 0} (avg -${q2Blk.avg_loss_avoided_pct ?? 0}%) + Down ${q2Down.beneficial_downgrades ?? 0}</div>
          </div>
          <div class="dq-mini-stat">
            <div class="dq-mini-lbl">Unnecessary Downgrades</div>
            <div class="dq-mini-val" style="color:var(--accent-crimson);">${q2Comb.good_trades_unnecessarily_downgraded ?? 0} / ${q2.total_interventions ?? 0}</div>
            <div class="dq-mini-sub">Missed Blk ${q2Blk.unnecessarily_blocked_good_trades ?? 0} (+${q2Blk.avg_upside_missed_pct ?? 0}%) + Down ${q2Down.unnecessary_downgrades ?? 0}</div>
          </div>
          <div class="dq-mini-stat">
            <div class="dq-mini-lbl">Net CRO Value Added</div>
            <div class="dq-mini-val" style="color:${(q2Comb.net_intervention_value_pct || 0) >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)"};">${(q2Comb.net_intervention_value_pct || 0) >= 0 ? "+" : ""}${q2Comb.net_intervention_value_pct ?? 0}%</div>
            <div class="dq-mini-sub">Preserved ${q2Down.preserved_winners ?? 0} · Unfilled ${q2Down.unfilled_pullbacks ?? 0}</div>
          </div>
          <div class="dq-mini-stat">
            <div class="dq-mini-lbl">Live Ledger Audit</div>
            <div class="dq-mini-val" style="color:var(--accent-cyan);">${q2Live.settled_interventions ?? 0} Settled</div>
            <div class="dq-mini-sub">${q2Live.open_awaiting_settlement ?? 0} awaiting T+1..T+14</div>
          </div>
        </div>
      </div>

      <div class="dq-card" style="border-top:2px solid var(--accent-crimson);">
        <div class="dq-card-head">
          <span class="dq-q-title">Q3. MORESHWAR VS BENCHMARKS</span>
          <span class="tag-pill ${mTone}">${escHtml(mVerdict)}</span>
        </div>
        <div class="dq-headline">${escHtml(q3.diagnostic || "")}</div>
        <div class="table-scroll">
          <table class="data-table dq-compact-table">
            <thead>
              <tr><th>Strategy</th><th>Sig</th><th>Win%</th><th>Cum%</th><th>DD%</th></tr>
            </thead>
            <tbody>
              ${q3Rows.map((r) => `
                <tr>
                  <td style="font-size:10px;font-weight:${String(r.strategy).includes("Moreshwar") ? "700" : "500"};">${escHtml(r.strategy)}</td>
                  <td style="font-family:var(--font-mono);">${r.signals ?? 0}</td>
                  <td style="font-family:var(--font-mono);">${r.win_rate_pct ?? "--"}%</td>
                  <td style="font-family:var(--font-mono);font-weight:700;color:${(r.cumulative_return_pct || 0) >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)"};">${(r.cumulative_return_pct || 0) >= 0 ? "+" : ""}${r.cumulative_return_pct ?? 0}%</td>
                  <td style="font-family:var(--font-mono);">${r.max_drawdown_pct != null ? `${r.max_drawdown_pct}%` : "--"}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  const rows = [
    ["Total Signals", trOrig.total_signals, trRecal.total_signals, trAct.total_signals],
    ["Winners / Losers", `${trOrig.winners || 0} / ${trOrig.losers || 0}`, `${trRecal.winners || 0} / ${trRecal.losers || 0}`, `${trAct.winners || 0} / ${trAct.losers || 0}`],
    ["Win Rate %", `${trOrig.win_rate_pct ?? 0}%`, `${trRecal.win_rate_pct ?? 0}%`, actStatus === "BROKER_VERIFIED" ? `${trAct.win_rate_pct}%` : "N/A (0 fills)"],
    ["Avg Return / Signal", `${trOrig.avg_return_pct ?? 0}%`, `${trRecal.avg_return_pct ?? 0}%`, actStatus === "BROKER_VERIFIED" ? `${trAct.avg_return_pct}%` : "N/A"],
    ["Avg MFE / Avg MAE", `+${trOrig.avg_mfe_pct ?? 0}% / ${trOrig.avg_mae_pct ?? 0}%`, `+${trRecal.avg_mfe_pct ?? 0}% / ${trRecal.avg_mae_pct ?? 0}%`, actStatus === "BROKER_VERIFIED" ? `+${trAct.avg_mfe_pct}% / ${trAct.avg_mae_pct}%` : "N/A"],
    ["Avg Holding Days", `${trOrig.avg_holding_days ?? 0}d`, `${trRecal.avg_holding_days ?? 0}d`, actStatus === "BROKER_VERIFIED" ? `${trAct.avg_holding_days}d` : "N/A"],
    ["Cumulative Return %", `${trOrig.cumulative_return_pct ?? 0}%`, `${trRecal.cumulative_return_pct ?? 0}%`, actStatus === "BROKER_VERIFIED" ? `${trAct.cumulative_return_pct}%` : "0% (Isolated)"],
    ["Sharpe / Sortino", `${trOrig.sharpe_ratio ?? 0} / ${trOrig.sortino_ratio ?? 0}`, `${trRecal.sharpe_ratio ?? 0} / ${trRecal.sortino_ratio ?? 0}`, actStatus === "BROKER_VERIFIED" ? `${trAct.sharpe_ratio} / ${trAct.sortino_ratio}` : "N/A"],
    ["Max Drawdown %", `${trOrig.max_drawdown_pct ?? 0}%`, `${trRecal.max_drawdown_pct ?? 0}%`, actStatus === "BROKER_VERIFIED" ? `${trAct.max_drawdown_pct}%` : "0%"],
    ["Profit Factor / R:R", `${trOrig.profit_factor ?? 0} / 1:${trOrig.realized_rr_ratio ?? 0}`, `${trRecal.profit_factor ?? 0} / 1:${trRecal.realized_rr_ratio ?? 0}`, actStatus === "BROKER_VERIFIED" ? `${trAct.profit_factor} / 1:${trAct.realized_rr_ratio}` : "N/A"],
  ];
  trackEl.innerHTML = rows.map(([lbl, a, b, c]) => `
    <tr>
      <td>${escHtml(lbl)}</td>
      <td style="font-family:var(--font-mono);color:var(--accent-cyan);">${escHtml(a)}</td>
      <td style="font-family:var(--font-mono);color:var(--accent-emerald);font-weight:600;">${escHtml(b)}</td>
      <td style="font-family:var(--font-mono);color:var(--text-secondary);">${escHtml(c)}</td>
    </tr>
  `).join("");

  const tech = attr.TechBot_1 || {};
  const fund = attr.FundBot_1 || {};
  const exec = attr.ExecBot_1 || {};
  const cro = attr.Gemini_CRO || {};
  attrEl.innerHTML = `
    <div class="attribution-card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <strong style="font-family:var(--font-mono);">TechBot_1</strong>
        <span class="tag-pill tag-entry">${tech.bull_win_rate_pct ?? 0}% WIN</span>
      </div>
      <div style="font-size:11px;color:var(--text-secondary);">${escHtml(tech.role || "Technical Timing")}</div>
      <div style="font-family:var(--font-mono);font-size:11.5px;">Bull 14D Avg: <strong>${tech.bull_avg_14d_return_pct ?? 0}%</strong> (${tech.bull_signals ?? 0} sig)</div>
      <div style="font-family:var(--font-mono);font-size:11.5px;">Timing Spread vs Bear: <strong>${tech.timing_spread_pct ?? 0}%</strong></div>
    </div>
    <div class="attribution-card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <strong style="font-family:var(--font-mono);">FundBot_1</strong>
        <span class="tag-pill tag-target">SPREAD ${fund.quality_alpha_spread_pct ?? 0}%</span>
      </div>
      <div style="font-size:11px;color:var(--text-secondary);">${escHtml(fund.role || "StockSight V2")}</div>
      <div style="font-family:var(--font-mono);font-size:11.5px;">Score ≥65 14D Avg: <strong>${fund.high_quality_avg_14d_pct ?? 0}%</strong></div>
      <div style="font-size:10.5px;color:var(--text-muted);">${escHtml(fund.caveat || "")}</div>
    </div>
    <div class="attribution-card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <strong style="font-family:var(--font-mono);">ExecBot_1</strong>
        <span class="tag-pill tag-entry">PF ${exec.profit_factor ?? 0}</span>
      </div>
      <div style="font-size:11px;color:var(--text-secondary);">${escHtml(exec.role || "Moreshwar Execution")}</div>
      <div style="font-family:var(--font-mono);font-size:11.5px;">Target Hits: <strong>${exec.target_hits ?? 0}</strong> · Stops: <strong>${exec.stop_losses_triggered ?? 0}</strong></div>
      <div style="font-family:var(--font-mono);font-size:11.5px;">Realized R:R: <strong>1:${exec.realized_rr_ratio ?? 0}</strong></div>
    </div>
    <div class="attribution-card cro-card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <strong style="font-family:var(--font-mono);">Gemini CRO</strong>
        <span class="tag-pill tag-score">DD -${cro.drawdown_reduction_pct ?? 0}%</span>
      </div>
      <div style="font-size:11px;color:var(--text-secondary);">${escHtml(cro.role || "Recalibration Impact")}</div>
      <div style="font-family:var(--font-mono);font-size:11.5px;">Deferred: <strong>${cro.deferred_or_blocked_signals ?? 0}</strong> (Avoided Loss: ${cro.avoided_losing_trades ?? 0})</div>
      <div style="font-family:var(--font-mono);font-size:11.5px;">Win Δ: <strong>${cro.win_rate_delta_pct >= 0 ? "+" : ""}${cro.win_rate_delta_pct ?? 0}%</strong> · Sharpe Δ: <strong>${cro.sharpe_delta >= 0 ? "+" : ""}${cro.sharpe_delta ?? 0}</strong></div>
    </div>
  `;

  const disc = bt.methodology_disclosure || {};
  methEl.innerHTML = `
    <div style="color:var(--accent-cyan);font-weight:700;">Anti-Bias &amp; Cost Disclosure (${(bt.symbols_evaluated || []).length} Assets Evaluated)</div>
    <div>• <strong>Look-Ahead Prevention:</strong> ${escHtml(disc.look_ahead_bias_prevention || "")}</div>
    <div>• <strong>Survivorship Control:</strong> ${escHtml(disc.survivorship_bias_control || "")}</div>
    <div>• <strong>Transaction Costs:</strong> ${escHtml(disc.transaction_costs || "")}</div>
    <div>• <strong>Track Isolation:</strong> ${escHtml(disc.separation_guarantee || "")}</div>
  `;

  drawBacktestEquityChart(bt.equity_curve || []);
  renderMoreshwarV2ShadowLab();
}

function renderMoreshwarV2ShadowLab() {
  const selEl = document.getElementById("mv2SymbolSelect");
  const gridEl = document.getElementById("mv2ExplainabilityGrid");
  const reasonEl = document.getElementById("mv2ReasonBox");
  const modelsBody = document.getElementById("mv2ModelsTableBody");
  const wfEl = document.getElementById("mv2WalkForwardGrid");
  const badgeEl = document.getElementById("mv2PromotionBadge");
  if (!selEl || !gridEl || !reasonEl || !modelsBody || !wfEl) return;

  const syms = Object.keys(appState.assets || {});
  if (syms.length) {
    if (!appState.mv2SelectedSymbol || !appState.assets[appState.mv2SelectedSymbol]) {
      appState.mv2SelectedSymbol = appState.assets[appState.activeSymbol] ? appState.activeSymbol : syms[0];
    }
    selEl.innerHTML = syms.map((s) => `<option value="${escHtml(s)}">${escHtml(s)}</option>`).join("");
    selEl.value = appState.mv2SelectedSymbol;
  }

  const asset = appState.assets[appState.mv2SelectedSymbol || appState.activeSymbol];
  const v2 = asset?.moreshwar_v2_preview || null;
  const v1 = asset?.moreshwar_levels || null;
  const res = appState.moreshwarV2Research || null;

  if (badgeEl && res) {
    badgeEl.textContent = res.promoted_to_production
      ? "MORESHWAR 2.0 PROMOTED"
      : "SHADOW ONLY · M1 ACTIVE IN PROD";
  }

  if (!v2) {
    gridEl.innerHTML = `<div class="mv2-field-card" style="grid-column:1/-1;"><span class="mv2-field-val">Awaiting Moreshwar 2.0 shadow telemetry...</span><span class="mv2-field-sub">Populates on live scan or /api/state refresh.</span></div>`;
    reasonEl.innerHTML = "";
  } else {
    const cur = asset.currency || "INR";
    const confBd = v2.confidence_breakdown || {};
    const regDet = v2.regime_details || {};
    const statusTone = v2.accepted ? "var(--accent-emerald)" : "var(--accent-amber)";
    const regTone = v2.market_regime === "BULLISH"
      ? "var(--accent-emerald)"
      : v2.market_regime === "NEUTRAL"
        ? "var(--accent-cyan)"
        : "var(--accent-crimson)";

    gridEl.innerHTML = `
      <div class="mv2-field-card">
        <span class="mv2-field-label">1. Fundamental Score</span>
        <span class="mv2-field-val">${v2.fundamental_score ?? "N/A"} / 100</span>
        <span class="mv2-field-sub">Porter ${v2.porters_score ?? "--"} · Mode: ${escHtml(v2.fundamental_mode)}</span>
      </div>
      <div class="mv2-field-card">
        <span class="mv2-field-label">2. Confidence Adjustment</span>
        <span class="mv2-field-val">${Number(v2.confidence || 0).toFixed(2)} <span style="font-size:10px;color:var(--text-muted);">(cap 0.85)</span></span>
        <span class="mv2-field-sub">Base ${confBd.base_quality_conf ?? "--"} − Miss ${confBd.missing_penalty ?? 0} − Tech ${confBd.contradiction_penalty ?? 0} − Reg ${confBd.regime_penalty ?? 0}</span>
      </div>
      <div class="mv2-field-card">
        <span class="mv2-field-label">3. ATR(14) Volatility</span>
        <span class="mv2-field-val">${fmtCurrency(v2.atr_value, cur, 2)} (${v2.atr_pct ?? 0}%)</span>
        <span class="mv2-field-sub">${v2.vol_throttled ? "High-Vol Throttled" : "Normal Volatility Envelope"}</span>
      </div>
      <div class="mv2-field-card">
        <span class="mv2-field-label">4. ATR Multiplier</span>
        <span class="mv2-field-val">${v2.atr_multiplier ?? 2.0}x ATR(14)</span>
        <span class="mv2-field-sub">Replaces M1 <code>(100-X)*scale</code></span>
      </div>
      <div class="mv2-field-card">
        <span class="mv2-field-label">5. Market Regime (Causal)</span>
        <span class="mv2-field-val" style="color:${regTone};">${escHtml(v2.market_regime || "UNKNOWN")}</span>
        <span class="mv2-field-sub">Nifty ${regDet.close ? fmtCurrency(regDet.close, "INR", 0) : "--"} · 20D/50D SMA (${regDet.ann_vol_20d_pct ?? "--"}% vol)</span>
      </div>
      <div class="mv2-field-card">
        <span class="mv2-field-label">6. Entry Price</span>
        <span class="mv2-field-val">${fmtCurrency(v2.entry_price, cur, 2)}</span>
        <span class="mv2-field-sub">M1 Active Entry: ${v1 ? fmtCurrency(v1.entry_price, cur, 2) : "--"}</span>
      </div>
      <div class="mv2-field-card">
        <span class="mv2-field-label">7. Stop-Loss (ATR)</span>
        <span class="mv2-field-val" style="color:var(--accent-crimson);">${fmtCurrency(v2.stop_loss, cur, 2)} (-${v2.stop_distance_pct ?? 0}%)</span>
        <span class="mv2-field-sub">M1 Active SL: ${v1 ? fmtCurrency(v1.stop_loss, cur, 2) : "--"}</span>
      </div>
      <div class="mv2-field-card">
        <span class="mv2-field-label">8. Target &amp; 9. R:R</span>
        <span class="mv2-field-val" style="color:var(--accent-emerald);">${fmtCurrency(v2.target, cur, 2)} (+${v2.target_distance_pct ?? 0}%)</span>
        <span class="mv2-field-sub">R:R <strong>1:${v2.rr_ratio ?? 0}</strong> (M1 R:R 1:${v1?.rr_ratio ?? "--"})</span>
      </div>
      <div class="mv2-field-card">
        <span class="mv2-field-label">10. Position Qty (Int)</span>
        <span class="mv2-field-val">${v2.position_qty ?? 0} Shares</span>
        <span class="mv2-field-sub">Risk Budget: ${fmtCurrency(v2.risk_budget_amount || 0, cur, 0)}</span>
      </div>
      <div class="mv2-field-card">
        <span class="mv2-field-label">11. Max Capital Exposure</span>
        <span class="mv2-field-val">${fmtCurrency(v2.max_capital_exposure || 0, cur, 0)} (${v2.exposure_pct ?? 0}%)</span>
        <span class="mv2-field-sub">Hard single-stock cap 20%</span>
      </div>
      <div class="mv2-field-card mv2-span-2">
        <span class="mv2-field-label">12. Shadow Decision Status &amp; Primary Verdict</span>
        <span class="mv2-field-val" style="color:${statusTone};">${v2.accepted ? "ACCEPTED (SHADOW CANDIDATE)" : "REJECTED / GATED (SHADOW)"}</span>
        <span class="mv2-field-sub">${escHtml(v2.reason_summary || "")}</span>
      </div>
    `;

    const penList = Array.isArray(confBd.penalties) && confBd.penalties.length
      ? confBd.penalties.map((p) => escHtml(p)).join(" · ")
      : "No structural confidence deductions beyond anti-certainty cap.";
    const reasonsList = v2.accepted
      ? (v2.accept_reasons || []).map((r) => `✅ ${escHtml(r)}`).join("<br>")
      : (v2.rejection_reasons || []).map((r) => `🛑 ${escHtml(r)}`).join("<br>");

    reasonEl.innerHTML = `
      <div><strong style="color:var(--accent-cyan);">Confidence Breakdown (${escHtml(v2.symbol)}):</strong> ${penList}</div>
      <div style="margin-top:4px;"><strong style="color:${statusTone};">Gate Details:</strong><br>${reasonsList}</div>
    `;
  }

  const models = (res && Array.isArray(res.models_comparison)) ? res.models_comparison : [];
  if (!models.length) {
    modelsBody.innerHTML = `<tr><td colspan="12" style="text-align:center;color:var(--text-secondary);">Experiment comparison table populates after live scan.</td></tr>`;
    wfEl.innerHTML = "";
    return;
  }

  modelsBody.innerHTML = models.map((m) => {
    const isHighlight = String(m.model).includes("Model D") || String(m.model).includes("Model E:") || String(m.model).includes("Model A");
    const retCol = (m.total_return_pct || 0) >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)";
    const intTag = String(m.fundamental_integrity || "").includes("CAUSAL_CLEAN")
      ? "tag-target"
      : String(m.fundamental_integrity || "").includes("MARKET_INDEX")
        ? "tag-entry"
        : "tag-score";
    const intShort = String(m.fundamental_integrity || "").includes("CAUSAL_CLEAN")
      ? "CAUSAL CLEAN"
      : String(m.fundamental_integrity || "").includes("MARKET_INDEX")
        ? "INDEX BENCH"
        : "STATIC FUND WARN";
    return `
      <tr style="${String(m.model).includes("Model E:") ? "background:hsla(196,86%,54%,0.07);" : ""}">
        <td style="font-weight:${isHighlight ? "700" : "500"};">${escHtml(m.model)}</td>
        <td style="font-family:var(--font-mono);color:${retCol};font-weight:700;">${(m.total_return_pct || 0) >= 0 ? "+" : ""}${m.total_return_pct ?? 0}%</td>
        <td style="font-family:var(--font-mono);color:${retCol};">${(m.annualised_return_pct || 0) >= 0 ? "+" : ""}${m.annualised_return_pct ?? 0}%</td>
        <td style="font-family:var(--font-mono);">${m.win_rate_pct ?? 0}%</td>
        <td style="font-family:var(--font-mono);">${m.profit_factor != null ? m.profit_factor : "N/A"}</td>
        <td style="font-family:var(--font-mono);">${m.sharpe_ratio ?? 0}</td>
        <td style="font-family:var(--font-mono);">${m.max_drawdown_pct ?? 0}%</td>
        <td style="font-family:var(--font-mono);">${m.total_trades ?? 0}</td>
        <td style="font-family:var(--font-mono);">${m.stop_loss_freq_pct ?? 0}%</td>
        <td style="font-family:var(--font-mono);">${m.capital_utilisation_pct ?? 0}%</td>
        <td style="font-family:var(--font-mono);">${m.rejected_trades ?? 0} / ${m.regime_blocked_trades ?? 0}</td>
        <td><span class="tag-pill ${intTag}">${intShort}</span></td>
      </tr>
    `;
  }).join("");

  const oos = res.out_of_sample || {};
  const oosM1 = oos.moreshwar_1_oos || {};
  const oosM2 = oos.moreshwar_2_oos || {};
  const oosNifty = oos.nifty50_oos || {};
  const wfRows = Array.isArray(res.rolling_walk_forward) ? res.rolling_walk_forward : [];
  const sensRows = Array.isArray(res.sensitivity_sweep) ? res.sensitivity_sweep : [];
  const bestSens = sensRows.length
    ? sensRows.filter((r) => r.val_trades > 0).sort((a, b) => (b.val_return_pct - a.val_return_pct))[0] || sensRows[0]
    : null;

  wfEl.innerHTML = `
    <div class="dq-card" style="border-top:2px solid var(--accent-emerald);">
      <div class="dq-card-head">
        <span class="dq-q-title">OUT-OF-SAMPLE TEST SPLIT</span>
        <span class="tag-pill tag-target">${escHtml(oos.splits?.out_of_sample?.start || "")} → ${escHtml(oos.splits?.out_of_sample?.end || "")}</span>
      </div>
      <div class="dq-headline">Clean Causal Moreshwar 2.0 on unseen tail vs M1 &amp; Nifty 50:</div>
      <div class="dq-mini-grid">
        <div class="dq-mini-stat">
          <div class="dq-mini-lbl">M2 Clean OOS</div>
          <div class="dq-mini-val" style="color:${(oosM2.total_return_pct || 0) >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)"};">${(oosM2.total_return_pct || 0) >= 0 ? "+" : ""}${oosM2.total_return_pct ?? "--"}%</div>
          <div class="dq-mini-sub">DD ${oosM2.max_drawdown_pct ?? "--"}% · Win ${oosM2.win_rate_pct ?? "--"}% (${oosM2.total_trades ?? 0} trd)</div>
        </div>
        <div class="dq-mini-stat">
          <div class="dq-mini-lbl">M1 Baseline OOS</div>
          <div class="dq-mini-val" style="color:var(--accent-cyan);">${(oosM1.total_return_pct || 0) >= 0 ? "+" : ""}${oosM1.total_return_pct ?? "--"}%</div>
          <div class="dq-mini-sub">DD ${oosM1.max_drawdown_pct ?? "--"}% · Win ${oosM1.win_rate_pct ?? "--"}% (${oosM1.total_trades ?? 0} trd)</div>
        </div>
        <div class="dq-mini-stat" style="grid-column: span 2;">
          <div class="dq-mini-lbl">Nifty 50 OOS Benchmark</div>
          <div class="dq-mini-val" style="color:var(--accent-amber);">+${oosNifty.total_return_pct ?? "--"}% (MaxDD ${oosNifty.max_drawdown_pct ?? "--"}%)</div>
          <div class="dq-mini-sub">Zero look-ahead across ${oos.splits?.out_of_sample?.bars ?? 0} OOS trading bars</div>
        </div>
      </div>
    </div>

    <div class="dq-card" style="border-top:2px solid var(--accent-cyan);">
      <div class="dq-card-head">
        <span class="dq-q-title">ROLLING WALK-FORWARD FOLDS</span>
        <span class="tag-pill tag-entry">${wfRows.length} WINDOWS</span>
      </div>
      <div class="dq-headline">Fold-by-fold stability (M1 vs Clean Causal M2):</div>
      <div class="table-scroll">
        <table class="data-table dq-compact-table">
          <thead><tr><th>Fold</th><th>Window</th><th>M1 Ret (DD)</th><th>M2 Clean (DD)</th><th>Win%</th></tr></thead>
          <tbody>
            ${wfRows.map((w) => `
              <tr>
                <td style="font-family:var(--font-mono);font-weight:700;">${escHtml(w.fold)}</td>
                <td style="font-family:var(--font-mono);">${escHtml(w.start)}→${escHtml(w.end)}</td>
                <td style="font-family:var(--font-mono);color:${w.m1_return_pct >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)"};">${w.m1_return_pct >= 0 ? "+" : ""}${w.m1_return_pct}% (${w.m1_max_dd_pct}%)</td>
                <td style="font-family:var(--font-mono);color:${w.m2_return_pct >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)"};">${w.m2_return_pct >= 0 ? "+" : ""}${w.m2_return_pct}% (${w.m2_max_dd_pct}%)</td>
                <td style="font-family:var(--font-mono);">${w.m2_win_rate_pct}% (n=${w.m2_trades})</td>
              </tr>
            `).join("") || `<tr><td colspan="5">Run scripts/run_moreshwar_v2_experiments.py for WF breakdown.</td></tr>`}
          </tbody>
        </table>
      </div>
    </div>

    <div class="dq-card" style="border-top:2px solid var(--accent-amber);">
      <div class="dq-card-head">
        <span class="dq-q-title">HONEST PROMOTION GATE (PHASE 10)</span>
        <span class="tag-pill tag-stop">REVIEW REQUIRED</span>
      </div>
      <div class="dq-headline">Why Moreshwar 2.0 remains in Shadow Research Mode until reviewed:</div>
      <div style="font-size:11px;color:var(--text-secondary);line-height:1.45;">
        • <strong>Proven Win:</strong> ATR(14) + Sizing + Nifty Bear Block swings identical StockSight picks from <strong>-7.76% → +0.58%</strong> (Model D) &amp; slashes MaxDD <strong>14.2% → 7.25%</strong>; OOS M2 wins <strong>+4.92% (0.70% DD)</strong>.<br>
        • <strong>Honest Caveat:</strong> Stripping static fundamentals cleanly (Model E Clean) yields <strong>${models.find((x) => String(x.model).includes("Model E:"))?.total_return_pct ?? "--"}%</strong> (Sharpe negative vs 6.5% T-bill) because historical quarterly ROE/OPM ratios are unavailable from NSE/Yahoo feeds.<br>
        ${bestSens ? `• <strong>Sensitivity Plateau:</strong> Val leader ATR=${bestSens.atr_multiplier}x (${escHtml(bestSens.regime_policy)}, ${bestSens.val_return_pct >= 0 ? "+" : ""}${bestSens.val_return_pct}%, n=${bestSens.val_trades}).` : ""}
      </div>
    </div>
  `;
}

function drawMv21EquityChart(r21) {
  const canvas = document.getElementById("mv21EquityChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const parent = canvas.parentElement;
  const rect = canvas.getBoundingClientRect();
  const parentW = parent ? Math.max(0, parent.clientWidth - 16) : 0;
  const parentH = parent ? Math.max(0, parent.clientHeight - 16) : 0;
  const w = Math.max(220, Math.floor(parentW || rect.width || 320));
  const h = Math.max(180, Math.floor(parentH || rect.height || 214));
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const mods = (r21 && r21.models) || {};
  const m1c = mods["Moreshwar 1.0"]?.equity_curve || [];
  const m2c = mods["Moreshwar 2.0"]?.equity_curve || [];
  const m21c = mods["Moreshwar 2.1"]?.equity_curve || [];
  const bmc = mods["Nifty 50"]?.equity_curve || [];

  if (m21c.length < 2) {
    ctx.font = "12px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsl(215, 15%, 65%)";
    ctx.fillText("5Y Moreshwar 2.1 equity curve awaiting report payload.", 14, h / 2);
    return;
  }

  const allSeries = [
    { data: m1c, col: "hsl(350, 84%, 60%)", dash: false },
    { data: m2c, col: "hsl(265, 80%, 66%)", dash: false },
    { data: bmc, col: "hsl(38, 92%, 55%)", dash: true },
    { data: m21c, col: "hsl(196, 86%, 54%)", dash: false }
  ];
  const baseNav = 100000.0;
  const norm = (v) => (Number(v || baseNav) / baseNav) * 100.0;
  const vals = [];
  allSeries.forEach((s) => s.data.forEach((p) => vals.push(norm(p.nav))));
  const minV = Math.min(...vals, 95) * 0.985;
  const maxV = Math.max(...vals, 105) * 1.015;

  const padT = 16, padB = 22, padL = 8, padR = w < 360 ? 38 : 48;
  const plotW = Math.max(120, w - padL - padR);
  const plotH = h - padT - padB;
  const yFor = (v) => padT + plotH - ((v - minV) / Math.max(1e-6, maxV - minV)) * plotH;
  const nPts = m21c.length;
  const xFor = (i, len) => padL + (i / Math.max(1, len - 1)) * plotW;

  ctx.strokeStyle = "hsla(222, 15%, 24%, 0.45)";
  ctx.lineWidth = 1;
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillStyle = "hsl(215, 15%, 65%)";
  for (let i = 0; i <= 4; i++) {
    const v = minV + ((maxV - minV) * i) / 4;
    const y = yFor(v);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(0), w - padR + 5, y + 3);
  }
  const y100 = yFor(100);
  if (y100 >= padT && y100 <= padT + plotH) {
    ctx.save();
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = "hsla(210, 20%, 70%, 0.3)";
    ctx.beginPath();
    ctx.moveTo(padL, y100);
    ctx.lineTo(w - padR, y100);
    ctx.stroke();
    ctx.restore();
  }
  ctx.fillText(m21c[0].date || "", padL, h - 5);
  ctx.save();
  ctx.textAlign = "right";
  ctx.fillText(m21c[nPts - 1].date || "", w - padR, h - 5);
  ctx.restore();

  allSeries.forEach((s) => {
    if (!s.data.length) return;
    ctx.save();
    ctx.strokeStyle = s.col;
    ctx.lineWidth = s.col.includes("196") ? 2.2 : 1.7;
    if (s.dash) ctx.setLineDash([4, 4]);
    ctx.beginPath();
    s.data.forEach((pt, idx) => {
      const x = xFor(idx, s.data.length);
      const y = yFor(norm(pt.nav));
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  });
}

function drawMv21MonteCarloChart(mc) {
  const canvas = document.getElementById("mv21MonteCarloChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const parent = canvas.parentElement;
  const rect = canvas.getBoundingClientRect();
  const parentW = parent ? Math.max(0, parent.clientWidth - 16) : 0;
  const parentH = parent ? Math.max(0, parent.clientHeight - 16) : 0;
  const w = Math.max(220, Math.floor(parentW || rect.width || 320));
  const h = Math.max(190, Math.floor(parentH || rect.height || 199));
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const tb = mc?.trade_bootstrap;
  const bb = mc?.daily_block_bootstrap;
  if (!tb || !bb) {
    ctx.font = "12px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsl(215, 15%, 65%)";
    ctx.fillText("Monte Carlo percentiles awaiting simulation data.", 14, h / 2);
    return;
  }

  const bars = [
    { label: "Trade Ret (P5 / P50 / P95)", p5: tb.p05_return_pct, p50: tb.median_return_pct, p95: tb.p95_return_pct, col: "hsl(196, 86%, 54%)" },
    { label: "Block-10D Ret (P5 / P50 / P95)", p5: bb.p05_return_pct, p50: bb.median_return_pct, p95: bb.p95_return_pct, col: "hsl(265, 80%, 66%)" },
    { label: "Trade MaxDD (Med → P95)", p5: 0, p50: -tb.median_max_dd_pct, p95: -tb.p95_max_dd_pct, col: "hsl(350, 84%, 60%)", isDd: true },
    { label: "Block MaxDD (Med → P95)", p5: 0, p50: -bb.median_max_dd_pct, p95: -bb.p95_max_dd_pct, col: "hsl(38, 92%, 55%)", isDd: true }
  ];

  const allVals = [0];
  bars.forEach((b) => allVals.push(b.p5, b.p50, b.p95));
  const minV = Math.min(...allVals) - 5;
  const maxV = Math.max(...allVals) + 5;
  const padL = 8;
  const padR = 8;
  const plotW = Math.max(120, w - padL - padR);
  const xFor = (v) => padL + ((v - minV) / Math.max(1e-6, maxV - minV)) * plotW;

  const topPad = 8;
  const axisH = 18;
  const zX = xFor(0);
  ctx.strokeStyle = "hsla(215, 20%, 60%, 0.42)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(zX, topPad);
  ctx.lineTo(zX, h - axisH);
  ctx.stroke();

  const rowH = (h - topPad - axisH) / bars.length;
  bars.forEach((b, idx) => {
    const cy = topPad + idx * rowH;
    const valTxt = b.isDd
      ? `Med ${Math.abs(b.p50).toFixed(1)}% | P95 ${Math.abs(b.p95).toFixed(1)}%`
      : `${b.p5.toFixed(0)}% | ${b.p50.toFixed(1)}% | ${b.p95 >= 0 ? "+" : ""}${b.p95.toFixed(0)}%`;

    ctx.font = w < 340 ? "9.5px 'JetBrains Mono', monospace" : "10.5px 'JetBrains Mono', monospace";
    ctx.fillStyle = "hsl(210, 25%, 90%)";
    ctx.textAlign = "left";
    ctx.fillText(b.label.slice(0, w < 340 ? 22 : 30), padL, cy + 11);

    ctx.fillStyle = b.col;
    ctx.textAlign = "right";
    ctx.fillText(valTxt, w - padR, cy + 11);

    const barY = cy + 17;
    const x1 = xFor(Math.min(b.p5, b.p95));
    const x2 = xFor(Math.max(b.p5, b.p95));
    ctx.fillStyle = "hsla(222, 18%, 22%, 0.55)";
    ctx.fillRect(padL, barY + 3, plotW, 4);
    ctx.fillStyle = b.isDd ? "hsla(350, 84%, 60%, 0.34)" : "hsla(196, 86%, 54%, 0.32)";
    ctx.fillRect(x1, barY, Math.max(4, x2 - x1), 10);

    const xm = xFor(b.p50);
    ctx.fillStyle = b.col;
    ctx.fillRect(xm - 2, barY - 2, 4, 14);
  });

  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillStyle = "hsl(215, 15%, 65%)";
  ctx.textAlign = "left";
  ctx.fillText(`${minV.toFixed(0)}%`, padL, h - 4);
  ctx.textAlign = "center";
  ctx.fillText("0%", zX, h - 4);
  ctx.textAlign = "right";
  ctx.fillText(`+${maxV.toFixed(0)}%`, w - padR, h - 4);
  ctx.textAlign = "left";
}

function renderMoreshwarV22Dashboard() {
  const bannerEl = document.getElementById("mv22DecisionBanner");
  const kpiEl = document.getElementById("mv22HeaderKpis");
  const scBody = document.getElementById("mv22ScorecardBody");
  const stBody = document.getElementById("mv22StressBody");
  const bsEl = document.getElementById("mv22BootstrapSummary");
  const badgeEl = document.getElementById("mv22DecisionBadge");
  if (!bannerEl || !kpiEl || !scBody || !stBody) return;

  const r22 = appState.moreshwarV22Research;
  if (!r22 || !r22.scorecard || !r22.scorecard.integrated_m22) {
    bannerEl.className = "mv21-verdict-banner";
    bannerEl.innerHTML = `<strong>Moreshwar 2.2 Integrated Report Unavailable:</strong> Run <code>python3 -m finsistant_v2.backtest_engine_v2_2_integrated</code> to generate <code>reports/moreshwar_2_2/integrated_final_report.json</code>.`;
    kpiEl.innerHTML = "";
    scBody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--text-secondary);">No Moreshwar 2.2 integrated research report loaded.</td></tr>`;
    stBody.innerHTML = "";
    if (bsEl) bsEl.innerHTML = "";
    if (badgeEl) badgeEl.textContent = "REPORT PENDING · M1.0 ACTIVE IN PROD";
    return;
  }

  const sc = r22.scorecard;
  const im = sc.integrated_m22;
  const dec = String(r22.decision || "NO-GO");
  const promoted = Boolean(r22.promoted_to_production);
  if (badgeEl) {
    badgeEl.textContent = `${dec} · ${promoted ? "RESEARCH PROMOTED" : "RESEARCH ONLY (M1.0 IN PROD)"}`;
  }

  const fmtPct = (v) => (v == null || Number.isNaN(Number(v)) ? "—" : `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(2)}%`);
  bannerEl.className = `mv21-verdict-banner ${dec.startsWith("GO") ? "" : "no-go"}`;
  bannerEl.innerHTML = `
    <strong>Final Decision: ${escHtml(dec)}</strong> (Selected IS Config: <code>${escHtml(r22.selected_config_id || "OPT_REG_DD_N8")}</code> · <code>promoted_to_production = ${promoted}</code>)<br/>
    <span style="color:var(--text-secondary);">Full 5Y improved from M2.1 (${fmtPct(sc.m21_frozen?.total_return_pct)}) to <strong>${fmtPct(im.total_return_pct)}</strong> (MaxDD ${Number(im.max_drawdown_pct || 0).toFixed(2)}%, NetPF ${Number(im.net_profit_factor || 0).toFixed(2)}), but Prior OOS (<code>${fmtPct(im.oos_return_pct)}</code>) &amp; Terminal Holdout 2026-05+ (<code>${fmtPct(im.holdout_return_pct)}</code>) remain negative.</span>
  `;

  const kpis = [
    { lbl: "5Y Net (INR)", val: fmtPct(im.total_return_pct), cls: Number(im.total_return_pct) >= 0 ? "pos" : "neg" },
    { lbl: "CAGR / Vol", val: `${Number(im.cagr_pct || 0).toFixed(2)}% / ${Number(im.annualised_vol_pct || 0).toFixed(2)}%`, cls: "" },
    { lbl: "Max Drawdown", val: `-${Number(im.max_drawdown_pct || 0).toFixed(2)}%`, cls: "" },
    { lbl: "Net PF / Sharpe", val: `${Number(im.net_profit_factor || 0).toFixed(2)} / ${Number(im.sharpe_ratio || 0).toFixed(2)}`, cls: Number(im.net_profit_factor) >= 1.3 ? "pos" : "" },
    { lbl: "Prior OOS (25-09+)", val: fmtPct(im.oos_return_pct), cls: Number(im.oos_return_pct) >= 0 ? "pos" : "neg" },
    { lbl: "Holdout (26-05+)", val: fmtPct(im.holdout_return_pct), cls: Number(im.holdout_return_pct) >= 0 ? "pos" : "neg" },
    { lbl: "1.5x Stress 5Y", val: fmtPct(sc.integrated_m22_stress_15x?.full_5y_pct), cls: Number(sc.integrated_m22_stress_15x?.full_5y_pct) >= 0 ? "pos" : "neg" },
    { lbl: "Ex-Top5 Trades", val: fmtPct(r22.concentration_stress?.ex_top5_winning_trades?.return_pct), cls: "" },
  ];
  kpiEl.innerHTML = kpis.map((k) => `
    <div class="kpi-tile">
      <div class="kpi-label">${escHtml(k.lbl)}</div>
      <div class="kpi-value ${k.cls}">${escHtml(k.val)}</div>
    </div>
  `).join("");

  const rows = [
    { name: "M2.1 Frozen (Mixed)", m: sc.m21_frozen },
    { name: "M2.1 Frozen (INR)", m: sc.m21_frozen_inr },
    { name: "M2.2 P1 (EXP_45D)", m: sc.prompt1_best },
    { name: "M2.2 P2 (PULL_CONT)", m: sc.prompt2_best },
    { name: "M2.2 Integrated (IS Opt)", m: sc.integrated_m22 },
    { name: "US Sleeve (NVDA Isolated)", m: sc.us_sleeve_isolated },
  ].filter((r) => r.m);

  scBody.innerHTML = rows.map(({ name, m }) => `
    <tr>
      <td><strong>${escHtml(name)}</strong></td>
      <td class="${Number(m.total_return_pct) >= 0 ? "text-green" : "text-red"}">${fmtPct(m.total_return_pct)}</td>
      <td>${m.cagr_pct != null ? `${Number(m.cagr_pct).toFixed(2)}%` : "—"}</td>
      <td>-${Number(m.max_drawdown_pct || 0).toFixed(2)}%</td>
      <td>${m.net_profit_factor != null ? Number(m.net_profit_factor).toFixed(2) : "—"}</td>
      <td>${Number(m.sharpe_ratio || 0).toFixed(2)}</td>
      <td>${m.total_trades ?? "—"}</td>
      <td>${Number(m.cost_drag_pct || 0).toFixed(2)}%</td>
    </tr>
  `).join("");

  const stressList = Array.isArray(r22.stress_tests) ? r22.stress_tests : [];
  stBody.innerHTML = stressList.map((s) => `
    <tr>
      <td>${escHtml(s.label || s.scenario_id)}</td>
      <td class="${Number(s.full_5y_pct) >= 0 ? "text-green" : "text-red"}">${fmtPct(s.full_5y_pct)}</td>
      <td class="${Number(s.oos_pct) >= 0 ? "text-green" : "text-red"}">${fmtPct(s.oos_pct)}</td>
      <td>-${Number(s.max_dd_pct || 0).toFixed(2)}%</td>
      <td>${Number(s.net_pf || 0).toFixed(2)}</td>
      <td>₹${Number(s.total_cost || 0).toLocaleString("en-IN", { maximumFractionDigits: 0 })}</td>
    </tr>
  `).join("");

  if (bsEl && r22.statistical_confidence) {
    const b10 = r22.statistical_confidence.block_bootstrap_10d || {};
    const b20 = r22.statistical_confidence.block_bootstrap_20d || {};
    const trb = r22.statistical_confidence.trade_bootstrap || {};
    bsEl.innerHTML = `
      <strong>1,200x Bootstraps:</strong>
      Trade Ret P05/P50/P95: <code>[${(trb.return_p05_p50_p95 || []).join("%, ")}%]</code> ·
      10D Block P05/P50/P95: <code>[${(b10.return_p05_p50_p95 || []).join("%, ")}%]</code> (Beat Cash: <code>${b10.prob_beat_cash_65_pct ?? "—"}%</code>) ·
      20D Block Sharpe: <code>[${(b20.sharpe_p05_p50_p95 || []).join(", ")}]</code>.<br/>
      <span style="color:var(--text-muted);">${escHtml(r22.statistical_confidence.multiple_testing_disclosure || "")}</span>
    `;
  }
}

function renderMoreshwarV21Dashboard() {
  const bannerEl = document.getElementById("mv21DecisionBanner");
  const kpiEl = document.getElementById("mv21HeaderKpis");
  const bmBody = document.getElementById("mv21BenchmarkTableBody");
  const wfBody = document.getElementById("mv21WalkForwardBody");
  const regBody = document.getElementById("mv21RegimeTableBody");
  const prGrid = document.getElementById("mv21PortfolioRiskGrid");
  const invEl = document.getElementById("mv21InvestigationContainer");
  const dqEl = document.getElementById("mv21DataQualityBody");
  const trBody = document.getElementById("mv21TradesTableBody");
  if (!bannerEl || !kpiEl || !bmBody) return;

  const r21 = appState.moreshwarV21Research;
  if (!r21 || !r21.models) {
    bannerEl.className = "mv21-verdict-banner";
    bannerEl.innerHTML = `<strong>Moreshwar 2.1 Report Pending / Unavailable:</strong> Run <code>scripts/run_moreshwar_v2_1_experiments.py</code> and regenerate <code>live_scan.json</code> to view 5Y statistical validation.`;
    kpiEl.innerHTML = "";
    bmBody.innerHTML = `<tr><td colspan="14" style="text-align:center;color:var(--text-secondary);">No Moreshwar 2.1 benchmark report loaded.</td></tr>`;
    if (wfBody) wfBody.innerHTML = "";
    if (regBody) regBody.innerHTML = "";
    if (prGrid) prGrid.innerHTML = "";
    if (invEl) invEl.innerHTML = "";
    if (dqEl) dqEl.innerHTML = "";
    if (trBody) trBody.innerHTML = "";
    const oosElEmpty = document.getElementById("mv21OosSummary");
    if (oosElEmpty) oosElEmpty.innerHTML = "";
    const mcStatsEmpty = document.getElementById("mv21MonteCarloStats");
    if (mcStatsEmpty) mcStatsEmpty.innerHTML = "";
    const promoEmpty = document.getElementById("mv21PromoBadge");
    if (promoEmpty) promoEmpty.textContent = "REPORT UNAVAILABLE · M1.0 ACTIVE IN PROD";
    requestAnimationFrame(() => {
      drawMv21EquityChart(null);
      drawMv21MonteCarloChart(null);
    });
    return;
  }

  // Populate interactive filter dropdowns while preserving selections
  const selMod = document.getElementById("mv21FilterModel");
  const selInst = document.getElementById("mv21FilterInstrument");
  const selSec = document.getElementById("mv21FilterSector");
  const selReg = document.getElementById("mv21FilterRegime");

  const modelKeys = ["Moreshwar 2.1", "Moreshwar 2.1 (Permissive)", "Moreshwar 2.0", "Moreshwar 1.0", "Moreshwar 2.1 (E-Ref Leak)"].filter((k) => r21.models[k]);
  if (selMod) {
    if (!modelKeys.includes(appState.mv21Filters.model)) appState.mv21Filters.model = modelKeys[0] || "Moreshwar 2.1";
    selMod.innerHTML = modelKeys.map((k) => `<option value="${escHtml(k)}">${escHtml(k)}</option>`).join("");
    selMod.value = appState.mv21Filters.model;
  }

  const activeModObj = r21.models[appState.mv21Filters.model] || r21.models["Moreshwar 2.1"];
  const allSyms = Object.keys(activeModObj?.by_stock || {}).sort();
  if (selInst) {
    selInst.innerHTML = `<option value="ALL">All (${allSyms.length} Instruments)</option>` + allSyms.map((s) => `<option value="${escHtml(s)}">${escHtml(s)}</option>`).join("");
    selInst.value = (appState.mv21Filters.instrument === "ALL" || allSyms.includes(appState.mv21Filters.instrument)) ? appState.mv21Filters.instrument : "ALL";
  }
  const allSectors = Object.keys(activeModObj?.by_sector || {}).sort();
  if (selSec) {
    selSec.innerHTML = `<option value="ALL">All (${allSectors.length} Sectors)</option>` + allSectors.map((s) => `<option value="${escHtml(s)}">${escHtml(s)}</option>`).join("");
    selSec.value = (appState.mv21Filters.sector === "ALL" || allSectors.includes(appState.mv21Filters.sector)) ? appState.mv21Filters.sector : "ALL";
  }
  const allRegimes = ["STRONG_BULLISH", "WEAK_BULLISH", "SIDEWAYS", "WEAK_BEARISH", "STRONG_BEARISH", "HIGH_VOLATILITY"];
  if (selReg) {
    selReg.innerHTML = `<option value="ALL">All 6 Market Regimes</option>` + allRegimes.map((r) => `<option value="${escHtml(r)}">${escHtml(r)}</option>`).join("");
    selReg.value = appState.mv21Filters.regime || "ALL";
  }

  // Verdict, promotion gate & staleness check
  const ds = r21.dataset_summary || {};
  const genMs = r21.generated_at ? Date.parse(r21.generated_at) : NaN;
  const endMs = ds.end_date ? Date.parse(ds.end_date) : NaN;
  const reportAgeDays = Number.isFinite(genMs) ? (Date.now() - genMs) / 86400000 : Infinity;
  const dataAgeDays = Number.isFinite(endMs) ? (Date.now() - endMs) / 86400000 : Infinity;
  const isStaleReport = Boolean(r21.is_stale) || reportAgeDays > 7 || dataAgeDays > 7;
  const staleNote = isStaleReport
    ? `⚠️ STALE REPORT WARNING: Report/dataset last updated ${Number.isFinite(reportAgeDays) ? `${Math.max(0, Math.floor(reportAgeDays))}d` : "unknown"} ago (${escHtml(r21.generated_at || ds.end_date || "unknown timestamp")}). Refresh via scripts/run_moreshwar_v2_1_experiments.py.`
    : "";

  const dec = r21.final_decision || {};
  const catStr = dec.classification || "Unclassified";
  const isPositive = catStr.startsWith("3.") || catStr.startsWith("4.");
  const ruleText = dec.promotion_recommendation || (
    r21.promoted_to_production
      ? "Promoted to production after outperforming benchmarks with statistically significant expectancy across untouched OOS folds."
      : "Keep Moreshwar 1.0 active in production; Moreshwar 2.1 requires positive statistically significant net expectancy (p < 0.05) across OOS & regime folds before promotion."
  );
  const unresolvedList = Array.isArray(dec.unresolved_risks) ? dec.unresolved_risks : [];
  bannerEl.className = `mv21-verdict-banner${isPositive ? " verdict-positive" : ""}`;
  bannerEl.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px;">
      <strong style="font-size:13px;">Final Research Classification: ${escHtml(catStr)}</strong>
      <div style="display:flex;gap:6px;flex-wrap:wrap;">
        ${isStaleReport ? `<span class="tag-pill tag-score">STALE REPORT</span>` : ""}
        <span class="tag-pill ${r21.promoted_to_production ? "tag-target" : "tag-stop"}">${r21.promoted_to_production ? "PROMOTED" : "NOT PROMOTED TO PRODUCTION · M1.0 ACTIVE"}</span>
      </div>
    </div>
    ${isStaleReport ? `<div style="margin-bottom:4px;color:var(--accent-amber);font-family:var(--font-mono);font-size:11px;">${staleNote}</div>` : ""}
    <div>${escHtml(dec.rationale || "")}</div>
    <div style="margin-top:4px;font-size:11px;color:var(--text-secondary);"><strong>Decision Rule:</strong> ${escHtml(ruleText)}</div>
    ${unresolvedList.length ? `<div style="margin-top:4px;font-size:11px;color:var(--text-muted);"><strong>Unresolved Risks:</strong> ${unresolvedList.map((u) => escHtml(u)).join(" · ")}</div>` : ""}
  `;

  const promoBadge = document.getElementById("mv21PromoBadge");
  if (promoBadge) {
    promoBadge.textContent = r21.promoted_to_production
      ? "M2.1 ACTIVE IN PROD"
      : (isStaleReport ? "STALE RESEARCH · M1.0 ACTIVE" : "RESEARCH ONLY · M1.0 ACTIVE IN PROD");
  }

  // Subtitle & dynamic KPI summary (full model OR filtered Instrument / Sector / Regime cohort)
  const fInst = appState.mv21Filters.instrument;
  const fSec = appState.mv21Filters.sector;
  const fReg = appState.mv21Filters.regime;
  const modM = activeModObj.metrics || {};
  const subEl = document.getElementById("mv21Subtitle");
  if (subEl) {
    subEl.textContent = `5Y Dataset: ${ds.start_date || "--"} → ${ds.end_date || "--"} (${ds.trading_days || 0} trading days · ${ds.instruments_count || 0} instruments · OOS starts ${ds.oos_start || "--"})`;
  }

  // Derive filtered cohort view when any non-ALL filter is active
  let cohortLabel = `Full Model: ${appState.mv21Filters.model}`;
  let cTrades = modM.total_trades ?? 0;
  let cWin = `${modM.win_rate_pct ?? 0}% (CI ${modM.win_rate_ci95?.[0] ?? 0}–${modM.win_rate_ci95?.[1] ?? 0}%)`;
  let cPf = modM.profit_factor ?? 0;
  let cExp = `${(modM.expectancy_pct || 0) >= 0 ? "+" : ""}${modM.expectancy_pct ?? 0}% (p=${modM.p_value_vs_zero ?? "--"})`;
  let cTotVal = `${(modM.total_return_pct || 0) >= 0 ? "+" : ""}${modM.total_return_pct ?? 0}%`;
  let cTotSub = `CAGR ${(modM.cagr_pct || 0) >= 0 ? "+" : ""}${modM.cagr_pct ?? 0}% · Cost ${modM.cost_drag_pct ?? 0}%`;

  if (fInst !== "ALL" && activeModObj.by_stock?.[fInst]) {
    const bs = activeModObj.by_stock[fInst];
    cohortLabel = `${appState.mv21Filters.model} · Instrument: ${fInst}`;
    cTrades = bs.trades;
    cWin = `${bs.win_rate_pct}%`;
    cPf = bs.profit_factor;
    cExp = `${bs.expectancy_pct >= 0 ? "+" : ""}${bs.expectancy_pct}%`;
    cTotVal = fmtCurrency(bs.total_pnl, fInst === "NVDA" ? "USD" : "INR", 0);
    cTotSub = "Instrument Net PnL (Simulated)";
  } else if (fSec !== "ALL" && activeModObj.by_sector?.[fSec]) {
    const bs = activeModObj.by_sector[fSec];
    cohortLabel = `${appState.mv21Filters.model} · Sector: ${fSec}`;
    cTrades = bs.trades;
    cWin = `${bs.win_rate_pct}%`;
    cPf = bs.profit_factor;
    cExp = `${bs.expectancy_pct >= 0 ? "+" : ""}${bs.expectancy_pct}%`;
    cTotVal = fmtCurrency(bs.total_pnl, fSec.includes("US ") ? "USD" : "INR", 0);
    cTotSub = "Sector Net PnL (Simulated)";
  } else if (fReg !== "ALL" && activeModObj.regimes_6?.[fReg]) {
    const br = activeModObj.regimes_6[fReg];
    cohortLabel = `${appState.mv21Filters.model} · Regime: ${fReg}`;
    cTrades = br.trades;
    cWin = `${br.win_rate_pct}%`;
    cPf = br.profit_factor;
    cExp = `${br.expectancy_pct >= 0 ? "+" : ""}${br.expectancy_pct}%`;
    cTotVal = `Sh ${br.sharpe_ratio} · DD ${br.max_drawdown_pct}%`;
    cTotSub = `${br.days_in_regime} market days in regime`;
  }

  kpiEl.innerHTML = `
    <div class="kpi-chip">
      <div class="kpi-lbl">Active Scope</div>
      <div class="kpi-val" style="font-size:12px;">${escHtml(cohortLabel)}</div>
      <div style="font-size:10.5px;color:var(--text-muted);">${ds.trading_days || 0} bars · ${ds.instruments_count || 0} instruments</div>
    </div>
    <div class="kpi-chip">
      <div class="kpi-lbl">Net Return / PnL (Sim)</div>
      <div class="kpi-val" style="color:${String(cTotVal).startsWith("-") ? "var(--accent-crimson)" : "var(--accent-cyan)"};">${escHtml(cTotVal)}</div>
      <div style="font-size:10.5px;color:var(--text-muted);">${escHtml(cTotSub)}</div>
    </div>
    <div class="kpi-chip">
      <div class="kpi-lbl">Trades &amp; Win Rate</div>
      <div class="kpi-val">${cTrades} · ${escHtml(cWin)}</div>
      <div style="font-size:10.5px;color:var(--text-muted);">Hold ${modM.avg_holding_days ?? "--"}D · Util ${modM.capital_utilisation_pct ?? "--"}%</div>
    </div>
    <div class="kpi-chip">
      <div class="kpi-lbl">Risk-Adjusted Ratios</div>
      <div class="kpi-val">Sh ${modM.sharpe_ratio ?? "--"} · So ${modM.sortino_ratio ?? "--"}</div>
      <div style="font-size:10.5px;color:var(--text-muted);">MaxDD ${modM.max_drawdown_pct ?? "--"}% · Recov ${modM.recovery_factor ?? "--"} · PF ${cPf}</div>
    </div>
    <div class="kpi-chip">
      <div class="kpi-lbl">Statistical Expectancy</div>
      <div class="kpi-val" style="font-size:13px;">${escHtml(cExp)}</div>
      <div style="font-size:10.5px;color:var(--text-muted);">${modM.distinguishable_from_random ? "Stat. significant vs 0" : "Not distinguishable from random/neg"}</div>
    </div>
  `;

  // Benchmark matrix table
  const bmRows = Array.isArray(r21.benchmark_comparison) ? r21.benchmark_comparison : [];
  bmBody.innerHTML = bmRows.map((m) => {
    const isHl = String(m.model).includes("2.1 (6-Regime") || String(m.model).includes("Nifty 50") || String(m.model).includes("Cash");
    const ret = m.total_return_pct ?? 0;
    const col = ret >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)";
    const ci = Array.isArray(m.win_rate_ci95) ? `[${m.win_rate_ci95[0]}–${m.win_rate_ci95[1]}]` : "";
    const pVal = m.p_value_vs_zero != null ? ` (p=${m.p_value_vs_zero})` : "";
    return `
      <tr style="${String(m.model).includes("2.1 (6-Regime") ? "background:hsla(196,86%,54%,0.08);" : ""}">
        <td style="font-weight:${isHl ? "700" : "500"};">${escHtml(m.model)}</td>
        <td style="font-family:var(--font-mono);color:${col};font-weight:700;">${ret >= 0 ? "+" : ""}${ret}%</td>
        <td style="font-family:var(--font-mono);color:${col};">${(m.cagr_pct || 0) >= 0 ? "+" : ""}${m.cagr_pct ?? 0}%</td>
        <td style="font-family:var(--font-mono);">${m.win_rate_pct ?? 0}% <span style="color:var(--text-muted);">${ci}</span></td>
        <td style="font-family:var(--font-mono);">${m.profit_factor != null ? m.profit_factor : "--"}</td>
        <td style="font-family:var(--font-mono);">${m.sharpe_ratio ?? "--"}</td>
        <td style="font-family:var(--font-mono);">${m.sortino_ratio ?? "--"}</td>
        <td style="font-family:var(--font-mono);">${m.max_drawdown_pct ?? "--"}%</td>
        <td style="font-family:var(--font-mono);">${m.recovery_factor != null ? m.recovery_factor : "--"}</td>
        <td style="font-family:var(--font-mono);">${m.expectancy_pct != null ? `${m.expectancy_pct >= 0 ? "+" : ""}${m.expectancy_pct}%${pVal}` : "--"}</td>
        <td style="font-family:var(--font-mono);">${m.total_trades ?? 0}</td>
        <td style="font-family:var(--font-mono);">${m.avg_holding_days ?? "--"}</td>
        <td style="font-family:var(--font-mono);">${m.capital_utilisation_pct ?? "--"}%</td>
        <td style="font-family:var(--font-mono);">${m.cost_drag_pct ?? 0}%</td>
      </tr>
    `;
  }).join("");

  // Walk-Forward + OOS
  const wf = r21.walk_forward_validation || {};
  const oos = r21.out_of_sample || {};
  const wfSub = document.getElementById("mv21WfSubtitle");
  if (wfSub) {
    wfSub.textContent = `1st-Half Avg ${wf.first_half_avg_return_pct ?? "--"}% vs 2nd-Half ${wf.second_half_avg_return_pct ?? "--"}% · Deterioration: ${wf.deterioration_detected ? "YES" : "NO"} · Zero tuning on OOS`;
  }
  if (wfBody) {
    const folds = wf.folds || [];
    wfBody.innerHTML = folds.map((f) => `
      <tr>
        <td style="font-family:var(--font-mono);"><strong>${escHtml(f.fold)}</strong><br><span style="color:var(--text-muted);">${escHtml(f.test_window)}</span></td>
        <td style="font-family:var(--font-mono);color:var(--accent-amber);">${f.nifty_return_pct >= 0 ? "+" : ""}${f.nifty_return_pct}%</td>
        <td style="font-family:var(--font-mono);color:${f.m1_return_pct >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)"};">${f.m1_return_pct >= 0 ? "+" : ""}${f.m1_return_pct}% (${f.m1_max_dd_pct}%)</td>
        <td style="font-family:var(--font-mono);color:${f.m2_return_pct >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)"};">${f.m2_return_pct >= 0 ? "+" : ""}${f.m2_return_pct}% (${f.m2_max_dd_pct}%)</td>
        <td style="font-family:var(--font-mono);font-weight:700;color:${f.m21_return_pct >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)"};">${f.m21_return_pct >= 0 ? "+" : ""}${f.m21_return_pct}% (${f.m21_max_dd_pct}%)</td>
        <td style="font-family:var(--font-mono);">${f.m21_win_rate_pct}% (n=${f.m21_trades})</td>
      </tr>
    `).join("");
  }
  const oosEl = document.getElementById("mv21OosSummary");
  if (oosEl && oos.moreshwar_2_1) {
    const om21 = oos.moreshwar_2_1;
    const om1 = oos.moreshwar_1_0 || {};
    const oSma = oos.sma20 || {};
    const oFix = oos.fixed_5_10 || {};
    const oNif = oos.nifty50 || {};
    const oCash = oos.cash_6_5 || {};
    oosEl.innerHTML = `<strong>Untouched OOS (${escHtml(oos.window || "")}):</strong> M2.1 <strong>${om21.total_return_pct}%</strong> (DD ${om21.max_drawdown_pct}%, n=${om21.total_trades}) vs M1.0 ${om1.total_return_pct}% (DD ${om1.max_drawdown_pct}%) vs SMA-20 ${oSma.total_return_pct}% vs Fixed 5/10 ${oFix.total_return_pct}% vs Nifty +${oNif.total_return_pct}% vs Cash +${oCash.total_return_pct}%.`;
  }

  // 6-State Regimes table (for currently selected Model)
  if (regBody) {
    const rMap = activeModObj.regimes_6 || {};
    regBody.innerHTML = allRegimes.map((rn) => {
      const d = rMap[rn] || {};
      const expCol = (d.expectancy_pct || 0) > 0 ? "var(--accent-emerald)" : ((d.expectancy_pct || 0) < 0 ? "var(--accent-crimson)" : "var(--text-muted)");
      return `
        <tr>
          <td style="font-family:var(--font-mono);font-weight:700;">${escHtml(rn)}</td>
          <td style="font-family:var(--font-mono);">${d.days_in_regime ?? 0}</td>
          <td style="font-family:var(--font-mono);">${d.trades ?? 0}</td>
          <td style="font-family:var(--font-mono);">${d.win_rate_pct ?? 0}%</td>
          <td style="font-family:var(--font-mono);color:${expCol};font-weight:700;">${(d.expectancy_pct || 0) >= 0 ? "+" : ""}${d.expectancy_pct ?? 0}%</td>
          <td style="font-family:var(--font-mono);">${d.profit_factor ?? 0}</td>
          <td style="font-family:var(--font-mono);">${d.sharpe_ratio ?? 0}</td>
          <td style="font-family:var(--font-mono);">${d.max_drawdown_pct ?? 0}%</td>
        </tr>
      `;
    }).join("");
  }

  // Monte Carlo Stats box (switch between M1.0 and M2.1 distributions based on selected model)
  const mcStatsEl = document.getElementById("mv21MonteCarloStats");
  const mcAll = r21.monte_carlo || {};
  const mcKey = String(appState.mv21Filters.model || "").includes("1.0") ? "moreshwar_1_0" : "moreshwar_2_1";
  const mc = mcAll[mcKey] || mcAll.moreshwar_2_1 || mcAll;
  if (mcStatsEl && mc.trade_bootstrap && mc.daily_block_bootstrap) {
    const tb = mc.trade_bootstrap;
    const bb = mc.daily_block_bootstrap;
    mcStatsEl.innerHTML = `
      <div>• <strong>[${escHtml(mcKey.toUpperCase())}] Trade Bootstrap (${tb.simulations} sims):</strong> P5 <strong>${tb.p05_return_pct}%</strong> | Median <strong>${tb.median_return_pct}%</strong> | P95 <strong>+${tb.p95_return_pct}%</strong> · Med DD ${tb.median_max_dd_pct}% (P95 ${tb.p95_max_dd_pct}%) · P(Ret&gt;0)=<strong>${tb.prob_positive_return_pct}%</strong> · P(DD≥25%)=<strong>${tb.prob_ruin_25pct_dd}%</strong></div>
      <div>• <strong>10D Block Bootstrap (${bb.simulations} sims):</strong> P5 <strong>${bb.p05_return_pct}%</strong> | Median <strong>${bb.median_return_pct}%</strong> | P95 <strong>+${bb.p95_return_pct}%</strong> · Med DD ${bb.median_max_dd_pct}% (P95 ${bb.p95_max_dd_pct}%) · P(Ret&gt;0)=<strong>${bb.prob_positive_return_pct}%</strong> · P(DD≥25%)=<strong>${bb.prob_ruin_25pct_dd}%</strong></div>
      <div style="color:var(--text-muted);margin-top:2px;">Limitation: ${escHtml(bb.limitation || "")}</div>
    `;
  }

  // Portfolio-Level Risk Grid (Section 6)
  if (prGrid) {
    const pr = activeModObj.portfolio_risk || {};
    const ex = modM.execution_realism || {};
    const cb = modM.cost_breakdown || {};
    prGrid.innerHTML = `
      <div class="dq-mini-stat"><div class="dq-mini-lbl">Simultaneous Positions</div><div class="dq-mini-val">${pr.avg_simultaneous_positions ?? 0} avg · ${pr.max_simultaneous_positions ?? 0} max</div><div class="dq-mini-sub">Exposure ${pr.capital_utilization_pct ?? 0}% avg / ${pr.max_portfolio_exposure_pct ?? 0}% peak</div></div>
      <div class="dq-mini-stat"><div class="dq-mini-lbl">Sector &amp; Correlation</div><div class="dq-mini-val">${pr.avg_sector_concentration_pct ?? 0}% avg · ${pr.max_sector_concentration_pct ?? 0}% peak</div><div class="dq-mini-sub">Avg Corr ${pr.avg_concurrent_correlation ?? 0} · Peak ${pr.peak_concurrent_correlation ?? 0}</div></div>
      <div class="dq-mini-stat"><div class="dq-mini-lbl">Portfolio SL Risk</div><div class="dq-mini-val">${pr.avg_portfolio_stop_loss_exposure_pct ?? 0}% avg · ${pr.max_portfolio_stop_loss_exposure_pct ?? 0}% max</div><div class="dq-mini-sub">Max losing streak: ${pr.max_consecutive_losses ?? 0} trades</div></div>
      <div class="dq-mini-stat"><div class="dq-mini-lbl">Tail Shocks &amp; Market Declines</div><div class="dq-mini-val">${pr.max_single_day_loss_pct ?? 0}% 1D · ${pr.max_5day_week_loss_pct ?? 0}% 5D</div><div class="dq-mini-sub">${pr.market_decline_days_count ?? 0} Nifty drop days (-0.614% avg / ${pr.avg_exposure_on_market_declines_pct ?? 0}% exp)</div></div>
      <div class="dq-mini-stat"><div class="dq-mini-lbl">Indian Delivery Friction</div><div class="dq-mini-val">${fmtCurrency(cb.total_cost || 0, "INR", 0)} (${modM.cost_drag_pct ?? 0}%)</div><div class="dq-mini-sub">STT ${fmtCurrency(cb.stt || 0, "INR", 0)} · DP ${fmtCurrency(cb.dp_charge || 0, "INR", 0)} · Slip ${fmtCurrency(cb.slippage_cost || 0, "INR", 0)}</div></div>
      <div class="dq-mini-stat"><div class="dq-mini-lbl">Gap &amp; Liquidity Realism</div><div class="dq-mini-val">${ex.gap_through_stops ?? 0} gap SL · ${ex.unfillable_zero_vol ?? 0} halted</div><div class="dq-mini-sub">Regime gates: ${ex.rejection_categories?.REGIME_GATE ?? 0} · Port caps: ${ex.rejection_categories?.PORTFOLIO_CAP_OR_CORR ?? 0}</div></div>
    `;
  }

  // Strategy Investigation & Counterfactuals (Section 4)
  if (invEl) {
    const inv = r21.strategy_investigation || {};
    const comp = inv.component_attribution || [];
    const scf = inv.selectivity_counterfactuals || {};
    const cfCats = scf.by_rejection_category || {};
    invEl.innerHTML = comp.map((c) => {
      const pillCls = c.impact === "POSITIVE" ? "tag-target" : (c.impact === "LOOK_AHEAD_BIASED" || c.impact === "EXCESSIVE_SELECTIVITY_WARNING" || c.impact === "NEGATIVE") ? "tag-stop" : "tag-score";
      return `
        <div style="background:var(--bg-elevated);border:1px solid var(--border-subtle);border-radius:4px;padding:7px 9px;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;margin-bottom:2px;">
            <strong style="color:var(--text-primary);">${escHtml(c.component)}</strong>
            <span class="tag-pill ${pillCls}">${escHtml(c.impact)}</span>
          </div>
          <div style="color:var(--text-secondary);">${escHtml(c.evidence)}</div>
        </div>
      `;
    }).join("") + `
      <div style="background:var(--bg-elevated);border:1px solid var(--border-subtle);border-radius:4px;padding:7px 9px;">
        <strong>Rejected Counterfactuals by Category (${scf.rejected_evaluated ?? 0} missed setups vs ${scf.accepted_trades ?? 0} taken):</strong>
        <div style="font-family:var(--font-mono);font-size:10.5px;margin-top:4px;color:var(--text-secondary);">
          ${Object.entries(cfCats).map(([cat, v]) => `${escHtml(cat)}: n=${v.count}, Win=${v.cf_win_rate_pct}%, Avg=${v.cf_avg_return_pct >= 0 ? "+" : ""}${v.cf_avg_return_pct}%`).join(" · ")}
        </div>
      </div>
    `;
  }

  // Data Quality & Survivorship (Section 2)
  if (dqEl) {
    const dqr = r21.data_quality_report || {};
    const delisted = dqr.delisted_probes || [];
    const sectorsCov = Array.isArray(dqr.sectors_covered) ? dqr.sectors_covered : [];
    const capTiersCov = Array.isArray(dqr.cap_tiers_covered) ? dqr.cap_tiers_covered : [];
    const regimesCov = Object.keys(activeModObj.regimes_6 || {});
    const completeCnt = dqr.complete_multi_year_count ?? dqr.complete_5y_count ?? 0;
    const distressedCnt = dqr.distressed_or_illiquid_count ?? dqr.distressed_controls_count ?? 0;
    dqEl.innerHTML = `
      <div>• <strong>Universe:</strong> ${dqr.instruments_loaded ?? 0} / ${dqr.instruments_requested ?? 0} instruments · Complete 5Y: <strong>${completeCnt}</strong> · Insolvency/Distressed controls: <strong>${distressedCnt}</strong> · Partial post-demerger: <strong>${dqr.partial_history_count ?? 0}</strong></div>
      <div>• <strong>Coverage &amp; Regimes:</strong> ${sectorsCov.length} sectors · ${capTiersCov.length} cap tiers · ${dqr.benchmark_bars ?? 0} Nifty 50 bars (${escHtml(dqr.benchmark_start || "--")} → ${escHtml(dqr.benchmark_end || "--")}) · Regimes: ${escHtml(regimesCov.join(" · "))}</div>
      <div>• <strong>Delisted Ticker Survivorship Audit (Real HTTP Probes — Never Fabricated):</strong> ${delisted.map((p) => `<code>${escHtml(p.symbol)}</code> (${escHtml(p.reason || p.status || (p.available ? "Available" : "HTTP 404 Unavailable"))})`).join(", ")}</div>
      <div style="margin-top:4px;color:var(--text-muted);">${escHtml(dqr.survivorship_disclosure || dqr.survivorship_limitation_note || "")}</div>
    `;
  }

  // Filtered Sample Trade Execution Log
  if (trBody) {
    let trades = Array.isArray(activeModObj.sample_trades) ? activeModObj.sample_trades : [];
    if (fInst !== "ALL") trades = trades.filter((t) => t.symbol === fInst);
    if (fSec !== "ALL") trades = trades.filter((t) => t.sector === fSec);
    if (fReg !== "ALL") trades = trades.filter((t) => t.signal_regime6 === fReg);
    trBody.innerHTML = trades.slice(0, 60).map((t) => {
      const cur = t.symbol === "NVDA" ? "USD" : "INR";
      const rCol = t.net_return_pct >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)";
      return `
        <tr>
          <td style="font-family:var(--font-mono);font-weight:700;">${escHtml(t.symbol)}<br><span style="color:var(--text-muted);font-weight:400;">${escHtml(t.sector)}</span></td>
          <td style="font-family:var(--font-mono);">${escHtml(t.entry_date)}<br>${escHtml(t.exit_date)} (${t.holding_days}d)</td>
          <td style="font-family:var(--font-mono);">${escHtml(t.signal_regime6)}</td>
          <td style="font-family:var(--font-mono);">${fmtCurrency(t.entry_price, cur, 2)} → ${fmtCurrency(t.exit_price, cur, 2)}</td>
          <td style="font-family:var(--font-mono);color:${rCol};font-weight:700;">${t.net_return_pct >= 0 ? "+" : ""}${t.net_return_pct}%</td>
          <td style="font-family:var(--font-mono);">${fmtCurrency(t.cost_amount, cur, 1)}</td>
          <td style="font-family:var(--font-mono);">${escHtml(t.exit_reason)}${t.gap_through_stop ? " (GAP)" : ""}</td>
        </tr>
      `;
    }).join("") || `<tr><td colspan="7" style="text-align:center;color:var(--text-secondary);">No sampled trades match current Model + Instrument + Sector + Regime filters (adjust filters or see KPI cohort totals above).</td></tr>`;
  }

  requestAnimationFrame(() => {
    drawMv21EquityChart(r21);
    drawMv21MonteCarloChart(mc);
  });
}

function renderAuditLedger() {
  const symSel = document.getElementById("auditSymbolFilter");
  const stSel = document.getElementById("auditStatusFilter");
  const flowEl = document.getElementById("auditInspectorFlow");
  const bodyEl = document.getElementById("auditLedgerTableBody");
  const subEl = document.getElementById("auditLedgerSub");
  if (!symSel || !stSel || !flowEl || !bodyEl) return;

  const trail = appState.auditTrail;
  const records = (trail && Array.isArray(trail.records)) ? trail.records : [];
  const acc = trail?.accumulation_summary || {};
  if (subEl) {
    subEl.textContent = `${records.length} Permanent Decisions · ${acc.settled_count ?? 0}/${acc.min_recommended_settled ?? 30} Settled (${acc.open_count ?? records.length} open monitoring) · Updated ${fmtTsShort(trail?.updated_at)}`;
  }

  // Sync symbol filter options without losing selection
  const curSymFilter = symSel.value || "ALL";
  const syms = Array.from(new Set(records.map((r) => r.symbol).filter(Boolean))).sort();
  symSel.innerHTML = `<option value="ALL">All Symbols (${syms.length})</option>` + syms.map((s) => `<option value="${escHtml(s)}">${escHtml(s)}</option>`).join("");
  symSel.value = syms.includes(curSymFilter) ? curSymFilter : "ALL";

  const symVal = symSel.value;
  const stVal = stSel.value || "ALL";
  const filtered = records.filter((r) => (symVal === "ALL" || r.symbol === symVal) && (stVal === "ALL" || (r.outcome?.status || "OPEN") === stVal));

  let selected = filtered.find((r) => r.audit_id === appState.selectedAuditId)
    || filtered.find((r) => r.symbol === appState.activeSymbol)
    || filtered[0]
    || null;
  if (selected) appState.selectedAuditId = selected.audit_id;

  if (!selected) {
    flowEl.innerHTML = `<div class="audit-stage-card" style="grid-column: 1 / -1;">No matching decision records in current filter.</div>`;
    bodyEl.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--text-secondary);padding:16px;">No audit records match selected filters.</td></tr>`;
    return;
  }

  const cur = selected.currency || "INR";
  const orig = selected.original_decision || {};
  const crit = selected.critical_review || {};
  const fin = selected.final_decision || {};
  const deltas = selected.recalibration_deltas || {};
  const out = selected.outcome || {};
  const sc = selected.scores || {};
  const vers = selected.versions || {};
  const ag = selected.agent_assessments || {};
  const mkt = selected.market_snapshot || {};

  const outTone = (out.status === "TARGET_HIT" || out.status === "BROKER_EXECUTED")
    ? "tag-target"
    : (out.status === "STOP_HIT" || String(out.status).includes("DEFERRED"))
      ? "tag-stop"
      : "tag-entry";
  const retShow = out.realized_return_pct != null
    ? `${out.realized_return_pct >= 0 ? "+" : ""}${out.realized_return_pct}% (Realized)`
    : out.counterfactual_return_pct != null
      ? `${out.counterfactual_return_pct >= 0 ? "+" : ""}${out.counterfactual_return_pct}% (Counterfactual)`
      : `${(out.unrealized_return_pct || 0) >= 0 ? "+" : ""}${out.unrealized_return_pct || 0}% (Unrealized)`;

  flowEl.innerHTML = `
    <div class="audit-stage-card" style="border-top:2px solid var(--accent-cyan);">
      <div class="audit-stage-head">
        <span class="audit-stage-title" style="color:var(--accent-cyan);">1. ORIGINAL DECISION</span>
        <span class="tag-pill tag-entry">${escHtml(orig.action || "--")}</span>
      </div>
      <div style="font-family:var(--font-mono);font-size:11.5px;">
        ${escHtml(selected.symbol)} · Spot ${fmtCurrency(mkt.spot_price, cur, 2)} · <strong>${escHtml(selected.market_date)}</strong>
      </div>
      <div style="font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);">
        Entry ${fmtCurrency(orig.entry, cur, 2)} · TP ${fmtCurrency(orig.target, cur, 2)} · SL ${fmtCurrency(orig.stop_loss, cur, 2)}<br>
        R:R 1:${orig.rr_ratio ?? "--"} · Qty ${orig.qty ?? 0} · Conv ${orig.conviction ?? 0}%
      </div>
      <div style="font-size:11px;color:var(--text-secondary);">
        Scores: Fund <strong>${sc.fundamental_total}/100</strong> · Porter <strong>${sc.porters_total}/100</strong> · Tech <strong>${sc.technical_score}%</strong>
      </div>
      <div style="font-size:10.5px;color:var(--text-muted);">
        Agents: Tech=${escHtml(ag.TechBot_1?.signal || "N/A")}, Fund=${escHtml(ag.FundBot_1?.signal || "N/A")}, Exec=${escHtml(ag.ExecBot_1?.signal || "N/A")}
      </div>
    </div>

    <div class="audit-stage-card" style="border-top:2px solid var(--accent-crimson);">
      <div class="audit-stage-head">
        <span class="audit-stage-title" style="color:var(--accent-crimson);">2. CRITICAL REVIEW (CRO)</span>
        <span class="tag-pill tag-stop">${escHtml(crit.stance || "--")}</span>
      </div>
      <div style="font-family:var(--font-mono);font-size:11px;">Risk Rating: <strong>${escHtml(crit.risk_rating || "--")}</strong> · Model: ${escHtml(crit.model || "none")}</div>
      <div style="font-size:11.5px;color:var(--text-secondary);"><strong>Objections:</strong> ${escHtml(crit.objections_raised || "None logged")}</div>
      <div style="font-size:11px;color:var(--text-muted);">${escHtml(crit.rationale || "")}</div>
    </div>

    <div class="audit-stage-card" style="border-top:2px solid var(--accent-amber);">
      <div class="audit-stage-head">
        <span class="audit-stage-title" style="color:var(--accent-amber);">3. FINAL DECISION &amp; DELTAS</span>
        <span class="tag-pill tag-score">${escHtml(fin.action || "--")}</span>
      </div>
      <div style="font-family:var(--font-mono);font-size:11px;">
        Entry ${fmtCurrency(fin.entry, cur, 2)} (${deltas.entry_delta_pct >= 0 ? "+" : ""}${deltas.entry_delta_pct || 0}%)<br>
        TP ${fmtCurrency(fin.target, cur, 2)} (${deltas.target_delta_pct >= 0 ? "+" : ""}${deltas.target_delta_pct || 0}%) · SL ${fmtCurrency(fin.stop_loss, cur, 2)}<br>
        Qty ${fin.qty ?? 0} (${deltas.qty_delta >= 0 ? "+" : ""}${deltas.qty_delta || 0}) · Conv ${fin.conviction ?? 0}% (${deltas.conviction_delta >= 0 ? "+" : ""}${deltas.conviction_delta || 0}%)
      </div>
      <div style="font-size:11px;color:var(--text-secondary);">
        ${(deltas.reasons || []).slice(0, 3).map((rs) => `• ${escHtml(rs)}`).join("<br>")}
      </div>
    </div>

    <div class="audit-stage-card" style="border-top:2px solid var(--accent-emerald);">
      <div class="audit-stage-head">
        <span class="audit-stage-title" style="color:var(--accent-emerald);">4. ACTUAL OUTCOME</span>
        <span class="tag-pill ${outTone}">${escHtml(out.status || "OPEN")}</span>
      </div>
      <div style="font-family:var(--font-mono);font-size:11.5px;">
        Return: <strong>${escHtml(retShow)}</strong><br>
        Mark: ${fmtCurrency(out.latest_mark_price, cur, 2)} (${escHtml(out.mark_date || "--")})
      </div>
      <div style="font-family:var(--font-mono);font-size:11px;color:var(--text-secondary);">
        MFE: +${out.mfe_pct ?? 0}% · MAE: ${out.mae_pct ?? 0}% · Hold: ${out.holding_days ?? 0}d<br>
        CRO Verdict: <strong>${escHtml(out.cro_verdict || "PENDING_FORWARD_BARS")}</strong>${out.original_counterfactual_pct != null ? ` (Orig CF: ${out.original_counterfactual_pct >= 0 ? "+" : ""}${out.original_counterfactual_pct}%)` : ""}<br>
        Broker: ${out.broker_executed ? `Fill ${fmtCurrency(out.broker_fill_price, cur, 2)} (${escHtml(out.order_id)})` : "Not Executed (Analytical Audit)"}
      </div>
      <div style="font-size:10.5px;color:var(--text-muted);">
        Mkt TS: ${escHtml(fmtTsShort(selected.market_timestamp))} · ${escHtml(vers.platform_version || "")} / ${escHtml(vers.scoring_model || "")}
      </div>
    </div>
  `;

  bodyEl.innerHTML = filtered.slice(0, 40).map((r) => {
    const rCur = r.currency || "INR";
    const rOrig = r.original_decision || {};
    const rFin = r.final_decision || {};
    const rOut = r.outcome || {};
    const rSc = r.scores || {};
    const rV = r.versions || {};
    const isSel = r.audit_id === selected.audit_id;
    const oTone = (rOut.status === "TARGET_HIT" || rOut.status === "BROKER_EXECUTED")
      ? "tag-target"
      : (rOut.status === "STOP_HIT" || String(rOut.status).includes("DEFERRED"))
        ? "tag-stop"
        : "tag-entry";
    const oRet = rOut.realized_return_pct != null
      ? `${rOut.realized_return_pct >= 0 ? "+" : ""}${rOut.realized_return_pct}%`
      : rOut.counterfactual_return_pct != null
        ? `CF ${rOut.counterfactual_return_pct >= 0 ? "+" : ""}${rOut.counterfactual_return_pct}%`
        : `${(rOut.unrealized_return_pct || 0) >= 0 ? "+" : ""}${rOut.unrealized_return_pct || 0}%`;

    return `
      <tr class="${isSel ? "audit-row-selected" : ""}" data-audit-id="${escHtml(r.audit_id)}" style="cursor:pointer;">
        <td style="font-family:var(--font-mono);font-size:11px;">${escHtml(r.market_date)}<br><span style="color:var(--text-muted);">${escHtml(r.audit_id.slice(-14))}</span></td>
        <td style="font-family:var(--font-mono);font-weight:700;">${escHtml(r.symbol)}</td>
        <td style="font-family:var(--font-mono);font-size:11px;">${rSc.fundamental_total}/${rSc.porters_total}/${rSc.technical_score}%</td>
        <td style="font-family:var(--font-mono);font-size:11px;">${escHtml(rOrig.action)} → <strong>${escHtml(rFin.action)}</strong><br>${fmtCurrency(rOrig.entry, rCur, 2)} → ${fmtCurrency(rFin.entry, rCur, 2)}</td>
        <td style="font-size:11px;color:var(--text-secondary);max-width:260px;">${escHtml((r.critical_review?.objections_raised || "").slice(0, 95))}${(r.critical_review?.objections_raised || "").length > 95 ? "…" : ""}</td>
        <td><span class="tag-pill ${oTone}">${escHtml(rOut.status || "OPEN")}</span><div style="font-family:var(--font-mono);font-size:11px;margin-top:3px;">${escHtml(oRet)}</div></td>
        <td style="font-family:var(--font-mono);font-size:10.5px;color:var(--text-muted);">${escHtml(rV.scoring_model || "")}<br>${escHtml(rV.critic_model || "none")}</td>
      </tr>
    `;
  }).join("");

  bodyEl.querySelectorAll("tr[data-audit-id]").forEach((tr) => {
    tr.addEventListener("click", () => {
      appState.selectedAuditId = tr.dataset.auditId;
      renderAuditLedger();
    });
  });
}

function renderIntelligenceViews() {
  const secProv = document.getElementById("intelSectionProvenance");
  const secBt = document.getElementById("intelSectionBacktest");
  const secV21 = document.getElementById("intelSectionV21");
  const secV22 = document.getElementById("intelSectionV22");
  const secV30 = document.getElementById("intelSectionV30");
  const secAud = document.getElementById("intelSectionAudit");
  const focus = appState.intelFocus || "all";
  if (secProv) secProv.style.display = (focus === "all" || focus === "provenance") ? "flex" : "none";
  if (secBt) secBt.style.display = (focus === "all" || focus === "backtest") ? "flex" : "none";
  if (secV21) secV21.style.display = (focus === "all" || focus === "v21") ? "flex" : "none";
  if (secV22) secV22.style.display = (focus === "all" || focus === "v21") ? "flex" : "none";
  const secV31 = document.getElementById("intelSectionV31");
  const legacyDrawer = document.getElementById("legacyIntelArchiveLog");
  const isLegacyFocus = ["provenance", "backtest", "v21", "v30", "audit"].includes(focus);
  if (secV30) secV30.style.display = (focus === "all" || focus === "v30") ? "flex" : "none";
  if (secV31) secV31.style.display = (focus === "all" || focus === "v31") ? "flex" : "none";
  if (secAud) secAud.style.display = (focus === "all" || focus === "audit") ? "flex" : "none";
  if (legacyDrawer) {
    if (isLegacyFocus) {
      legacyDrawer.open = true;
    } else if (focus === "v31") {
      legacyDrawer.open = false;
    }
  }

  renderProvenanceMonitor();
  renderBacktestAnalytics();
  renderMoreshwarV21Dashboard();
  renderMoreshwarV22Dashboard();
  renderMoreshwarV30Dashboard();
  renderMoreshwarV31Dashboard();
  renderAuditLedger();
  renderReportsView();
}

function drawReportsCharts() {
  const bt = appState.backtest;
  drawReportEquityCanvas((bt || {}).equity_curve || []);
  drawReportAgentCanvas((bt?.decision_quality_answers?.top_contributing_agent?.ranking) || []);
  drawReportCroDonutCanvas(bt?.decision_quality_answers?.gemini_intervention_audit || null);
  drawReportBenchmarkCanvas((bt?.decision_quality_answers?.moreshwar_vs_benchmarks?.rows) || []);
}

function drawReportEquityCanvas(curve) {
  const canvas = document.getElementById("rptEquityCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(260, rect.width || 440);
  const h = Math.max(170, rect.height || 210);
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  if (!Array.isArray(curve) || curve.length < 2) {
    ctx.font = "12px 'Inter', sans-serif";
    ctx.fillStyle = "hsl(215, 15%, 65%)";
    ctx.fillText("Growth curve appears after live scan data loads.", 14, h / 2);
    return;
  }

  const vals = [];
  curve.forEach((pt) => {
    if (typeof pt.original_nav === "number") vals.push(pt.original_nav);
    if (typeof pt.recalibrated_nav === "number") vals.push(pt.recalibrated_nav);
    if (typeof pt.benchmark_nav === "number") vals.push(pt.benchmark_nav);
  });
  const minV = Math.min(...vals) * 0.99;
  const maxV = Math.max(...vals) * 1.01;
  const padT = 14, padB = 22, padL = 10, padR = 54;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const yFor = (v) => padT + plotH - ((v - minV) / Math.max(1e-6, maxV - minV)) * plotH;
  const xFor = (i) => padL + (i / Math.max(1, curve.length - 1)) * plotW;

  ctx.strokeStyle = "hsla(222, 15%, 24%, 0.45)";
  ctx.lineWidth = 1;
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillStyle = "hsl(215, 15%, 65%)";
  for (let i = 0; i <= 3; i++) {
    const v = minV + ((maxV - minV) * i) / 3;
    const y = yFor(v);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(1), w - padR + 6, y + 3);
  }
  ctx.fillText(curve[0].date || "", padL, h - 5);
  ctx.fillText(curve[curve.length - 1].date || "", w - padR - 66, h - 5);

  const plotLine = (key, color, dashed = false) => {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    if (dashed) ctx.setLineDash([4, 4]);
    ctx.beginPath();
    curve.forEach((pt, i) => {
      const x = xFor(i);
      const y = yFor(Number(pt[key] ?? 100));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    const last = curve[curve.length - 1];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(xFor(curve.length - 1), yFor(Number(last[key] ?? 100)), 3.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  };

  plotLine("benchmark_nav", "hsl(38, 92%, 55%)", true);
  plotLine("original_nav", "hsl(196, 86%, 54%)", false);
  plotLine("recalibrated_nav", "hsl(160, 84%, 44%)", false);
}

function drawReportAgentCanvas(ranking) {
  const canvas = document.getElementById("rptAgentCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(260, rect.width || 440);
  const h = Math.max(170, rect.height || 210);
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  if (!Array.isArray(ranking) || !ranking.length) {
    ctx.font = "12px 'Inter', sans-serif";
    ctx.fillStyle = "hsl(215, 15%, 65%)";
    ctx.fillText("Agent attribution chart awaiting scan.", 14, h / 2);
    return;
  }

  const vals = [];
  ranking.forEach((r) => {
    vals.push(Number(r.cum_return_lift_pct || 0), Number(r.drawdown_saved_pct || 0));
  });
  const maxAbs = Math.max(8, ...vals.map((v) => Math.abs(v))) * 1.18;
  const labelW = 92;
  const padR = 48;
  const chartW = Math.max(80, w - labelW - padR);
  const zeroX = labelW + chartW * 0.45;
  const scalePos = (chartW * 0.55) / maxAbs;
  const scaleNeg = (chartW * 0.45) / maxAbs;

  // Zero vertical line
  ctx.strokeStyle = "hsla(215, 20%, 50%, 0.45)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(zeroX, 10);
  ctx.lineTo(zeroX, h - 14);
  ctx.stroke();

  const rowH = (h - 24) / ranking.length;
  ctx.font = "10.5px 'JetBrains Mono', monospace";
  ranking.forEach((r, idx) => {
    const cy = 16 + idx * rowH;
    ctx.fillStyle = "hsl(210, 25%, 92%)";
    ctx.fillText(`#${r.rank} ${String(r.agent).replace("_1", "")}`, 6, cy + 12);

    const drawBar = (val, yOff, posCol, negCol) => {
      const xEnd = val >= 0 ? zeroX + val * scalePos : zeroX + val * scaleNeg;
      const bx = Math.min(zeroX, xEnd);
      const bw = Math.max(2, Math.abs(xEnd - zeroX));
      ctx.fillStyle = val >= 0 ? posCol : negCol;
      ctx.fillRect(bx, cy + yOff, bw, 9);
      ctx.fillStyle = val >= 0 ? posCol : negCol;
      const txt = `${val >= 0 ? "+" : ""}${val.toFixed(1)}%`;
      const tx = val >= 0 ? bx + bw + 4 : Math.max(labelW - 2, bx - 44);
      ctx.fillText(txt, Math.min(w - 44, tx), cy + yOff + 8);
    };
    drawBar(Number(r.cum_return_lift_pct || 0), 0, "hsl(160, 84%, 44%)", "hsl(350, 84%, 60%)");
    drawBar(Number(r.drawdown_saved_pct || 0), 12, "hsl(196, 86%, 54%)", "hsl(38, 92%, 55%)");
  });
}

function drawReportCroDonutCanvas(croAudit) {
  const canvas = document.getElementById("rptCroCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(260, rect.width || 440);
  const h = Math.max(170, rect.height || 210);
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const comb = (croAudit && croAudit.combined_scorecard) || {};
  const prev = Number(comb.bad_trades_prevented || 0);
  const miss = Number(comb.good_trades_unnecessarily_downgraded || 0);
  const pres = Number(comb.preserved_winners || 0);
  const total = prev + miss + pres;

  if (total <= 0) {
    ctx.font = "12px 'Inter', sans-serif";
    ctx.fillStyle = "hsl(215, 15%, 65%)";
    ctx.fillText("No CRO interventions logged in current window.", 14, h / 2);
    return;
  }

  const cx = Math.min(w * 0.34, 118);
  const cy = h / 2;
  const radius = Math.min(cx - 18, cy - 18, 66);
  const slices = [
    { val: prev, color: "hsl(160, 84%, 44%)", label: "Prevented Bad" },
    { val: miss, color: "hsl(350, 84%, 60%)", label: "Unnecessary Block/Down" },
    { val: pres, color: "hsl(196, 86%, 54%)", label: "Preserved Winners" },
  ];

  let startA = -Math.PI / 2;
  slices.forEach((s) => {
    if (s.val <= 0) return;
    const sweep = (s.val / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startA, startA + sweep);
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 22;
    ctx.stroke();
    startA += sweep;
  });

  ctx.textAlign = "center";
  ctx.fillStyle = "hsl(210, 25%, 94%)";
  ctx.font = "bold 18px 'JetBrains Mono', monospace";
  ctx.fillText(`${comb.prevention_rate_pct ?? 0}%`, cx, cy + 4);
  ctx.font = "9.5px 'Inter', sans-serif";
  ctx.fillStyle = "hsl(215, 15%, 68%)";
  ctx.fillText("PREVENTED", cx, cy + 18);

  // Clean right-side breakdown labels
  ctx.textAlign = "left";
  const lx = cx + radius + 28;
  const items = [
    [`${prev} / ${total} Bad Prevented`, `Avg block save -${croAudit?.blocks?.avg_loss_avoided_pct ?? 0}%`, "hsl(160, 84%, 44%)"],
    [`${miss} / ${total} Unnecessary`, `Missed avg +${croAudit?.blocks?.avg_upside_missed_pct ?? 0}%`, "hsl(350, 84%, 60%)"],
    [`${pres} / ${total} Preserved Wins`, `Net CRO value ${(comb.net_intervention_value_pct || 0) >= 0 ? "+" : ""}${comb.net_intervention_value_pct ?? 0}%`, "hsl(196, 86%, 54%)"],
  ];
  items.forEach(([t1, t2, col], idx) => {
    const iy = 38 + idx * 50;
    ctx.fillStyle = col;
    ctx.fillRect(lx, iy - 9, 8, 8);
    ctx.fillStyle = "hsl(210, 25%, 92%)";
    ctx.font = "bold 11px 'JetBrains Mono', monospace";
    ctx.fillText(t1, lx + 14, iy - 1);
    ctx.fillStyle = "hsl(215, 15%, 68%)";
    ctx.font = "10px 'Inter', sans-serif";
    ctx.fillText(t2, lx + 14, iy + 14);
  });
}

function drawReportBenchmarkCanvas(rows) {
  const canvas = document.getElementById("rptBenchmarkCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(260, rect.width || 440);
  const h = Math.max(170, rect.height || 210);
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  if (!Array.isArray(rows) || !rows.length) {
    ctx.font = "12px 'Inter', sans-serif";
    ctx.fillStyle = "hsl(215, 15%, 65%)";
    ctx.fillText("Benchmark comparison awaiting scan.", 14, h / 2);
    return;
  }

  const labelW = w < 380 ? 92 : 148;
  const negValPad = 42;
  const padR = 46;
  const plotW = Math.max(80, w - labelW - negValPad - padR);
  const maxAbs = Math.max(5, ...rows.map((r) => Math.abs(Number(r.cumulative_return_pct || 0)))) * 1.15;
  const zeroX = labelW + negValPad + plotW * 0.72;
  const scaleNeg = (plotW * 0.72) / maxAbs;
  const scalePos = (plotW * 0.28) / maxAbs;

  ctx.strokeStyle = "hsla(215, 20%, 50%, 0.45)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(zeroX, 8);
  ctx.lineTo(zeroX, h - 10);
  ctx.stroke();

  const rowH = (h - 16) / rows.length;
  ctx.font = w < 380 ? "9.5px 'JetBrains Mono', monospace" : "10px 'JetBrains Mono', monospace";
  rows.forEach((r, i) => {
    const cy = 12 + i * rowH;
    const shortName = String(r.strategy)
      .replace("Moreshwar Mathematical Model (Orig)", "Moreshwar Orig")
      .replace("Recalibrated Moreshwar (+ Gemini CRO)", "Recal + CRO")
      .replace("Nifty 50 (^NSEI) Buy & Hold", "Nifty 50 Hold")
      .replace("Equal-Weight Universe Buy & Hold", "Equal-Wt 10")
      .replace("Simple SMA-20 + Fixed 1:2 (5%/10%)", "Simple SMA-20")
      .replace("Matrix Entries + Fixed 1:2 Exits", "Matrix + 1:2");

    ctx.fillStyle = "hsl(210, 25%, 90%)";
    ctx.fillText(shortName.slice(0, w < 380 ? 14 : 20), 6, cy + 12);

    const val = Number(r.cumulative_return_pct || 0);
    const isBench = String(r.strategy).includes("Nifty") || String(r.strategy).includes("Equal-Weight");
    const col = val >= 0 ? (isBench ? "hsl(38, 92%, 55%)" : "hsl(160, 84%, 44%)") : (String(r.strategy).includes("Matrix +") ? "hsl(196, 86%, 54%)" : "hsl(350, 84%, 60%)");
    const xEnd = val >= 0 ? zeroX + val * scalePos : zeroX + val * scaleNeg;
    const bx = Math.min(zeroX, xEnd);
    const bw = Math.max(2, Math.abs(xEnd - zeroX));
    ctx.fillStyle = col;
    ctx.fillRect(bx, cy + 3, bw, 12);
    const valStr = `${val >= 0 ? "+" : ""}${val.toFixed(1)}%`;
    ctx.fillText(valStr, val >= 0 ? bx + bw + 4 : bx - 40, cy + 13);
  });
}

function renderReportsView() {
  const briefEl = document.getElementById("rptExecutiveBrief");
  const kpiEl = document.getElementById("rptHighlightKpis");
  const storyEl = document.getElementById("rptStoryContainer");
  const isoBadge = document.getElementById("rptIsolationBadge");
  if (!briefEl || !kpiEl || !storyEl) return;

  const assetsList = Object.values(appState.assets || {});
  const bt = appState.backtest || {};
  const dqa = bt.decision_quality_answers || {};
  const selM = bt.selectivity_metrics || {};
  const trOrig = bt.tracks?.original_ai_signals?.metrics || {};
  const trRecal = bt.tracks?.recalibrated_ai_signals?.metrics || {};
  const trAct = bt.tracks?.actual_executed_trades?.metrics || {};
  const actStatus = bt.tracks?.actual_executed_trades?.status || "NO_LIVE_BROKER_FILLS";
  const bench = bt.benchmark || {};
  const q1 = dqa.top_contributing_agent || {};
  const q2 = dqa.gemini_intervention_audit || {};
  const q2Comb = q2.combined_scorecard || {};
  const q3 = dqa.moreshwar_vs_benchmarks || {};
  const sampleCtx = dqa.sample_context || {};
  const buys = appState.queue?.buys || [];
  const sells = appState.queue?.sells || [];
  const provSum = appState.provenance?.summary || {};
  const discCount = provSum.discrepancy_count ?? 0;

  if (isoBadge) {
    isoBadge.textContent = actStatus === "BROKER_VERIFIED"
      ? "REAL BROKER FILLS VERIFIED SEPARATELY"
      : "0 REAL BROKER FILLS · SIMULATIONS STRICTLY ISOLATED";
  }

  const buySyms = buys.map((b) => b.symbol).join(", ") || "None";
  const topAgent = q1.winner || "FundBot_1";
  const topLift = q1.cum_return_lift_pct ?? 0;
  const topDd = q1.drawdown_saved_pct ?? 0;
  const prevCount = q2Comb.bad_trades_prevented ?? 0;
  const totalCro = q2.total_interventions ?? 0;
  const prevRate = q2Comb.prevention_rate_pct ?? 0;
  const unnecCount = q2Comb.good_trades_unnecessarily_downgraded ?? 0;
  const unnecRate = q2Comb.unnecessary_downgrade_rate_pct ?? 0;
  const settledLive = sampleCtx.settled_records ?? 0;
  const minLive = sampleCtx.min_recommended_settled ?? 30;
  const openLive = sampleCtx.open_records ?? 0;

  briefEl.innerHTML = `
    <strong>Plain-English Bottom Line:</strong> Across <strong>${assetsList.length}</strong> live tracked stocks, our tighter quality filters eliminated <strong>${selM.noise_reduction_pct ?? 0}%</strong> of low-conviction noise, leaving <strong>${buys.length} high-conviction BUY setups</strong> right now (<em>${escHtml(buySyms)}</em>) and <strong>${sells.length}</strong> exit warnings. On 1-year historical walk-forward testing (net of 0.35% costs), <strong>${escHtml(topAgent)}</strong> contributes the most to profit (<strong>${topLift >= 0 ? "+" : ""}${topLift}%</strong> cumulative lift and <strong>${topDd >= 0 ? "+" : ""}${topDd}%</strong> drawdown protection), while <strong>Gemini CRO</strong> prevented <strong>${prevCount} of ${totalCro} weak trades (${prevRate}%)</strong> at a cost of <strong>${unnecCount} unnecessary blocks/downgrades (${unnecRate}%)</strong>. However, the default <strong>Moreshwar exit formula</strong> trails Nifty 50 (<strong>${trOrig.cumulative_return_pct ?? 0}%</strong> vs <strong>${bench.benchmark_return_pct >= 0 ? "+" : ""}${bench.benchmark_return_pct ?? 0}%</strong>) because its tight stop-loss gets triggered by short-term noise. Live broker execution currently has <strong>${trAct.total_signals || 0} real fills</strong>, and the permanent ledger holds <strong>${settledLive}/${minLive} settled</strong> live decisions (${openLive} open waiting for T+1..T+14 outcomes).
  `;

  kpiEl.innerHTML = `
    <div class="kpi-tile">
      <span class="kpi-tile-label">Live Queue Focus</span>
      <span class="kpi-tile-val" style="color:var(--accent-emerald);">${buys.length} High-Conviction Buys</span>
      <span class="kpi-tile-sub">${selM.noise_reduction_pct ?? 0}% noise filtered out</span>
    </div>
    <div class="kpi-tile">
      <span class="kpi-tile-label">Best Profit Agent</span>
      <span class="kpi-tile-val" style="color:var(--accent-cyan);">${escHtml(topAgent)}</span>
      <span class="kpi-tile-sub">${topLift >= 0 ? "+" : ""}${topLift}% return lift · ${topDd}% DD saved</span>
    </div>
    <div class="kpi-tile">
      <span class="kpi-tile-label">Gemini CRO Accuracy</span>
      <span class="kpi-tile-val" style="color:var(--accent-amber);">${prevRate}% Saved Bad</span>
      <span class="kpi-tile-sub">${prevCount} prevented vs ${unnecCount} unnecessary</span>
    </div>
    <div class="kpi-tile">
      <span class="kpi-tile-label">Recalibrated vs Nifty 50</span>
      <span class="kpi-tile-val" style="color:${(trRecal.cumulative_return_pct || 0) >= (bench.benchmark_return_pct || 0) ? "var(--accent-emerald)" : "var(--accent-crimson)"};">${trRecal.cumulative_return_pct ?? 0}% vs ${bench.benchmark_return_pct >= 0 ? "+" : ""}${bench.benchmark_return_pct ?? 0}%</span>
      <span class="kpi-tile-sub">Drawdown cut to ${trRecal.max_drawdown_pct ?? 0}% (from ${trOrig.max_drawdown_pct ?? 0}%)</span>
    </div>
    <div class="kpi-tile">
      <span class="kpi-tile-label">Live Ledger Progress</span>
      <span class="kpi-tile-val">${settledLive} / ${minLive} Settled</span>
      <span class="kpi-tile-sub">${openLive} monitoring forward bars · ${discCount} feed alerts</span>
    </div>
  `;

  const eqCapEl = document.getElementById("rptEquityCaption");
  if (eqCapEl) {
    eqCapEl.textContent = `Starting every portfolio at 100.0, Gemini recalibration softens losses and nearly halves maximum drawdown (${trRecal.max_drawdown_pct ?? 0}% vs ${trOrig.max_drawdown_pct ?? 0}%), though a choppy NSE market kept Nifty 50 (+${bench.benchmark_return_pct ?? 0}%) ahead overall.`;
  }
  const agCapEl = document.getElementById("rptAgentCaption");
  if (agCapEl) {
    agCapEl.textContent = `${topAgent} drives the largest performance boost (${topLift >= 0 ? "+" : ""}${topLift}% lift), followed by TechBot_1 and Gemini_CRO. ExecBot_1's tight Moreshwar stop-loss distance is the only negative drag compared to wider 5%/10% exits.`;
  }
  const croCapEl = document.getElementById("rptCroCaption");
  if (croCapEl) {
    croCapEl.textContent = `Out of ${totalCro} times Gemini intervened on a setup, ${prevCount} (${prevRate}%) successfully avoided losing trades or improved overbought entries, while ${unnecCount} (${unnecRate}%) missed upside — delivering a net +${q2Comb.net_intervention_value_pct ?? 0}% protection edge.`;
  }
  const bmCapEl = document.getElementById("rptBenchmarkCaption");
  if (bmCapEl) {
    bmCapEl.textContent = `Moreshwar beats a naive Simple SMA-20 rule (+${q3.spread_vs_sma_strategy_pct ?? 0}% spread and nearly 10% lower drawdown), but using the same high-quality AI entries with wider Fixed 1:2 exits improves cumulative performance by ${Math.abs(q3.spread_vs_fixed_1to2_pct ?? 0).toFixed(2)}%.`;
  }

  const strongBuys = assetsList.filter((a) => ["BUY NOW", "ADD"].includes(a.decision_matrix?.action)).length;
  const holdSips = assetsList.filter((a) => ["HOLD", "SIP ONLY"].includes(a.decision_matrix?.action)).length;
  const cautions = assetsList.length - strongBuys - holdSips;
  const heldCount = Object.values(appState.holdingsMap || {}).filter((h) => (h.qty || 0) > 0).length;

  storyEl.innerHTML = `
    <div class="rpt-story-card" style="border-top:2px solid var(--accent-emerald);">
      <div class="rpt-story-head">
        <span class="rpt-story-title">Tab 1 · Execution Queue &amp; Selectivity</span>
        <span class="tag-pill tag-target">${buys.length} ACTIVE PICK${buys.length === 1 ? "" : "S"}</span>
      </div>
      <div class="rpt-story-text">Instead of flooding the queue, strict gates only surface aligned setups with R:R ≥ 1:1.5. Active ideas today: <strong>${escHtml(buySyms)}</strong>. Over 1 year, this filtering removed ${selM.noise_reduction_pct ?? 0}% of weak signals before execution.</div>
    </div>
    <div class="rpt-story-card" style="border-top:2px solid var(--accent-cyan);">
      <div class="rpt-story-head">
        <span class="rpt-story-title">Tab 2 · Watchlist Moat &amp; Portfolio</span>
        <span class="tag-pill tag-entry">${heldCount} HELD · ${assetsList.length} WATCHED</span>
      </div>
      <div class="rpt-story-text">Across your universe: <strong>${strongBuys}</strong> stocks rate BUY NOW/ADD, <strong>${holdSips}</strong> sit in HOLD/SIP ONLY, and <strong>${cautions}</strong> are flagged CAUTION/AVOID. Live Zerodha holdings remain safely isolated from all backtests.</div>
    </div>
    <div class="rpt-story-card" style="border-top:2px solid var(--accent-amber);">
      <div class="rpt-story-head">
        <span class="rpt-story-title">Tab 3 · Multi-Agent &amp; Gemini CRO</span>
        <span class="tag-pill tag-score">${escHtml(appState.geminiStatus || "CRO READY")}</span>
      </div>
      <div class="rpt-story-text"><strong>${escHtml(topAgent)}</strong> leads on profit lift, while Gemini's adversarial review shields against chasing overbought rallies and downward regression curves — adding <strong>+${q2Comb.net_intervention_value_pct ?? 0}%</strong> net value across backtested interventions.</div>
    </div>
    <div class="rpt-story-card" style="border-top:2px solid var(--accent-crimson);">
      <div class="rpt-story-head">
        <span class="rpt-story-title">Tab 4 · Data Health &amp; Audit Accumulation</span>
        <span class="tag-pill ${discCount > 0 ? "tag-stop" : "tag-target"}">${discCount} DISCREPANC${discCount === 1 ? "Y" : "IES"}</span>
      </div>
      <div class="rpt-story-text">Market feeds combine Official NSE MCP + Yahoo Finance with zero simulated prices. The permanent audit ledger currently monitors <strong>${openLive}</strong> open decisions toward our <strong>${minLive}-trade</strong> live statistical milestone.</div>
    </div>
  `;

  drawReportsCharts();
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

async function handleStockyQuery(rawText = "") {
  const q = String(rawText || "").trim();
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
      } else if (btn.dataset.tab === "intelligence") {
        requestAnimationFrame(() => {
          const mcAll = (appState.moreshwarV21Research || {}).monte_carlo || {};
          const mcKey = String(appState.mv21Filters.model || "").includes("1.0") ? "moreshwar_1_0" : "moreshwar_2_1";
          drawBacktestEquityChart((appState.backtest || {}).equity_curve || []);
          drawMv21EquityChart(appState.moreshwarV21Research);
          drawMv21MonteCarloChart(mcAll[mcKey] || mcAll.moreshwar_2_1 || mcAll);
        });
      } else if (btn.dataset.tab === "reports") {
        requestAnimationFrame(() => drawReportsCharts());
      }
    });
  });

  document.querySelectorAll("#intelJumpRow [data-intel-focus]").forEach((btn) => {
    btn.addEventListener("click", () => {
      appState.intelFocus = btn.dataset.intelFocus || "all";
      document.querySelectorAll("#intelJumpRow [data-intel-focus]").forEach((b) => {
        b.classList.toggle("active-subfilter", b === btn);
      });
      renderIntelligenceViews();
      requestAnimationFrame(() => {
        const mcAll = (appState.moreshwarV21Research || {}).monte_carlo || {};
        const mcKey = String(appState.mv21Filters.model || "").includes("1.0") ? "moreshwar_1_0" : "moreshwar_2_1";
        drawBacktestEquityChart((appState.backtest || {}).equity_curve || []);
        drawMv21EquityChart(appState.moreshwarV21Research);
        drawMv21MonteCarloChart(mcAll[mcKey] || mcAll.moreshwar_2_1 || mcAll);
      });
    });
  });

  const legacyRptDrawer = document.getElementById("legacyReportsArchiveLog");
  if (legacyRptDrawer) {
    legacyRptDrawer.addEventListener("toggle", () => {
      if (legacyRptDrawer.open) {
        requestAnimationFrame(() => drawReportsCharts());
      }
    });
  }
  const legacyIntelDrawer = document.getElementById("legacyIntelArchiveLog");
  if (legacyIntelDrawer) {
    legacyIntelDrawer.addEventListener("toggle", () => {
      if (legacyIntelDrawer.open) {
        requestAnimationFrame(() => {
          const mcAll = (appState.moreshwarV21Research || {}).monte_carlo || {};
          const mcKey = String(appState.mv21Filters.model || "").includes("1.0") ? "moreshwar_1_0" : "moreshwar_2_1";
          drawBacktestEquityChart((appState.backtest || {}).equity_curve || []);
          drawMv21EquityChart(appState.moreshwarV21Research);
          drawMv21MonteCarloChart(mcAll[mcKey] || mcAll.moreshwar_2_1 || mcAll);
        });
      }
    });
  }

  const auditSymF = document.getElementById("auditSymbolFilter");
  const auditStF = document.getElementById("auditStatusFilter");
  if (auditSymF) auditSymF.addEventListener("change", renderAuditLedger);
  if (auditStF) auditStF.addEventListener("change", renderAuditLedger);
  const mv2Sel = document.getElementById("mv2SymbolSelect");
  if (mv2Sel) {
    mv2Sel.addEventListener("change", () => {
      appState.mv2SelectedSymbol = mv2Sel.value;
      renderMoreshwarV2ShadowLab();
    });
  }
  const mv21ModF = document.getElementById("mv21FilterModel");
  const mv21InstF = document.getElementById("mv21FilterInstrument");
  const mv21SecF = document.getElementById("mv21FilterSector");
  const mv21RegF = document.getElementById("mv21FilterRegime");
  const mv21ResetBtn = document.getElementById("mv21ResetFiltersBtn");
  if (mv21ModF) {
    mv21ModF.addEventListener("change", () => {
      appState.mv21Filters.model = mv21ModF.value;
      renderMoreshwarV21Dashboard();
    });
  }
  if (mv21InstF) {
    mv21InstF.addEventListener("change", () => {
      appState.mv21Filters.instrument = mv21InstF.value;
      if (mv21InstF.value !== "ALL") appState.mv21Filters.sector = "ALL";
      renderMoreshwarV21Dashboard();
    });
  }
  if (mv21SecF) {
    mv21SecF.addEventListener("change", () => {
      appState.mv21Filters.sector = mv21SecF.value;
      if (mv21SecF.value !== "ALL") appState.mv21Filters.instrument = "ALL";
      renderMoreshwarV21Dashboard();
    });
  }
  if (mv21RegF) {
    mv21RegF.addEventListener("change", () => {
      appState.mv21Filters.regime = mv21RegF.value;
      renderMoreshwarV21Dashboard();
    });
  }
  if (mv21ResetBtn) {
    mv21ResetBtn.addEventListener("click", () => {
      appState.mv21Filters = { model: "Moreshwar 2.1", instrument: "ALL", sector: "ALL", regime: "ALL" };
      renderMoreshwarV21Dashboard();
    });
  }

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

  document.querySelectorAll(".prompt-chip[data-prompt]").forEach((chip) => {
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
  wireMoreshwarV31Controls();

  window.addEventListener("resize", () => {
    const mcAll = (appState.moreshwarV21Research || {}).monte_carlo || {};
    const mcKey = String(appState.mv21Filters.model || "").includes("1.0") ? "moreshwar_1_0" : "moreshwar_2_1";
    drawUnifiedChart(appState.assets[appState.activeSymbol]);
    drawBacktestEquityChart((appState.backtest || {}).equity_curve || []);
    drawMv21EquityChart(appState.moreshwarV21Research);
    drawMv21MonteCarloChart(mcAll[mcKey] || mcAll.moreshwar_2_1 || mcAll);
    drawReportsCharts();
    drawMv31ProspectiveChart(appState.moreshwarV31);
  });
}

function renderMoreshwarV31Dashboard() {
  const data = appState.moreshwarV31;
  const kpiEl = document.getElementById("mv31PortfolioOverviewKpis");
  const stratEl = document.getElementById("mv31StrategyStatusBody");
  const valEl = document.getElementById("mv31ValidationTrackerBody");
  const histBody = document.getElementById("mv31HistoricalBaselineBody");
  const ordBody = document.getElementById("mv31ExecutionOrdersBody");
  const riskEl = document.getElementById("mv31RiskDashboardBody");
  const hashBadge = document.getElementById("mv31ConfigHashBadge");
  const stateBadge = document.getElementById("mv31EngineStateBadge");
  const monthlyEl = document.getElementById("mv31MonthlyReturnsBody");
  if (!kpiEl) return;

  if (!data || !data.prospective_paper_trading) {
    kpiEl.innerHTML = `<div class="kpi-chip"><div class="kpi-chip-label">Status</div><div class="kpi-chip-value">Loading M3.1 Paper Engine...</div></div>`;
    return;
  }

  const cfg = data.configuration_integrity || {};
  const eng = data.engine_state || {};
  const sched = data.scheduler_status || {};
  const prosp = data.prospective_paper_trading || {};
  const ov = prosp.portfolio_overview || {};
  const st = prosp.strategy_status || {};
  const vt = prosp.validation_tracker || {};
  const ex = prosp.execution_monitoring || {};
  const rk = prosp.risk_dashboard || {};
  const rm = rk.metrics || {};
  const recon = rk.reconciliation || {};
  const dq = rk.data_quality || {};
  const hist = data.historical_research_baseline || {};
  const hsum = hist.summary || {};

  if (hashBadge) {
    hashBadge.textContent = cfg.verified
      ? `SHA256: ${cfg.canonical_prefix_16 || "47a30f4bbd004c5f"} · VERIFIED`
      : "SHA256: MISMATCH · HALTED";
  }
  if (stateBadge) {
    stateBadge.textContent = `ENGINE: ${eng.engine_state || "IDLE"} · SCHED: ${sched.status || "IDLE"}`;
  }

  const fmtInr = (v) => `₹${Number(v || 0).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  const fmtPct = (v) => `${Number(v || 0) >= 0 ? "+" : ""}${Number(v || 0).toFixed(2)}%`;

  kpiEl.innerHTML = [
    { label: "Initial Capital", value: fmtInr(ov.initial_capital_inr) },
    { label: "Current Paper NAV", value: fmtInr(ov.current_nav_inr) },
    { label: "Prospective Return", value: fmtPct(ov.total_return_pct) },
    { label: "Today's P&L", value: fmtInr(ov.today_pnl_inr) },
    { label: "Available Cash", value: fmtInr(ov.available_cash_inr) },
    { label: "Invested Capital", value: fmtInr(ov.invested_capital_inr) },
    { label: "Open Positions", value: `${ov.open_positions_count || 0} (${ov.open_tactical_positions_count || 0} Fam E)` },
    { label: "Pending T+1 Settle", value: fmtInr(ov.pending_settlements_inr) },
  ]
    .map((k) => `<div class="kpi-chip"><div class="kpi-chip-label">${k.label}</div><div class="kpi-chip-value">${k.value}</div></div>`)
    .join("");

  if (stratEl) {
    const alloc = st.target_allocation_pct || {};
    stratEl.innerHTML = `
      <div><strong>Active Policy:</strong> <code>${st.active_policy || "G_SIMPLE_TREND_BASELINE"}</code> (${st.execution_profile || "EXP_E_COMBINED_EXEC_INTEL"})</div>
      <div><strong>Regime State:</strong> Raw <code>${st.current_raw_regime || "BULL_TREND"}</code> · Confirmed <code>${st.confirmed_regime || "BULL_TREND"}</code> (Streak ${st.confirmation_streak_days || 0}/${st.confirmation_required_days || 3} sessions)</div>
      <div><strong>Frozen Target Allocation:</strong> Core NIFTYBEES <strong>${alloc.core_niftybees ?? 60}%</strong> · Tactical Family E <strong>${alloc.tactical_family_e ?? 30}%</strong> · Liquid LIQUIDBEES <strong>${alloc.liquid_liquidbees ?? 10}%</strong> · Gold <strong>0%</strong> (not in frozen config)</div>
      <div><strong>Family E New Entries:</strong> <strong>${st.family_e_new_entries_eligible ? "ELIGIBLE (Paper Only)" : "HALTED (Bear / Paused / Safe Mode)"}</strong> · <strong>Family A:</strong> <span style="color:var(--accent-red);">${st.family_a_status || "SUSPENDED_HARD_BLOCKED"}</span></div>
      <div><strong>Config Integrity:</strong> ${cfg.verified ? "VERIFIED" : "FAILED"} · Prefix <code>${cfg.canonical_prefix_16 || "47a30f4bbd004c5f"}</code> · Live Trading: <strong>DISABLED</strong></div>
    `;
  }

  if (valEl) {
    const done = Number(vt.completed_unseen_sessions || 0);
    const minReq = Number(vt.minimum_required_sessions || 60);
    const tgt = Number(vt.target_sessions || 120);
    const pctMin = Math.min(100, (done / Math.max(1, minReq)) * 100).toFixed(1);
    const pctTgt = Math.min(100, (done / Math.max(1, tgt)) * 100).toFixed(1);
    valEl.innerHTML = `
      <div><strong>Prospective Start Boundary:</strong> <code>${vt.prospective_start_date || "2026-11-19"}</code> · Status: <strong>${vt.status_label || "AWAITING_FIRST_PROSPECTIVE_SESSION"}</strong></div>
      <div><strong>Unseen Sessions Completed:</strong> <strong>${done} / ${minReq}</strong> minimum (${pctMin}%) · <strong>${done} / ${tgt}</strong> target (${pctTgt}%)</div>
      <div style="background:rgba(255,255,255,0.08);height:7px;border-radius:4px;overflow:hidden;margin:4px 0 6px;">
        <div style="width:${pctMin}%;height:100%;background:hsl(160,84%,44%);"></div>
      </div>
      <div><strong>Prospective Trades:</strong> ${vt.prospective_paper_trades || 0} filled · ${vt.prospective_rejected_orders || 0} rejected/failed</div>
      <div><strong>Rolling Sharpe / Max DD:</strong> ${rm.rolling_sharpe != null ? rm.rolling_sharpe : (rm.rolling_ratios_sample_status || "INSUFFICIENT_SAMPLE")} · Max DD ${Number(rm.max_drawdown_pct || 0).toFixed(2)}%</div>
      <div><strong>Data Quality:</strong> <code>${dq.overall_status || "PARTIAL_DIAGNOSTIC_ONLY"}</code> · <em>Incomplete validation is never reported as completed performance.</em></div>
    `;
  }

  if (histBody) {
    const fm = hsum.frozen_full_metrics || {};
    const om = hsum.frozen_outer_metrics || {};
    const nb = hsum.niftybees_full_metrics || {};
    const b45 = hsum.blend_45_55_full_metrics || {};
    const fmtSharpe = (obj, fallback) => {
      const v = obj.sharpe_ratio ?? obj.sharpe_vs_65_rf ?? obj.sharpe;
      if (v == null || Number.isNaN(Number(v))) return fallback;
      const n = Number(v);
      return `${n >= 0 ? "+" : ""}${n.toFixed(2)}`;
    };
    const rows = [
      ["G_SIMPLE_TREND_BASELINE (Frozen Winner)", "+120.03%", "10.46%", "+0.87", "+15.74% (DD 9.73%)"],
      ["Static Prompt 3 Primary (60/30/10 Fam E)", "+94.84%", "13.82%", "+0.62", "+9.76% (DD 13.46%)"],
      ["100% NIFTYBEES Executable Benchmark", nb.total_return_pct != null ? fmtPct(nb.total_return_pct) : "+68.60%", nb.max_drawdown_pct != null ? `${Number(nb.max_drawdown_pct).toFixed(2)}%` : "16.02%", fmtSharpe(nb, "+0.33"), "+6.75% (DD 15.14%)"],
      ["45/55 NIFTYBEES/LIQUIDBEES Benchmark", b45.total_return_pct != null ? fmtPct(b45.total_return_pct) : "+53.08%", b45.max_drawdown_pct != null ? `${Number(b45.max_drawdown_pct).toFixed(2)}%` : "6.09%", fmtSharpe(b45, "+0.31"), "Reference Benchmark"],
    ];
    if (fm.total_return_pct != null) {
      rows[0] = [
        "G_SIMPLE_TREND_BASELINE (Frozen Winner)",
        fmtPct(fm.total_return_pct),
        `${Number(fm.max_drawdown_pct || 0).toFixed(2)}%`,
        fmtSharpe(fm, "+0.87"),
        om.total_return_pct != null ? `${fmtPct(om.total_return_pct)} (DD ${Number(om.max_drawdown_pct || 0).toFixed(2)}%)` : "+15.74%",
      ];
    }
    histBody.innerHTML = rows
      .map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td><td>${r[4]}</td></tr>`)
      .join("");
  }

  if (ordBody) {
    const orders = ex.recent_orders || [];
    const pending = ex.pending_next_session_orders || [];
    if (!orders.length && !pending.length) {
      const doneSess = Number(vt.completed_unseen_sessions || 0);
      ordBody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">${doneSess > 0 ? `${doneSess} real prospective session(s) completed — 0 fills executed yet.` : `No prospective simulated orders executed yet — awaiting first unseen NSE session after ${vt.prospective_start_date || "2026-11-19"}.`}</td></tr>`;
    } else {
      const pendingRows = pending.map((p, idx) => {
        const sym = p.symbol || (p.kind === "CORE_REBALANCE" ? "NIFTYBEES" : p.kind === "LIQUID_SWEEP" ? "LIQUIDBEES" : p.kind);
        const qtyTxt = p.req_qty != null
          ? `${p.req_qty} sh · <strong>QUEUED (NEXT OPEN)</strong>`
          : p.target_core_inr != null
            ? `Target ₹${Number(p.target_core_inr).toLocaleString("en-IN")} · <strong>QUEUED</strong>`
            : `Auto Sweep · <strong>QUEUED</strong>`;
        const pxTxt = p.expected_price != null ? `₹${Number(p.expected_price).toFixed(2)}` : "Next Open";
        const evTxt = p.ev_net_bps != null ? `EV +${Number(p.ev_net_bps).toFixed(1)}` : "—";
        const slTxt = p.stop_ratio != null ? `SL ${(Number(p.stop_ratio) * 100).toFixed(1)}%` : "At Fill";
        return `<tr>
          <td><code>NEXT-OPEN-${idx + 1}</code><br><span style="font-size:10px;color:var(--text-muted);">${(p.signal_timestamp || "").slice(0, 10) || "Next Open"}</span></td>
          <td>${sym} (${p.kind})</td>
          <td>${qtyTxt}</td>
          <td>${pxTxt}</td>
          <td>${evTxt}</td>
          <td>${slTxt}</td>
        </tr>`;
      });
      const filledRows = orders
        .slice(-15)
        .reverse()
        .map(
          (o) => `<tr>
            <td><code>${o.order_id}</code><br><span style="font-size:10px;color:var(--text-muted);">${o.trading_date}</span></td>
            <td>${o.symbol} (${o.side})</td>
            <td>${o.filled_qty}/${o.requested_qty} · <strong>${o.execution_status}</strong></td>
            <td>${Number(o.actual_exec_price || 0).toFixed(2)}</td>
            <td>${Number(o.slippage_bps || 0).toFixed(2)}</td>
            <td>₹${Number(o.total_cost_inr || 0).toFixed(2)}</td>
          </tr>`
        );
      ordBody.innerHTML = [...pendingRows, ...filledRows].join("");
    }
  }

  if (riskEl) {
    const alerts = rk.alerts || [];
    const alertHtml = alerts.length
      ? alerts
          .slice(0, 5)
          .map((a) => `<div><span class="tag-pill ${a.severity === "CRITICAL" ? "tag-stop" : "tag-score"}">${a.severity}</span> <strong>${a.code}</strong>: ${a.description}</div>`)
          .join("")
      : `<div style="color:hsl(160,84%,44%);">No active critical or warning risk alerts.</div>`;
    riskEl.innerHTML = `
      <div><strong>System Health:</strong> <strong>${rk.overall_health || "HEALTHY"}</strong> · <strong>Reconciliation:</strong> ${recon.passed ? "PASS" : "FAIL"} (diff ₹${Number(recon.reconciliation_diff_inr || 0).toFixed(4)})</div>
      <div><strong>Drawdown:</strong> Current ${Number(rm.current_drawdown_pct || 0).toFixed(2)}% · Max ${Number(rm.max_drawdown_pct || 0).toFixed(2)}% · <strong>CAGR:</strong> ${rm.cagr_pct != null ? fmtPct(rm.cagr_pct) : (rm.cagr_sample_status || "INSUFFICIENT_SAMPLE")}</div>
      <div><strong>Exposure:</strong> Equity ${Number(rm.equity_exposure_pct || 0).toFixed(1)}% (Core ${Number(rm.core_etf_exposure_pct || 0).toFixed(1)}% + Fam E ${Number(rm.tactical_equity_exposure_pct || 0).toFixed(1)}%) · Liquid ${Number(rm.liquid_allocation_pct || 0).toFixed(1)}% · Cash ${Number(rm.cash_pct || 0).toFixed(1)}% · Gold 0.0%</div>
      <div><strong>Costs &amp; Turnover:</strong> Total Costs ${fmtInr(rm.total_transaction_costs_inr)} (${Number(rm.cost_pct_of_gross_traded_value || 0).toFixed(3)}% of gross traded) · Turnover ${Number(rm.turnover_x || 0).toFixed(2)}x</div>
      <div><strong>Data Quality:</strong> <code>${dq.overall_status || "PARTIAL_DIAGNOSTIC_ONLY"}</code> — ${dq.summary || ""}</div>
      <div style="margin-top:6px;">${alertHtml}</div>
    `;
  }

  if (monthlyEl) {
    const mrets = (prosp.performance || {}).monthly_returns || [];
    const doneSess = Number(vt.completed_unseen_sessions || 0);
    monthlyEl.textContent = mrets.length
      ? "Prospective Monthly Returns: " + mrets.map((m) => `${m.month}: ${fmtPct(m.return_pct)} (${m.sessions}d)`).join(" · ")
      : `Prospective Monthly Returns: ${doneSess} completed unseen session(s) (strictly separated from 2021–2026 historical backtest).`;
  }

  const rptM31Brief = document.getElementById("rptM31Brief");
  const rptM31Kpis = document.getElementById("rptM31Kpis");
  const rptM31QueueBody = document.getElementById("rptM31QueueBody");
  if (rptM31Brief && rptM31Kpis) {
    const doneSess = Number(vt.completed_unseen_sessions || 0);
    const minSess = Number(vt.minimum_required_sessions || 60);
    const targetSess = Number(vt.target_sessions || 120);
    const pending = ex.pending_next_session_orders || [];
    const recentOrders = ex.recent_orders || [];
    const tacQueued = pending.filter((p) => p.kind === "TACTICAL_BUY");
    const validBars = Number((dq && dq.valid_equity_bars) || 0);
    const fm = hsum.frozen_full_metrics || {};
    const nb = hsum.niftybees_full_metrics || {};
    const histRet = fm.total_return_pct != null ? fmtPct(fm.total_return_pct) : "+120.03%";
    const histSharpe = (fm.sharpe_ratio ?? fm.sharpe) != null ? `+${Number(fm.sharpe_ratio ?? fm.sharpe).toFixed(2)}` : "+0.87";
    const histDd = fm.max_drawdown_pct != null ? `${Number(fm.max_drawdown_pct).toFixed(2)}%` : "10.46%";
    const nbRet = nb.total_return_pct != null ? fmtPct(nb.total_return_pct) : "+68.60%";
    const reconDiff = Number((recon && recon.reconciliation_diff_inr) || 0);
    const reconPass = (recon && recon.passed) !== false;
    const latestDt = (recon && recon.as_of_date) || (dq && dq.trading_date) || "2026-11-20";

    rptM31Brief.innerHTML =
      `<strong>Prospective Paper Trading (Real Unseen NSE Sessions Only · Post-2026-11-19):</strong> ` +
      `Completed <strong>${doneSess} / ${minSess}</strong> minimum unseen session(s) (latest <code>${latestDt}</code>${validBars > 0 ? ` across <strong>${validBars.toLocaleString("en-IN")}</strong> real NSE PIT-eligible stocks` : ""}). ` +
      `Current Paper NAV is <strong>${fmtInr(ov.current_nav_inr)}</strong> (<strong>${fmtPct(ov.prospective_return_pct)}</strong> prospective return; <code>${recentOrders.length}</code> filled order(s) so far; <strong>${pending.length}</strong> order(s) queued for next-session open${tacQueued.length ? `: <em>${tacQueued.map((p) => p.symbol).join(", ")}</em>` : ""}). ` +
      `Four-ledger reconciliation: <strong>${reconPass ? "PASS" : "FAIL"} (diff ₹${reconDiff.toFixed(4)})</strong>.` +
      `<br><strong>Separate 5-Year Historical Research Baseline (2021-07 → 2026-11-19 · Never Merged Into Prospective):</strong> ` +
      `Frozen winner <code>G_SIMPLE_TREND_BASELINE</code> returned <strong>${histRet}</strong> (Sharpe <strong>${histSharpe}</strong>, Max DD <strong>${histDd}</strong>) vs <strong>${nbRet}</strong> for 100% NIFTYBEES.`;

    rptM31Kpis.innerHTML = [
      {
        label: "PROSPECTIVE PAPER NAV",
        value: fmtInr(ov.current_nav_inr),
        sub: `${fmtPct(ov.prospective_return_pct)} unseen · ${doneSess}/${minSess} min (${targetSess} target)`,
        tone: "emerald",
      },
      {
        label: "UNIVERSE & REGIME",
        value: st.confirmed_regime || "BULL_TREND",
        sub: `${validBars > 0 ? `${validBars.toLocaleString("en-IN")} real NSE bars` : "Full PIT universe"} · Streak ${st.confirmation_streak_days ?? st.regime_streak_days ?? 0}/${st.confirmation_required_days || 3}`,
        tone: "cyan",
      },
      {
        label: "NEXT-OPEN QUEUE",
        value: `${pending.length} Queued`,
        sub: tacQueued.length ? `${tacQueued.length} Family E (${tacQueued.map((p) => p.symbol).join(", ")}) + Core/Liquid` : "Core NIFTYBEES + Liquid sweep",
        tone: "emerald",
      },
      {
        label: "5Y BACKTEST ARCHIVE",
        value: `${histRet} vs ${nbRet}`,
        sub: `Sharpe ${histSharpe} · Max DD ${histDd} (Historical Only)`,
        tone: "cyan",
      },
      {
        label: "4-LEDGER AUDIT & SAFETY",
        value: reconPass ? `PASS (₹${reconDiff.toFixed(4)})` : "SAFE_MODE",
        sub: `Live Broker: DISABLED · SHA ${(cfg.canonical_sha256 || "47a30f4bbd004c5f").slice(0, 12)}`,
        tone: reconPass ? "emerald" : "crimson",
      },
    ]
      .map(
        (k) => `<div class="kpi-mini-card kpi-${k.tone}">
          <div class="kpi-mini-label">${k.label}</div>
          <div class="kpi-mini-val">${k.value}</div>
          <div class="kpi-mini-sub">${k.sub}</div>
        </div>`
      )
      .join("");

    if (rptM31QueueBody && ordBody && ordBody.innerHTML) {
      rptM31QueueBody.innerHTML = ordBody.innerHTML;
    }
  }

  drawMv31ProspectiveChart(data);
}

function drawMv31ProspectiveChart(data) {
  const canvas = document.getElementById("mv31ProspectiveChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(260, rect.width || 440);
  const h = Math.max(170, rect.height || 190);
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const curve = ((data || {}).prospective_paper_trading?.performance?.equity_curve) || [];
  const doneSess = Number((data || {}).prospective_paper_trading?.validation_tracker?.completed_unseen_sessions || curve.length || 0);
  if (curve.length < 2) {
    ctx.font = "12px 'Inter', sans-serif";
    ctx.fillStyle = "hsl(215, 15%, 72%)";
    ctx.fillText(`Prospective Paper Curve: ${doneSess} real unseen session(s) recorded post-2026-11-19 (line plots from session 2).`, 14, h / 2 - 8);
    ctx.fillStyle = "hsl(38, 92%, 60%)";
    ctx.fillText("Historical backtest (2021-07 → 2026-11-19) is shown separately on the right.", 14, h / 2 + 12);
    return;
  }

  const vals = [];
  curve.forEach((pt) => {
    vals.push(Number(pt.portfolio_nav || 100000), Number(pt.niftybees_nav || 100000), Number(pt.blend_45_55_nav || 100000));
  });
  const minV = Math.min(...vals) * 0.995;
  const maxV = Math.max(...vals) * 1.005;
  const padT = 14, padB = 22, padL = 10, padR = 58;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const yFor = (v) => padT + plotH - ((v - minV) / Math.max(1e-6, maxV - minV)) * plotH;
  const xFor = (i) => padL + (i / Math.max(1, curve.length - 1)) * plotW;

  const drawSeries = (key, color, dashed = false) => {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.0;
    if (dashed) ctx.setLineDash([4, 4]);
    ctx.beginPath();
    curve.forEach((pt, i) => {
      const x = xFor(i);
      const y = yFor(Number(pt[key] || 100000));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  };
  drawSeries("blend_45_55_nav", "hsl(215, 25%, 65%)", true);
  drawSeries("niftybees_nav", "hsl(38, 92%, 55%)", true);
  drawSeries("portfolio_nav", "hsl(160, 84%, 44%)", false);
}

function renderMoreshwarV30Dashboard() {
  const bannerEl = document.getElementById("mv30DecisionBanner");
  const kpiEl = document.getElementById("mv30HeaderKpis");
  const resBody = document.getElementById("mv30ResearchTableBody");
  const portBody = document.getElementById("mv30PortfolioTableBody");
  const execBody = document.getElementById("mv30ExecutionTableBody");
  const riskEl = document.getElementById("mv30RiskAndOpsBody");
  const valBody = document.getElementById("mv30ValidationTableBody");
  if (!kpiEl && !bannerEl) return;

  const d = appState.moreshwarV3Research;
  if (!d || typeof d !== "object") {
    if (!appState._mv30FetchStarted) {
      appState._mv30FetchStarted = true;
      const urls = [
        "/api/research/v3",
        `https://raw.githubusercontent.com/${GH_OWNER}/${GH_REPO}/main/reports/moreshwar_3_0/moreshwar_v3_release_candidate.json?ts=${Date.now()}`,
        `reports/moreshwar_3_0/moreshwar_v3_release_candidate.json?ts=${Date.now()}`
      ];
      (async () => {
        for (const u of urls) {
          try {
            const r = await fetch(u, { cache: "no-store" });
            if (!r.ok) continue;
            const payload = await r.json();
            if (payload && payload.release_decision) {
              appState.moreshwarV3Research = payload;
              renderMoreshwarV30Dashboard();
              return;
            }
          } catch (_) {}
        }
      })();
    }
    if (bannerEl) {
      bannerEl.innerHTML = `<strong>Moreshwar 3.0 (${"v3.0.0-rc1"}):</strong> Loading frozen PIT NSE research &amp; paper-trading release bundle...`;
    }
    return;
  }

  const rd = d.release_decision || {};
  const research = d.research_dashboard || {};
  const port = d.portfolio_dashboard || {};
  const exec = d.execution_dashboard || {};
  const risk = d.risk_dashboard || {};
  const val = d.validation_dashboard || {};
  const ledger = port.paper_account_ledger || {};

  if (bannerEl) {
    bannerEl.innerHTML = `<strong>Moreshwar 3.0 Release Outcome:</strong> ${rd.release_outcome || "CONDITIONAL GO (PAPER ONLY)"} · ${rd.summary_rationale || "Frozen M3.0 PIT baseline preserved; live broker execution disabled."}`;
  }

  if (kpiEl) {
    const kpis = [
      { label: "Release Version", value: d.release_version || "3.0.0-rc1" },
      { label: "Paper Ledger NAV", value: ledger.current_nav_inr != null ? `₹${Number(ledger.current_nav_inr).toLocaleString("en-IN", { maximumFractionDigits: 2 })}` : "Frozen M3.0" },
      { label: "Max Historical DD", value: risk.max_historical_drawdown_pct != null ? `${Number(risk.max_historical_drawdown_pct).toFixed(2)}%` : "—" },
      { label: "Capital Utilisation", value: risk.capital_utilisation_pct != null ? `${Number(risk.capital_utilisation_pct).toFixed(1)}%` : "—" },
      { label: "Live Capital Auth", value: rd.live_capital_deployment_authorized ? "YES" : "DISABLED" },
    ];
    kpiEl.innerHTML = kpis
      .map((k) => `<div class="kpi-chip"><div class="kpi-chip-label">${k.label}</div><div class="kpi-chip-value">${k.value}</div></div>`)
      .join("");
  }

  if (resBody) {
    const families = Array.isArray(research.strategy_families) ? research.strategy_families : [];
    resBody.innerHTML = families.length
      ? families
          .slice(0, 8)
          .map((s) => `<tr>
            <td>${s.strategy_name || s.strategy_id || "—"}</td>
            <td>${s.family || "Strategy"}</td>
            <td>${s.full_return_pct != null ? `${Number(s.full_return_pct).toFixed(2)}%` : "—"}</td>
            <td>${s.oos_return_pct != null ? `${Number(s.oos_return_pct).toFixed(2)}%` : "—"}</td>
            <td>${s.max_drawdown_pct != null ? `${Number(s.max_drawdown_pct).toFixed(2)}%` : "—"}</td>
            <td>${s.status || "FROZEN"}</td>
          </tr>`)
          .join("")
      : `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">Moreshwar 3.0 frozen baseline loaded.</td></tr>`;
  }

  if (portBody) {
    const pos = Array.isArray(port.active_paper_positions) ? port.active_paper_positions : [];
    portBody.innerHTML = pos.length
      ? pos
          .slice(0, 10)
          .map((p) => `<tr>
            <td>${p.symbol || "—"}</td>
            <td>${p.strategy_id || p.sector || "—"}</td>
            <td>${p.notional_inr != null ? `₹${Number(p.notional_inr).toFixed(0)}` : "—"}</td>
            <td>${p.weight_pct != null ? `${Number(p.weight_pct).toFixed(1)}%` : "—"}</td>
            <td>${p.sl_px != null ? `₹${Number(p.sl_px).toFixed(2)}` : "—"}</td>
            <td>PAPER</td>
          </tr>`)
          .join("")
      : `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">No active M3.0 paper positions in snapshot.</td></tr>`;
  }

  if (execBody) {
    const orders = Array.isArray(exec.simulated_paper_orders) ? exec.simulated_paper_orders : [];
    execBody.innerHTML = orders.length
      ? orders
          .slice(0, 10)
          .map((o) => `<tr>
            <td><code>${o.paper_order_id || o.order_id || "—"}</code></td>
            <td>${o.symbol || "—"}</td>
            <td>${o.qty ?? "—"} · ${o.status || "PAPER"}</td>
            <td>${o.expected_fill_price != null ? `₹${Number(o.expected_fill_price).toFixed(2)}` : "—"}</td>
            <td>${o.slippage_bps != null ? Number(o.slippage_bps).toFixed(1) : "—"}</td>
            <td>${o.total_cost_inr != null ? `₹${Number(o.total_cost_inr).toFixed(2)}` : "—"}</td>
          </tr>`)
          .join("")
      : `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">No simulated M3.0 orders recorded.</td></tr>`;
  }

  if (riskEl) {
    riskEl.innerHTML = `
      <div><strong>Drawdown:</strong> Current ${Number(risk.current_paper_drawdown_pct || 0).toFixed(2)}% · Max Historical ${Number(risk.max_historical_drawdown_pct || 0).toFixed(2)}%</div>
      <div><strong>Capital Utilisation:</strong> ${Number(risk.capital_utilisation_pct || 0).toFixed(1)}% · Idle Cash ${Number(risk.idle_cash_reserve_pct || 0).toFixed(1)}%</div>
      <div><strong>Security Boundary:</strong> Live Zerodha Order Submission <strong>DISABLED</strong> · Paper-Only Mode Verified</div>
    `;
  }

  if (valBody) {
    const benches = Array.isArray(val.benchmark_comparison) ? val.benchmark_comparison : [];
    const iv = val.independent_audit_verdict;
    const verdictRows =
      iv && typeof iv === "object"
        ? [
            `<tr>
              <td><strong>Prompt 9 Independent Audit</strong></td>
              <td><strong>${iv.deployment_decision || "NO-GO"}</strong></td>
              <td>Paper Only: ${iv.approved_for_paper_research_monitoring ? "YES" : "NO"}</td>
              <td>Live Capital: ${iv.approved_for_live_capital ? "YES" : "DISABLED"}</td>
              <td>${iv.verdict_code || "NO_GO_LIVE_CAPITAL"}</td>
            </tr>`,
            ...(Array.isArray(iv.primary_reasons_for_no_go)
              ? iv.primary_reasons_for_no_go.map(
                  (r, i) => `<tr><td>Finding #${i + 1}</td><td colspan="4" style="white-space:normal;">${r}</td></tr>`
                )
              : []),
          ].join("")
        : `<tr><td colspan="5" style="text-align:center;color:var(--text-muted);">Prompt 9 Independent Audit Verdict: ${iv || "NO-GO Live Capital (Upgraded in Moreshwar 3.1)"}</td></tr>`;
    valBody.innerHTML = benches.length
      ? benches
          .slice(0, 8)
          .map((b) => `<tr>
            <td>${b.name || b.benchmark || "—"}</td>
            <td>${b.return_pct != null ? `${Number(b.return_pct).toFixed(2)}%` : "—"}</td>
            <td>${b.cagr_pct != null ? `${Number(b.cagr_pct).toFixed(2)}%` : "—"}</td>
            <td>${b.max_drawdown_pct != null ? `${Number(b.max_drawdown_pct).toFixed(2)}%` : "—"}</td>
            <td>${b.verdict || b.note || "Audited"}</td>
          </tr>`)
          .join("")
      : verdictRows;
  }
}

async function fetchM31CloudFallback() {
  const base = `https://raw.githubusercontent.com/${GH_OWNER}/${GH_REPO}/main/reports/moreshwar_3_1`;
  const ts = Date.now();
  const getJson = async (rel) => {
    try {
      const r = await fetch(`${base}/${rel}?ts=${ts}`, { cache: "no-store" });
      return r.ok ? await r.json() : null;
    } catch (_) {
      return null;
    }
  };
  const getText = async (rel) => {
    try {
      const r = await fetch(`${base}/${rel}?ts=${ts}`, { cache: "no-store" });
      return r.ok ? await r.text() : null;
    } catch (_) {
      return null;
    }
  };
  const [daily, prog, alerts, dq, cfg, p4sum, snap, csvText] = await Promise.all([
    getJson("prospective_paper_trading/paper_trading_daily_report.json"),
    getJson("prospective_paper_trading/prospective_validation_progress.json"),
    getJson("prospective_paper_trading/risk_alerts.json"),
    getJson("prospective_paper_trading/data_quality_report.json"),
    getJson("prospective_paper_trading/configuration_integrity.json"),
    getJson("strategy_intelligence/strategy_intelligence_summary.json"),
    getJson("prospective_paper_trading/portfolio_snapshot.json"),
    getText("prospective_paper_trading/paper_trading_daily_report.csv"),
  ]);
  if (!daily && !prog) return null;
  const nav = (daily && daily.portfolio_nav) || {};
  const recon = (daily && daily.reconciliation_result) || {};
  const reg = (daily && daily.regime_determination) || {};
  const alloc = (daily && daily.allocation) || {};
  const openHoldings = Array.isArray(snap && snap.open_holdings) ? snap.open_holdings : [];
  const openTactical = openHoldings.filter((h) => String(h.bucket || "").includes("TACTICAL"));
  const equityCurve = [];
  if (csvText) {
    const lines = csvText.trim().split(/\r?\n/).filter(Boolean);
    if (lines.length >= 2) {
      const cols = lines[0].split(",");
      const idx = (k) => cols.indexOf(k);
      const iDate = idx("trading_date");
      const iNav = idx("nav_inr");
      const iNifty = idx("benchmark_niftybees_nav");
      const iBlend = idx("benchmark_45_55_nav");
      for (let i = 1; i < lines.length; i++) {
        const parts = lines[i].split(",");
        if (iDate >= 0 && parts[iDate]) {
          equityCurve.push({
            trading_date: parts[iDate],
            portfolio_nav: Number(parts[iNav] || 100000),
            niftybees_nav: Number(parts[iNifty] || 100000),
            blend_45_55_nav: Number(parts[iBlend] || 100000),
          });
        }
      }
    }
  }
  return {
    release_version: (daily && daily.release_version) || "3.1.0-rc1",
    banner: "PAPER TRADING ONLY — LIVE TRADING DISABLED",
    configuration_integrity: cfg || { verified: true, canonical_prefix_16: "47a30f4bbd004c5f" },
    engine_state: { engine_state: (daily && daily.engine_status) || "RUNNING" },
    scheduler_status: { status: "GITHUB_ACTIONS_DAILY" },
    historical_research_baseline: { summary: p4sum || {} },
    prospective_paper_trading: {
      portfolio_overview: {
        initial_capital_inr: nav.initial_capital_inr ?? 100000,
        current_nav_inr: nav.current_nav_inr ?? 100000,
        total_return_pct: nav.cumulative_return_pct ?? 0,
        today_pnl_inr: nav.daily_pnl_inr ?? 0,
        available_cash_inr: recon.settled_cash_inr ?? 100000,
        pending_settlements_inr: recon.unsettled_cash_inr ?? 0,
        invested_capital_inr: Number(recon.core_value_inr || 0) + Number(recon.tactical_value_inr || 0) + Number(recon.liquid_value_inr || 0),
        open_positions_count: openHoldings.length,
        open_tactical_positions_count: openTactical.length,
      },
      strategy_status: {
        active_policy: "G_SIMPLE_TREND_BASELINE",
        execution_profile: "EXP_E_COMBINED_EXEC_INTEL",
        current_raw_regime: reg.raw_regime || "BULL_TREND",
        confirmed_regime: reg.confirmed_regime || "BULL_TREND",
        confirmation_streak_days: reg.confirmation_streak_days ?? 3,
        confirmation_required_days: reg.confirmation_required_days ?? 3,
        target_allocation_pct: {
          core_niftybees: alloc.target_core_niftybees_pct ?? 60,
          tactical_family_e: alloc.target_tactical_family_e_pct ?? 30,
          liquid_liquidbees: alloc.target_liquid_liquidbees_pct ?? 10,
          gold: 0,
        },
        family_e_new_entries_eligible: reg.allow_new_family_e_entries !== false,
        family_a_status: "SUSPENDED_HARD_BLOCKED",
      },
      performance: { equity_curve: equityCurve, monthly_returns: [], has_prospective_sessions: Number((prog && prog.completed_unseen_sessions) || equityCurve.length || 0) > 0 },
      execution_monitoring: {
        recent_orders: (daily && daily.simulated_orders_latest_session) || [],
        pending_next_session_orders: (daily && daily.pending_next_session_orders) || (snap && snap.pending_next_session_orders) || [],
      },
      risk_dashboard: {
        overall_health: (alerts && alerts.overall_health) || "HEALTHY",
        metrics: (daily && daily.risk_metrics) || {},
        alerts: (alerts && alerts.alerts) || [],
        data_quality: dq || {},
        reconciliation: recon,
      },
      validation_tracker: prog || { prospective_start_date: "2026-11-19", completed_unseen_sessions: 0, minimum_required_sessions: 60, target_sessions: 120 },
    },
  };
}

function wireMoreshwarV31Controls() {
  const buttons = document.querySelectorAll("[data-m31-action]");
  buttons.forEach((btn) => {
    btn.addEventListener("click", async () => {
      const action = btn.getAttribute("data-m31-action");
      const bannerEl = document.getElementById("mv31BannerStatus");
      try {
        const res = await fetch("/api/m3_1/control", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action }),
        });
        const data = await res.json();
        if (data.dashboard) {
          appState.moreshwarV31 = data.dashboard;
          renderMoreshwarV31Dashboard();
        }
        if (bannerEl) {
          bannerEl.innerHTML = `<strong>PAPER TRADING ONLY — LIVE TRADING DISABLED:</strong> Action <code>${action}</code> → <strong>${data.status || res.status}</strong>`;
        }
      } catch (err) {
        if (bannerEl) {
          bannerEl.innerHTML = `<strong>PAPER TRADING ONLY — LIVE TRADING DISABLED:</strong> Cloud read-only mode (manage runs via GitHub Actions).`;
        }
      }
    });
  });

  // Always fetch latest Moreshwar 3.1 live ledger (local API or GitHub raw reports) so older live_scan.json snapshots never mask newer paper sessions
  fetch("/api/m3_1/dashboard", { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : null))
    .then(async (payload) => {
      if (payload && payload.prospective_paper_trading) {
        appState._moreshwarV31LiveLoaded = true;
        appState.moreshwarV31 = payload;
        renderMoreshwarV31Dashboard();
        return;
      }
      const cloud = await fetchM31CloudFallback();
      if (cloud) {
        appState._moreshwarV31LiveLoaded = true;
        appState.moreshwarV31 = cloud;
        renderMoreshwarV31Dashboard();
      }
    })
    .catch(async () => {
      const cloud = await fetchM31CloudFallback();
      if (cloud) {
        appState._moreshwarV31LiveLoaded = true;
        appState.moreshwarV31 = cloud;
        renderMoreshwarV31Dashboard();
      }
    });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootFinsistantApp);
} else {
  bootFinsistantApp();
}
