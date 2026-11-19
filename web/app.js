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
  const secAud = document.getElementById("intelSectionAudit");
  const focus = appState.intelFocus || "all";
  if (secProv) secProv.style.display = (focus === "all" || focus === "provenance") ? "flex" : "none";
  if (secBt) secBt.style.display = (focus === "all" || focus === "backtest") ? "flex" : "none";
  if (secAud) secAud.style.display = (focus === "all" || focus === "audit") ? "flex" : "none";

  renderProvenanceMonitor();
  renderBacktestAnalytics();
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
        requestAnimationFrame(() => drawBacktestEquityChart((appState.backtest || {}).equity_curve || []));
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
      requestAnimationFrame(() => drawBacktestEquityChart((appState.backtest || {}).equity_curve || []));
    });
  });

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

  window.addEventListener("resize", () => {
    drawUnifiedChart(appState.assets[appState.activeSymbol]);
    drawBacktestEquityChart((appState.backtest || {}).equity_curve || []);
    drawReportsCharts();
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootFinsistantApp);
} else {
  bootFinsistantApp();
}
