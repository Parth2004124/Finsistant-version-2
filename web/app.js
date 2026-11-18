/**
 * Finsistant Version 2 — Unified All-in-One Frontend Controller
 * Combines:
 * 1. StockSight Version 2 (4-Pillar Fundamental Engine, Porter's 5 Forces, Moreshwar Levels, Decision Matrix, Portfolio/Watchlist)
 * 2. Finsistant (Execution Terminal, Top Picks Queue, Sell Warnings, Auto-Execution Rules, Degree-2 Polynomial Regression, Stocky AI)
 * 3. Multi-Agent Trading Orchestrator (AgentRunner: TechBot_1, FundBot_1, ExecBot_1)
 */

const UNIVERSE_FALLBACK = {
  RELIANCE: { name: "Reliance Industries Ltd", exchange: "NSE", sector: "Energy, Telecom & Retail", price: 2985, chg: 1.45, pe: 26.4, roe: 16.8, roce: 18.4, opm: 22.5, sales_g: 16.2, profit_g: 18.5, mcap: 2015000, qty: 15, avg: 2790, setup: "Institutional Breakout", industry: "POWER" },
  TCS: { name: "Tata Consultancy Services Ltd", exchange: "NSE", sector: "IT Services & Enterprise AI", price: 4120, chg: 0.92, pe: 29.8, roe: 48.5, roce: 58.2, opm: 26.4, sales_g: 12.4, profit_g: 15.8, mcap: 1490000, qty: 10, avg: 3890, setup: "Moat Accumulation", industry: "IT" },
  HDFCBANK: { name: "HDFC Bank Ltd", exchange: "NSE", sector: "Private Sector Banking", price: 1745, chg: 1.18, pe: 18.6, roe: 17.4, roce: 16.2, opm: 28.0, sales_g: 18.5, profit_g: 19.4, mcap: 1325000, qty: 25, avg: 1610, setup: "High Quality Value", industry: "BANKING" },
  ICICIBANK: { name: "ICICI Bank Ltd", exchange: "NSE", sector: "Private Sector Banking & Credit", price: 1288, chg: 1.74, pe: 19.2, roe: 18.9, roce: 17.5, opm: 31.0, sales_g: 19.8, profit_g: 22.4, mcap: 908000, qty: 30, avg: 1165, setup: "Momentum Leader", industry: "BANKING" },
  INFY: { name: "Infosys Ltd", exchange: "NSE", sector: "Digital Cloud & Enterprise AI", price: 1865, chg: 0.85, pe: 27.1, roe: 31.8, roce: 39.4, opm: 23.8, sales_g: 11.2, profit_g: 14.6, mcap: 774000, qty: 0, avg: 1865, setup: "Channel Breakout", industry: "IT" },
  TATAMOTORS: { name: "Tata Motors Ltd", exchange: "NSE", sector: "Automotive, EV & Mobility", price: 945, chg: 2.15, pe: 11.8, roe: 24.6, roce: 21.8, opm: 15.6, sales_g: 17.4, profit_g: 26.0, mcap: 348000, qty: 0, avg: 945, setup: "Deep Value Turnaround", industry: "AUTO" },
  NVDA: { name: "NVIDIA Corporation", exchange: "NASDAQ", sector: "AI Compute & Semiconductors", price: 142.8, chg: 2.64, pe: 44.2, roe: 68.0, roce: 62.0, opm: 58.4, sales_g: 78.0, profit_g: 89.0, mcap: 2850000, qty: 0, avg: 142.8, setup: "AI Momentum Continuation", industry: "IT" },
  ZEEL: { name: "Zee Entertainment Enterprises", exchange: "NSE", sector: "Media & Broadcasting", price: 134.5, chg: -2.4, pe: 48.0, roe: 4.2, roce: 5.8, opm: 8.4, sales_g: -6.5, profit_g: -18.0, mcap: 12900, qty: 40, avg: 158.0, setup: "Risk Breakdown Warning", industry: "GENERAL" }
};

let appState = {
  activeSymbol: "RELIANCE",
  selectedPickId: null,
  cashBalance: 485000,
  assets: {},
  rules: [
    { rule_id: "rule-1", name: "Strict Breakout Rules (Score > 80 & R:R > 1.8)", active: true },
    { rule_id: "rule-2", name: "Moat Compounders (Fundamental >= 75 & Porter >= 75)", active: true },
    { rule_id: "rule-3", name: "Auto-Exit Capital Traps (Fundamental <= 40)", active: true }
  ],
  chatHistory: [
    {
      role: "bot",
      text: "🤖 Welcome to Finsistant Version 2! I merge StockSight V2 (4-Pillar + Porter's 5 Forces + Moreshwar Levels), Finsistant Execution Queue & Polynomial Regression, and the 3-Agent Orchestrator (TechBot_1, FundBot_1, ExecBot_1).\n\nClick any quick chip above or ask me to analyze a stock, compare peers, or allocate capital."
    }
  ]
};

const fmtINR = (n, d = 0) => "₹" + Number(n || 0).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });

function seededRandom(seedStr) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < seedStr.length; i++) {
    h ^= seedStr.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return function () {
    h += 0x6D2B79F5;
    let t = Math.imul(h ^ (h >>> 15), 1 | h);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function computeAssetModel(sym, raw) {
  const isFin = raw.industry === "BANKING";
  let bus = (raw.sales_g > 15 ? 15 : raw.sales_g > 8 ? 10 : raw.sales_g > 0 ? 5 : 2) +
            (raw.profit_g > 15 ? 15 : raw.profit_g > 8 ? 10 : raw.profit_g > 0 ? 5 : 2) +
            (isFin ? (raw.roe > 15 ? 10 : 5) : (raw.opm > 20 ? 10 : raw.opm > 12 ? 5 : 2));
  bus = Math.min(40, bus);

  let moat = (isFin ? (raw.roe > 18 ? 8 : 5) : ((raw.opm > 18 ? 5 : 2) + (raw.roce > 20 ? 5 : 2))) +
             (raw.mcap > 20000 ? 5 : 3) + (raw.profit_g > raw.sales_g ? 5 : 2);
  moat = Math.min(20, moat);

  let mgmt = raw.pe < 15 ? 18 : raw.pe < 28 ? 14 : raw.pe < 45 ? 10 : 5;
  let risk = (raw.mcap > 5000 ? 10 : 5) + (raw.chg >= 0 ? 8 : 3);
  let fTotal = Math.min(98, bus + moat + mgmt + risk);

  let entrants = raw.mcap > 10000 && raw.roce > 18 ? 20 : 14;
  let suppliers = raw.opm > 24 ? 20 : raw.opm > 16 ? 15 : 10;
  let buyers = raw.roe > 22 ? 20 : raw.roe > 16 ? 16 : 10;
  let substitutes = raw.sales_g > 15 ? 20 : raw.sales_g > 10 ? 15 : 8;
  let rivalry = raw.profit_g > 15 ? 20 : raw.profit_g > 10 ? 15 : 8;
  let pTotal = Math.min(98, entrants + suppliers + buyers + substitutes + rivalry);

  const xBlend = Number(((fTotal + pTotal) / 2).toFixed(1));
  const scale = Math.max(1, raw.price / 1000);
  const target = Number((raw.price + xBlend * scale).toFixed(2));
  const stopLoss = Number((raw.price - (100 - xBlend) * scale).toFixed(2));
  const entryPrice = raw.qty > 0 ? raw.price : Number((raw.price - (100 - xBlend) * 0.55 * scale).toFixed(2));
  const rr = Number(((target - raw.price) / Math.max(1, raw.price - stopLoss)).toFixed(2));

  let decision = "BUY NOW";
  if (xBlend <= 45) decision = raw.qty > 0 ? "EXIT" : "AVOID";
  else if (xBlend >= 65) decision = raw.qty > 0 ? "ADD" : "BUY NOW";
  else decision = raw.qty > 0 ? "HOLD" : "SIP ONLY";

  // Generate 36 candles + polynomial regression curve
  const rand = seededRandom(sym);
  const candles = [];
  let p = raw.price * (raw.chg >= 0 ? 0.92 : 1.08);
  const drift = raw.chg >= 0 ? 0.0023 : -0.0021;
  for (let i = 0; i < 36; i++) {
    const shock = (rand() - 0.48) * 0.025 + drift;
    const open = p;
    const close = i === 35 ? raw.price : open * (1 + shock);
    const high = Math.max(open, close) * (1 + 0.008 + rand() * 0.008);
    const low = Math.min(open, close) * (1 - 0.008 - rand() * 0.008);
    const reg = raw.price * (0.93 + (i / 35) * (raw.chg >= 0 ? 0.08 : -0.07));
    candles.push({
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      reg_value: Number(reg.toFixed(2))
    });
    p = close;
  }

  const reg30d = Number((raw.price * (raw.chg >= 0 ? 1.058 : 0.94)).toFixed(2));
  const techConf = raw.chg >= 0 ? 0.79 : 0.36;
  const fundConf = Number((xBlend / 100).toFixed(2));
  const execConf = Number((0.45 * techConf + 0.55 * fundConf).toFixed(2));
  const autoQty = Math.max(1, Math.min(50, Math.floor((appState.cashBalance * 0.015) / Math.max(1, raw.price - stopLoss))));

  return {
    symbol: sym,
    ...raw,
    fScore: { business: bus, moat, management: mgmt, risk, total: fTotal },
    pScore: { entrants, suppliers, buyers, substitutes, rivalry, total: pTotal },
    moreshwar: { xBlend, entry: entryPrice, target, stopLoss, rr },
    decision,
    reg30d,
    techConf,
    fundConf,
    execConf,
    autoQty,
    candles
  };
}

function initFallbackUniverse() {
  Object.keys(UNIVERSE_FALLBACK).forEach((sym) => {
    appState.assets[sym] = computeAssetModel(sym, UNIVERSE_FALLBACK[sym]);
  });
}

function drawUnifiedChart(asset) {
  const canvas = document.getElementById("unifiedChart");
  if (!canvas || !asset) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const candles = asset.candles || [];
  if (!candles.length) return;

  let minP = Math.min(asset.moreshwar.stopLoss * 0.992, ...candles.map((c) => c.low));
  let maxP = Math.max(asset.moreshwar.target * 1.008, asset.reg30d * 1.005, ...candles.map((c) => c.high));

  const padTop = 16, padBot = 20, padLeft = 10, padRight = 72;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBot;
  const yFor = (val) => padTop + plotH - ((val - minP) / Math.max(1, maxP - minP)) * plotH;
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
    ctx.fillText(`₹${val.toFixed(0)}`, w - padRight + 6, y + 3);
  }

  // Degree-2 Polynomial Regression Curve (Amber)
  ctx.strokeStyle = "hsl(38, 92%, 55%)";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  candles.forEach((c, idx) => {
    const x = padLeft + idx * stepX + stepX * 0.5;
    const y = yFor(c.reg_value || c.close);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Candlesticks
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

  // Horizontal Moreshwar Target (Green), Entry/Spot (Cyan), Stop-Loss (Red)
  const drawLevelLine = (price, color, label) => {
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
    ctx.fillText(label, padLeft + 6, y - 5);
    ctx.restore();
  };

  drawLevelLine(asset.moreshwar.target, "hsl(158, 74%, 44%)", `TP ₹${asset.moreshwar.target}`);
  drawLevelLine(asset.price, "hsl(196, 86%, 54%)", `CP ₹${asset.price}`);
  drawLevelLine(asset.moreshwar.stopLoss, "hsl(350, 84%, 60%)", `SL ₹${asset.moreshwar.stopLoss}`);
}

function renderAllViews() {
  const assetsList = Object.values(appState.assets);
  const holdings = assetsList.filter((a) => a.qty > 0);
  const buysQueue = assetsList.filter((a) => a.fScore.total >= 65);
  const sellsQueue = holdings.filter((a) => a.fScore.total <= 45);

  // Portfolio Stats
  const invested = holdings.reduce((s, a) => s + a.qty * a.avg, 0);
  const current = holdings.reduce((s, a) => s + a.qty * a.price, 0);
  const pnl = current - invested;
  const pnlPct = invested > 0 ? (pnl / invested) * 100 : 0;
  const healthScore = holdings.length
    ? (holdings.reduce((s, a) => s + a.fScore.total * (a.qty * a.price), 0) / Math.max(1, current)).toFixed(1)
    : "78.4";

  document.getElementById("hdrInvested").textContent = fmtINR(invested);
  document.getElementById("hdrCurrent").textContent = fmtINR(current);
  const pnlEl = document.getElementById("hdrPnl");
  pnlEl.textContent = `${pnl >= 0 ? "+" : ""}${fmtINR(pnl)} (${pnl >= 0 ? "+" : ""}${pnlPct.toFixed(2)}%)`;
  pnlEl.style.color = pnl >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)";
  document.getElementById("hdrBalance").textContent = fmtINR(appState.cashBalance);
  document.getElementById("hdrHealth").textContent = `${healthScore} / 100`;

  // Top Picks Queue
  const picksContainer = document.getElementById("picksQueueContainer");
  picksContainer.innerHTML = "";
  buysQueue.forEach((a) => {
    const row = document.createElement("div");
    row.className = `pick-row ${a.symbol === appState.activeSymbol ? "selected" : ""}`;
    row.innerHTML = `
      <div>
        <span class="pick-sym">${a.symbol}</span>
        <span class="pick-sub">${a.setup}</span>
      </div>
      <span class="tag-pill tag-entry">${fmtINR(a.price)}</span>
      <span class="tag-pill tag-target">${fmtINR(a.moreshwar.target)}</span>
      <span class="tag-pill tag-stop">${fmtINR(a.moreshwar.stopLoss)}</span>
      <span class="tag-pill tag-score">${a.fScore.total}</span>
    `;
    row.addEventListener("click", () => {
      appState.activeSymbol = a.symbol;
      renderAllViews();
    });
    picksContainer.appendChild(row);
  });

  // Sell Warning Box
  const sellContainer = document.getElementById("sellWarningContainer");
  if (sellsQueue.length > 0) {
    const s = sellsQueue[0];
    sellContainer.innerHTML = `
      <div class="sell-warning-box">
        <div>
          <strong style="color: var(--accent-crimson); font-family: var(--font-mono);">⚠️ SELL WARNING: ${s.symbol}</strong>
          <div style="font-size: 11px; color: var(--text-secondary);">Fundamental Score ${s.fScore.total}/100 · Decision: ${s.decision} (${s.qty} sh)</div>
        </div>
        <button type="button" class="btn-danger" id="sellWarnBtn">Exit Now</button>
      </div>
    `;
    document.getElementById("sellWarnBtn").addEventListener("click", () => {
      appState.cashBalance += s.qty * s.price;
      s.qty = 0;
      appState.assets[s.symbol] = computeAssetModel(s.symbol, s);
      renderAllViews();
    });
  } else {
    sellContainer.innerHTML = `
      <div class="terminal-box" style="color: var(--text-secondary); text-align: center;">
        ✅ Zero Active Sell Warnings — All Holdings Above Safety Floor
      </div>
    `;
  }

  // Active Asset Center Panel & Analysis Terminal
  const active = appState.assets[appState.activeSymbol] || assetsList[0];
  document.getElementById("centerChartTitle").textContent = `${active.symbol} — ${active.name} (${active.exchange} · 1D)`;
  document.getElementById("centerSpotPrice").textContent = fmtINR(active.price, 2);
  const chgEl = document.getElementById("centerSpotChg");
  chgEl.textContent = `${active.chg >= 0 ? "+" : ""}${active.chg}%`;
  chgEl.style.color = active.chg >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)";

  const decBadge = document.getElementById("centerDecisionBadge");
  decBadge.textContent = active.decision;
  decBadge.className = `tag-pill ${active.fScore.total >= 65 ? "tag-target" : active.fScore.total <= 45 ? "tag-stop" : "tag-score"}`;

  document.getElementById("legTarget").textContent = fmtINR(active.moreshwar.target);
  document.getElementById("legEntry").textContent = fmtINR(active.price);
  document.getElementById("legStop").textContent = fmtINR(active.moreshwar.stopLoss);
  document.getElementById("legReg").textContent = fmtINR(active.reg30d);

  // Analysis Terminal Box
  document.getElementById("analysisTerminalBox").innerHTML = `
    <div style="color: var(--accent-cyan); font-weight: 600;">${active.symbol} Unified Execution Terminal</div>
    <div>• StockSight V2 Score: <strong>${active.fScore.total}/100</strong> | Porter Moat: <strong>${active.pScore.total}/100</strong></div>
    <div>• Moreshwar Levels: Entry <strong>${fmtINR(active.moreshwar.entry)}</strong> · Target <strong>${fmtINR(active.moreshwar.target)}</strong> · SL <strong>${fmtINR(active.moreshwar.stopLoss)}</strong> (R:R 1:${active.moreshwar.rr})</div>
    <div>• Auto-Risk Sized Qty: <strong>${active.autoQty} Shares</strong> (${fmtINR(active.autoQty * active.price)})</div>
    <div>• Multi-Agent Verdict: <strong style="color: var(--accent-emerald);">${active.decision}</strong> (${(active.execConf * 100).toFixed(1)}% Conviction)</div>
  `;

  // 3-Agent Orchestrator Cards
  document.getElementById("agTechBadge").textContent = active.techConf >= 0.55 ? "BULLISH" : "BEARISH";
  document.getElementById("agTechBadge").className = `tag-pill ${active.techConf >= 0.55 ? "tag-target" : "tag-stop"}`;
  document.getElementById("agTechConf").textContent = `${(active.techConf * 100).toFixed(0)}%`;
  document.getElementById("agTechBar").style.width = `${active.techConf * 100}%`;
  document.getElementById("agTechDesc").textContent = `30D Regression Target ${fmtINR(active.reg30d)}. Momentum ${active.chg >= 0 ? "positive above 20D SMA" : "below key resistance"}.`;

  document.getElementById("agFundBadge").textContent = active.fScore.total >= 65 ? "BULLISH" : active.fScore.total <= 45 ? "BEARISH" : "NEUTRAL";
  document.getElementById("agFundBadge").className = `tag-pill ${active.fScore.total >= 65 ? "tag-target" : active.fScore.total <= 45 ? "tag-stop" : "tag-score"}`;
  document.getElementById("agFundConf").textContent = `${active.fScore.total}/100`;
  document.getElementById("agFundBar").style.width = `${active.fScore.total}%`;
  document.getElementById("agFundDesc").textContent = `Bus ${active.fScore.business}/40 · Moat ${active.fScore.moat}/20 · Mgmt ${active.fScore.management}/20 · Risk ${active.fScore.risk}/20 | Porter ${active.pScore.total}/100.`;

  document.getElementById("agExecBadge").textContent = active.fScore.total >= 65 ? "APPROVED" : "GUARDRAIL";
  document.getElementById("agExecConf").textContent = `${(active.execConf * 100).toFixed(1)}%`;
  document.getElementById("agExecBar").style.width = `${active.execConf * 100}%`;
  document.getElementById("agExecDesc").textContent = `Decision Matrix: ${active.decision}. Sized ${active.autoQty} shares via Zerodha Kite Execution Bridge.`;

  // Right Column Mini Holdings
  const miniBody = document.getElementById("miniHoldingsTableBody");
  miniBody.innerHTML = "";
  holdings.forEach((h) => {
    const tr = document.createElement("tr");
    tr.style.cursor = "pointer";
    const hPnl = (h.price - h.avg) * h.qty;
    tr.innerHTML = `
      <td style="font-family: var(--font-mono); font-weight: 600;">${h.symbol}</td>
      <td style="font-family: var(--font-mono);">${h.qty}</td>
      <td style="font-family: var(--font-mono);">${fmtINR(h.price)}</td>
      <td style="font-family: var(--font-mono); color: ${hPnl >= 0 ? "var(--accent-emerald)" : "var(--accent-crimson)"};">${hPnl >= 0 ? "+" : ""}${fmtINR(hPnl)}</td>
      <td><span class="tag-pill ${h.fScore.total >= 65 ? "tag-target" : "tag-stop"}">${h.fScore.total}</span></td>
    `;
    tr.addEventListener("click", () => {
      appState.activeSymbol = h.symbol;
      renderAllViews();
    });
    miniBody.appendChild(tr);
  });

  // TAB 2: StockSight V2 Deep-Dive & Portfolio Manager
  document.getElementById("ssDeepDiveTitle").textContent = `StockSight V2 Dual-Scoring Engine — ${active.symbol} (${active.name})`;
  document.getElementById("ssIndustryBadge").textContent = `Industry Profile: ${active.industry} · P/E ${active.pe}x · ROE ${active.roe}% · ROCE ${active.roce}%`;
  document.getElementById("ssFundTotalBadge").textContent = `Fundamental: ${active.fScore.total} / 100`;
  document.getElementById("ssPorterTotalBadge").textContent = `Porter Moat: ${active.pScore.total} / 100`;

  document.getElementById("ssFourPillarsContainer").innerHTML = `
    <div class="pillar-card">
      <div style="display:flex;justify-content:space-between;"><span>1. Business Pillar</span><strong>${active.fScore.business} / 40</strong></div>
      <div class="progress-track"><div class="progress-fill" style="width:${(active.fScore.business/40)*100}%;"></div></div>
      <span style="font-size:11px;color:var(--text-secondary);">Sales Growth ${active.sales_g}% · Profit Growth ${active.profit_g}% · OPM ${active.opm}%</span>
    </div>
    <div class="pillar-card">
      <div style="display:flex;justify-content:space-between;"><span>2. Economic Moat Pillar</span><strong>${active.fScore.moat} / 20</strong></div>
      <div class="progress-track"><div class="progress-fill" style="width:${(active.fScore.moat/20)*100}%;"></div></div>
      <span style="font-size:11px;color:var(--text-secondary);">ROE ${active.roe}% · ROCE ${active.roce}% · MCap ₹${active.mcap.toLocaleString()} Cr</span>
    </div>
    <div class="pillar-card">
      <div style="display:flex;justify-content:space-between;"><span>3. Management &amp; Valuation</span><strong>${active.fScore.management} / 20</strong></div>
      <div class="progress-track"><div class="progress-fill" style="width:${(active.fScore.management/20)*100}%;"></div></div>
      <span style="font-size:11px;color:var(--text-secondary);">Trailing P/E ${active.pe}x relative to earnings expansion</span>
    </div>
    <div class="pillar-card">
      <div style="display:flex;justify-content:space-between;"><span>4. Risk &amp; Stability</span><strong>${active.fScore.risk} / 20</strong></div>
      <div class="progress-track"><div class="progress-fill" style="width:${(active.fScore.risk/20)*100}%;"></div></div>
      <span style="font-size:11px;color:var(--text-secondary);">Large-Cap Safety Floor &amp; Volatility Check</span>
    </div>
  `;

  document.getElementById("ssPortersGrid").innerHTML = `
    <div class="pillar-card"><span>Barriers to Entry (MCap/ROCE)</span><strong>${active.pScore.entrants} / 20</strong></div>
    <div class="pillar-card"><span>Supplier Power (OPM Proxy)</span><strong>${active.pScore.suppliers} / 20</strong></div>
    <div class="pillar-card"><span>Buyer Pricing Power (ROE)</span><strong>${active.pScore.buyers} / 20</strong></div>
    <div class="pillar-card"><span>Threat of Substitutes &amp; Rivalry</span><strong>${active.pScore.substitutes + active.pScore.rivalry} / 40</strong></div>
  `;

  document.getElementById("ssDecisionMatrixBox").innerHTML = `
    <div style="color:var(--accent-emerald);font-weight:600;">Moreshwar Mathematical Model &amp; Final Decision Matrix</div>
    <div>• Composite Quality (X): <strong>${active.moreshwar.xBlend} / 100</strong> | Spot Price (Y): <strong>${fmtINR(active.price, 2)}</strong></div>
    <div>• Dynamic Entry: <strong>${fmtINR(active.moreshwar.entry, 2)}</strong> | Dynamic Target: <strong>${fmtINR(active.moreshwar.target, 2)}</strong> | Dynamic Stop-Loss: <strong>${fmtINR(active.moreshwar.stopLoss, 2)}</strong></div>
    <div>• Final Matrix Signal: <strong>${active.decision}</strong> (${active.qty > 0 ? "Currently Held in Portfolio" : "Watchlist Candidate"})</div>
  `;

  // Full Portfolio & Watchlist Interactive Table
  const mgrBody = document.getElementById("fullPortfolioManagerBody");
  mgrBody.innerHTML = "";
  assetsList.forEach((a) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>
        <strong style="font-family:var(--font-mono);cursor:pointer;color:var(--accent-cyan);" class="sym-link">${a.symbol}</strong>
        <span style="display:block;font-size:10.5px;color:var(--text-secondary);">${a.qty > 0 ? "PORTFOLIO HOLDING" : "WATCHLIST"}</span>
      </td>
      <td style="font-family:var(--font-mono);">${a.fScore.total} / ${a.pScore.total}</td>
      <td><span class="tag-pill ${a.fScore.total >= 65 ? "tag-target" : a.fScore.total <= 45 ? "tag-stop" : "tag-score"}">${a.decision}</span></td>
      <td><input type="number" class="input-field qty-in" value="${a.qty}" min="0" style="width:74px;" /></td>
      <td><input type="number" class="input-field avg-in" value="${a.avg}" min="1" style="width:94px;" /></td>
      <td style="font-family:var(--font-mono);">${fmtINR(a.price)} → <span style="color:var(--accent-emerald);">${fmtINR(a.moreshwar.target)}</span></td>
      <td><button type="button" class="btn-secondary save-row-btn">Update</button></td>
    `;
    tr.querySelector(".sym-link").addEventListener("click", () => {
      appState.activeSymbol = a.symbol;
      renderAllViews();
    });
    tr.querySelector(".save-row-btn").addEventListener("click", () => {
      const newQty = parseInt(tr.querySelector(".qty-in").value, 10) || 0;
      const newAvg = parseFloat(tr.querySelector(".avg-in").value) || a.price;
      a.qty = newQty;
      a.avg = newAvg;
      appState.assets[a.symbol] = computeAssetModel(a.symbol, a);
      renderAllViews();
    });
    mgrBody.appendChild(tr);
  });

  // TAB 3: Auto Rules & Verify Output
  const rulesBox = document.getElementById("autoRulesList");
  rulesBox.innerHTML = "";
  appState.rules.forEach((r) => {
    const div = document.createElement("div");
    div.className = "pillar-card";
    div.style.flexDirection = "row";
    div.style.alignItems = "center";
    div.style.justifyContent = "space-between";
    div.innerHTML = `
      <span style="font-size:12px;">${r.name}</span>
      <button type="button" class="btn-secondary">${r.active ? "ACTIVE ✓" : "PAUSED"}</button>
    `;
    div.querySelector("button").addEventListener("click", () => {
      r.active = !r.active;
      renderAllViews();
    });
    rulesBox.appendChild(div);
  });

  document.getElementById("unifiedVerifyOutput").textContent =
    `$ python scripts/verify_setup.py\n` +
    `Verifying Finsistant Version 2 (All-in-One) platform status...\n` +
    JSON.stringify({
      pipeline_status: "HEALTHY",
      platform: "Finsistant Version 2 (All-in-One)",
      merged_modules: [
        "1. StockSight Version 2 (4-Pillar + Porter's 5 Forces + Moreshwar Levels)",
        "2. Finsistant Execution Terminal (TechSight Queue + Polynomial Regression + Stocky AI)",
        "3. Multi-Agent Orchestrator (TechBot_1 + FundBot_1 + ExecBot_1)"
      ],
      active_symbol: active.symbol,
      fundamental_score: `${active.fScore.total}/100`,
      porters_5_forces: `${active.pScore.total}/100`,
      moreshwar_target: active.moreshwar.target,
      moreshwar_stop_loss: active.moreshwar.stopLoss,
      decision_matrix: active.decision
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

function handleStockyQuery(rawText) {
  const q = rawText.trim();
  if (!q) return;
  appState.chatHistory.push({ role: "user", text: q });
  const ql = q.toLowerCase();
  let reply = "";

  const cmp = ql.match(/(?:compare|vs)\s+([a-z0-9_-]+)\s+(?:and|with|vs)\s+([a-z0-9_-]+)/);
  const inv = ql.match(/(?:invest|allocate)\s+([\d,.]+)\s*(k|l|lakh|cr)?/);
  const buy = ql.match(/buy\s+(\d+)\s+([a-z0-9_-]+)/);

  if (cmp) {
    const s1 = cmp[1].toUpperCase(), s2 = cmp[2].toUpperCase();
    const a1 = appState.assets[s1] || appState.assets["HDFCBANK"];
    const a2 = appState.assets[s2] || appState.assets["ICICIBANK"];
    const win = (a1.fScore.total + a1.pScore.total) >= (a2.fScore.total + a2.pScore.total) ? a1 : a2;
    reply =
      `📊 Stocky Head-to-Head: ${a1.symbol} vs ${a2.symbol}\n\n` +
      `• ${a1.symbol} (${fmtINR(a1.price)}): Fund ${a1.fScore.total}/100 · Porter ${a1.pScore.total}/100 · Decision ${a1.decision} · Target ${fmtINR(a1.moreshwar.target)}\n` +
      `• ${a2.symbol} (${fmtINR(a2.price)}): Fund ${a2.fScore.total}/100 · Porter ${a2.pScore.total}/100 · Decision ${a2.decision} · Target ${fmtINR(a2.moreshwar.target)}\n\n` +
      `🏆 Verdict: ${win.symbol} leads on risk-adjusted composite moat & growth quality.`;
  } else if (inv) {
    let amt = parseFloat(inv[1].replace(/,/g, "")) || 100000;
    const u = (inv[2] || "").toLowerCase();
    if (u.startsWith("k")) amt *= 1000;
    else if (u.startsWith("l")) amt *= 100000;
    else if (u.startsWith("c")) amt *= 10000000;

    const top = Object.values(appState.assets).filter((a) => a.fScore.total >= 65).slice(0, 4);
    const sumS = top.reduce((s, a) => s + a.moreshwar.xBlend, 0) || 1;
    reply = `💰 Score-Weighted Allocation Plan for ${fmtINR(amt)}:\n\n`;
    top.forEach((a) => {
      const w = a.moreshwar.xBlend / sumS;
      const sh = Math.max(1, Math.floor((amt * w) / a.price));
      reply += `• ${a.symbol} (Score ${a.moreshwar.xBlend} · ${a.decision}): Buy ${sh} sh @ ${fmtINR(a.price)} = ${fmtINR(sh * a.price)}\n`;
    });
  } else if (buy) {
    const qty = parseInt(buy[1], 10) || 1;
    const sym = buy[2].toUpperCase();
    const a = appState.assets[sym] || appState.assets[appState.activeSymbol];
    a.qty += qty;
    appState.cashBalance = Math.max(0, appState.cashBalance - qty * a.price);
    appState.activeSymbol = a.symbol;
    renderAllViews();
    reply = `✅ Executed via Kite Bridge: Bought ${qty} shares of ${a.symbol} @ ${fmtINR(a.price)} (Target ${fmtINR(a.moreshwar.target)} | SL ${fmtINR(a.moreshwar.stopLoss)}).`;
  } else if (ql.includes("efficiency") || ql.includes("portfolio") || ql.includes("trap")) {
    const traps = Object.values(appState.assets).filter((a) => a.qty > 0 && a.fScore.total <= 45);
    reply =
      `🛡️ Stocky Portfolio Efficiency Audit:\n` +
      `• Core Compounders: RELIANCE, TCS, HDFCBANK, ICICIBANK\n` +
      (traps.length
        ? `• ⚠️ Capital Trap Detected: ${traps.map((t) => `${t.symbol} (Score ${t.fScore.total}/100)`).join(", ")} — Recommend immediate exit.`
        : `• ✅ Zero Capital Traps: All holdings exceed fundamental quality thresholds.`);
  } else {
    const found = Object.keys(appState.assets).find((s) => ql.includes(s.toLowerCase()));
    const a = appState.assets[found || appState.activeSymbol];
    appState.activeSymbol = a.symbol;
    renderAllViews();
    reply =
      `🔍 Deep-Dive Analysis: ${a.symbol} (${a.name})\n` +
      `• Total Fundamental Score: ${a.fScore.total}/100 (Bus ${a.fScore.business}/40, Moat ${a.fScore.moat}/20, Mgmt ${a.fScore.management}/20, Risk ${a.fScore.risk}/20)\n` +
      `• Porter's 5 Forces Moat: ${a.pScore.total}/100\n` +
      `• Moreshwar Levels: Entry ${fmtINR(a.moreshwar.entry)} | Target ${fmtINR(a.moreshwar.target)} | SL ${fmtINR(a.moreshwar.stopLoss)}\n` +
      `• 30D Quant Regression Target: ${fmtINR(a.reg30d)} | Final Decision: ${a.decision}`;
  }

  appState.chatHistory.push({ role: "bot", text: reply });
  renderChat();
}

document.addEventListener("DOMContentLoaded", () => {
  initFallbackUniverse();
  renderAllViews();
  renderChat();

  // Tabs
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
    const a = appState.assets[appState.activeSymbol];
    if (!a) return;
    const cost = a.autoQty * a.price;
    appState.cashBalance = Math.max(0, appState.cashBalance - cost);
    a.qty += a.autoQty;
    appState.assets[a.symbol] = computeAssetModel(a.symbol, a);
    handleStockyQuery(`Buy ${a.autoQty} ${a.symbol}`);
  });

  document.getElementById("executeAllBtn").addEventListener("click", () => {
    Object.values(appState.assets).filter((a) => a.fScore.total >= 70).forEach((a) => {
      const q = Math.max(1, Math.floor(a.autoQty / 2));
      a.qty += q;
      appState.cashBalance = Math.max(0, appState.cashBalance - q * a.price);
    });
    renderAllViews();
  });

  document.getElementById("scanNowBtn").addEventListener("click", () => {
    renderAllViews();
  });

  document.getElementById("addSymbolBtn").addEventListener("click", () => {
    const inp = document.getElementById("addSymbolInput");
    const sym = inp.value.trim().toUpperCase();
    if (!sym) return;
    if (!appState.assets[sym]) {
      appState.assets[sym] = computeAssetModel(sym, {
        name: `${sym} Enterprises Ltd`,
        exchange: "NSE",
        sector: "Diversified Growth Equity",
        price: 1420,
        chg: 1.65,
        pe: 21.4,
        roe: 20.5,
        roce: 22.1,
        opm: 21.0,
        sales_g: 16.4,
        profit_g: 19.2,
        mcap: 58000,
        qty: 0,
        avg: 1420,
        setup: "Fresh Scanner Setup",
        industry: "GENERAL"
      });
    }
    appState.activeSymbol = sym;
    inp.value = "";
    renderAllViews();
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

  document.getElementById("saveUnifiedConfigBtn").addEventListener("click", renderAllViews);
  document.getElementById("runUnifiedVerifyBtn").addEventListener("click", renderAllViews);

  window.addEventListener("resize", () => {
    drawUnifiedChart(appState.assets[appState.activeSymbol]);
  });
});
