/* ==========================================================================
   FINSISTANT V2 — MOBILE-FIRST ZERODHA KITE SHELL CONTROLLER (ANDROID APK ONLY)
   Loaded exclusively inside mobile-app/app/src/main/assets/index.html.
   Web interface (root & web/) is 100% untouched.
   ========================================================================== */

(function initKiteMobileShell() {
  "use strict";

  const TAB_TITLES = {
    mission: "Watchlist",
    intelligence: "Mission 3.2",
    stocksight: "Portfolio",
    reports: "Reports",
    agents: "Setup",
  };

  let activeTab = "mission";
  let missionView = "list"; // "list" | "detail"
  let searchQuery = "";
  let aiOpen = false;
  let lastRenderedSig = "";

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function buildChrome() {
    if (document.getElementById("kmTopbar")) return;

    // 1. Kite Top App Bar
    const topbar = document.createElement("div");
    topbar.id = "kmTopbar";
    topbar.className = "km-topbar";
    topbar.innerHTML = `
      <div class="km-topbar-left">
        <button type="button" class="km-back-btn" id="kmBackBtn" aria-label="Back">&#8592;</button>
        <div class="km-title-wrap">
          <div class="km-title" id="kmTopTitle">Watchlist</div>
          <div class="km-subtitle"><span class="km-live-dot"></span><span id="kmTopSub">GitHub Cloud · Live NSE</span></div>
        </div>
      </div>
      <div class="km-topbar-actions">
        <button type="button" class="km-icon-btn accent" id="kmOpenAiBtn" aria-label="Stocky AI">AI</button>
        <button type="button" class="km-icon-btn" id="kmRefreshBtn" aria-label="Refresh">&#8635;</button>
      </div>
    `;
    document.body.prepend(topbar);

    // 2. Kite Fixed Bottom Navigation Bar (5 Tabs)
    const nav = document.createElement("nav");
    nav.id = "kmBottomNav";
    nav.className = "km-bottom-nav";
    nav.setAttribute("aria-label", "Mobile Navigation");
    nav.innerHTML = `
      <button type="button" class="km-nav-btn active" data-km-tab="mission">
        <svg viewBox="0 0 24 24"><path d="M5 7h14M5 12h14M5 17h10"/></svg>
        <span>Watchlist</span>
      </button>
      <button type="button" class="km-nav-btn" data-km-tab="intelligence">
        <svg viewBox="0 0 24 24"><path d="M3 17l6-6 4 4 8-8"/><path d="M17 7h4v4"/></svg>
        <span>Mission 3.2</span>
      </button>
      <button type="button" class="km-nav-btn" data-km-tab="stocksight">
        <svg viewBox="0 0 24 24"><rect x="3" y="7" width="18" height="13" rx="2"/><path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
        <span>Portfolio</span>
      </button>
      <button type="button" class="km-nav-btn" data-km-tab="reports">
        <svg viewBox="0 0 24 24"><path d="M6 3h9l5 5v13H6z"/><path d="M15 3v5h5"/><path d="M9 13h6M9 17h6"/></svg>
        <span>Reports</span>
      </button>
      <button type="button" class="km-nav-btn" data-km-tab="agents">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.09a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.09a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
        <span>Setup</span>
      </button>
    `;
    document.body.appendChild(nav);

    // 3. Kite Watchlist View Container inside #panel-mission
    const panelMission = document.getElementById("panel-mission");
    if (panelMission) {
      const wl = document.createElement("div");
      wl.id = "kmWatchlistContainer";
      wl.className = "km-watchlist-wrap";
      wl.innerHTML = `
        <div class="km-mission-strip" id="kmMissionBanner">
          <div><strong id="kmBannerMission">M3.2 Mock Test · 1 / 80</strong> · <span id="kmBannerNav">₹1,00,000</span></div>
          <span class="km-link">Open &#8250;</span>
        </div>
        <div class="km-search-bar">
          <input type="search" id="kmSearchInput" class="km-search-input" placeholder="Search today's NSE orchestrator picks..." />
        </div>
        <div id="kmWatchlistRows"></div>
      `;
      panelMission.prepend(wl);
    }

    // 4. Kite Underline Segmented Bar for Mission 3.2 (Tab 4)
    const panelIntel = document.getElementById("panel-intelligence");
    if (panelIntel) {
      const seg = document.createElement("div");
      seg.className = "km-segbar";
      seg.innerHTML = `
        <button type="button" class="km-seg-btn active" data-km-m32="picks">M3.2 Picks</button>
        <button type="button" class="km-seg-btn" data-km-m32="learn">Brain &amp; Gates</button>
        <button type="button" class="km-seg-btn" data-km-m32="m31">M3.1 Baseline</button>
      `;
      panelIntel.prepend(seg);
      seg.querySelectorAll(".km-seg-btn").forEach((b) => {
        b.addEventListener("click", () => {
          seg.querySelectorAll(".km-seg-btn").forEach((x) => x.classList.remove("active"));
          b.classList.add("active");
          const mode = b.getAttribute("data-km-m32");
          document.body.classList.remove("km-m32-seg-picks", "km-m32-seg-learn", "km-m32-seg-m31");
          document.body.classList.add(`km-m32-seg-${mode}`);
          window.scrollTo(0, 0);
        });
      });
    }

    // 5. Kite Underline Segmented Bar for Portfolio / StockSight (Tab 2)
    const panelSs = document.getElementById("panel-stocksight");
    if (panelSs) {
      const seg = document.createElement("div");
      seg.className = "km-segbar";
      seg.innerHTML = `
        <button type="button" class="km-seg-btn active" data-km-ss="score">4-Pillar Score</button>
        <button type="button" class="km-seg-btn" data-km-ss="list">All 12 Picks</button>
      `;
      panelSs.prepend(seg);
      seg.querySelectorAll(".km-seg-btn").forEach((b) => {
        b.addEventListener("click", () => {
          seg.querySelectorAll(".km-seg-btn").forEach((x) => x.classList.remove("active"));
          b.classList.add("active");
          const mode = b.getAttribute("data-km-ss");
          document.body.classList.remove("km-ss-seg-score", "km-ss-seg-list");
          document.body.classList.add(`km-ss-seg-${mode}`);
          window.scrollTo(0, 0);
        });
      });
    }

    // 6. Stocky AI Slide-Up Bottom Sheet (relocates existing Stocky panel cleanly without breaking IDs)
    const backdrop = document.createElement("div");
    backdrop.id = "kmAiBackdrop";
    backdrop.className = "km-ai-backdrop";
    const sheet = document.createElement("div");
    sheet.id = "kmAiSheet";
    sheet.className = "km-ai-sheet";
    sheet.innerHTML = `
      <div class="km-ai-handle"></div>
      <div class="km-ai-head">
        <div>
          <div style="font-size:15px;font-weight:600;color:#E5EAF1;">Stocky AI Assistant</div>
          <div style="font-size:11px;color:#8D99AA;">Live Quant, Fundamental &amp; Execution Assistant</div>
        </div>
        <button type="button" class="km-icon-btn" id="kmCloseAiBtn">&#10005;</button>
      </div>
      <div class="km-ai-body" id="kmAiBody"></div>
    `;
    document.body.appendChild(backdrop);
    document.body.appendChild(sheet);

    const stockyMsg = document.getElementById("stockyChatMessages");
    const stockyPanel = stockyMsg ? stockyMsg.closest(".panel") : null;
    if (stockyPanel) {
      const aiBody = document.getElementById("kmAiBody");
      Array.from(stockyPanel.children).forEach((child) => {
        if (!child.classList.contains("panel-header")) {
          aiBody.appendChild(child);
        }
      });
    }

    // Wire events
    nav.querySelectorAll(".km-nav-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tab = btn.getAttribute("data-km-tab");
        if (tab === "mission" && activeTab === "mission" && missionView === "detail") {
          setMissionView("list");
        } else {
          switchMobileTab(tab);
        }
      });
    });

    document.getElementById("kmBackBtn").addEventListener("click", () => {
      if (missionView === "detail") setMissionView("list");
    });

    document.getElementById("kmOpenAiBtn").addEventListener("click", () => setAiSheet(true));
    document.getElementById("kmCloseAiBtn").addEventListener("click", () => setAiSheet(false));
    backdrop.addEventListener("click", () => setAiSheet(false));

    document.getElementById("kmRefreshBtn").addEventListener("click", () => {
      const btn = document.getElementById("kmRefreshBtn");
      btn.style.opacity = "0.5";
      if (typeof window.refreshFinsistantMobileScan === "function") {
        Promise.resolve(window.refreshFinsistantMobileScan()).finally(() => {
          btn.style.opacity = "1";
          setTimeout(renderKiteWatchlist, 600);
        });
      } else {
        location.reload();
      }
    });

    const mBanner = document.getElementById("kmMissionBanner");
    if (mBanner) {
      mBanner.addEventListener("click", () => switchMobileTab("intelligence"));
    }

    const searchInput = document.getElementById("kmSearchInput");
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        searchQuery = String(e.target.value || "").trim().toUpperCase();
        lastRenderedSig = "";
        renderKiteWatchlist();
      });
    }

    // Initial body state classes
    document.body.classList.add("km-mission-list", "km-m32-seg-picks", "km-ss-seg-score");
  }

  function setAiSheet(open) {
    aiOpen = Boolean(open);
    const bd = document.getElementById("kmAiBackdrop");
    const sh = document.getElementById("kmAiSheet");
    if (bd) bd.classList.toggle("open", aiOpen);
    if (sh) sh.classList.toggle("open", aiOpen);
  }

  function setMissionView(view, symLabel) {
    missionView = view;
    document.body.classList.toggle("km-mission-list", view === "list");
    document.body.classList.toggle("km-mission-detail", view === "detail");
    const backBtn = document.getElementById("kmBackBtn");
    const titleEl = document.getElementById("kmTopTitle");
    if (backBtn) backBtn.classList.toggle("visible", activeTab === "mission" && view === "detail");
    if (titleEl && activeTab === "mission") {
      titleEl.textContent = view === "detail" && symLabel ? symLabel : "Watchlist";
    }
    window.scrollTo(0, 0);
    if (view === "detail") {
      // Trigger canvas redraw now that chart container is visible
      setTimeout(() => window.dispatchEvent(new Event("resize")), 80);
    }
  }

  function switchMobileTab(tab) {
    const valid = TAB_TITLES[tab] ? tab : "mission";
    activeTab = valid;
    const desktopBtn = document.querySelector(`.nav-tab[data-tab="${valid}"]`);
    if (desktopBtn) desktopBtn.click();

    document.querySelectorAll("#kmBottomNav .km-nav-btn").forEach((b) => {
      b.classList.toggle("active", b.getAttribute("data-km-tab") === valid);
    });

    const backBtn = document.getElementById("kmBackBtn");
    const titleEl = document.getElementById("kmTopTitle");
    if (valid === "mission") {
      setMissionView(missionView === "detail" ? "list" : "list");
    } else {
      if (backBtn) backBtn.classList.remove("visible");
      if (titleEl) titleEl.textContent = TAB_TITLES[valid] || "MOVIQ";
      window.scrollTo(0, 0);
    }
  }

  function updateHeaderTelemetry() {
    const subEl = document.getElementById("kmTopSub");
    const srcStatus = document.getElementById("liveSourceSubtitle");
    if (subEl && srcStatus) {
      const txt = srcStatus.textContent || "";
      const m = txt.match(/Updated\s+(.+)$/i);
      subEl.textContent = m ? `Live · ${m[1].replace(/,\s*/, " ")}` : "GitHub Cloud · Live NSE";
    }
    const mBadge = document.getElementById("mv32MissionBadge");
    const bMission = document.getElementById("kmBannerMission");
    if (mBadge && bMission) {
      const m = (mBadge.textContent || "").match(/(\d+\s*\/\s*\d+)/);
      if (m) bMission.textContent = `M3.2 Mock Test · ${m[1]}`;
    }
  }

  function renderKiteWatchlist() {
    updateHeaderTelemetry();
    const rowsEl = document.getElementById("kmWatchlistRows");
    const picksContainer = document.getElementById("picksQueueContainer");
    if (!rowsEl || !picksContainer) return;

    const pickCards = Array.from(picksContainer.querySelectorAll(".pick-row, [data-symbol]"));
    const items = [];
    const seen = new Set();

    // Read top-level appState.assets directly from app.js classic script scope
    const stateAssets = (typeof appState !== "undefined" && appState && appState.assets) ? appState.assets : null;
    if (stateAssets && Object.keys(stateAssets).length > 0) {
      Object.values(stateAssets).forEach((a) => {
        if (!a || !a.symbol || seen.has(a.symbol)) return;
        seen.add(a.symbol);
        items.push({
          symbol: a.symbol,
          exchange: a.exchange || "NSE",
          sector: a.sector || "",
          price: Number(a.price || 0),
          changePct: Number(a.change_pct || 0),
          score: a.fundamental_score ? a.fundamental_score.total : null,
          decision: a.decision_matrix ? a.decision_matrix.action : "",
        });
      });
    } else {
      pickCards.forEach((el) => {
        const txt = (el.innerText || "").trim();
        const lines = txt.split(/\n+/).map((s) => s.trim()).filter(Boolean);
        const sym = el.getAttribute("data-symbol") || (lines[0] || "").split(/\s+/)[0];
        if (!sym || seen.has(sym)) return;
        seen.add(sym);
        const priceMatch = txt.match(/₹([\d,.]+)/);
        const scoreMatch = txt.match(/\b(\d{2})\s*$/);
        items.push({
          symbol: sym,
          exchange: "NSE",
          sector: lines[1] || "Live NSE Scan",
          price: priceMatch ? Number(priceMatch[1].replace(/,/g, "")) : 0,
          changePct: 0,
          score: scoreMatch ? Number(scoreMatch[1]) : null,
          decision: "",
          sourceEl: el,
        });
      });
    }

    const filtered = searchQuery
      ? items.filter((it) => it.symbol.toUpperCase().includes(searchQuery) || String(it.sector).toUpperCase().includes(searchQuery))
      : items;

    const sig = filtered.map((it) => `${it.symbol}:${it.price}:${it.changePct}`).join("|");
    if (sig && sig === lastRenderedSig) return;
    lastRenderedSig = sig;

    if (!filtered.length) {
      rowsEl.innerHTML = `<div style="padding:28px 16px;text-align:center;color:#647184;">Loading live NSE orchestrator picks...</div>`;
      return;
    }

    rowsEl.innerHTML = filtered
      .map((it) => {
        const up = it.changePct >= 0;
        const chgStr = `${up ? "+" : ""}${it.changePct.toFixed(2)}%`;
        const pxStr = it.price > 0 ? `₹${it.price.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "--";
        const decTag = it.decision ? ` · ${it.decision}` : "";
        const scoreTxt = it.score != null ? ` · Score ${it.score}` : "";
        return `
          <div class="km-wl-row" data-km-sym="${esc(it.symbol)}">
            <div class="km-wl-left">
              <div class="km-wl-sym-line">
                <span class="km-wl-sym">${esc(it.symbol)}</span>
                <span class="km-wl-exch">${esc(it.exchange)}</span>
              </div>
              <div class="km-wl-sub">${esc(it.sector || "NSE Equity")}${esc(decTag)}</div>
            </div>
            <div class="km-wl-right">
              <div class="km-wl-price ${up ? "up" : "down"}">${pxStr}</div>
              <div class="km-wl-chg">${chgStr}${esc(scoreTxt)}</div>
            </div>
          </div>
        `;
      })
      .join("");

    rowsEl.querySelectorAll(".km-wl-row").forEach((row) => {
      row.addEventListener("click", () => {
        const sym = row.getAttribute("data-km-sym");
        // Click matching row in hidden desktop queue so app.js updates chart, agents, CRO & StockSight cleanly
        const matchingPick = Array.from(picksContainer.children).find((c) => {
          const ds = c.getAttribute("data-symbol");
          if (ds && ds.toUpperCase() === sym.toUpperCase()) return true;
          const first = (c.innerText || "").trim().split(/\s+/)[0];
          return first && first.toUpperCase() === sym.toUpperCase();
        });
        if (matchingPick) matchingPick.click();
        setMissionView("detail", sym);
      });
    });
  }

  function observeUpdates() {
    const picksContainer = document.getElementById("picksQueueContainer");
    if (picksContainer) {
      new MutationObserver(() => renderKiteWatchlist()).observe(picksContainer, { childList: true, subtree: true });
    }
    const hdr = document.getElementById("liveSourceSubtitle");
    if (hdr) {
      new MutationObserver(() => updateHeaderTelemetry()).observe(hdr, { childList: true, characterData: true, subtree: true });
    }
    const m32Badge = document.getElementById("mv32MissionBadge");
    if (m32Badge) {
      new MutationObserver(() => updateHeaderTelemetry()).observe(m32Badge, { childList: true, characterData: true, subtree: true });
    }
    // Initial sync passes after async GitHub raw fetch settles
    setTimeout(renderKiteWatchlist, 600);
    setTimeout(renderKiteWatchlist, 2000);
    setTimeout(renderKiteWatchlist, 4500);
  }

  // Expose native Android MainActivity.kt hooks
  window.getActiveMobileTab = () => activeTab;
  window.switchMobileTab = (tab) => switchMobileTab(tab === "picks" ? "mission" : tab);
  window.handleFinsistantMobileBack = () => {
    if (aiOpen) {
      setAiSheet(false);
      return true;
    }
    if (activeTab === "mission" && missionView === "detail") {
      setMissionView("list");
      return true;
    }
    return false;
  };

  function boot() {
    buildChrome();
    observeUpdates();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
