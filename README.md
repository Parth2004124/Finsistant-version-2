# Finsistant Version 2 — All-in-One Quant, Fundamental & Multi-Agent Platform

**Finsistant Version 2** merges three complete trading & portfolio intelligence systems into a single unified platform (cloned cleanly without touching any of the original repositories):

1. **StockSight Version 2 (`finsistant_v2/scoring_engine.py`)**
   - **4-Pillar Fundamental Scoring System (Max 100 Points):** Business (`/40`), Economic Moat (`/20`), Management & Valuation (`/20`), and Risk & Stability (`/20`).
   - **Industry Normalization Matrix:** Dynamic sector weighting across `BANKING`, `IT`, `FMCG`, `PHARMA`, `AUTO`, `POWER`, `REALTY`, and `GENERAL`.
   - **Porter's 5 Forces Qualitative Moat Score (Max 100 Points):** Threat of New Entrants (`/20`), Supplier Power (`/20`), Buyer Power (`/20`), Threat of Substitutes (`/20`), and Competitive Rivalry (`/20`).
   - **Moreshwar Dynamic Target / Entry / Stop-Loss Levels:** Mathematical model blending Fundamental Score and Porter Moat Score (`X`) around Spot Price (`Y`).
   - **Final Decision Tree Matrix:** Actionable signals (`BUY NOW`, `ADD`, `HOLD`, `SIP ONLY`, `WAIT`, `REVIEW`, `REDUCE`, `EXIT`, `AVOID`).
   - **Interactive Portfolio & Watchlist Manager:** Edit `Qty` and `Avg Price` to move assets between Watchlist and Portfolio with real-time P&L and Portfolio Health Score.

2. **Finsistant Execution Terminal (`finsistant_v2/quant_engine.py` & `finsistant_v2/stocky_ai.py`)**
   - **TechSight Top Picks Queue & Scanner:** Populates high-conviction setups with Entry, Target, Stop-Loss, and Score.
   - **1-Click & Batch Execution Bridge:** `EXECUTE SELECTED` and `EXECUTE ALL` buttons with Auto-Risk Position Sizing (`1.5%` risk budget divided by Entry-to-Stop distance) and Zerodha Kite Connect execution flow.
   - **Sell Warnings Box:** Automatically flags holdings whose Fundamental Score drops below safety floors (`<= 45`) for immediate 1-click capital preservation exit.
   - **Degree-2 Polynomial Regression Engine:** Fits a quadratic regression curve (`y = ax² + bx + c`) with 30-day forward price projection and volatility channels overlaid on the OHLCV candlestick chart.
   - **Stocky AI Assistant:** Built-in natural-language assistant supporting `Analyze <SYMBOL>`, `Compare <SYM1> vs <SYM2>`, `Invest <Amount>` (score-weighted capital allocator), `Portfolio efficiency check` (Capital Trap detector), and `Buy <Qty> <SYMBOL>`.

3. **Multi-Agent Trading Orchestrator (`finsistant_v2/agents.py`)**
   - **`AgentRunner` Execution Bus:** Coordinates three specialized agents on every asset:
     - `TechBot_1` (`TechnicalAnalyst`): Evaluates RSI-14, MACD, 20D SMA, Bollinger Bands, and 30-Day Polynomial Regression slope.
     - `FundBot_1` (`FundamentalAnalyst`): Evaluates the StockSight V2 4-Pillar Score + Porter's 5 Forces Moat.
     - `ExecBot_1` (`ExecutionAgent`): Synthesizes conviction, calculates Moreshwar levels & auto-risk share quantity, enforces `MAX_CONCURRENT_TRADES`, and routes orders.

---

## Quickstart

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify all 3 merged engines
cp .env.example .env
python scripts/verify_setup.py

# 4. Launch the All-in-One Web Platform
python server.py
```
