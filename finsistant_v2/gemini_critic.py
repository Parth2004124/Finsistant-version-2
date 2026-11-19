"""Gemini Critical Trade Analyst for Finsistant Version 2 (Zero Simulation).

Calls the live Google Gemini API (generativelanguage.googleapis.com) to critically
evaluate every suggested trade from the 3-Agent Orchestrator and StockSight V2
engine, surfacing bull catalysts, adversarial red flags/risks, and an actionable
institutional rationale.

Zero simulation policy:
- If GEMINI_API_KEY is missing or rejected by Google Gemini API, returns an
  explicit `unavailable` / `error` payload with the real reason instead of
  fabricating synthetic AI commentary.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
import re
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
GEMINI_FALLBACK_MODELS = ("gemini-2.5-flash", "gemini-2.0-flash")


def _redact_key(text: str, api_key: str) -> str:
    cleaned = str(text or "")
    if api_key and len(api_key) >= 8:
        cleaned = cleaned.replace(api_key, "***")
    cleaned = re.sub(r"key=[^&\s\"']+", "key=***", cleaned)
    cleaned = re.sub(r"AIza[0-9A-Za-z_-]{10,}", "AIza***", cleaned)
    return cleaned


def _currency_prefix(currency: str) -> str:
    return "$" if str(currency or "INR").upper() == "USD" else "₹"


def _extract_json_block(raw_text: str) -> dict[str, Any]:
    txt = (raw_text or "").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", txt, re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Gemini did not return valid JSON structure")


def _call_gemini_generate(prompt: str, api_key: str, response_mime_type: str = "application/json", timeout: int = 25) -> tuple[str, str]:
    """Call Gemini generateContent REST API and return (text, model_used)."""
    last_error = "Unknown Gemini API failure"
    for model in GEMINI_FALLBACK_MODELS:
        encoded_key = urllib.parse.quote(api_key, safe="")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={encoded_key}"
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.25,
                "maxOutputTokens": 650,
            },
        }
        if response_mime_type:
            payload["generationConfig"]["responseMimeType"] = response_mime_type

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                candidates = body.get("candidates") or []
                if not candidates:
                    last_error = f"{model}: empty candidates response"
                    continue
                parts = candidates[0].get("content", {}).get("parts") or []
                text = "".join(str(p.get("text", "")) for p in parts).strip()
                if not text:
                    last_error = f"{model}: empty text in candidate"
                    continue
                return text, model
        except urllib.error.HTTPError as exc:
            raw_err = exc.read().decode("utf-8", errors="replace")
            try:
                err_msg = json.loads(raw_err).get("error", {}).get("message", raw_err)
            except Exception:
                err_msg = raw_err
            err_msg = _redact_key(" ".join(str(err_msg).split()), api_key)
            last_error = f"HTTP {exc.code} ({model}): {err_msg}"
            # Invalid / revoked key should fail fast rather than retrying another model
            if exc.code in (400, 401, 403) and "not found" not in err_msg.lower():
                raise RuntimeError(_redact_key(last_error, api_key)) from exc
            if exc.code == 429:
                time.sleep(1.0)
        except Exception as exc:
            last_error = f"{type(exc).__name__} ({model}): {_redact_key(str(exc), api_key)}"

    raise RuntimeError(_redact_key(last_error, api_key))


def build_trade_critique_prompt(
    symbol: str,
    analysis: dict[str, Any],
    pipeline: dict[str, Any] | None = None,
    trade_pick: dict[str, Any] | None = None,
) -> str:
    cur = analysis.get("currency", "INR")
    sym_p = _currency_prefix(cur)
    f = analysis.get("fundamental_score", {})
    r = f.get("ratios", {})
    p = analysis.get("porters_score", {})
    m = analysis.get("moreshwar_levels", {})
    d = analysis.get("decision_matrix", {})
    msgs = (pipeline or {}).get("messages") or []
    tech_msg = msgs[0] if len(msgs) > 0 else {}
    fund_msg = msgs[1] if len(msgs) > 1 else {}
    exec_msg = msgs[2] if len(msgs) > 2 else {}

    tx_type = (trade_pick or {}).get("transaction_type", "WATCH" if d.get("action") == "WAIT" else "BUY")
    qty = (trade_pick or {}).get("qty") or (pipeline or {}).get("quantity") or 0
    conv_pct = round(float((pipeline or {}).get("conviction_score", 0.0)) * 100.0, 1)

    return f"""You are the Chief Risk Officer & Critical Trade Analyst for Finsistant Version 2.
Critically analyze the following live suggested trade setup. DO NOT rubber-stamp it. Challenge every bullish premise with technical overextension, fundamental weaknesses, valuation/beta risks, regression curve conflicts, or Risk:Reward flaws, then deliver an objective execution rationale.

ASSET & SUGGESTED TRADE:
- Symbol: {symbol} ({analysis.get('name', symbol)}) | Exchange: {analysis.get('exchange', '')} | Sector: {analysis.get('sector', '')}
- Suggested Action: {tx_type} | Matrix Verdict: {d.get('action', 'N/A')} ({d.get('conviction', 'N/A')}) | Setup: {analysis.get('setup_type', 'N/A')}
- Sized Qty: {qty} shares | Spot Price: {sym_p}{analysis.get('price', 0)} ({analysis.get('change_pct', 0):+.2f}% today)
- Moreshwar Levels: Entry {sym_p}{m.get('entry_price', 0)} | Target {sym_p}{m.get('target', 0)} | Stop-Loss {sym_p}{m.get('stop_loss', 0)} | R:R 1:{m.get('rr_ratio', 0)}
- Degree-2 Polynomial Regression: {analysis.get('regression_curve', 'N/A')} | 30D Quant Target: {sym_p}{analysis.get('projected_30d_target', 0)}

TECHNICAL TELEMETRY (TechBot_1):
- Signal: {tech_msg.get('signal', 'N/A')} (Conf {round(float(tech_msg.get('confidence', 0))*100)}%)
- RSI(14): {analysis.get('rsi_14', 'N/A')} | MACD: {analysis.get('macd', 'N/A')} vs Signal {analysis.get('macd_signal', 'N/A')} | 20D SMA: {sym_p}{analysis.get('sma_20', 0)}
- Tech Summary: {tech_msg.get('summary', 'N/A')}

FUNDAMENTAL & MOAT TELEMETRY (StockSight V2 / FundBot_1):
- Total Fundamental Score: {f.get('total', 0)}/100 (Business {f.get('business', 0)}/40, Moat {f.get('moat', 0)}/20, Management/Valuation {f.get('management', 0)}/20, Risk/Stability {f.get('risk', 0)}/20)
- Ratios: Trailing P/E={r.get('pe', 'N/A')}x, ROE={r.get('roe', 'N/A')}%, ROCE={r.get('roce', 'N/A')}%, Sales Growth={r.get('sales_growth', 'N/A')}%, Profit Growth={r.get('profit_growth', 'N/A')}%, OPM={r.get('opm', 'N/A')}%, Beta={r.get('beta', 'N/A')}
- Porter's 5 Forces Moat: {p.get('total', 0)}/100 (Entrants {p.get('entrants', 0)}, Suppliers {p.get('suppliers', 0)}, Buyers {p.get('buyers', 0)}, Substitutes {p.get('substitutes', 0)}, Rivalry {p.get('rivalry', 0)})
- Risk Explanation: {f.get('explanation', '')}
- ExecBot_1 Conviction: {conv_pct}% | Summary: {exec_msg.get('summary', 'N/A')}

Return ONLY a valid JSON object with these exact string keys:
{{
  "stance": "ONE OF: HIGH-CONVICTION BUY | CAUTIOUS ACCUMULATE | WAIT FOR PULLBACK | HOLD / MONITOR | HIGH RISK / AVOID | EXIT WARNING",
  "risk_rating": "ONE OF: LOW | MEDIUM | HIGH | EXTREME",
  "bull_case": "1-2 crisp sentences citing exact supporting numbers.",
  "critical_risks": "1-2 sharp adversarial sentences calling out overbought RSI, low fundamental score, weak pillar, downward curve conflict, high P/E/Beta, or tight R:R.",
  "rationale": "2-3 sentences giving the decisive critical trade rationale with exact Entry/Target/SL/sizing discipline."
}}"""


def analyze_trade_with_gemini(
    symbol: str,
    analysis: dict[str, Any],
    pipeline: dict[str, Any] | None = None,
    trade_pick: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run live Gemini critical trade analysis for a single symbol."""
    key = (api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")).strip()
    now_iso = datetime.now(timezone.utc).isoformat()

    if not key or key == "your_gemini_api_key_here":
        return {
            "symbol": symbol,
            "status": "unavailable",
            "stance": "GEMINI KEY MISSING",
            "risk_rating": "UNKNOWN",
            "bull_case": "",
            "critical_risks": "GEMINI_API_KEY is not configured in GitHub Actions Secrets or .env.",
            "rationale": "Configure a valid GEMINI_API_KEY secret to enable live critical trade analysis (zero simulation policy).",
            "model": None,
            "generated_at": now_iso,
            "error": "GEMINI_API_KEY not configured",
        }

    prompt = build_trade_critique_prompt(symbol, analysis, pipeline=pipeline, trade_pick=trade_pick)
    try:
        raw_text, model_used = _call_gemini_generate(prompt, api_key=key, response_mime_type="application/json")
        parsed = _extract_json_block(raw_text)
        return {
            "symbol": symbol,
            "status": "live",
            "stance": str(parsed.get("stance", "CAUTIOUS ACCUMULATE")).strip().upper(),
            "risk_rating": str(parsed.get("risk_rating", "MEDIUM")).strip().upper(),
            "bull_case": str(parsed.get("bull_case", "")).strip(),
            "critical_risks": str(parsed.get("critical_risks", "")).strip(),
            "rationale": str(parsed.get("rationale", "")).strip(),
            "model": model_used,
            "generated_at": now_iso,
            "error": None,
        }
    except Exception as exc:
        err = _redact_key(str(exc), key)
        return {
            "symbol": symbol,
            "status": "error",
            "stance": "GEMINI API ERROR",
            "risk_rating": "UNKNOWN",
            "bull_case": "",
            "critical_risks": err,
            "rationale": f"Live Gemini critical analysis failed: {err}. Zero simulated fallback applied.",
            "model": GEMINI_DEFAULT_MODEL,
            "generated_at": now_iso,
            "error": err,
        }


def critique_suggested_trades(
    assets: dict[str, dict[str, Any]],
    pipelines: dict[str, dict[str, Any]],
    trade_queue: list[dict[str, Any]],
    api_key: str | None = None,
    max_workers: int = 4,
) -> dict[str, dict[str, Any]]:
    """Critically analyze every suggested trade and watchlist asset via live Gemini API."""
    key = (api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")).strip()
    pick_by_sym = {str(t.get("symbol", "")).upper(): t for t in trade_queue if t.get("symbol")}
    symbols = list(assets.keys())
    if not symbols:
        return {}

    if not key or key == "your_gemini_api_key_here":
        return {
            sym: analyze_trade_with_gemini(sym, assets[sym], pipelines.get(sym), pick_by_sym.get(sym), api_key="")
            for sym in symbols
        }

    # Probe the first symbol sequentially so an invalid/revoked key fails fast in 1 call instead of N parallel failures
    results: dict[str, dict[str, Any]] = {}
    first_sym = symbols[0]
    first_res = analyze_trade_with_gemini(
        first_sym, assets[first_sym], pipelines.get(first_sym), pick_by_sym.get(first_sym), api_key=key
    )
    results[first_sym] = first_res
    if first_res.get("status") == "error" and any(code in str(first_res.get("error", "")) for code in ("HTTP 400", "HTTP 401", "HTTP 403")):
        for sym in symbols[1:]:
            results[sym] = {
                **first_res,
                "symbol": sym,
            }
        return results

    remaining = symbols[1:]
    if not remaining:
        return results

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(remaining)))) as pool:
        future_map = {
            pool.submit(
                analyze_trade_with_gemini,
                sym,
                assets[sym],
                pipelines.get(sym),
                pick_by_sym.get(sym),
                key,
            ): sym
            for sym in remaining
        }
        for fut in as_completed(future_map):
            sym = future_map[fut]
            try:
                results[sym] = fut.result()
            except Exception as exc:
                results[sym] = {
                    "symbol": sym,
                    "status": "error",
                    "stance": "GEMINI API ERROR",
                    "risk_rating": "UNKNOWN",
                    "bull_case": "",
                    "critical_risks": _redact_key(str(exc), key),
                    "rationale": f"Live Gemini request failed: {_redact_key(str(exc), key)}",
                    "model": GEMINI_DEFAULT_MODEL,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "error": _redact_key(str(exc), key),
                }
    return results


def ask_gemini_stocky(message: str, context_summary: str, api_key: str | None = None) -> str | None:
    """Optional live Gemini backing for Stocky AI interactive chat when key is valid."""
    key = (api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")).strip()
    if not key or key == "your_gemini_api_key_here":
        return None
    prompt = (
        "You are Stocky AI inside Finsistant Version 2, a strict quantitative & fundamental trading terminal.\n"
        "Answer concisely using ONLY the live market context below.\n\n"
        f"LIVE PORTFOLIO & MARKET CONTEXT:\n{context_summary}\n\n"
        f"USER QUESTION: {message}"
    )
    text, _ = _call_gemini_generate(prompt, api_key=key, response_mime_type="", timeout=20)
    return text
