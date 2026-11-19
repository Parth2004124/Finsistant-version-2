"""Integrated Moreshwar 2.2 Quantitative Architecture (Prompt 3 of 3).

Combines ONLY the independently validated components from Prompts 1 & 2 into five
cleanly separated, non-redundant responsibilities:
1. SelectionIntelligence: Causal PIT quality/liquidity filters + non-redundant
   cross-sectional factor ranking (no volume_confirmation, no alphabetical bias).
2. EntryIntelligence: Pullback + Momentum Continuation (`PULL_CONT`) with
   anti-extension bounds (pure 20D breakouts rejected per Prompt 2 failure analysis).
3. ExitIntelligence: Volatility-normalized initial ATR SL + cost-covered Breakeven
   Lock (`+0.65%` @ `+2.0x ATR`) + optional regime-deterioration tightening & 60D
   runaway room (zero fixed TP clipping).
4. RiskIntelligence: Equal-weight, Risk-Budget, or Inverse-Volatility (`INV_VOL`)
   sizing, causal regime exposure throttling, and Portfolio Drawdown Monitor.
5. PortfolioIntelligence: Ranked slot allocation (`N=4..8`), sector/correlation
   limits, stop-loss re-entry cooldown, and cash preservation.

Strictly isolated from frozen Moreshwar 1.0, 2.0, and 2.1. Never triggers live trades.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .moreshwar_v2_1 import compute_pairwise_return_correlation
from .moreshwar_v2_2 import DeliveryCostConfig
from .moreshwar_v2_2_p2 import evaluate_entry_archetypes_p2, MoreshwarV22P2Config


@dataclass(frozen=True)
class IntegratedM22Config:
    """Versioned system configuration for Integrated Moreshwar 2.2."""

    model_version: str = "2.2.0-integrated"
    # Entry & Exit validated backbone
    entry_mode: str = "PULL_CONT"  # Validated Phase A winners (Breakout omitted)
    exit_mode: str = "P1_BREAKEVEN_LOCK"
    horizon_days: int = 60
    initial_atr_mult: float = 2.0
    breakeven_trigger_atr: float = 2.0
    breakeven_fee_buffer_pct: float = 0.65
    min_stop_pct: float = 2.5
    max_stop_pct: float = 10.0

    # Portfolio & Risk Intelligence
    sizing_mode: str = "RISK_BUDGET"  # "RISK_BUDGET" | "EQUAL_WEIGHT" | "INV_VOL"
    max_open_positions: int = 6
    max_sector_positions: int = 2
    max_pairwise_corr: float = 0.78
    risk_per_trade_pct: float = 1.0
    max_portfolio_stop_risk_pct: float = 6.0
    max_single_stock_pct: float = 18.0
    max_portfolio_exposure_pct: float = 85.0
    reentry_cooldown_days: int = 5
    max_daily_volume_participation: float = 0.01

    # Regime Exposure & Drawdown Protection
    use_dynamic_regime_exposure: bool = False
    use_drawdown_throttle: bool = False
    dd_throttle_level1_pct: float = 8.0
    dd_throttle_level2_pct: float = 12.0
    dd_recovery_pct: float = 5.0
    tighten_be_on_bear_regime: bool = False

    # Execution & Stress toggles (for Stress Suite only)
    entry_delay_bars: int = 1  # Normal = 1 (signal t -> next-bar open t+1); 2 = delayed stress
    gap_shock_extra_pct: float = 0.0  # Extra % gap-through-stop slippage stress
    require_pit_filters: bool = True
    require_bull_regime_entry: bool = False


class SelectionAndEntryEngine:
    """SelectionIntelligence + EntryIntelligence delegating strictly causally."""

    @staticmethod
    def evaluate_universe_day(
        *,
        valid_syms: list[str],
        open_positions: dict[str, dict[str, Any]],
        cooldown_until_idx: dict[str, int],
        d_idx: int,
        dt: str,
        features: dict[str, dict[str, dict[str, Any]]],
        uni: dict[str, dict[str, Any]],
        regime6_obj: dict[str, Any],
        cfg: IntegratedM22Config,
        tariff: DeliveryCostConfig,
    ) -> list[dict[str, Any]]:
        p2_cfg = MoreshwarV22P2Config(
            entry_mode=cfg.entry_mode,
            exit_mode=cfg.exit_mode,
            horizon_days=cfg.horizon_days,
            initial_atr_mult=cfg.initial_atr_mult,
            min_stop_pct=cfg.min_stop_pct,
            max_stop_pct=cfg.max_stop_pct,
            breakeven_trigger_atr=cfg.breakeven_trigger_atr,
            breakeven_fee_buffer_pct=cfg.breakeven_fee_buffer_pct,
            require_pit_filters=cfg.require_pit_filters,
            require_bull_regime_entry=cfg.require_bull_regime_entry,
            max_open_positions=cfg.max_open_positions,
            max_sector_positions=cfg.max_sector_positions,
            max_pairwise_corr=cfg.max_pairwise_corr,
            risk_per_trade_pct=cfg.risk_per_trade_pct,
            max_portfolio_stop_risk_pct=cfg.max_portfolio_stop_risk_pct,
            max_single_stock_pct=cfg.max_single_stock_pct,
            max_portfolio_exposure_pct=cfg.max_portfolio_exposure_pct,
            reentry_cooldown_days=cfg.reentry_cooldown_days,
            max_daily_volume_participation=cfg.max_daily_volume_participation,
        )
        candidates: list[dict[str, Any]] = []
        for sym in valid_syms:
            if sym in open_positions or dt not in features[sym]:
                continue
            if d_idx < cooldown_until_idx.get(sym, -1):
                continue
            ft = features[sym][dt]
            b_idx = ft["idx"]
            c_win = uni[sym]["candles"][max(0, b_idx - 61) : b_idx + 1]
            ev = evaluate_entry_archetypes_p2(
                symbol=sym,
                feats=ft,
                candles_window=c_win,
                regime6=regime6_obj,
                cfg=p2_cfg,
                tariff=tariff,
            )
            if ev["matched"]:
                ev["sector"] = uni[sym].get("sector", "Equity")
                ev["signal_date"] = dt
                ev["recent_closes"] = [float(b["close"]) for b in c_win[-61:]]
                candidates.append(ev)

        candidates.sort(key=lambda x: (-x["rank_score"], -x["trend_strength"], x["symbol"]))
        return candidates


class PortfolioRiskAllocator:
    """RiskIntelligence + PortfolioIntelligence: sizing, concentration & drawdown guards."""

    @staticmethod
    def allocate(
        *,
        candidates: list[dict[str, Any]],
        open_positions: dict[str, dict[str, Any]],
        cash: float,
        nav: float,
        current_dd_pct: float,
        dd_throttled: bool,
        regime6_name: str,
        cfg: IntegratedM22Config,
    ) -> list[dict[str, Any]]:
        if cfg.use_dynamic_regime_exposure:
            if regime6_name == "STRONG_BULLISH":
                reg_cap_pct = cfg.max_portfolio_exposure_pct
                reg_risk_mult = 1.0
            elif regime6_name == "WEAK_BULLISH":
                reg_cap_pct = min(cfg.max_portfolio_exposure_pct, 65.0)
                reg_risk_mult = 0.80
            elif regime6_name == "SIDEWAYS":
                reg_cap_pct = min(cfg.max_portfolio_exposure_pct, 40.0)
                reg_risk_mult = 0.50
            else:
                return []  # Preserve 100% cash in bearish / high-vol regimes
        else:
            reg_cap_pct = cfg.max_portfolio_exposure_pct
            reg_risk_mult = 1.0 if regime6_name == "STRONG_BULLISH" else (0.80 if regime6_name == "WEAK_BULLISH" else 0.60)

        dd_mult = 1.0
        if cfg.use_drawdown_throttle and dd_throttled:
            dd_mult = 0.25 if current_dd_pct >= cfg.dd_throttle_level2_pct else 0.50

        tentative_open = [
            {
                "symbol": s,
                "sector": p["sector"],
                "entry_price": p["entry_price"],
                "sl": p["sl"],
                "qty": p["qty"],
                "recent_closes": p.get("recent_closes", []),
                "notional": p["last_price"] * p["qty"],
            }
            for s, p in open_positions.items()
        ]
        deployed_amt = sum(p["notional"] for p in tentative_open)
        avail_cash = cash
        orders: list[dict[str, Any]] = []

        for cand in candidates:
            if len(tentative_open) >= cfg.max_open_positions:
                break
            sec = cand["sector"]
            if sum(1 for p in tentative_open if p["sector"] == sec) >= cfg.max_sector_positions:
                continue

            max_corr = 0.0
            for op_p in tentative_open:
                if op_p["recent_closes"] and cand["recent_closes"]:
                    max_corr = max(
                        max_corr,
                        compute_pairwise_return_correlation(cand["recent_closes"], op_p["recent_closes"], window=60),
                    )
            if max_corr > cfg.max_pairwise_corr:
                continue

            existing_sl = sum(max(0.0, (p["entry_price"] - p["sl"]) * p["qty"]) for p in tentative_open)
            rem_sl = max(0.0, nav * (cfg.max_portfolio_stop_risk_pct / 100.0) - existing_sl)
            spot = cand["spot"]
            risk_sh = max(0.01, spot * cand["sl_ratio"])
            if rem_sl < risk_sh:
                continue

            rem_cap = max(0.0, nav * (reg_cap_pct / 100.0) - deployed_amt)
            cap_limit = min(nav * (cfg.max_single_stock_pct / 100.0), rem_cap, avail_cash * 0.96)

            corr_mult = 0.65 if max_corr >= 0.65 else 1.0
            if cfg.sizing_mode == "EQUAL_WEIGHT":
                target_notional = (nav * (reg_cap_pct / 100.0)) / max(1, cfg.max_open_positions) * dd_mult
                target_notional = min(target_notional, cap_limit)
                qty = int(target_notional // max(1e-6, spot))
                # Still respect aggregate SL risk ceiling
                qty = min(qty, int(rem_sl // risk_sh))
            elif cfg.sizing_mode == "INV_VOL":
                # Lower-volatility names (near 1.5% ATR) receive larger target weight; noisy names smaller
                base_notional = (nav * (reg_cap_pct / 100.0)) / max(1, cfg.max_open_positions)
                vol_weight = max(0.45, min(1.35, 2.2 / max(0.8, cand["atr_pct"])))
                target_notional = min(base_notional * vol_weight * reg_risk_mult * dd_mult * corr_mult, cap_limit)
                qty = int(target_notional // max(1e-6, spot))
                qty = min(qty, int(rem_sl // risk_sh))
            else:  # "RISK_BUDGET"
                eff_risk = min(nav * (cfg.risk_per_trade_pct / 100.0) * reg_risk_mult * corr_mult * dd_mult, rem_sl)
                qty = max(0, min(int(eff_risk // risk_sh), int(cap_limit // max(1e-6, spot))))

            if qty < 1 or (qty * spot) < 1500.0:
                continue

            deployed_amt += qty * spot
            avail_cash -= qty * spot
            tentative_open.append({
                "symbol": cand["symbol"],
                "sector": sec,
                "entry_price": spot,
                "sl": spot * (1.0 - cand["sl_ratio"]),
                "qty": qty,
                "recent_closes": cand["recent_closes"],
                "notional": qty * spot,
            })
            orders.append({**cand, "req_qty": qty})

        return orders
