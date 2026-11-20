"""Moreshwar 3.1 (Prompt 2/5) — 4-Tier Deterministic Sector & Industry Classification Hierarchy.

Resolves the Moreshwar 3.0 Prompt 1 forensic finding where 2,085 of 2,832 panel equities
(73.62%) collapsed into a single fallback sector label ("Unclassified (Historical/Non-Total-Market)"),
causing `max_sector_positions = 2` to falsely block 114 trades in `PORT_RISK_BUDGETED`.

Hierarchy:
1. TIER_1_VALIDATED_NSE_SECTOR: Authoritative NSE index macro-sector classification already present.
2. TIER_2_REFERENCE_MAPPING: Maintained corporate reference table for major merged, delisted, DVR,
   and well-known NSE listed companies omitted from the current Nifty Total Market snapshot.
3. TIER_3_COMPANY_NAME_TAXONOMY: Deterministic, auditable industry/sector classification derived
   strictly from official NSE `security_master.company_name` keywords (zero future data).
4. TIER_4_UNKNOWN_CONTROLLED: Explicit `UNKNOWN_SECTOR` / `UNKNOWN_INDUSTRY` governed by an
   independent `max_unknown_sector_exposure_pct` and individual stock cap rather than consuming
   one shared 2-stock sector slot.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np

from finsistant_v2.m3.schema import DEFAULT_M3_DB_PATH, connect_m3_db

UNKNOWN_SECTOR_LABEL = "UNKNOWN_SECTOR"
UNKNOWN_INDUSTRY_LABEL = "UNKNOWN_INDUSTRY"

# Tier 2 Maintained Corporate Reference Mapping (merged, delisted, DVR, and prominent non-snapshot NSE equities)
MAINTAINED_SYMBOL_SECTOR_MAP: dict[str, tuple[str, str]] = {
    "HDFC": ("Financial Services", "Housing Finance"),
    "ISEC": ("Financial Services", "Capital Markets"),
    "IDFC": ("Financial Services", "Investment & Holding Companies"),
    "IBULHSGFIN": ("Financial Services", "Housing Finance"),
    "DHFL": ("Financial Services", "Housing Finance"),
    "RELCAPITAL": ("Financial Services", "Diversified Financial Services"),
    "SRTRANSFIN": ("Financial Services", "Non Banking Financial Company"),
    "PRUDENT": ("Financial Services", "Wealth & Advisory Services"),
    "GEOJITFSL": ("Financial Services", "Capital Markets"),
    "EQUITAS": ("Financial Services", "Small Finance & Holdings"),
    "UJJIVAN": ("Financial Services", "Microfinance & Holdings"),
    "MINDTREE": ("Information Technology", "IT Consulting & Software"),
    "LTI": ("Information Technology", "IT Consulting & Software"),
    "PROTEAN": ("Information Technology", "Digital Public Infrastructure"),
    "TREJHARA": ("Information Technology", "IT Software & Solutions"),
    "TATAMTRDVR": ("Automobile and Auto Components", "Passenger & Commercial Vehicles"),
    "MOTHERSUMI": ("Automobile and Auto Components", "Auto Components"),
    "JBCHEPHARM": ("Healthcare", "Pharmaceuticals"),
    "CADILAHC": ("Healthcare", "Pharmaceuticals"),
    "LAURUSLABS": ("Healthcare", "Pharmaceuticals"),
    "BIKAJI": ("Fast Moving Consumer Goods", "Packaged Foods"),
    "FELDVR": ("Consumer Services", "Retail"),
    "STOVEKRAFT": ("Consumer Durables", "Household Appliances"),
    "ZEEL": ("Media Entertainment & Publication", "Media & Broadcasting"),
    "SUNTV": ("Media Entertainment & Publication", "Media & Broadcasting"),
    "INFRATEL": ("Telecommunication", "Telecom Infrastructure"),
    "AVANTEL": ("Telecommunication", "Telecom & Defence Electronics"),
    "HARIOMPIPE": ("Metals & Mining", "Iron & Steel Products"),
    "RAJMET": ("Metals & Mining", "Metals Trading & Processing"),
    "INTERARCH": ("Construction", "Pre-Engineered Steel Buildings"),
    "SVLL": ("Services", "Logistics"),
    "JETAIRWAYS": ("Services", "Aviation"),
    "RCOM": ("Telecommunication", "Telecom Services"),
    "SINTEX": ("Textiles", "Textiles & Plastics"),
    "SPENTEX": ("Textiles", "Textiles & Yarn"),
    "DHARMAJ": ("Chemicals", "Agrochemicals"),
    "OSWALSEEDS": ("Fast Moving Consumer Goods", "Agricultural Products"),
    "JISLDVREQS": ("Capital Goods", "Micro Irrigation & Agro Equipment"),
    "AHLADA": ("Capital Goods", "Industrial Steel Products"),
    "SGIL": ("Capital Goods", "Castings & Wind Energy Components"),
    "AMDIND": ("Capital Goods", "Packaging Products"),
}

# Tier 3 Deterministic Company-Name Keyword Rules -> (Sector, Industry)
_NAME_TAXONOMY_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"\b(BANK|FINANC|FINSERV|FINVEST|FINCORP|CAPITAL|SECURIT|BROKING|INVESTMENT|HOUSING|CREDIT|LEASING|INSURANCE|ASSET MANAGEMENT|MUTUAL|WEALTH|MERCANTILE|FACTORING|HOLDINGS?\b|STOCK)\w*\b", re.I),
        "Financial Services",
        "Financial Services & NBFC",
    ),
    (
        re.compile(r"\b(PHARMA|LABORATOR|DRUG|HEALTH|HOSPITAL|MEDICAL|DIAGNOSTIC|BIOTECH|LIFESCIENCE|LIFE SCIENCE|REMED|CLINICAL|SURGICAL|AYURVED|MEDICARE|FORMULATION)\w*\b", re.I),
        "Healthcare",
        "Pharmaceuticals & Healthcare",
    ),
    (
        re.compile(r"\b(SOFTWARE|INFOTECH|TECHNOLOG|SYSTEMS|DIGITAL|CYBER|CLOUD|E-?COMMERCE|INTERNET|IT SERVICES|COMPUT|SOLUTIONS|INFOSYS|DATA\s+SYSTEMS|TECH\b|INFO\b)\w*\b", re.I),
        "Information Technology",
        "IT Software & Services",
    ),
    (
        re.compile(r"\b(CHEMIC|PETROCHEM|FERTILIZ|FERTILIS|AGROCHEM|PESTICID|DYESTUFF|PIGMENT|PAINT|RESIN|POLYMER|ALKALI|ACID|SOLVENT|COATING|ADHESIVE|PLASTIC|POLYPLEX|CARBON|NITRO|CHLOR|FLUORO|AROMATIC|ORGANICS?)\w*\b", re.I),
        "Chemicals",
        "Specialty & Commodity Chemicals",
    ),
    (
        re.compile(r"\b(STEEL|METAL|ALUMIN|COPPER|ZINC|IRON|ALLOY|MINING|MINERAL|FORGING|FOUNDRY|CASTING|TUBE|PIPE|WIRE|SMELT|FERRO|COAL|ORE\b|ISPAT|METALLURG|TINPLATE)\w*\b", re.I),
        "Metals & Mining",
        "Metals & Metal Products",
    ),
    (
        re.compile(r"\b(AUTO|MOTOR|TYRE|TIRE|TRACTOR|VEHICLE|CYCLE|GEAR|AXLE|CLUTCH|BRAKE|SPRING|PISTON|BEARING|TRANSMISSION|MOBILITY|WHEEL|FORGE)\w*\b", re.I),
        "Automobile and Auto Components",
        "Auto & Auto Components",
    ),
    (
        re.compile(r"\b(FOOD|AGRO|SUGAR|DAIRY|BEVERAGE|COFFEE|SPICE|FLOUR|RICE|EDIBLE|BREW|DISTILL|WINE|SPIRIT|CONSUMER|PERSONAL CARE|SOAP|COSMETIC|NUTRITION|BISCUIT|BAKERY|SEED|POULTRY|SEAFOOD|FEED|TEA\b|MILLS?\b|AGRI|PLANTATION|FRUIT|GRAIN|OIL\s+MILLS)\w*\b", re.I),
        "Fast Moving Consumer Goods",
        "Food, Agro & Consumer Goods",
    ),
    (
        re.compile(r"\b(TEXTIL|SPINNING|COTTON|YARN|FABRIC|APPAREL|GARMENT|WEAVING|WOOL|SILK|DENIM|KNIT|LOOM|SYNTHETIC|FIBRE|FIBER|FASHION|CLOTHING|FOOTWEAR|LEATHER|SYNTHETICS|SUITING|SHIRTING|SPINTEX|TEX\b)\w*\b", re.I),
        "Textiles",
        "Textiles & Apparel",
    ),
    (
        re.compile(r"\b(CEMENT|CONCRETE|CERAMIC|TILE|MARBLE|GRANITE|GLASS|BRICK|REFRACTOR|BUILDING MATERIAL|PLYWOOD|LAMINATE|TIMBER|WOOD|SANITARYWARE|GRANITO|VITRIFIED)\w*\b", re.I),
        "Construction Materials",
        "Cement & Building Materials",
    ),
    (
        re.compile(r"\b(REALTY|REAL ESTATE|PROPERT|ESTATE|DEVELOPER|HOUSING DEV|TOWNSHIP|BUILDCON|HOMES|LANDMARK|RESIDENC)\w*\b", re.I),
        "Realty",
        "Real Estate Development",
    ),
    (
        re.compile(r"\b(INFRA|CONSTRUCTION|PROJECT|BUILD|CONTRACTOR|ROAD|HIGHWAY|BRIDGE|CIVIL|EPC\b)\w*\b", re.I),
        "Construction",
        "Infrastructure & Construction",
    ),
    (
        re.compile(r"\b(POWER|ENERGY|ELECTRIC|SOLAR|WIND|HYDRO|THERMAL|UTILITY|GRID|RENEWABLE)\w*\b", re.I),
        "Power",
        "Power Generation & Transmission",
    ),
    (
        re.compile(r"\b(PETROLEUM|REFINER|FUEL|LUBRICANT|OIL\b|GAS\b|LPG\b|LNG\b)\w*\b", re.I),
        "Oil Gas & Consumable Fuels",
        "Oil, Gas & Petroleum",
    ),
    (
        re.compile(r"\b(TELECOM|COMMUNICATION|NETWORK|WIRELESS|BROADBAND|SATELLITE|TOWER)\w*\b", re.I),
        "Telecommunication",
        "Telecom Equipment & Services",
    ),
    (
        re.compile(r"\b(MEDIA|ENTERTAINMENT|BROADCAST|TELEVISION|CINEMA|FILM|MUSIC|PUBLISH|NEWS|PRINT|ADVERTISING|STUDIO)\w*\b", re.I),
        "Media Entertainment & Publication",
        "Media & Publishing",
    ),
    (
        re.compile(r"\b(HOTEL|RESORT|HOSPITALITY|TOURISM|TRAVEL|RESTAURANT|RETAIL|STORE|EDUCATION|LEARNING|SCHOOL|AVIATION|AIRLINE|LEISURE|MART\b)\w*\b", re.I),
        "Consumer Services",
        "Hospitality, Retail & Services",
    ),
    (
        re.compile(r"\b(LOGISTIC|SHIPPING|TRANSPORT|CARGO|FREIGHT|WAREHOUS|EXPRESS|COURIER|SUPPLY CHAIN|DREDGING|TRADING|CONSULTING|STAFFING|SECURITY SERVICES|PORT\b|PORTS\b|MARITIME|SERVICES\b)\w*\b", re.I),
        "Services",
        "Logistics & Commercial Services",
    ),
    (
        re.compile(r"\b(JEWEL|GOLD|DIAMOND|APPLIANCE|ELECTRONIC|LIGHTING|WATCH|FURNITURE|KITCHEN|AIR CONDITION|COOLER|CONSUMER DURABLE|HOME\b|FAN\b|FANS\b)\w*\b", re.I),
        "Consumer Durables",
        "Consumer Durables & Lifestyle",
    ),
    (
        re.compile(r"\b(ENGINEER|MACHIN|EQUIPMENT|CABLE|TRANSFORMER|SWITCHGEAR|PUMP|VALVE|COMPRESSOR|TURBINE|AEROSPACE|DEFENCE|DEFENSE|ROBOT|AUTOMATION|TOOL|INDUSTRIAL|BATTERY|BATTERIES|PAPER|PULP|PACKAG|CONTAINER|RUBBER|PRECISION|FABRICAT|BOILER|CRANE|ELEVATOR)\w*\b", re.I),
        "Capital Goods",
        "Industrial & Electrical Equipment",
    ),
]


def build_4tier_sector_classification(
    panel: dict[str, Any],
    *,
    db_path: Path = DEFAULT_M3_DB_PATH,
) -> dict[str, Any]:
    """Classify all panel equities through the 4-tier hierarchy and return enriched arrays + audit rows."""
    sec_ids = [str(s) for s in panel["sec_ids"]]
    syms = [str(s) for s in panel["current_symbols"]]
    orig_sectors = [str(s) for s in panel["sectors"]]
    orig_industries = [str(s) for s in panel["industries"]]
    elig = panel["is_pit_elig"]

    comp_names: dict[str, str] = {}
    statuses: dict[str, str] = {}
    if db_path.exists():
        conn = connect_m3_db(db_path)
        for r in conn.execute("SELECT sec_id, company_name, status FROM security_master"):
            comp_names[str(r["sec_id"])] = str(r["company_name"] or "")
            statuses[str(r["sec_id"])] = str(r["status"] or "")
        conn.close()

    n_secs = len(sec_ids)
    resolved_sectors = np.empty(n_secs, dtype=object)
    resolved_industries = np.empty(n_secs, dtype=object)
    resolution_tiers = np.empty(n_secs, dtype=object)
    is_unknown = np.zeros(n_secs, dtype=bool)

    tier_counts = {
        "TIER_1_VALIDATED_NSE_SECTOR": 0,
        "TIER_2_REFERENCE_MAPPING": 0,
        "TIER_3_COMPANY_NAME_TAXONOMY": 0,
        "TIER_4_UNKNOWN_CONTROLLED": 0,
    }
    audit_rows: list[dict[str, Any]] = []

    for j in range(n_secs):
        sid = sec_ids[j]
        sym = syms[j].upper().strip()
        o_sec = orig_sectors[j].strip()
        o_ind = orig_industries[j].strip()
        cname = comp_names.get(sid, sym)
        elig_days = int(np.sum(elig[:, j]))

        if o_sec and "Unclassified" not in o_sec and o_sec != UNKNOWN_SECTOR_LABEL:
            r_sec = o_sec
            r_ind = o_ind if (o_ind and "Unclassified" not in o_ind) else o_sec
            tier = "TIER_1_VALIDATED_NSE_SECTOR"
        elif sym in MAINTAINED_SYMBOL_SECTOR_MAP:
            r_sec, r_ind = MAINTAINED_SYMBOL_SECTOR_MAP[sym]
            tier = "TIER_2_REFERENCE_MAPPING"
        else:
            matched_sec: str | None = None
            matched_ind: str | None = None
            search_text = f"{cname} {sym}"
            for rx, m_sec, m_ind in _NAME_TAXONOMY_RULES:
                if rx.search(search_text):
                    matched_sec = m_sec
                    matched_ind = m_ind
                    break
            if matched_sec is not None and matched_ind is not None:
                r_sec = matched_sec
                r_ind = matched_ind
                tier = "TIER_3_COMPANY_NAME_TAXONOMY"
            else:
                r_sec = UNKNOWN_SECTOR_LABEL
                r_ind = UNKNOWN_INDUSTRY_LABEL
                tier = "TIER_4_UNKNOWN_CONTROLLED"
                is_unknown[j] = True

        resolved_sectors[j] = r_sec
        resolved_industries[j] = r_ind
        resolution_tiers[j] = tier
        tier_counts[tier] += 1

        audit_rows.append({
            "sec_id": sid,
            "symbol": sym,
            "company_name": cname,
            "lifecycle_status": statuses.get(sid, "UNKNOWN"),
            "pit_eligible_days": elig_days,
            "original_m3_sector": o_sec,
            "resolved_m3_1_sector": r_sec,
            "resolved_m3_1_industry": r_ind,
            "classification_tier": tier,
            "is_unknown_sector": bool(is_unknown[j]),
        })

    tot_elig_days = max(1, int(np.sum(elig)))
    orig_unclass_mask = np.array(["Unclassified" in s for s in orig_sectors], dtype=bool)
    orig_unclass_elig_pct = round(float(np.sum(elig[:, orig_unclass_mask])) / tot_elig_days * 100.0, 2)
    new_unknown_elig_pct = round(float(np.sum(elig[:, is_unknown])) / tot_elig_days * 100.0, 2)

    sector_distribution: dict[str, int] = {}
    for s in resolved_sectors:
        sector_distribution[str(s)] = sector_distribution.get(str(s), 0) + 1

    return {
        "sectors": resolved_sectors,
        "industries": resolved_industries,
        "tiers": resolution_tiers,
        "is_unknown": is_unknown,
        "audit_rows": audit_rows,
        "summary": {
            "total_panel_securities": n_secs,
            "original_unclassified_count": int(np.sum(orig_unclass_mask)),
            "original_unclassified_pct": round(float(np.sum(orig_unclass_mask)) / max(1, n_secs) * 100.0, 2),
            "original_unclassified_elig_stock_days_pct": orig_unclass_elig_pct,
            "tier_counts": tier_counts,
            "remaining_unknown_count": int(np.sum(is_unknown)),
            "remaining_unknown_pct": round(float(np.sum(is_unknown)) / max(1, n_secs) * 100.0, 2),
            "remaining_unknown_elig_stock_days_pct": new_unknown_elig_pct,
            "reclassified_securities_count": int(np.sum(orig_unclass_mask) - np.sum(is_unknown)),
            "resolved_sector_distribution": dict(sorted(sector_distribution.items(), key=lambda kv: -kv[1])),
        },
    }
