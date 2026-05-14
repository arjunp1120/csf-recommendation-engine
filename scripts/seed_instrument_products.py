"""Seed ``instrument_products`` from ``trade_history`` distincts (plan Step 0.10).

For every distinct ``(instrument_name, symbol)`` pair currently appearing
in ``trade_history``, derive a canonical product mapping:

  1. **Deterministic path** — regex/lookup over the CME symbol grammar
     (CL→WTI, BZ→Brent, HO→Heating Oil, RB→RBOB, NG→Natural Gas, GO→Gasoil,
     plus TAS variants CLT/HOT/RBT) and the human-readable name grammar
     (``<Product> <Mon><YY>``, ``Calendar``, ``Spread``, ``Butterfly``,
     ``Crack``, ``Swap``, ``Strip``). Writes with ``source='regex'``.

  2. **LLM fallback** — for anything the deterministic path returns None
     for, call the Instrument Resolver agent
     (``settings.daf_instrument_resolver_agent_id``) for a structured
     mapping. Writes with ``source='llm-suggested'``.

The script targets the **local DB only** (via ``DATABASE_URL`` in ``.env``).
Use ``--dry-run`` to preview without writing.

Manual step after run (per plan §0.10): operator reviews rows where
``source = 'llm-suggested'`` and sets ``reviewed_by`` / ``reviewed_at``.
The nightly unresolved-flag job (Step 0.9) handles new instruments
arriving after this initial seed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import asyncpg
from dotenv import load_dotenv

# Make the seed script importable from the repo root.
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

from csf_recommendation_engine.core.config import get_settings  # noqa: E402
from csf_recommendation_engine.domain.intelligence.intelligence_service import (  # noqa: E402
    IntelligenceService,
)
from csf_recommendation_engine.infra.db import instrument_products  # noqa: E402

logger = logging.getLogger("seed_instrument_products")


# ---------------------------------------------------------------------------
# Canonical vocabularies — derived from live DB inspection (2026-05-13)
# ---------------------------------------------------------------------------

ALLOWED_PRODUCTS: dict[str, str] = {
    # canonical product_name -> product_family
    "WTI Crude":   "Crude",
    "Brent Crude": "Crude",
    "Heating Oil": "Distillates",
    "RBOB":        "Refined",
    "Natural Gas": "Nat Gas",
    "Gasoil":      "Distillates",
}

ALLOWED_STRUCTURES: frozenset[str] = frozenset(
    {"Outright", "Spread", "Butterfly", "Crack", "Swap", "Strip", "Unknown"}
)

# Ordered ticker / human-name token mapping. Longer tokens FIRST to avoid
# "WTI" matching inside "WTI Crude" before "WTI Crude" matches. Tickers
# come after human names so "Heating Oil" wins over "HO" inside the same
# string. Tickers with TAS suffix (CLT/HOT/RBT) come before the parent
# ticker to disambiguate.
_PRODUCT_TOKENS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bHeating Oil\b", re.IGNORECASE),  "Heating Oil"),
    (re.compile(r"\bNatural Gas\b", re.IGNORECASE),  "Natural Gas"),
    (re.compile(r"\bNat\s*Gas\b",   re.IGNORECASE),  "Natural Gas"),
    (re.compile(r"\bWTI\s+Crude\b", re.IGNORECASE),  "WTI Crude"),
    (re.compile(r"\bBrent\s+Crude\b", re.IGNORECASE), "Brent Crude"),
    (re.compile(r"\bGasoil\b",      re.IGNORECASE),  "Gasoil"),
    (re.compile(r"\bRBOB\b",        re.IGNORECASE),  "RBOB"),
    (re.compile(r"\bWTI\b",         re.IGNORECASE),  "WTI Crude"),
    (re.compile(r"\bBrent\b",       re.IGNORECASE),  "Brent Crude"),
    # CME / ICE tickers (TAS variants first)
    (re.compile(r"\bCLT\b"),                          "WTI Crude"),
    (re.compile(r"\bHOT\b"),                          "Heating Oil"),
    (re.compile(r"\bRBT\b"),                          "RBOB"),
    (re.compile(r"\bCL\b"),                           "WTI Crude"),
    (re.compile(r"\bBZ\b"),                           "Brent Crude"),
    (re.compile(r"\bBRN\b"),                          "Brent Crude"),
    (re.compile(r"\bHO\b"),                           "Heating Oil"),
    (re.compile(r"\bRB\b"),                           "RBOB"),
    (re.compile(r"\bNG\b"),                           "Natural Gas"),
    (re.compile(r"\bGO\b"),                           "Gasoil"),
]

_MONTH_LETTER_CODES: dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
_MONTH_NAME_CODES: dict[str, int] = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
_NAME_MONTH_RE = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d{4}|\d{2})\b",
    re.IGNORECASE,
)
_SYMBOL_MONTH_RE = re.compile(r"([FGHJKMNQUVXZ])(\d{1,2})")


def _year_from_digits(yd: int) -> int:
    """1-digit (5..9, 0..4) → 2020s/2030s; 2-digit (25..) → 20XX directly."""
    if yd < 10:
        return 2020 + yd
    return 2000 + yd


def _parse_month_letter(letter: str, year_digits: str) -> str | None:
    if letter not in _MONTH_LETTER_CODES or not year_digits.isdigit():
        return None
    m = _MONTH_LETTER_CODES[letter]
    y = _year_from_digits(int(year_digits))
    return f"{y:04d}-{m:02d}"


def _parse_month_name(name: str, year_digits: str) -> str | None:
    key = name[:3].title()
    if key not in _MONTH_NAME_CODES or not year_digits.isdigit():
        return None
    m = _MONTH_NAME_CODES[key]
    yd = int(year_digits)
    y = _year_from_digits(yd) if len(year_digits) <= 2 else yd
    return f"{y:04d}-{m:02d}"


def _extract_month_from_name(text: str) -> str | None:
    match = _NAME_MONTH_RE.search(text)
    if not match:
        return None
    return _parse_month_name(match.group(1), match.group(2))


def _extract_month_from_symbol(sym: str) -> str | None:
    match = _SYMBOL_MONTH_RE.search(sym)
    if not match:
        return None
    return _parse_month_letter(match.group(1), match.group(2))


def _find_product(text: str) -> str | None:
    """Return the canonical product mentioned in ``text``.

    Picks the **leftmost** match across all tokens — this is what makes
    inter-product spreads like 'Brent/WTI Spread Jan 26' resolve to
    'Brent Crude' rather than 'WTI Crude' (first-named leg wins). On
    same-position ties (e.g. 'WTI Crude' vs 'WTI' both starting at 0),
    the LONGER match wins.
    """
    matches: list[tuple[int, int, str]] = []
    for compiled, product in _PRODUCT_TOKENS:
        m = compiled.search(text)
        if m is None:
            continue
        matches.append((m.start(), -(m.end() - m.start()), product))
    if not matches:
        return None
    matches.sort()  # leftmost start, then longest-length (via negative)
    return matches[0][2]


def _detect_structure(name: str, symbol: str | None) -> str:
    """Classify into one of the ALLOWED_STRUCTURES. Order matters —
    Strip / Swap / Butterfly / Crack are specific signals that must
    win over the more general Spread / Outright."""
    name_lower = name.lower()
    sym = (symbol or "").upper()

    if "butterfly" in name_lower or ":BF" in sym:
        return "Butterfly"
    if "crack" in name_lower or ":C1" in sym:
        return "Crack"
    if "swap" in name_lower:
        return "Swap"
    if "strip" in name_lower or re.match(r"^[A-Z]+CAL\d+$", sym):
        return "Strip"
    if "calendar" in name_lower or "spread" in name_lower:
        return "Spread"
    # Symbol-only spread signals
    if "-" in sym and re.match(r"^[A-Z]+[FGHJKMNQUVXZ]?\d", sym):
        # e.g. "CLM6-CLN6", "HOZ6-HOM7", "BRN-CL"
        return "Spread"
    if "/" in sym and re.match(r"^[A-Z]+[FGHJKMNQUVXZ]\d", sym):
        # e.g. "CLF6/CLG6"
        return "Spread"
    return "Outright"


# ---------------------------------------------------------------------------
# Deterministic resolver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolved:
    """Internal representation of one mapping, ready to upsert."""
    product_name: str
    product_family: str
    structure_type: str
    contract_month: str | None
    confidence: float
    source: Literal["regex", "llm-suggested"]
    reasoning: str


def deterministic_resolve(instrument_name: str, symbol: str | None) -> Resolved | None:
    """Try to resolve the (name, symbol) pair without an LLM call.

    Returns a :class:`Resolved` on success, ``None`` if the deterministic
    rules can't confidently pick a product (caller then falls back to LLM).
    """
    if not instrument_name:
        return None

    structure = _detect_structure(instrument_name, symbol)

    # Crack: product = refined leg (HO or RB), regardless of token order
    if structure == "Crack":
        refined = re.search(
            r"\b(HO|RB|RBOB|Heating Oil)\b", instrument_name, re.IGNORECASE
        )
        if refined is None:
            return None
        tok = refined.group(1).upper()
        product = "Heating Oil" if tok in ("HO", "HEATING OIL") else "RBOB"
    else:
        product = _find_product(instrument_name)
        if product is None and symbol:
            product = _find_product(symbol)
        if product is None:
            return None

    family = ALLOWED_PRODUCTS[product]

    month: str | None = None
    if structure != "Strip":
        month = _extract_month_from_name(instrument_name)
        if month is None and symbol:
            month = _extract_month_from_symbol(symbol)

    # Confidence calibration — high for fully unambiguous symbol/name pairs.
    if (
        symbol
        and structure == "Outright"
        and re.match(r"^[A-Z]{2,3}[FGHJKMNQUVXZ]\d{1,2}$", symbol)
    ):
        confidence = 1.0
    elif structure in ("Spread", "Butterfly", "Crack", "Swap", "Strip"):
        confidence = 0.9
    else:
        confidence = 0.95

    reasoning = (
        f"deterministic: product={product}, structure={structure}, "
        f"month={month or 'n/a'}, signals={{name='{instrument_name}', "
        f"symbol={symbol!r}}}"
    )
    return Resolved(
        product_name=product,
        product_family=family,
        structure_type=structure,
        contract_month=month,
        confidence=confidence,
        source="regex",
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------


async def llm_resolve(
    service: IntelligenceService,
    instrument_name: str,
    symbol: str | None,
) -> Resolved | None:
    """Call the Instrument Resolver agent. Returns ``None`` on transport
    failure, JSON-parse failure, schema mismatch, or when the agent
    returns ``product_name=None`` (declined to guess)."""
    resp = await service.resolve_instrument(instrument_name, symbol)
    if resp is None or not resp.product_name:
        return None
    if resp.product_name not in ALLOWED_PRODUCTS:
        logger.warning(
            "LLM returned out-of-vocab product %r for %r — dropping",
            resp.product_name, instrument_name,
        )
        return None
    structure = resp.structure_type if (resp.structure_type in ALLOWED_STRUCTURES) else "Unknown"
    return Resolved(
        product_name=resp.product_name,
        product_family=resp.product_family or ALLOWED_PRODUCTS[resp.product_name],
        structure_type=structure,
        contract_month=resp.contract_month,
        confidence=float(resp.confidence),
        source="llm-suggested",
        reasoning=resp.reasoning or "(agent returned no reasoning)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def fetch_distinct_pairs(
    conn: asyncpg.Connection,
) -> list[tuple[str, str | None]]:
    """Distinct (instrument_name, symbol) pairs in trade_history, sorted
    by ``instrument_name`` ascending. NULL symbols become Python None."""
    rows = await conn.fetch(
        """
        SELECT DISTINCT instrument_name, symbol
        FROM public.trade_history
        WHERE instrument_name IS NOT NULL
        ORDER BY instrument_name
        """
    )
    return [(str(r["instrument_name"]), (str(r["symbol"]) if r["symbol"] is not None else None)) for r in rows]


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed instrument_products from trade_history (Step 0.10)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve everything but do not write to instrument_products."
    )
    parser.add_argument(
        "--llm-budget", type=int, default=200,
        help="Max LLM calls per run; stop fallback after this many. Default 200."
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip the LLM fallback entirely; deterministic only."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N distinct pairs (for testing)."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    settings = get_settings()
    if not settings.database_url:
        logger.error("DATABASE_URL is empty — refusing to run.")
        return 2

    llm_available = bool(settings.daf_instrument_resolver_agent_id) and not args.no_llm
    if not llm_available:
        logger.warning(
            "LLM fallback disabled (agent id empty or --no-llm). "
            "Long-tail entries that deterministic rules can't handle will be skipped."
        )

    service = IntelligenceService(settings)
    conn = await asyncpg.connect(settings.database_url, ssl=False)
    counts: dict[str, int] = {"regex": 0, "llm-suggested": 0, "unresolved": 0, "skipped": 0}
    unresolved: list[tuple[str, str | None]] = []
    llm_calls = 0

    try:
        pairs = await fetch_distinct_pairs(conn)
        if args.limit is not None:
            pairs = pairs[: args.limit]
        logger.info("Loaded %d distinct (instrument_name, symbol) pairs", len(pairs))

        for idx, (name, sym) in enumerate(pairs, start=1):
            resolved = deterministic_resolve(name, sym)

            if resolved is None and llm_available and llm_calls < args.llm_budget:
                llm_calls += 1
                resolved = await llm_resolve(service, name, sym)

            if resolved is None:
                counts["unresolved"] += 1
                unresolved.append((name, sym))
                logger.warning(
                    "[%d/%d] UNRESOLVED  name=%r symbol=%r",
                    idx, len(pairs), name, sym,
                )
                continue

            counts[resolved.source] = counts.get(resolved.source, 0) + 1

            if args.dry_run:
                logger.info(
                    "[%d/%d] DRY-RUN     %s -> %s / %s / %s (conf=%.2f, source=%s)",
                    idx, len(pairs), name,
                    resolved.product_name, resolved.structure_type,
                    resolved.contract_month or "n/a",
                    resolved.confidence, resolved.source,
                )
            else:
                mapping_id = await instrument_products.upsert(
                    conn,
                    instrument_name=name,
                    product_name=resolved.product_name,
                    product_family=resolved.product_family,
                    structure_type=resolved.structure_type,
                    contract_month=resolved.contract_month,
                    confidence=resolved.confidence,
                    source=resolved.source,
                )
                logger.info(
                    "[%d/%d] %-13s %s -> %s / %s / %s (conf=%.2f, id=%s)",
                    idx, len(pairs), resolved.source.upper(), name,
                    resolved.product_name, resolved.structure_type,
                    resolved.contract_month or "n/a",
                    resolved.confidence, mapping_id,
                )

        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("  regex (deterministic): %d", counts["regex"])
        logger.info("  llm-suggested:         %d", counts["llm-suggested"])
        logger.info("  unresolved:            %d", counts["unresolved"])
        logger.info("  total LLM calls:       %d (budget=%d)", llm_calls, args.llm_budget)
        if unresolved:
            logger.warning("Unresolved names (operator review):")
            for name, sym in unresolved:
                logger.warning("    name=%r symbol=%r", name, sym)
        if args.dry_run:
            logger.info("DRY RUN — no rows were written.")
        return 0 if counts["unresolved"] == 0 else 1

    finally:
        await service.aclose()
        await conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
