"""Unit tests for ``ProductResolver`` (plan Step 0.9, exercised in Step 0.10).

Fixture data only — no DB. Verifies lookup precedence (exact wins over
regex), first-regex-wins on ties, malformed-pattern skip, empty-input
handling, and that the returned ``ProductMatch`` carries all the column
metadata from the source row.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from csf_recommendation_engine.domain.products import ProductMatch, ProductResolver


def _row(**overrides: object) -> dict:
    """Build a synthetic ``instrument_products`` row dict matching the
    shape returned by ``infra/db/instrument_products.load_all``."""
    base: dict[str, object] = {
        "mapping_id": "00000000-0000-0000-0000-000000000001",
        "instrument_name": None,
        "symbol_pattern": None,
        "product_name": "WTI Crude",
        "product_family": "Crude",
        "contract_month": None,
        "expiry_date": None,
        "structure_type": "Outright",
        "source": "manual",
        "confidence": 1.0,
        "reviewed_at": None,
        "reviewed_by": None,
        "created_at": None,
        "updated_at": None,
    }
    base.update(overrides)
    return base


class TestEmptyResolver:
    def test_empty_rows(self) -> None:
        r = ProductResolver([])
        assert r.resolve("anything") is None
        assert r.size == {"row_count": 0, "exact_entries": 0, "regex_entries": 0}
        assert len(r) == 0


class TestExactLookup:
    def test_basic_exact_match(self) -> None:
        rows = [_row(instrument_name="CL Jul26", product_name="WTI Crude")]
        r = ProductResolver(rows)
        m = r.resolve("CL Jul26")
        assert m is not None
        assert m.product_name == "WTI Crude"
        assert m.via == "exact"

    def test_case_sensitive(self) -> None:
        """Exact lookup is case-sensitive — matches the partial unique
        index in the DB (which doesn't lower-case)."""
        rows = [_row(instrument_name="CL Jul26", product_name="WTI Crude")]
        r = ProductResolver(rows)
        assert r.resolve("cl jul26") is None
        assert r.resolve("CL Jul26") is not None

    def test_unmappable_returns_none(self) -> None:
        rows = [_row(instrument_name="CL Jul26", product_name="WTI Crude")]
        r = ProductResolver(rows)
        assert r.resolve("absolutely_unknown_xyz") is None


class TestRegexLookup:
    def test_basic_regex_match(self) -> None:
        rows = [_row(symbol_pattern=r"^CL[FGHJKMNQUVXZ]\d{1,2}$", product_name="WTI Crude")]
        r = ProductResolver(rows)
        m = r.resolve("CLN6")
        assert m is not None
        assert m.product_name == "WTI Crude"
        assert m.via == "regex"

    def test_regex_uses_search_not_match(self) -> None:
        """``re.Pattern.search`` is used so a pattern can match anywhere
        in the string. Operators relying on full-string matches must
        anchor their patterns with ^...$ explicitly."""
        rows = [_row(symbol_pattern=r"BF", product_name="WTI Crude")]
        r = ProductResolver(rows)
        assert r.resolve("CL:BF M6-N6-Q6") is not None


class TestPrecedence:
    def test_exact_wins_over_regex(self) -> None:
        rows = [
            _row(
                mapping_id="11111111-1111-1111-1111-111111111111",
                symbol_pattern=r"^CL[A-Z]\d+$",
                product_name="WTI Crude (regex)",
            ),
            _row(
                mapping_id="22222222-2222-2222-2222-222222222222",
                instrument_name="CLM6",
                product_name="WTI Crude (exact)",
            ),
        ]
        r = ProductResolver(rows)
        m = r.resolve("CLM6")
        assert m is not None
        assert m.product_name == "WTI Crude (exact)"
        assert m.via == "exact"

    def test_first_regex_wins_on_multi_match(self) -> None:
        """Two regex rows both match → first by insertion order wins."""
        rows = [
            _row(mapping_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                 symbol_pattern=r"^CL", product_name="WTI Crude (first)"),
            _row(mapping_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                 symbol_pattern=r"^CLN6$", product_name="WTI Crude (second)"),
        ]
        r = ProductResolver(rows)
        m = r.resolve("CLN6")
        assert m is not None
        assert m.product_name == "WTI Crude (first)"


class TestMalformedPatternSkip:
    def test_invalid_regex_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        rows = [
            _row(mapping_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                 symbol_pattern="invalid[regex", product_name="X"),
            _row(mapping_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
                 symbol_pattern=r"^CL", product_name="WTI Crude"),
        ]
        with caplog.at_level("WARNING"):
            r = ProductResolver(rows)

        # The bad row was skipped; the good one was kept.
        assert r.size == {"row_count": 2, "exact_entries": 0, "regex_entries": 1}
        m = r.resolve("CLM6")
        assert m is not None
        assert m.product_name == "WTI Crude"

        # A warning was logged naming the malformed pattern.
        assert any(
            "Skipping invalid symbol_pattern" in record.message
            for record in caplog.records
        )


class TestEdgeCases:
    def test_empty_string_returns_none(self) -> None:
        rows = [_row(instrument_name="CL Jul26", product_name="WTI Crude")]
        r = ProductResolver(rows)
        assert r.resolve("") is None

    def test_none_input_returns_none(self) -> None:
        """The annotation is ``str``, but the guard catches falsy inputs
        defensively so a caller passing ``None`` doesn't crash the
        ``re.search`` loop."""
        rows = [_row(symbol_pattern=r"^CL", product_name="WTI Crude")]
        r = ProductResolver(rows)
        assert r.resolve(None) is None  # type: ignore[arg-type]

    def test_rows_without_either_key_are_inert(self) -> None:
        """Rows with neither ``instrument_name`` nor ``symbol_pattern``
        sit in the index but contribute no lookups. They're allowed
        because the underlying DB schema permits them (defensive)."""
        rows = [_row(instrument_name=None, symbol_pattern=None, product_name="WTI Crude")]
        r = ProductResolver(rows)
        assert r.resolve("anything") is None
        assert r.size == {"row_count": 1, "exact_entries": 0, "regex_entries": 0}


class TestMatchMetadata:
    def test_match_carries_all_columns(self) -> None:
        rows = [
            _row(
                mapping_id="dddddddd-dddd-dddd-dddd-dddddddddddd",
                instrument_name="WTI Cal 26 Strip",
                product_name="WTI Crude",
                product_family="Crude",
                contract_month=None,
                structure_type="Strip",
                source="regex",
                confidence=0.95,
            )
        ]
        r = ProductResolver(rows)
        m = r.resolve("WTI Cal 26 Strip")
        assert isinstance(m, ProductMatch)
        assert m.product_name == "WTI Crude"
        assert m.product_family == "Crude"
        assert m.structure_type == "Strip"
        assert m.contract_month is None
        assert m.source == "regex"
        assert m.confidence == 0.95
        assert m.mapping_id == "dddddddd-dddd-dddd-dddd-dddddddddddd"
        assert m.via == "exact"

    def test_match_is_frozen(self) -> None:
        """``ProductMatch`` is a frozen dataclass — mutations must fail."""
        rows = [_row(instrument_name="CL Jul26", product_name="WTI Crude")]
        r = ProductResolver(rows)
        m = r.resolve("CL Jul26")
        assert m is not None
        with pytest.raises(Exception):
            m.product_name = "Brent Crude"  # type: ignore[misc]
