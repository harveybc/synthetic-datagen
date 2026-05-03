"""Test the financial OHLCV column-classification schema utility."""
from sdg_plugins.schema.financial_ohlcv_schema import classify_columns


def test_classify_basic():
    cols = ["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
            "TYPICAL_PRICE", "RSI_14", "MACD", "ATR_14", "REGIME_FLAG",
            "MY_CUSTOM_FEATURE"]
    out = classify_columns(cols)
    assert "DATE_TIME" in out.contextual
    assert set(out.primitive) == {"OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"}
    for c in ("TYPICAL_PRICE", "RSI_14", "MACD", "ATR_14"):
        assert c in out.derived
    assert "REGIME_FLAG" in out.contextual
    # Unknown columns must be conservatively classified as derived.
    assert "MY_CUSTOM_FEATURE" in out.derived
    assert "MY_CUSTOM_FEATURE" in out.unknown


def test_no_primitive_in_derived():
    out = classify_columns(["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])
    assert not out.derived
    assert not out.unknown
