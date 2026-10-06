"""Yahoo download robustness, without network."""

import numpy as np
import pandas as pd
import pytest

from data_platform.sources import yahoo

DATES = pd.bdate_range("2024-01-01", periods=5)


def frame(base: float = 100.0) -> pd.DataFrame:
    close = pd.Series(np.linspace(base, base + 4, 5), index=DATES)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "Close": close,
            "Adj Close": close * 0.9,
            "Volume": 1e6,
            "Stock Splits": 0.0,
        }
    )


def test_split_by_symbol_handles_both_multiindex_orientations():
    a, b = frame(10), frame(20)
    by_ticker = pd.concat({"AAA": a, "BBB": b}, axis=1)  # (Ticker, Price)
    by_field = by_ticker.swaplevel(axis=1)  # (Price, Ticker)
    for raw in (by_ticker, by_field):
        out = yahoo.split_by_symbol(raw, ["AAA", "BBB", "CCC"])
        assert set(out) == {"AAA", "BBB"}
        assert out["BBB"]["Close"].iloc[0] == 20


def test_assemble_prices_separates_traded_and_adjusted_series():
    p = yahoo.assemble_prices({"AAA": frame(10)}, ["AAA", "ZZZ"])
    assert p.failed == ["ZZZ"]
    assert p.price_close["AAA"].iloc[0] == 10
    assert p.close["AAA"].iloc[0] == pytest.approx(9.0)
    assert p.open["AAA"].iloc[0] == pytest.approx(9.5 * 0.9)


def test_retries_transient_failures_and_reports_real_misses(tmp_path):
    calls = []

    def flaky(chunk, start, end):
        calls.append(list(chunk))
        if len(calls) == 1:
            raise OSError("curl: (6) Could not resolve host")  # outage on first try
        return {s: frame() for s in chunk if s != "DEAD"}

    p = yahoo.download_prices(
        ["AAA", "BBB", "DEAD"],
        "2024-01-01",
        "2024-01-08",
        cache_dir=tmp_path,
        downloader=flaky,
        sleep=lambda s: None,
        max_rounds=3,
    )
    assert set(p.close.columns) == {"AAA", "BBB"}
    assert p.failed == ["DEAD"]
    store = tmp_path / "prices" / "2024-01-01_2024-01-08"
    assert (store / "DEAD.missing").exists()  # chunk worked, so DEAD truly has no data
    assert (store / "AAA.parquet").exists()


def test_outage_is_not_cached_as_missing_and_resume_uses_cache(tmp_path):
    def down(chunk, start, end):
        raise OSError("network down")

    p = yahoo.download_prices(
        ["AAA"],
        "2024-01-01",
        "2024-01-08",
        cache_dir=tmp_path,
        downloader=down,
        sleep=lambda s: None,
        max_rounds=2,
    )
    assert p.failed == ["AAA"]
    store = tmp_path / "prices" / "2024-01-01_2024-01-08"
    assert not (store / "AAA.missing").exists()

    calls = []

    def ok(chunk, start, end):
        calls.append(list(chunk))
        return {s: frame() for s in chunk}

    yahoo.download_prices(
        ["AAA"], "2024-01-01", "2024-01-08", cache_dir=tmp_path, downloader=ok, sleep=lambda s: None
    )
    yahoo.download_prices(
        ["AAA", "BBB"],
        "2024-01-01",
        "2024-01-08",
        cache_dir=tmp_path,
        downloader=ok,
        sleep=lambda s: None,
    )
    assert calls == [["AAA"], ["BBB"]]  # second run only fetches what is not cached


def test_benchmark_failure_has_actionable_message(tmp_path, monkeypatch):
    def down(chunk, start, end):
        raise OSError("network down")

    monkeypatch.setattr(yahoo, "_yf_downloader", lambda threads, cache_dir: down)
    monkeypatch.setattr(yahoo.time, "sleep", lambda s: None)
    with pytest.raises(yahoo.YahooDownloadError, match="network/rate-limit"):
        yahoo.download_benchmark("SPY", "2024-01-01", "2024-01-08", cache_dir=tmp_path)
