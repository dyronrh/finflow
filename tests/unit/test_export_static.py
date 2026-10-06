import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load():
    sys.path.insert(0, str(ROOT / "pipelines"))
    spec = importlib.util.spec_from_file_location(
        "export_static", ROOT / "pipelines" / "export_static.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_columnar_candles_round_trip_shape():
    export = _load()
    body = {
        "security_id": "AAA",
        "candles": [
            {
                "time": "2024-01-02",
                "open": 10.123,
                "high": 10.5,
                "low": 9.9,
                "close": 10.4,
                "volume": 1234.7,
            },
            {
                "time": "2024-01-03",
                "open": 0.51234,
                "high": 0.6,
                "low": 0.5,
                "close": 0.55557,
                "volume": 99.0,
            },
        ],
    }
    out = export.compact_candles(body)
    assert out["format"] == "columnar"
    assert out["time"] == ["2024-01-02", "2024-01-03"]
    assert out["open"] == [10.12, 0.5123]  # cents, or 4 decimals under $1
    assert out["close"][1] == 0.5556
    assert out["volume"] == [1234, 99]
    assert all(len(out[f]) == 2 for f in export.FIELDS)
