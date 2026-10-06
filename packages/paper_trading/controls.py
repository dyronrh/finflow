"""Trading controls and kill switch (README §15.3).

Execution is allowed only when *all* of these hold:

* ``GLOBAL_TRADING_ENABLED=true`` (master switch, default false);
* ``PAPER_TRADING_ENABLED=true`` (default true);
* no kill-switch file exists (``data/local_dev_only/KILL_SWITCH``).

Live trading is not implemented: ``LIVE_TRADING_ENABLED=true`` is refused, so
no configuration mistake can route orders to a real-money account.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KILL_FILE = ROOT / "data" / "local_dev_only" / "KILL_SWITCH"


class TradingDisabled(RuntimeError):
    """Raised when a control blocks execution."""


def _flag(name: str, default: bool, env: dict[str, str]) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().strip("'\"").lower() in {"1", "true", "yes", "on"}


def _dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#"):
            out[key.strip()] = value.strip()
    return out


@dataclass(frozen=True)
class TradingControls:
    global_enabled: bool
    paper_enabled: bool
    live_enabled: bool
    kill_file: Path

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, kill_file: Path | None = None):
        merged = {**_dotenv(ROOT / ".env"), **(env if env is not None else os.environ)}
        return cls(
            global_enabled=_flag("GLOBAL_TRADING_ENABLED", False, merged),
            paper_enabled=_flag("PAPER_TRADING_ENABLED", True, merged),
            live_enabled=_flag("LIVE_TRADING_ENABLED", False, merged),
            kill_file=Path(kill_file or merged.get("KILL_SWITCH_FILE", DEFAULT_KILL_FILE)),
        )

    @property
    def kill_switch_active(self) -> bool:
        return self.kill_file.exists()

    def assert_can_trade(self, mode: str = "paper") -> None:
        if mode != "paper" or self.live_enabled:
            raise TradingDisabled(
                "live trading is not implemented in this platform; only paper trading is allowed"
            )
        if self.kill_switch_active:
            reason = self.kill_file.read_text().strip() or "no reason recorded"
            raise TradingDisabled(f"KILL SWITCH active ({self.kill_file}): {reason}")
        if not self.global_enabled:
            raise TradingDisabled("GLOBAL_TRADING_ENABLED is false")
        if not self.paper_enabled:
            raise TradingDisabled("PAPER_TRADING_ENABLED is false")

    def state(self) -> dict[str, object]:
        return {
            "global_trading_enabled": self.global_enabled,
            "paper_trading_enabled": self.paper_enabled,
            "live_trading_enabled": self.live_enabled,
            "kill_switch_active": self.kill_switch_active,
        }


def activate_kill_switch(reason: str, by: str, kill_file: Path = DEFAULT_KILL_FILE) -> Path:
    kill_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"reason": reason, "by": by, "at": datetime.now(UTC).isoformat()}
    kill_file.write_text(json.dumps(payload))
    return kill_file


def deactivate_kill_switch(kill_file: Path = DEFAULT_KILL_FILE) -> bool:
    if kill_file.exists():
        kill_file.unlink()
        return True
    return False
