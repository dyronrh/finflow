from __future__ import annotations

import pandas as pd
import pytest

from data_platform.synthetic import SyntheticMarket, generate_synthetic_market
from quant_core.config import StrategyConfig, default_strategy_config


@pytest.fixture(scope="session")
def market() -> SyntheticMarket:
    return generate_synthetic_market(n_securities=80, start="2022-01-01", end="2024-12-31", seed=7)


@pytest.fixture(scope="session")
def config() -> StrategyConfig:
    return default_strategy_config()


@pytest.fixture(scope="session")
def as_of(market: SyntheticMarket) -> pd.Timestamp:
    return market.dates[-1]
