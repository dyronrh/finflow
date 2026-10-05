"""Feature membership of each factor family (README §8.1).

``True`` means a higher raw value is better; ``False`` flips the sign before
ranking (e.g. EV/EBITDA, leverage).
"""

from __future__ import annotations

from quant_core.config import FactorFamily

FACTOR_DEFINITIONS: dict[FactorFamily, dict[str, bool]] = {
    "value": {
        "earnings_yield": True,
        "fcf_yield": True,
        "ev_ebitda": False,
    },
    "growth": {
        "revenue_growth": True,
        "eps_growth": True,
    },
    "profitability": {
        "roic": True,
        "gross_margin": True,
        "fcf_margin": True,
        "net_debt_ebitda": False,
    },
    "momentum": {
        "mom_12_1": True,
        "mom_6m": True,
        "mom_3m": True,
    },
    "revisions": {
        "eps_rev_30d": True,
        "eps_rev_90d": True,
        "revisions_up_ratio": True,
    },
}

FACTOR_LABELS_ES: dict[FactorFamily, str] = {
    "value": "valoración",
    "growth": "crecimiento",
    "profitability": "rentabilidad/calidad",
    "momentum": "momentum",
    "revisions": "revisiones de EPS",
}


def all_feature_columns() -> list[str]:
    return [feature for features in FACTOR_DEFINITIONS.values() for feature in features]
