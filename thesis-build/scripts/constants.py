"""Prices that come from outside the experiments, shared by figures, macros and tables."""

import csv
from pathlib import Path

_PRICES = Path(__file__).resolve().parents[1] / "data" / "azure_vm_prices.csv"


def _azure(meter: str) -> float:
    with _PRICES.open(newline="") as fh:
        return next(float(r["usd_per_hour"]) for r in csv.DictReader(fh) if r["meter"] == meter)


# Azure's published price of the node type the pipeline ran on (NC24ads_A100_v4, one A100
# 80 GB, Japan East), from the Retail Prices API (make data).
SPOT_USD = _azure("NC24ads_A100_v4 Spot")
PAYG_USD = _azure("NC24ads_A100_v4")
