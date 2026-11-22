"""Generate a SYNTHETIC UPI-style transaction dataset (no real people, UPI IDs or accounts).

    python dataset/generate_sample_data.py                      # 30,000 rows, ~4% fraud
    python dataset/generate_sample_data.py --rows 50000 --fraud-rate 0.03

The label is drawn from a hidden logistic risk model plus noise, so the problem is realistic
(overlapping classes, imperfect separability) rather than trivially solvable. The generator
encodes plausible *assumptions* (night-time activity, balance drain, new payees, risky devices,
collect-request scams ...). It is NOT real UPI data and metrics on it do not transfer to production.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.preprocessing import NIGHT_HOURS, TARGET_COLUMN, TRANSACTION_TYPES  # noqa: E402

TYPE_PROBS = [0.35, 0.30, 0.12, 0.13, 0.07, 0.02, 0.01]
TYPE_AMOUNT_SCALE = {"P2P": 1.0, "P2M": 0.7, "BILL_PAYMENT": 1.3, "QR_PAYMENT": 0.5,
                     "COLLECT_REQUEST": 1.1, "CASH_OUT": 2.0, "CASH_IN": 1.5}
TYPE_EFFECT = {"P2P": 0.4, "P2M": -0.6, "BILL_PAYMENT": -0.9, "QR_PAYMENT": -0.2,
               "COLLECT_REQUEST": 1.4, "CASH_OUT": 1.1, "CASH_IN": -0.3}
NEW_PAYEE_PROB = {"P2P": 0.35, "COLLECT_REQUEST": 0.5}
HOUR_WEIGHTS = np.array([0.6, 0.4, 0.3, 0.3, 0.4, 0.8, 1.5, 2.5, 3.5, 4.5, 5.0, 5.5,
                         6.0, 5.5, 5.0, 5.0, 5.5, 6.5, 7.0, 7.5, 7.0, 5.5, 3.5, 1.5])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_rows: int = 30_000, fraud_rate: float = 0.04, seed: int = 42,
             signal: float = 2.2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_rows

    ttype = rng.choice(TRANSACTION_TYPES, size=n, p=TYPE_PROBS)
    scale = np.array([TYPE_AMOUNT_SCALE[t] for t in ttype])
    sender_balance = np.round(rng.lognormal(9.6, 1.0, n), 2)
    receiver_balance = np.round(rng.lognormal(9.2, 1.2, n), 2)
    receiver_balance[rng.random(n) < 0.20] = 0.0  # new / empty receiver accounts

    amount = rng.lognormal(6.5, 1.15, n) * scale
    amount = np.minimum(amount, sender_balance * rng.uniform(0.02, 1.0, n))  # cannot exceed balance
    amount = np.round(np.clip(amount, 1.0, 100_000.0), 2)  # UPI per-transaction limit is typically 1 lakh

    hour = rng.choice(24, size=n, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum())
    frequency = rng.poisson(2.2, n)
    device_risk = np.round(rng.beta(1.4, 9.0, n), 3)
    new_device = (rng.random(n) < 0.07).astype(int)
    new_payee = np.array([rng.random() < NEW_PAYEE_PROB.get(t, 0.12) for t in ttype]).astype(int)
    failed = rng.poisson(0.12, n)

    ratio = amount / (sender_balance + 1.0)
    night = np.isin(hour, NIGHT_HOURS).astype(int)
    recv_zero = (receiver_balance == 0).astype(int)
    type_effect = np.array([TYPE_EFFECT[t] for t in ttype])

    z = (2.2 * ratio + 0.9 * night + 0.18 * np.clip(frequency - 3, 0, None) + 3.2 * device_risk
         + 1.3 * new_device + 0.9 * new_payee + 0.6 * failed + 0.8 * recv_zero
         + 0.3 * np.log10(amount) + type_effect)
    # `signal` sharpens the hidden risk model; the noise term keeps classes overlapping (imperfect).
    z = signal * z + rng.normal(0, 0.9, n)

    # Choose the intercept so the overall fraud rate matches the requested rate.
    lo, hi = -30.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > fraud_rate:
            hi = mid
        else:
            lo = mid
    p = _sigmoid(z + (lo + hi) / 2)
    is_fraud = (rng.random(n) < p).astype(int)

    return pd.DataFrame({
        "transaction_id": [f"SYN{i:07d}" for i in range(1, n + 1)],
        "amount": amount,
        "transaction_type": ttype,
        "transaction_hour": hour,
        "transaction_frequency": frequency,
        "sender_balance": sender_balance,
        "receiver_balance": receiver_balance,
        "device_risk_score": device_risk,
        "is_new_device": new_device,
        "is_new_payee": new_payee,
        "failed_attempts_24h": failed,
        TARGET_COLUMN: is_fraud,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic UPI-style dataset.")
    parser.add_argument("--rows", type=int, default=30_000)
    parser.add_argument("--fraud-rate", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--signal", type=float, default=2.2,
                        help="strength of the hidden risk model (higher = easier problem)")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset" / "upi_synthetic.csv")
    args = parser.parse_args()
    if not (0.001 <= args.fraud_rate <= 0.5):
        parser.error("--fraud-rate must be between 0.001 and 0.5")
    df = generate(args.rows, args.fraud_rate, args.seed, args.signal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows to {args.output}  (fraud rows: {int(df[TARGET_COLUMN].sum()):,}, "
          f"rate {df[TARGET_COLUMN].mean():.2%})")


if __name__ == "__main__":
    main()
