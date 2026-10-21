"""Compare live predictions (logs/predictions.jsonl) with the training baseline.

A model does not break loudly. It silently gets worse when the data changes,
e.g. spammers invent a new disguise. This script raises warnings when the
live traffic looks different from what the model saw at training time.

Usage:
    python -m src.monitor
"""
import json
import sys

import pandas as pd

from src.config import MODEL_INFO_PATH, PREDICTION_LOG_PATH

# Scores between these values mean the model is unsure.
UNCERTAIN = (0.3, 0.7)
MAX_RATE_CHANGE = 0.15
MAX_UNCERTAIN_SHARE = 0.10


def summarize(log, baseline):
    """Return (summary dict, list of warning strings)"""
    judol_rate = float((log["label"] == "judol").mean())
    uncertain = float(log["score"].between(*UNCERTAIN).mean())
    summary = {
        "n_predictions": len(log),
        "judol_rate": judol_rate,
        "baseline_judol_rate": baseline["judol_rate"],
        "uncertain_share": uncertain,
        "median_score": float(log["score"].median()),
        "model_versions": sorted(log["model_version"].unique().tolist()),
    }
    warnings = []
    if abs(judol_rate - baseline["judol_rate"]) > MAX_RATE_CHANGE:
        warnings.append(
            f"judol rate {judol_rate:.1%} differs from baseline {baseline['judol_rate']:.1%}: "
            "spam wave, or the model is missing/over-flagging something"
        )
    if uncertain > MAX_UNCERTAIN_SHARE:
        warnings.append(
            f"{uncertain:.1%} of scores are in {UNCERTAIN}: model is unsure, "
            "possibly a new spam style. Label a sample and retrain."
        )
    return summary, warnings


def load_log(path=PREDICTION_LOG_PATH):
    return pd.read_json(path, lines=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if not PREDICTION_LOG_PATH.exists():
        print(f"No prediction log yet at {PREDICTION_LOG_PATH}. Call the API first.")
        return
    baseline = json.loads(MODEL_INFO_PATH.read_text())["baseline"]
    log = load_log()
    log["day"] = pd.to_datetime(log["ts"]).dt.date
    summary, warnings = summarize(log, baseline)

    print(json.dumps(summary, indent=2))
    print("\nPer day:")
    print(log.groupby("day").agg(n=("score", "size"), judol_rate=("label", lambda s: (s == "judol").mean())))
    print("\n" + ("\n".join(f"WARNING: {w}" for w in warnings) if warnings else "OK: no drift detected"))


if __name__ == "__main__":
    main()
