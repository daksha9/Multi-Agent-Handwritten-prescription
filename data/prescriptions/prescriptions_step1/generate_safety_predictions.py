#!/usr/bin/env python3
"""
Safety Screening Predictions Generator for MedAgent.

Evaluates the 30 synthetic safety cases against the MedAgent clinical rule engine
and exports decisions to `safety_predictions.csv` with columns: `case_id,system_action`.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Ensure UTF-8 stdout on Windows
reconfigure_fn = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure_fn):
    reconfigure_fn(encoding="utf-8")

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.verification_agent import PrescriptionVerificationAgent


def parse_list(val: str | None) -> list[str]:
    if not val or val.strip().lower() in ("none", "null", "[]"):
        return []
    return [x.strip() for x in val.split(";") if x.strip()]


def generate_safety_predictions():
    print("=" * 80)
    print(" 🛡️ RUNNING SAFETY SCREENING PREDICTIONS GENERATOR")
    print("=" * 80)

    cases_path = SCRIPT_DIR / "safety_cases.csv"
    if not cases_path.exists():
        print(f"❌ Error: safety_cases.csv not found at {cases_path}")
        sys.exit(1)

    with open(cases_path, mode="r", encoding="utf-8-sig") as f:
        cases = list(csv.DictReader(f))

    print(f" Loaded {len(cases)} synthetic safety benchmark cases.")

    agent = PrescriptionVerificationAgent()
    predictions = []

    for c in cases:
        case_id = c["case_id"]
        prescribed = parse_list(c.get("prescribed_medicines"))
        allergies = parse_list(c.get("patient_allergies"))
        conditions = parse_list(c.get("patient_conditions"))
        active_meds = parse_list(c.get("active_medications"))

        # Deterministic verification
        res = agent.verify_prescriptions(
            prescribed_drugs=prescribed,
            patient_allergies=allergies,
            patient_conditions=conditions,
            active_medications=active_meds,
            consult_llm=False,
        )

        critical_count = res.get("critical_count", 0)
        warning_count = res.get("warning_count", 0)

        # Trigger review if any clinical safety rule or alert fired
        if critical_count > 0 or warning_count > 0 or len(res.get("alerts", [])) > 0:
            system_action = "review"
        else:
            system_action = "no_rule_triggered"

        predictions.append({
            "case_id": case_id,
            "system_action": system_action,
        })

    out_csv = SCRIPT_DIR / "safety_predictions.csv"
    with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "system_action"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f" ✅ Generated {len(predictions)} safety predictions in: {out_csv}")


if __name__ == "__main__":
    generate_safety_predictions()
