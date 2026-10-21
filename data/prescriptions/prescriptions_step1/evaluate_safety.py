#!/usr/bin/env python3
"""
Clinical Safety Screening Benchmark Evaluator for MedAgent.

Evaluates safety screening decisions against human/pharmacist ground-truth cases:
1. Calculates Screening Misses (False Negatives: expected review, but system failed to flag)
2. Calculates False Alarms (False Positives: expected no_rule_triggered, but system flagged)
3. Evaluates Sensitivity, Specificity, Precision, Recall, and F1
4. Generates a clinical safety audit report JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

# Ensure stdout handles UTF-8 on Windows
reconfigure_fn = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure_fn):
    reconfigure_fn(encoding="utf-8")


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, mode="r", encoding="utf-8-sig", errors="replace") as f:
        lines = [line for line in f if line.strip() and not line.strip().startswith("#")]
    reader = csv.DictReader(lines)
    return list(reader)


def run_safety_evaluation(
    cases_path: str | Path,
    predictions_path: str | Path,
    out_path: str | Path,
) -> dict[str, Any]:
    cases_rows = load_csv(cases_path)
    pred_rows = load_csv(predictions_path)

    cases_by_id = {r["case_id"].strip(): r for r in cases_rows if r.get("case_id")}
    preds_by_id = {r["case_id"].strip(): r for r in pred_rows if r.get("case_id")}

    case_ids = set(cases_by_id.keys())
    pred_ids = set(preds_by_id.keys())

    matched_ids = sorted(list(case_ids & pred_ids))
    unmatched_cases = sorted(list(case_ids - pred_ids))
    unmatched_preds = sorted(list(pred_ids - case_ids))

    if not matched_ids:
        print(f"❌ Error: No matching case_ids found between {cases_path} and {predictions_path}!")
        sys.exit(1)

    tp = 0  # Expected review, predicted review
    tn = 0  # Expected no_rule_triggered, predicted no_rule_triggered
    misses_list = []  # Expected review, predicted no_rule_triggered (FN)
    false_alarms_list = []  # Expected no_rule_triggered, predicted review (FP)

    per_case_eval = []

    for cid in matched_ids:
        case = cases_by_id[cid]
        pred = preds_by_id[cid]

        exp_act = case.get("expected_action", "").strip().lower()
        sys_act = pred.get("system_action", "").strip().lower()

        is_match = (exp_act == sys_act)

        if exp_act == "review":
            if sys_act == "review":
                tp += 1
                status = "TRUE_POSITIVE"
            else:
                misses_list.append({
                    "case_id": cid,
                    "expected": exp_act,
                    "predicted": sys_act,
                    "clinical_rationale": case.get("clinical_rationale", ""),
                    "prescribed": case.get("prescribed_medicines", ""),
                })
                status = "MISS_FALSE_NEGATIVE"
        elif exp_act == "no_rule_triggered":
            if sys_act == "no_rule_triggered":
                tn += 1
                status = "TRUE_NEGATIVE"
            else:
                false_alarms_list.append({
                    "case_id": cid,
                    "expected": exp_act,
                    "predicted": sys_act,
                    "clinical_rationale": case.get("clinical_rationale", ""),
                    "prescribed": case.get("prescribed_medicines", ""),
                })
                status = "FALSE_ALARM_POSITIVE"
        else:
            status = "UNKNOWN_ACTION"

        per_case_eval.append({
            "case_id": cid,
            "prescribed_medicines": case.get("prescribed_medicines", ""),
            "patient_allergies": case.get("patient_allergies", ""),
            "patient_conditions": case.get("patient_conditions", ""),
            "active_medications": case.get("active_medications", ""),
            "expected_action": exp_act,
            "system_action": sys_act,
            "match": is_match,
            "classification": status,
            "clinical_rationale": case.get("clinical_rationale", ""),
        })

    n_total = len(matched_ids)
    fn = len(misses_list)
    fp = len(false_alarms_list)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0

    report = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cases_file": str(cases_path),
            "predictions_file": str(predictions_path),
            "total_cases_evaluated": n_total,
            "unmatched_cases": unmatched_cases,
            "unmatched_predictions": unmatched_preds,
            "environment": {
                "os": platform.platform(),
                "python_version": platform.python_version(),
            },
        },
        "summary_metrics": {
            "total_cases": n_total,
            "expected_review_cases": tp + fn,
            "expected_no_rule_triggered_cases": tn + fp,
            "true_positives": tp,
            "true_negatives": tn,
            "screening_misses_count": fn,
            "false_alarms_count": fp,
            "screening_miss_rate_percent": round((fn / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0, 2),
            "false_alarm_rate_percent": round((fp / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0, 2),
            "sensitivity_recall_percent": round(sensitivity * 100.0, 2),
            "specificity_percent": round(specificity * 100.0, 2),
            "precision_percent": round(precision * 100.0, 2),
            "f1_score_percent": round(f1 * 100.0, 2),
            "overall_accuracy_percent": round(accuracy * 100.0, 2),
        },
        "critical_screening_misses": misses_list,
        "false_alarms": false_alarms_list,
        "per_case_evaluation": per_case_eval,
    }

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # -------------------------------------------------------------------------
    # PRINT FORMATTED TERMINAL REPORT
    # -------------------------------------------------------------------------
    print("=" * 80)
    print(" 🛡️ MEDAGENT CLINICAL SAFETY SCREENING EVALUATION REPORT")
    print("=" * 80)
    print(f" Cases Evaluated    : {n_total} synthetic non-patient scenarios")
    print(f" Execution Date     : {report['metadata']['generated_at']}")
    print("=" * 80)

    print("\n[1. CONFUSION MATRIX & WORKFLOW METRICS]")
    print("-" * 80)
    print(f"  {'Classification Metric':<40} | {'Count / Total':<15} | {'Percentage':<15}")
    print("  " + "-" * 78)
    print(f"  {'True Positives (Correct Reviews)':<40} | {tp:>3} / {tp+fn:<7}   | {(tp/(tp+fn)*100.0):>5.2f} %")
    print(f"  {'True Negatives (No Rule Triggered)':<40} | {tn:>3} / {tn+fp:<7}   | {(tn/(tn+fp)*100.0):>5.2f} %")
    print(f"  {'Screening Misses (False Negatives)':<40} | {fn:>3} / {tp+fn:<7}   | {(fn/(tp+fn)*100.0 if (tp+fn)>0 else 0.0):>5.2f} % (Zero tolerance)")
    print(f"  {'False Alarms (False Positives)':<40} | {fp:>3} / {tn+fp:<7}   | {(fp/(tn+fp)*100.0 if (tn+fp)>0 else 0.0):>5.2f} %")
    print("  " + "-" * 78)
    print(f"  {'Overall Workflow Decision Accuracy':<40} | {tp+tn:>3} / {n_total:<7}   | {accuracy*100.0:>5.2f} %")

    print("\n[2. CLINICAL SAFETY PERFORMANCE METRICS]")
    print("-" * 80)
    sm = report["summary_metrics"]
    print(f"  {'Performance Metric':<40} | {'Score / Rate':<15} | {'Clinical Benchmark':<20}")
    print("  " + "-" * 78)
    print(f"  {'Sensitivity / Safety Recall':<40} | {sm['sensitivity_recall_percent']:>5.2f} %         | Target: 100% on criticals")
    print(f"  {'Specificity':<40} | {sm['specificity_percent']:>5.2f} %         | Low alert fatigue")
    print(f"  {'Precision':<40} | {sm['precision_percent']:>5.2f} %         | Actionable alert rate")
    print(f"  {'F1-Score':<40} | {sm['f1_score_percent']:>5.2f} %         | Harmonic mean")

    if fn == 0:
        print("\n  ✅ ZERO SCREENING MISSES: All high-risk combinations escalated for clinical review.")
    else:
        print(f"\n  ⚠️ WARNING: {fn} screening misses detected! Details recorded in JSON.")

    print("\n" + "=" * 80)
    print(f" 📄 Full safety report saved to: {out_p.resolve()}")
    print("=" * 80)

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedAgent Safety Screening Performance.")
    parser.add_argument("--cases", required=False, default="safety_cases.csv", help="Path to safety cases CSV")
    parser.add_argument("--predictions", required=False, default="safety_predictions.csv", help="Path to safety predictions CSV")
    parser.add_argument("--out", required=False, default="safety_report.json", help="Output JSON safety report path")

    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists() and (Path(".") / "safety_cases.csv").exists():
        cases_path = Path(".") / "safety_cases.csv"

    preds_path = Path(args.predictions)
    if not preds_path.exists() and (Path(".") / "safety_predictions.csv").exists():
        preds_path = Path(".") / "safety_predictions.csv"

    out_path = Path(args.out)

    run_safety_evaluation(
        cases_path=cases_path,
        predictions_path=preds_path,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()

