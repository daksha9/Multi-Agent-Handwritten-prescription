#!/usr/bin/env python3
"""
Prescription OCR & Entity Extraction Evaluator for MedAgent.

Evaluates pipeline predictions against human-annotated ground-truth labels:
1. Character Error Rate (CER) via Levenshtein edit distance
2. Position-wise Word Accuracy
3. Field Accuracy across all 4 target clinical fields (medicines, dose, frequency, duration)
4. Medicine Entity Precision, Recall, and F1 (set-based matching with case/alias normalization)
5. Pipeline Stage Timing benchmarks from high-resolution monotonic telemetry
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure stdout handles UTF-8 on Windows
reconfigure_fn = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure_fn):
    reconfigure_fn(encoding="utf-8")


def normalize_text(text: Any) -> str:
    """Collapses whitespace and folds case."""
    if text is None:
        return ""
    text_str = str(text)
    # Collapse whitespace and lowercase
    return re.sub(r"\s+", " ", text_str).strip().lower()


def normalize_medicines_set(med_str: Any) -> set[str]:
    """Splits semicolon-separated medicines, strips punctuation, and normalizes."""
    if not med_str:
        return set()
    cleaned = str(med_str).strip()
    if not cleaned or cleaned.lower() in ("none", "null", "[]"):
        return set()
    parts = cleaned.split(";")
    result = set()
    for part in parts:
        norm = normalize_text(part)
        # Strip common trailing dosage words if in drug name
        norm = re.sub(r"^(tab|tablet|cap|capsule|syr|syrup|inj|injection)\s+", "", norm)
        # Strip parenthetical annotations e.g. (vitamin b12)
        norm = re.sub(r"\(.*?\)", "", norm).strip()
        if norm:
            result.add(norm)
    return result


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates Levenshtein distance using standard dynamic programming."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)

    for i, c1 in enumerate(s1):
        v1[0] = i + 1
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0

    return v0[len(s2)]


def compute_cer(predicted: str, reference: str) -> tuple[float, int, int]:
    """
    Computes Character Error Rate = edit_distance / len(reference).
    Returns (cer, edit_distance, reference_length).
    """
    p_norm = normalize_text(predicted)
    r_norm = normalize_text(reference)

    ref_len = len(r_norm)
    if ref_len == 0:
        if len(p_norm) == 0:
            return 0.0, 0, 0
        return 1.0, len(p_norm), 0

    dist = levenshtein_distance(p_norm, r_norm)
    cer = dist / ref_len
    return cer, dist, ref_len


def compute_word_accuracy(predicted: str, reference: str) -> tuple[float, int, int]:
    """
    Position-wise token agreement accuracy.
    Tokenizes on whitespace; compares tokens at identical index positions.
    """
    p_norm = normalize_text(predicted)
    r_norm = normalize_text(reference)

    p_tokens = p_norm.split()
    r_tokens = r_norm.split()

    ref_count = len(r_tokens)
    if ref_count == 0:
        return (1.0 if len(p_tokens) == 0 else 0.0), 0, 0

    matches = 0
    for i in range(ref_count):
        if i < len(p_tokens) and p_tokens[i] == r_tokens[i]:
            matches += 1

    acc = matches / ref_count
    return acc, matches, ref_count


def load_csv(path: str | Path) -> list[dict[str, str]]:
    """Loads a CSV into a list of row dictionaries."""
    with open(path, mode="r", encoding="utf-8-sig", errors="replace") as f:
        # Filter comments or blank lines
        lines = [line for line in f if line.strip() and not line.strip().startswith("#")]
    reader = csv.DictReader(lines)
    return list(reader)


def parse_timings(path: str | Path | None) -> dict[str, Any]:
    """Aggregates latency telemetry per stage from timings.csv."""
    if not path or not os.path.exists(path):
        return {}

    rows = load_csv(path)
    stage_latencies = defaultdict(list)
    per_rx_total = defaultdict(float)

    for r in rows:
        rx_id = r.get("prescription_id", "").strip()
        stage = r.get("stage", "").strip()
        try:
            sec = float(r.get("seconds", 0.0))
        except ValueError:
            sec = 0.0

        stage_latencies[stage].append(sec)
        if stage != "total_end_to_end":
            per_rx_total[rx_id] += sec

    summary = {}
    for stg, vals in stage_latencies.items():
        if not vals:
            continue
        vals_sorted = sorted(vals)
        n = len(vals)
        median_val = vals_sorted[n // 2] if n % 2 != 0 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2.0
        summary[stg] = {
            "invocations": n,
            "mean_seconds": round(sum(vals) / n, 4),
            "median_seconds": round(median_val, 4),
            "min_seconds": round(min(vals), 4),
            "max_seconds": round(max(vals), 4),
            "total_seconds": round(sum(vals), 4),
        }

    return summary


def run_evaluation(
    labels_path: str | Path,
    predictions_path: str | Path,
    timings_path: str | Path | None,
    out_path: str | Path,
) -> dict[str, Any]:
    """Runs the complete evaluation and produces JSON + terminal summary."""
    labels_rows = load_csv(labels_path)
    pred_rows = load_csv(predictions_path)

    # Index by prescription_id
    labels_by_id = {r.get("prescription_id", "").strip(): r for r in labels_rows if r.get("prescription_id")}
    preds_by_id = {r.get("prescription_id", "").strip(): r for r in pred_rows if r.get("prescription_id")}

    label_ids = set(labels_by_id.keys())
    pred_ids = set(preds_by_id.keys())

    matched_ids = sorted(list(label_ids & pred_ids))
    unmatched_labels = sorted(list(label_ids - pred_ids))
    unmatched_preds = sorted(list(pred_ids - label_ids))

    if not matched_ids:
        print(f"❌ Error: No matching prescription_ids found between {labels_path} and {predictions_path}!")
        sys.exit(1)

    per_case_results = []
    total_cer_sum = 0.0
    total_word_acc_sum = 0.0

    field_match_counts = {"medicines": 0, "dose": 0, "frequency": 0, "duration": 0, "all_4": 0}

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    macro_prec_list = []
    macro_rec_list = []
    macro_f1_list = []

    for pid in matched_ids:
        lbl = labels_by_id[pid]
        prd = preds_by_id[pid]

        # Field keys with fallback tolerance
        ref_text = lbl.get("expected_text") or lbl.get("text") or ""
        pred_text = prd.get("predicted_text") or prd.get("text") or ""

        ref_meds = lbl.get("expected_medicines") or lbl.get("medicines") or ""
        pred_meds = prd.get("predicted_medicines") or prd.get("medicines") or ""

        ref_dose = lbl.get("expected_dose") or lbl.get("dose") or ""
        pred_dose = prd.get("predicted_dose") or prd.get("dose") or ""

        ref_freq = lbl.get("expected_frequency") or lbl.get("frequency") or ""
        pred_freq = prd.get("predicted_frequency") or prd.get("frequency") or ""

        ref_dur = lbl.get("expected_duration") or lbl.get("duration") or ""
        pred_dur = prd.get("predicted_duration") or prd.get("duration") or ""

        # 1. CER
        cer_val, edit_dist, ref_chars = compute_cer(pred_text, ref_text)
        total_cer_sum += cer_val

        # 2. Word Accuracy
        word_acc, w_matches, w_ref_tokens = compute_word_accuracy(pred_text, ref_text)
        total_word_acc_sum += word_acc

        # 3. Field matching
        ref_med_set = normalize_medicines_set(ref_meds)
        pred_med_set = normalize_medicines_set(pred_meds)
        meds_match = ref_med_set == pred_med_set

        dose_match = normalize_text(pred_dose) == normalize_text(ref_dose)
        freq_match = normalize_text(pred_freq) == normalize_text(ref_freq)
        dur_match = normalize_text(pred_dur) == normalize_text(ref_dur)

        all_4_match = meds_match and dose_match and freq_match and dur_match

        if meds_match:
            field_match_counts["medicines"] += 1
        if dose_match:
            field_match_counts["dose"] += 1
        if freq_match:
            field_match_counts["frequency"] += 1
        if dur_match:
            field_match_counts["duration"] += 1
        if all_4_match:
            field_match_counts["all_4"] += 1

        # 4. Medicine Precision / Recall / F1
        tp = len(pred_med_set & ref_med_set)
        fp = len(pred_med_set - ref_med_set)
        fn = len(ref_med_set - pred_med_set)

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        case_prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not ref_med_set else 0.0)
        case_rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not pred_med_set else 0.0)
        case_f1 = (2 * case_prec * case_rec / (case_prec + case_rec)) if (case_prec + case_rec) > 0 else 0.0

        macro_prec_list.append(case_prec)
        macro_rec_list.append(case_rec)
        macro_f1_list.append(case_f1)

        per_case_results.append(
            {
                "prescription_id": pid,
                "cer": round(cer_val, 4),
                "edit_distance": edit_dist,
                "reference_characters": ref_chars,
                "word_accuracy": round(word_acc, 4),
                "exact_4_field_match": all_4_match,
                "field_matches": {
                    "medicines": meds_match,
                    "dose": dose_match,
                    "frequency": freq_match,
                    "duration": dur_match,
                },
                "expected": {
                    "text": ref_text,
                    "medicines": ref_meds,
                    "dose": ref_dose,
                    "frequency": ref_freq,
                    "duration": ref_dur,
                },
                "predicted": {
                    "text": pred_text,
                    "medicines": pred_meds,
                    "dose": pred_dose,
                    "frequency": pred_freq,
                    "duration": pred_dur,
                },
            }
        )

    n_cases = len(matched_ids)
    mean_cer = total_cer_sum / n_cases
    mean_word_acc = total_word_acc_sum / n_cases

    # Micro & Macro F1
    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 1.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 1.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

    macro_prec = sum(macro_prec_list) / n_cases
    macro_rec = sum(macro_rec_list) / n_cases
    macro_f1 = sum(macro_f1_list) / n_cases

    timings_summary = parse_timings(timings_path)

    report_data = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "labels_file": str(labels_path),
            "predictions_file": str(predictions_path),
            "timings_file": str(timings_path) if timings_path else None,
            "total_labels": len(label_ids),
            "total_predictions": len(pred_ids),
            "matched_cases": n_cases,
            "unmatched_labels": unmatched_labels,
            "unmatched_predictions": unmatched_preds,
            "environment": {
                "os": platform.platform(),
                "python_version": platform.python_version(),
                "processor": platform.processor() or "Local Host",
            },
        },
        "aggregate_metrics": {
            "mean_cer_percent": round(mean_cer * 100.0, 2),
            "mean_word_accuracy_percent": round(mean_word_acc * 100.0, 2),
            "exact_4_field_accuracy_percent": round((field_match_counts["all_4"] / n_cases) * 100.0, 2),
            "individual_field_accuracies_percent": {
                "medicines": round((field_match_counts["medicines"] / n_cases) * 100.0, 2),
                "dose": round((field_match_counts["dose"] / n_cases) * 100.0, 2),
                "frequency": round((field_match_counts["frequency"] / n_cases) * 100.0, 2),
                "duration": round((field_match_counts["duration"] / n_cases) * 100.0, 2),
            },
            "medicine_entity_metrics": {
                "micro_precision_percent": round(micro_prec * 100.0, 2),
                "micro_recall_percent": round(micro_rec * 100.0, 2),
                "micro_f1_percent": round(micro_f1 * 100.0, 2),
                "macro_precision_percent": round(macro_prec * 100.0, 2),
                "macro_recall_percent": round(macro_rec * 100.0, 2),
                "macro_f1_percent": round(macro_f1 * 100.0, 2),
            },
        },
        "timings_summary": timings_summary,
        "per_case_results": per_case_results,
    }

    # Write output JSON
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # -------------------------------------------------------------------------
    # PRINT FORMATTED TERMINAL REPORT
    # -------------------------------------------------------------------------
    print("=" * 80)
    print(" 🏥 MEDAGENT PRESCRIPTION PIPELINE EVALUATION REPORT")
    print("=" * 80)
    print(f" Evaluated Cohort   : {n_cases} prescriptions (Matched: {len(label_ids & pred_ids)})")
    if unmatched_labels:
        print(f" Unmatched Labels   : {unmatched_labels}")
    if unmatched_preds:
        print(f" Unmatched Preds    : {unmatched_preds}")
    print(f" Execution Date     : {report_data['metadata']['generated_at']}")
    print("=" * 80)

    print("\n[1. OCR & TEXT LEVEL ACCURACY]")
    print("-" * 80)
    print(f"  {'Metric':<40} | {'Score / Value':<15} | {'Interpretation':<20}")
    print("  " + "-" * 78)
    print(f"  {'Character Error Rate (CER)':<40} | {report_data['aggregate_metrics']['mean_cer_percent']:>5.2f} %         | Lower is better")
    print(f"  {'Word Accuracy (Position-wise)':<40} | {report_data['aggregate_metrics']['mean_word_accuracy_percent']:>5.2f} %         | Exact token order")

    print("\n[2. CLINICAL FIELD ACCURACY]")
    print("-" * 80)
    print(f"  {'Field Evaluated':<40} | {'Exact Match %':<15} | {'Matched / Total':<20}")
    print("  " + "-" * 78)
    ifa = report_data["aggregate_metrics"]["individual_field_accuracies_percent"]
    print(f"  {'Medicine Names (Set Match)':<40} | {ifa['medicines']:>5.2f} %         | {field_match_counts['medicines']}/{n_cases}")
    print(f"  {'Prescribed Dose':<40} | {ifa['dose']:>5.2f} %         | {field_match_counts['dose']}/{n_cases}")
    print(f"  {'Frequency (Sig)':<40} | {ifa['frequency']:>5.2f} %         | {field_match_counts['frequency']}/{n_cases}")
    print(f"  {'Duration':<40} | {ifa['duration']:>5.2f} %         | {field_match_counts['duration']}/{n_cases}")
    print("  " + "-" * 78)
    print(f"  {'All 4 Fields Exact Match':<40} | {report_data['aggregate_metrics']['exact_4_field_accuracy_percent']:>5.2f} %         | {field_match_counts['all_4']}/{n_cases}")

    print("\n[3. MEDICINE ENTITY EXTRACTION METRICS]")
    print("-" * 80)
    mem = report_data["aggregate_metrics"]["medicine_entity_metrics"]
    print(f"  {'Metric':<40} | {'Micro-Average':<15} | {'Macro-Average':<15}")
    print("  " + "-" * 78)
    print(f"  {'Precision':<40} | {mem['micro_precision_percent']:>5.2f} %         | {mem['macro_precision_percent']:>5.2f} %")
    print(f"  {'Recall':<40} | {mem['micro_recall_percent']:>5.2f} %         | {mem['macro_recall_percent']:>5.2f} %")
    print(f"  {'F1-Score':<40} | {mem['micro_f1_percent']:>5.2f} %         | {mem['macro_f1_percent']:>5.2f} %")

    if timings_summary:
        print("\n[4. PIPELINE LATENCY & STAGE BENCHMARKS]")
        print("-" * 80)
        print(f"  {'Pipeline Stage':<35} | {'Mean (s)':<10} | {'Median (s)':<10} | {'Total (s)':<10}")
        print("  " + "-" * 78)
        for stg, t_info in timings_summary.items():
            print(f"  {stg:<35} | {t_info['mean_seconds']:>8.4f} s | {t_info['median_seconds']:>8.4f} s | {t_info['total_seconds']:>8.2f} s")

    print("\n" + "=" * 80)
    print(f" 📄 Full evaluation report saved to: {out_p.resolve()}")
    print("=" * 80)

    return report_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate Prescription OCR and Clinical Entity Extraction.")
    parser.add_argument("--labels", required=False, default="labels.csv", help="Path to ground truth labels CSV")
    parser.add_argument("--predictions", required=False, default="predictions.csv", help="Path to pipeline predictions CSV")
    parser.add_argument("--timings", required=False, default=None, help="Path to timings CSV (optional)")
    parser.add_argument("--out", required=False, default="evaluation_report.json", help="Output JSON evaluation report path")

    args = parser.parse_args()

    # Fallback to look in current folder if path not found
    labels_path = Path(args.labels)
    if not labels_path.exists() and (Path(".") / "labels.csv").exists():
        labels_path = Path(".") / "labels.csv"

    preds_path = Path(args.predictions)
    if not preds_path.exists() and (Path(".") / "predictions.csv").exists():
        preds_path = Path(".") / "predictions.csv"

    timings_path = Path(args.timings) if args.timings else None
    if timings_path and not timings_path.exists() and (Path(".") / "timings.csv").exists():
        timings_path = Path(".") / "timings.csv"

    out_path = Path(args.out)

    run_evaluation(
        labels_path=labels_path,
        predictions_path=preds_path,
        timings_path=timings_path,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()

