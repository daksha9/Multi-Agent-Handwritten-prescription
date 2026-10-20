"""
Medication Impact & Longitudinal Patient Report Analyzer for MedAgent.
Computes before/after medication comparisons, trajectory differences,
percentage changes, and descriptive clinical interpretations.
"""

from typing import Any

import pandas as pd


class MedicationImpactAnalyzer:
    """
    Analyzes multi-visit laboratory reports across medication change intervals.
    Provides chronological difference tracking, percentage changes,
    and descriptive trend interpretations without claiming unsupported causality.
    """

    # Reference bounds for common clinical parameters to determine clinical direction
    # 'lower_is_better': True for glucose, BP, LDL; False for eGFR, HDL
    PARAMETER_METRICS = {
        "HbA1c": {
            "unit": "%",
            "normal_min": 4.0,
            "normal_max": 5.6,
            "lower_is_better": True,
            "condition": "Type 2 Diabetes",
        },
        "Fasting Blood Glucose": {
            "unit": "mg/dL",
            "normal_min": 70.0,
            "normal_max": 99.0,
            "lower_is_better": True,
            "condition": "Type 2 Diabetes",
        },
        "Postprandial Blood Glucose": {
            "unit": "mg/dL",
            "normal_min": 70.0,
            "normal_max": 140.0,
            "lower_is_better": True,
            "condition": "Type 2 Diabetes",
        },
        "Systolic Blood Pressure": {
            "unit": "mmHg",
            "normal_min": 90.0,
            "normal_max": 120.0,
            "lower_is_better": True,
            "condition": "Hypertension",
        },
        "Diastolic Blood Pressure": {
            "unit": "mmHg",
            "normal_min": 60.0,
            "normal_max": 80.0,
            "lower_is_better": True,
            "condition": "Hypertension",
        },
        "Total Cholesterol": {
            "unit": "mg/dL",
            "normal_min": 125.0,
            "normal_max": 200.0,
            "lower_is_better": True,
            "condition": "Dyslipidemia",
        },
        "LDL Cholesterol": {
            "unit": "mg/dL",
            "normal_min": 50.0,
            "normal_max": 100.0,
            "lower_is_better": True,
            "condition": "Dyslipidemia",
        },
        "HDL Cholesterol": {
            "unit": "mg/dL",
            "normal_min": 40.0,
            "normal_max": 60.0,
            "lower_is_better": False,
            "condition": "Dyslipidemia",
        },
        "Serum Triglycerides": {
            "unit": "mg/dL",
            "normal_min": 50.0,
            "normal_max": 150.0,
            "lower_is_better": True,
            "condition": "Dyslipidemia",
        },
        "Serum Creatinine": {
            "unit": "mg/dL",
            "normal_min": 0.7,
            "normal_max": 1.3,
            "lower_is_better": True,
            "condition": "Renal Function",
        },
        "eGFR": {
            "unit": "mL/min/1.73m²",
            "normal_min": 90.0,
            "normal_max": 120.0,
            "lower_is_better": False,
            "condition": "Renal Function",
        },
        "Body Temperature": {
            "unit": "°F",
            "normal_min": 97.8,
            "normal_max": 99.1,
            "lower_is_better": True,
            "condition": "Infection/Fever",
        },
        "Heart Rate": {
            "unit": "bpm",
            "normal_min": 60.0,
            "normal_max": 100.0,
            "lower_is_better": True,
            "condition": "Vitals",
        },
        "SpO2": {
            "unit": "%",
            "normal_min": 95.0,
            "normal_max": 100.0,
            "lower_is_better": False,
            "condition": "Respiratory",
        },
    }

    def __init__(self):
        pass

    def compare_medication_impact(
        self, lab_reports: list[dict[str, Any]], medication_timeline: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """
        Alias for compute_parameter_comparison_table for compatibility with clinical dashboards.
        """
        return self.compute_parameter_comparison_table(lab_reports, medication_timeline)

    def compute_parameter_comparison_table(
        self, lab_reports: list[dict[str, Any]], medication_timeline: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """
        Calculates chronological before/after differences and percentage changes for every parameter.
        Returns rows matching Objective 2 requirements:
        - Test/parameter name
        - Date of test
        - Previous value
        - Current value
        - Unit
        - Reference/normal range
        - Difference/change
        - Percentage change
        - Associated medication
        - Condition being monitored
        - Interpretation of trend
        """
        if not lab_reports:
            return []

        df = pd.DataFrame(lab_reports)
        if "date" not in df.columns or "parameter" not in df.columns or "value" not in df.columns:
            return []

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"]).sort_values(by=["parameter", "date"])
        df = df.drop_duplicates(subset=["parameter", "date"], keep="last")

        comparison_records = []

        for param_name_raw, group in df.groupby("parameter"):
            param_name = str(param_name_raw)
            records = group.to_dict(orient="records")
            metric_info = self.PARAMETER_METRICS.get(param_name, {})

            for i in range(len(records)):
                curr = records[i]
                curr_val = curr["value"]
                curr_date = curr["date"].strftime("%Y-%m-%d")
                unit = curr.get("unit") or metric_info.get("unit", "")
                ref_range = (
                    curr.get("reference_range")
                    or f"{metric_info.get('normal_min', '')} - {metric_info.get('normal_max', '')}"
                )
                condition = curr.get("condition") or metric_info.get("condition", "General Health")
                associated_med = curr.get("associated_medication") or self._resolve_medication_for_date(
                    curr_date, medication_timeline
                )

                if i == 0:
                    # Baseline entry
                    prev_val = None
                    diff = 0.0
                    pct_change = 0.0
                    trend = "Baseline Measurement"
                    status = "Baseline"
                else:
                    prev = records[i - 1]
                    prev_val = prev["value"]
                    diff = round(curr_val - prev_val, 2)
                    pct_change = round(((curr_val - prev_val) / prev_val) * 100, 2) if prev_val != 0 else 0.0

                    # Clinical descriptive trend interpretation
                    trend, status = self._interpret_trend(
                        param_name=param_name,
                        prev_val=prev_val,
                        curr_val=curr_val,
                        diff=diff,
                        pct_change=pct_change,
                        metric_info=metric_info,
                    )

                comparison_records.append(
                    {
                        "Parameter": param_name,
                        "Date of Test": curr_date,
                        "Previous Value": f"{prev_val:.2f}" if prev_val is not None else "— (Baseline)",
                        "Current Value": f"{curr_val:.2f}",
                        "Unit": unit,
                        "Reference Normal Range": ref_range,
                        "Difference / Change": f"{'+' if diff > 0 else ''}{diff:.2f}" if prev_val is not None else "—",
                        "Percentage Change": f"{'+' if pct_change > 0 else ''}{pct_change:.1f}%"
                        if prev_val is not None
                        else "—",
                        "Associated Medication": associated_med,
                        "Condition Monitored": condition,
                        "Trend Interpretation": trend,
                        "Status Indicator": status,
                        "raw_prev": prev_val,
                        "raw_curr": curr_val,
                        "raw_diff": diff,
                        "raw_pct": pct_change,
                    }
                )

        # Sort chronologically by date descending or parameter
        comparison_records.sort(key=lambda x: (x["Date of Test"], x["Parameter"]), reverse=True)
        return comparison_records

    def _resolve_medication_for_date(self, date_str: str, medication_timeline: list[dict[str, Any]] | None) -> str:
        """Determines which medication regimen was active during a given lab report date."""
        if not medication_timeline:
            return "Unassigned / General Care"
        try:
            target_date = pd.to_datetime(date_str)
            for period in medication_timeline:
                start = pd.to_datetime(period.get("start_date", "1900-01-01"))
                end = pd.to_datetime(period.get("end_date", "2100-01-01"))
                if start <= target_date <= end:
                    meds = period.get("medications", [])
                    return ", ".join(meds) if isinstance(meds, list) else str(meds)
        except Exception:
            pass
        return "General Care"

    def _interpret_trend(
        self,
        param_name: str,
        prev_val: float,
        curr_val: float,
        diff: float,
        pct_change: float,
        metric_info: dict[str, Any],
    ) -> tuple[str, str]:
        """
        Produces strictly descriptive, evidence-based interpretations of parameter trends.
        Does not claim medication causality, adhering to medical safety instructions.
        """
        normal_min = metric_info.get("normal_min")
        normal_max = metric_info.get("normal_max")
        lower_is_better = metric_info.get("lower_is_better", True)

        # Check stability threshold (less than 2% change)
        if abs(pct_change) < 2.0:
            in_range = ""
            if normal_min is not None and normal_max is not None:
                in_range = (
                    " (within normal range)" if (normal_min <= curr_val <= normal_max) else " (above target range)"
                )
            return f"Relatively stable parameter value (Δ {pct_change:+.1f}%){in_range}.", "Stable"

        if lower_is_better:
            if diff < 0:
                # Decreased (improvement if started high)
                if normal_max is not None and curr_val <= normal_max:
                    msg = f"Observed reduction of {abs(pct_change):.1f}% ({diff:.2f} {metric_info.get('unit', '')}), returning to target normal range."
                    status = "Improved (In Range)"
                else:
                    msg = f"Observed reduction of {abs(pct_change):.1f}% ({diff:.2f} {metric_info.get('unit', '')}) moving closer to reference threshold."
                    status = "Trending Downward"
            else:
                # Increased (worsening if lower is better)
                msg = f"Elevated by {abs(pct_change):.1f}% ({diff:+.2f} {metric_info.get('unit', '')}) above previous observation."
                status = "Elevated"
        else:
            # Higher is better (e.g. eGFR, SpO2, HDL)
            if diff > 0:
                msg = f"Observed increase of {abs(pct_change):.1f}% ({diff:+.2f} {metric_info.get('unit', '')}) showing positive trajectory."
                status = "Improved"
            else:
                msg = f"Observed decrease of {abs(pct_change):.1f}% ({diff:.2f} {metric_info.get('unit', '')}) below previous observation."
                status = "Declined"

        return msg, status

    def get_time_series_data_by_parameter(self, lab_reports: list[dict[str, Any]], parameter: str) -> pd.DataFrame:
        """Extracts chronological observations for a single parameter for plotting."""
        if not lab_reports:
            return pd.DataFrame()
        df = pd.DataFrame(lab_reports)
        if "parameter" not in df.columns or "date" not in df.columns:
            return pd.DataFrame()
        df_param = df[df["parameter"] == parameter].copy()
        if df_param.empty:
            return pd.DataFrame()
        df_param["date"] = pd.to_datetime(df_param["date"])
        df_param = df_param.sort_values(by="date")  # type: ignore
        df_param["date_str"] = df_param["date"].dt.strftime("%Y-%m-%d")
        return df_param

    def predict_parameter_trajectory(
        self,
        lab_reports: list[dict[str, Any]],
        parameter: str,
        forecast_horizons: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Computes predictive trajectory forecasting for a clinical biomarker
        over 30, 60, and 90-day future horizons based on observed regression rate.
        """
        if forecast_horizons is None:
            forecast_horizons = [30, 60, 90]

        df_param = self.get_time_series_data_by_parameter(lab_reports, parameter)
        metric_info = self.PARAMETER_METRICS.get(parameter, {})
        norm_max = metric_info.get("normal_max")
        norm_min = metric_info.get("normal_min")
        lower_is_better = metric_info.get("lower_is_better", True)
        unit = metric_info.get("unit", "")

        if df_param.empty:
            return {
                "parameter": parameter,
                "status": "no_data",
                "forecast_points": [],
                "target_achievement_estimate": "Insufficient laboratory data for forecasting.",
                "efficacy_rating": "Indeterminate",
            }

        curr_row = df_param.iloc[-1]
        curr_val = float(curr_row["value"])
        curr_date = pd.to_datetime(curr_row["date"])

        # Calculate daily velocity rate
        daily_rate = 0.0
        if len(df_param) >= 2:
            first_row = df_param.iloc[0]
            first_val = float(first_row["value"])
            first_date = pd.to_datetime(first_row["date"])
            days_span = max(1, (curr_date - first_date).days)
            # Dampen extreme daily swings for physiological plausibility
            daily_rate = (curr_val - first_val) / float(days_span)
            # Cap maximum monthly rate to 15% change
            max_monthly_delta = curr_val * 0.15
            max_daily_rate = max_monthly_delta / 30.0
            daily_rate = max(-max_daily_rate, min(max_daily_rate, daily_rate))

        forecast_points = []
        for days in forecast_horizons:
            f_date = curr_date + pd.Timedelta(days=days)
            f_val = curr_val + (daily_rate * days)
            # Apply physiological clamps
            if lower_is_better and norm_min is not None:
                f_val = max(norm_min * 0.9, f_val)

            uncertainty = (days / 90.0) * (0.05 * curr_val)
            forecast_points.append(
                {
                    "horizon_days": days,
                    "forecast_date": f_date.strftime("%Y-%m-%d"),
                    "forecast_value": round(f_val, 2),
                    "lower_bound": round(max(0.1, f_val - uncertainty), 2),
                    "upper_bound": round(f_val + uncertainty, 2),
                    "unit": unit,
                }
            )

        # Target normal achievement projection
        target_achievement = "Stabilized within target bounds."
        if lower_is_better and norm_max is not None:
            if curr_val <= norm_max:
                target_achievement = f"Currently in optimal target range (≤ {norm_max} {unit}). Projected maintenance."
            elif daily_rate < 0:
                needed_drop = curr_val - norm_max
                days_needed = int(needed_drop / abs(daily_rate))
                weeks_needed = max(1, round(days_needed / 7.0))
                target_achievement = f"Projected to attain normal threshold (≤ {norm_max} {unit}) in approx {weeks_needed} weeks under current regimen."
            else:
                target_achievement = f"Upward trajectory observed (+{daily_rate * 30:.1f} {unit}/month). Clinical regimen re-evaluation advised."
        elif not lower_is_better and norm_min is not None:
            if curr_val >= norm_min:
                target_achievement = f"Currently in optimal reference range (≥ {norm_min} {unit})."
            elif daily_rate > 0:
                needed_gain = norm_min - curr_val
                days_needed = int(needed_gain / daily_rate)
                weeks_needed = max(1, round(days_needed / 7.0))
                target_achievement = (
                    f"Projected to reach normal threshold (≥ {norm_min} {unit}) in approx {weeks_needed} weeks."
                )
            else:
                target_achievement = "Downstream trend observed. Clinical re-assessment indicated."

        # Medication efficacy rating
        if (lower_is_better and daily_rate < -0.01) or (not lower_is_better and daily_rate > 0.01):
            efficacy = "High Therapeutic Response"
        elif abs(daily_rate) <= 0.01:
            efficacy = "Stable / Maintenance Response"
        else:
            efficacy = "Suboptimal Response / Regimen Review Suggested"

        return {
            "parameter": parameter,
            "status": "success",
            "current_value": curr_val,
            "current_date": curr_date.strftime("%Y-%m-%d"),
            "monthly_velocity": round(daily_rate * 30.0, 2),
            "forecast_points": forecast_points,
            "target_achievement_estimate": target_achievement,
            "efficacy_rating": efficacy,
        }
