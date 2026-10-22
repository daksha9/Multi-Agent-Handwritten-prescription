import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .history_agent import PatientHistorySynthesisAgent
from .parser import ClinicalDocumentParser
from .prediction_agent import DiseasePredictionAgent
from .preprocessor import PrescriptionPreprocessor
from .prescription_vision import PrescriptionVisionExtractor
from .verification_agent import PrescriptionVerificationAgent
from .voice_agent import PatientVoiceHistoryAgent


class ClinicalAgentOrchestrator:
    """
    Multi-Agent Clinical Orchestrator & Telemetry Engine.
    Coordinates specialized agents (Verification, Prediction, History, Document Parser),
    maintains global clinical context across sessions, and generates structured datasets
    for Power BI dashboards.
    """

    def __init__(self, export_dir: str | Path | None = None):
        if export_dir is None:
            self.export_dir = Path(__file__).resolve().parents[2] / "outputs" / "exports"
        else:
            self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-agents
        self.parser = ClinicalDocumentParser()
        self.verification_agent = PrescriptionVerificationAgent()
        self.prediction_agent = DiseasePredictionAgent()
        self.history_agent = PatientHistorySynthesisAgent()

        # 4-stage pipeline agents
        self.preprocessor = PrescriptionPreprocessor()
        self.vision_extractor = PrescriptionVisionExtractor()
        self.voice_agent = PatientVoiceHistoryAgent()

        # Telemetry storage
        self.processed_prescriptions: list[dict[str, Any]] = []
        self.processed_disease_predictions: list[dict[str, Any]] = []
        self.processed_biomarkers: list[dict[str, Any]] = []
        self.patient_timelines: list[dict[str, Any]] = []

    def process_prescription_verification(
        self,
        extracted_drugs: list[str],
        ocr_confidence_scores: list[float] | None = None,
        patient_allergies: list[str] | None = None,
        patient_conditions: list[str] | None = None,
        active_medications: list[str] | None = None,
    ) -> dict[str, Any]:
        """Runs the verification agent and records telemetry."""
        start_time = time.time()
        verification_result = self.verification_agent.verify_prescriptions(
            prescribed_drugs=extracted_drugs,
            patient_allergies=patient_allergies,
            patient_conditions=patient_conditions,
            active_medications=active_medications,
        )
        elapsed_time = round(time.time() - start_time, 3)

        # Log telemetry record
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        avg_ocr_score = (
            round(sum(ocr_confidence_scores) / len(ocr_confidence_scores), 1)
            if ocr_confidence_scores and len(ocr_confidence_scores) > 0
            else 85.0
        )

        for drug in extracted_drugs:
            if not drug or drug.lower() == "not matched":
                continue
            self.processed_prescriptions.append(
                {
                    "timestamp": timestamp,
                    "drug_name": drug,
                    "ocr_confidence": avg_ocr_score,
                    "safety_status": verification_result["overall_status"],
                    "critical_alerts": verification_result["critical_count"],
                    "warning_alerts": verification_result["warning_count"],
                    "total_alerts": verification_result["total_alerts"],
                    "execution_latency_sec": elapsed_time,
                }
            )

        verification_result["execution_latency_sec"] = elapsed_time
        return verification_result

    # ============================================================
    # 4-STAGE AGENT PIPELINE
    # ============================================================

    def run_4stage_pipeline(
        self,
        image_input,
        audio_path: str | Path | None = None,
        audio_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        """
        End-to-end 4-stage agent pipeline:
          Stage 1: Image Preprocessing (CLAHE, binarization, de-skew)
          Stage 2: Handwritten OCR & Text Extraction (Gemini Vision + drug normalization)
          Stage 3: Patient Voice History & Speech (Gemini Audio transcription + NER)
          Stage 4: Clinical Reasoning & Doctor Safety (cross-reference + risk report)

        Parameters
        ----------
        image_input : PIL.Image.Image or str or Path
            Raw handwritten prescription image.
        audio_path : str or Path, optional
            Path to patient's audio recording (.wav / .mp3).
        audio_bytes : bytes, optional
            Raw bytes of on-the-spot voice recording. If neither provided,
            Stage 3 returns an empty history and Stage 4 skips voice-based checks.

        Returns
        -------
        dict
            {
              "stage1_preprocessing": {...},
              "stage2_ocr_extraction": {...},
              "stage3_voice_history": {...},
              "stage4_doctor_dashboard": {...},
              "execution_telemetry": {...}
            }
        """
        pipeline_start = time.time()
        telemetry: dict[str, Any] = {"stages": {}}

        # ── STAGE 1: Image Preprocessing ──
        t0 = time.time()
        stage1 = self.preprocessor.run_stage1(image_input)
        telemetry["stages"]["stage1"] = {
            "status": stage1.get("status"),
            "latency_sec": round(time.time() - t0, 3),
        }

        if stage1.get("status") != "SUCCESS":
            return {
                "stage1_preprocessing": stage1,
                "stage2_ocr_extraction": None,
                "stage3_voice_history": None,
                "stage4_doctor_dashboard": None,
                "execution_telemetry": telemetry,
                "error": f"Stage 1 failed: {stage1.get('error')}",
            }

        enhanced_image = stage1["enhanced_image"]

        # ── STAGE 2: Handwritten OCR & Text Extraction ──
        t0 = time.time()
        stage2 = self.vision_extractor.extract_stage2(enhanced_image)
        telemetry["stages"]["stage2"] = {
            "medications_count": len(stage2.get("prescribed_medications", [])),
            "latency_sec": round(time.time() - t0, 3),
        }

        # ── STAGE 3: Patient Voice History & Speech ──
        t0 = time.time()
        if audio_bytes:
            stage3 = self.voice_agent.transcribe_and_extract(audio_bytes)
        elif audio_path:
            stage3 = self.voice_agent.transcribe_and_extract(audio_path)
        else:
            stage3 = {
                "raw_transcript": "",
                "patient_reported_history": {
                    "known_allergies": [],
                    "chronic_conditions": [],
                    "current_active_medications": [],
                    "past_reported_side_effects": [],
                },
                "note": "No audio provided — voice history skipped.",
            }
        telemetry["stages"]["stage3"] = {
            "transcript_length": len(stage3.get("raw_transcript", "")),
            "latency_sec": round(time.time() - t0, 3),
        }

        # ── STAGE 4: Clinical Reasoning & Doctor Safety ──
        t0 = time.time()
        stage4 = self.verification_agent.run_stage4(stage2, stage3)
        telemetry["stages"]["stage4"] = {
            "alert_level": stage4.get("overall_safety_alert_level"),
            "risk_count": len(stage4.get("clinical_risk_analysis", [])),
            "latency_sec": round(time.time() - t0, 3),
        }

        telemetry["total_latency_sec"] = round(time.time() - pipeline_start, 3)

        return {
            "stage1_preprocessing": {
                "status": stage1["status"],
                "processed_image_path": stage1["processed_image_path"],
            },
            "stage2_ocr_extraction": stage2,
            "stage3_voice_history": stage3,
            "stage4_doctor_dashboard": stage4,
            "execution_telemetry": telemetry,
        }

    # Alias for 3-stage agent architecture
    run_3stage_pipeline = run_4stage_pipeline

    def run_post_medication_analysis(
        self,
        feedback_text: str | None = None,
        audio_bytes: bytes | None = None,
        audio_path: str | Path | None = None,
        prescribed_medications: list[dict[str, Any]] | list[str] | None = None,
        patient_allergies: list[str] | None = None,
        patient_conditions: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Coordinates post-medication voice & text analysis:
        1. Transcribes audio if provided.
        2. Merges with typed text.
        3. Calls verification agent to perform ADR causality & doctor recommendations.
        """
        transcribed_text = ""
        if audio_bytes:
            transcribed_text = self.voice_agent.transcribe_audio_bytes(audio_bytes)
        elif audio_path:
            transcribed_text = self.voice_agent.transcribe_audio(audio_path)

        combined_feedback = ""
        if transcribed_text and feedback_text:
            combined_feedback = f"{feedback_text.strip()} (Spoken note: {transcribed_text.strip()})"
        elif transcribed_text:
            combined_feedback = transcribed_text.strip()
        elif feedback_text:
            combined_feedback = feedback_text.strip()

        meds = prescribed_medications or []
        analysis = self.verification_agent.analyze_post_medication_feedback(
            prescribed_medications=meds,
            patient_allergies=patient_allergies,
            patient_conditions=patient_conditions,
            feedback_text=combined_feedback,
        )
        analysis["spoken_transcript"] = transcribed_text
        analysis["typed_text"] = feedback_text or ""
        return analysis

    def process_patient_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Parses multiple clinical documents, synthesizes history, and runs disease prediction."""
        start_time = time.time()

        # 1. Parse documents
        parsed_docs = self.parser.parse_multiple(documents)

        # Sort documents by date to ensure the most recent biomarkers win during aggregation
        # We assume parsed_docs contains 'date' or we fallback to a constant for sorting
        try:
            parsed_docs.sort(key=lambda x: x.get("date", "1900-01-01"))
        except Exception:
            pass

        # 2. Synthesize longitudinal clinical narrative & timeline
        history_result = self.history_agent.synthesize_history(parsed_docs)

        # 3. Aggregate latest biomarkers & vitals for disease prediction
        latest_biomarkers = {}
        latest_vitals = {}
        patient_info = {"age": 50, "gender": "Unknown"}

        for doc in parsed_docs:
            latest_biomarkers.update(doc.get("biomarkers", {}))
            latest_vitals.update(doc.get("vitals", {}))
            if doc.get("patient", {}).get("age"):
                patient_info["age"] = doc["patient"]["age"]
            if doc.get("patient", {}).get("gender"):
                patient_info["gender"] = doc["patient"]["gender"]

        # 4. Predict diseases with probability %
        prediction_result = self.prediction_agent.predict_diseases(
            biomarkers=latest_biomarkers,
            vitals=latest_vitals,
            patient_info=patient_info,
            diagnoses_history=history_result["all_diagnoses"],
        )

        elapsed_time = round(time.time() - start_time, 3)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Log disease predictions telemetry
        for pred in prediction_result.get("predictions", []):
            self.processed_disease_predictions.append(
                {
                    "timestamp": timestamp,
                    "patient_age": patient_info["age"],
                    "patient_gender": patient_info["gender"],
                    "disease": pred["disease"],
                    "probability_percent": pred["probability_percent"],
                    "risk_tier": pred["risk_tier"],
                    "primary_contributing_factor": pred["contributing_factors"][0]
                    if pred["contributing_factors"]
                    else "",
                }
            )

        # Log biomarkers telemetry
        for bm in prediction_result.get("biomarker_table", []):
            self.processed_biomarkers.append(
                {
                    "timestamp": timestamp,
                    "biomarker": bm["biomarker"],
                    "value": bm["value"],
                    "unit": bm["unit"],
                    "reference_range": bm["reference_range"],
                    "status": bm["status"],
                    "severity": bm["severity"],
                }
            )

        # Log timeline telemetry
        for tl in history_result.get("timeline", []):
            self.patient_timelines.append(
                {
                    "timestamp": timestamp,
                    "date": tl["date"],
                    "document": tl["document"],
                    "diagnoses": "; ".join(tl.get("diagnoses", [])),
                    "medications": "; ".join(tl.get("medications", [])),
                }
            )

        return {
            "parsed_documents": parsed_docs,
            "history": history_result,
            "predictions": prediction_result,
            "patient_info": patient_info,
            "execution_latency_sec": elapsed_time,
        }

    def export_powerbi_datasets(self) -> dict[str, str]:
        """
        Exports all telemetry into clean, structured CSV files ready for
        immediate connection with Power BI Desktop.
        """
        paths = {}

        # 1. Prescriptions Telemetry
        df_rx = (
            pd.DataFrame(self.processed_prescriptions)
            if self.processed_prescriptions
            else pd.DataFrame(
                columns=[
                    "timestamp",
                    "drug_name",
                    "ocr_confidence",
                    "safety_status",
                    "critical_alerts",
                    "warning_alerts",
                    "total_alerts",
                    "execution_latency_sec",
                ]
            )
        )
        rx_path = self.export_dir / "prescriptions_telemetry.csv"
        df_rx.to_csv(rx_path, index=False)
        paths["prescriptions"] = str(rx_path)

        # 2. Disease Predictions Telemetry
        df_pred = (
            pd.DataFrame(self.processed_disease_predictions)
            if self.processed_disease_predictions
            else pd.DataFrame(
                columns=[
                    "timestamp",
                    "patient_age",
                    "patient_gender",
                    "disease",
                    "probability_percent",
                    "risk_tier",
                    "primary_contributing_factor",
                ]
            )
        )
        pred_path = self.export_dir / "disease_predictions.csv"
        df_pred.to_csv(pred_path, index=False)
        paths["predictions"] = str(pred_path)

        # 3. Biomarkers Telemetry
        df_bm = (
            pd.DataFrame(self.processed_biomarkers)
            if self.processed_biomarkers
            else pd.DataFrame(
                columns=["timestamp", "biomarker", "value", "unit", "reference_range", "status", "severity"]
            )
        )
        bm_path = self.export_dir / "biomarkers_telemetry.csv"
        df_bm.to_csv(bm_path, index=False)
        paths["biomarkers"] = str(bm_path)

        # 4. Patient Timeline
        df_tl = (
            pd.DataFrame(self.patient_timelines)
            if self.patient_timelines
            else pd.DataFrame(columns=["timestamp", "date", "document", "diagnoses", "medications"])
        )
        tl_path = self.export_dir / "patient_timeline.csv"
        df_tl.to_csv(tl_path, index=False)
        paths["timeline"] = str(tl_path)

        return paths

    def export_longitudinal_powerbi_datasets(
        self,
        patient_info: dict[str, Any],
        comparison_records: list[dict[str, Any]],
        prescriptions: list[dict[str, Any]],
        lab_reports: list[dict[str, Any]],
    ) -> dict[str, str]:
        """
        Exports full longitudinal patient profile, medication impact comparison,
        and prescription history for Power BI Desktop.
        """
        paths = {}

        # 1. Patient Overview
        df_patient = pd.DataFrame(
            [
                {
                    "Patient_ID": patient_info.get("patient_id", "P001"),
                    "Name": patient_info.get("name", "Unknown"),
                    "Age": patient_info.get("age", "N/A"),
                    "Gender": patient_info.get("gender", "N/A"),
                    "Conditions": "; ".join(patient_info.get("known_conditions", [])),
                    "Allergies": "; ".join(patient_info.get("known_allergies", [])),
                    "Adverse_Reactions": "; ".join(patient_info.get("adverse_reactions", [])),
                    "Current_Medications": "; ".join(patient_info.get("current_medications", [])),
                    "Medical_History": patient_info.get("medical_history", ""),
                }
            ]
        )
        p_path = self.export_dir / "patient_overview_telemetry.csv"
        df_patient.to_csv(p_path, index=False)
        paths["patient_overview"] = str(p_path)

        # 2. Medication Impact Comparison
        df_comp = (
            pd.DataFrame(comparison_records)
            if comparison_records
            else pd.DataFrame(
                columns=[
                    "Parameter",
                    "Date of Test",
                    "Previous Value",
                    "Current Value",
                    "Unit",
                    "Reference Normal Range",
                    "Difference / Change",
                    "Percentage Change",
                    "Associated Medication",
                    "Condition Monitored",
                    "Trend Interpretation",
                    "Status Indicator",
                ]
            )
        )
        comp_path = self.export_dir / "medication_impact_comparison.csv"
        df_comp.to_csv(comp_path, index=False)
        paths["medication_impact"] = str(comp_path)

        # 3. Prescriptions Telemetry
        rx_rows = []
        for rx in prescriptions:
            rx_date = rx.get("date", "")
            dr = rx.get("doctor_name", "")
            clinic = rx.get("clinic_name", "")
            cond = rx.get("diagnosed_condition", "")
            for med in rx.get("medicines", []):
                rx_rows.append(
                    {
                        "Prescription_Date": rx_date,
                        "Doctor": dr,
                        "Clinic": clinic,
                        "Diagnosed_Condition": cond,
                        "Medicine_Name": med.get("medicine_name", ""),
                        "Form": med.get("form", "Tablet"),
                        "Dosage": med.get("dosage", ""),
                        "Frequency": med.get("frequency", ""),
                        "Duration": med.get("duration", ""),
                        "Instructions": med.get("instructions", ""),
                        "Certainty": med.get("certainty", "High"),
                    }
                )
        df_rx_full = (
            pd.DataFrame(rx_rows)
            if rx_rows
            else pd.DataFrame(
                columns=[
                    "Prescription_Date",
                    "Doctor",
                    "Clinic",
                    "Diagnosed_Condition",
                    "Medicine_Name",
                    "Form",
                    "Dosage",
                    "Frequency",
                    "Duration",
                    "Instructions",
                    "Certainty",
                ]
            )
        )
        rx_full_path = self.export_dir / "prescriptions_telemetry.csv"
        df_rx_full.to_csv(rx_full_path, index=False)
        paths["prescriptions"] = str(rx_full_path)

        # 4. Biomarkers / Lab Reports
        df_labs = (
            pd.DataFrame(lab_reports)
            if lab_reports
            else pd.DataFrame(
                columns=["date", "parameter", "value", "unit", "reference_range", "condition", "associated_medication"]
            )
        )
        labs_path = self.export_dir / "biomarkers_telemetry.csv"
        df_labs.to_csv(labs_path, index=False)
        paths["biomarkers"] = str(labs_path)

        return paths

    def generate_unified_clinical_json(
        self,
        patient_info: dict[str, Any],
        comparison_records: list[dict[str, Any]],
        prescriptions: list[dict[str, Any]],
        lab_reports: list[dict[str, Any]],
        rx_res: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Synthesizes clinical data into the standard clinical intelligence JSON structure.

        KEY IMPROVEMENTS (v2):
        - Reports are grouped by (date + associated_medication) instead of date only, preventing
          multiple distinct reports from being merged when collected on the same date.
        - Each report receives a unique REP_###_MED ID for unambiguous identification.
        - Status calculations use numeric reference range parsing instead of string-condition guessing.
        - Delta comparisons now track ALL chronological phases per parameter, not just binary baseline vs single follow-up.
        - Medication efficiency evaluation computes a weighted therapeutic-score across all parameters
          to identify the regimen with the strongest aggregate evidence of normalizing effect.
        - Prescription_extraction and longitudinal_report_analysis are fully isolated:
          longitudinal_report_analysis contains ZERO prescription-derived fields.
        - Defensive try/except wrappers prevent malformed records from breaking the full payload.
        """
        # ──────────────────────────────────────────────────────────────────────
        # 1. SIDEBAR PATIENT PROFILE  (blank-aware)
        # ──────────────────────────────────────────────────────────────────────
        raw_allergies = patient_info.get("known_allergies") or []
        raw_conditions = patient_info.get("known_conditions") or []
        raw_adverse = patient_info.get("adverse_reactions") or []

        def _join_or_none(lst: list[str]) -> str | None:
            cleaned = [s for s in lst if s and str(s).strip()]
            return ", ".join(cleaned) if cleaned else None

        allergies_str = _join_or_none(raw_allergies)
        conditions_str = _join_or_none(raw_conditions)
        adverse_str = _join_or_none(raw_adverse)

        safety_alerts: list[str] = []
        if allergies_str and not any(
            a.lower() in allergies_str.lower() for a in ["none documented", "no known drug allergies", "nka"]
        ):
            safety_alerts.append(f"Documented Allergies: {allergies_str}")
        if conditions_str:
            safety_alerts.append(f"Active Monitoring: {conditions_str}")
        safety_measures = " | ".join(safety_alerts) if safety_alerts else "Routine monitoring and scheduled follow-ups."

        patient_name_raw = patient_info.get("name")
        sidebar_profile = {
            "patient_name": patient_name_raw if patient_name_raw else None,
            "known_allergies": allergies_str,
            "known_conditions": conditions_str,
            "adverse_effects_history": adverse_str,
            "safety_measures_and_alerts": safety_measures if safety_alerts else None,
        }

        # ──────────────────────────────────────────────────────────────────────
        # 2. PRESCRIPTION EXTRACTION BLOCK  (only if rx_res or prescriptions provided)
        # ──────────────────────────────────────────────────────────────────────
        target_rx: dict[str, Any] = {}
        if rx_res and rx_res.get("status") == "success":
            target_rx = dict(rx_res)
        elif prescriptions:
            target_rx = dict(prescriptions[-1])

        meds_raw: list[dict[str, Any]] = target_rx.get("medicines", []) if target_rx else []
        prescribed_meds: list[dict[str, Any]] = []
        clinical_explanations: list[dict[str, Any]] = []

        try:
            for m in meds_raw:
                m_name = m.get("medicine_name") or "Unknown Medication"
                prescribed_meds.append(
                    {
                        "medicine_name": m_name,
                        "dosage": m.get("dosage") or "Standard dose",
                        "frequency": m.get("frequency") or "As directed",
                        "duration": m.get("duration") or "As directed",
                        "instructions": m.get("instructions") or "Take as directed by physician",
                        "confidence_level": m.get("certainty") or "High",
                    }
                )
                diag_cond = target_rx.get("diagnosed_condition") or "patient condition"
                clinical_explanations.append(
                    {
                        "medicine_name": m_name,
                        "therapeutic_use": m.get("verification_notes")
                        or f"Therapeutic management for {diag_cond}.",
                        "mechanism_of_action": (
                            f"Pharmacological agent targeting receptors and physiological pathways "
                            f"implicated in {diag_cond}."
                        ),
                    }
                )
        except Exception:
            prescribed_meds = []
            clinical_explanations = []

        def _blank_if_missing(val: Any) -> Any:
            if val is None:
                return None
            if isinstance(val, str) and not val.strip():
                return None
            return val

        rx_patient_name = target_rx.get("patient_name") or patient_info.get("name")
        prescription_extraction = {
            "patient_info": {
                "name": _blank_if_missing(rx_patient_name),
                "age": target_rx.get("patient_age") or patient_info.get("age"),
                "gender": _blank_if_missing(target_rx.get("patient_gender") or patient_info.get("gender")),
                "date": _blank_if_missing(target_rx.get("date")),
            },
            "doctor_info": {
                "name": _blank_if_missing(target_rx.get("doctor_name")),
                "qualification": _blank_if_missing(
                    target_rx.get("doctor_registration") or "Consultant Physician"
                ),
            },
            "diagnosed_condition": _blank_if_missing(target_rx.get("diagnosed_condition")),
            "prescribed_medications": prescribed_meds,
            "unreadable_fields": target_rx.get("unreadable_fields", []) or [],
            "clinical_explanations": clinical_explanations,
            "safety_and_disclaimers": {
                "warnings_and_interactions": [
                    (
                        "Cross-check drug interactions with active regimen: "
                        f"{', '.join(patient_info.get('current_medications', [])) or 'None documented'}."
                    ),
                    "Ensure adequate hydration and regular liver and renal function panels.",
                ],
                "potential_adverse_effects": [
                    "Mild gastrointestinal symptoms or headache may occur during therapeutic initiation.",
                    "Discontinue and report immediately if any rash, facial swelling, or breathing difficulty occurs.",
                ],
                "general_disclaimer": (
                    "AI-generated extraction powered by Gemini 3.8 Flash Vision. "
                    "Verify with a licensed healthcare provider before administration."
                ),
            },
            "patient_care_tips": [
                "Maintain strict adherence to prescribed dosage timing.",
                "Store medications in a cool, dry place away from direct sunlight.",
                "Record daily blood pressure and blood glucose in a personal health log.",
            ],
        }

        # ──────────────────────────────────────────────────────────────────────
        # 3. LONGITUDINAL REPORT ANALYSIS BLOCK   (report-only; NO prescription data)
        # ──────────────────────────────────────────────────────────────────────
        from .impact_analyzer import MedicationImpactAnalyzer  # lazy import to avoid circular top-level
        _analyzer = MedicationImpactAnalyzer()

        def _parse_ref_range(ref_str: str | None) -> tuple[float | None, float | None]:
            """Safely extract numeric (low, high) from strings like '70 - 99' or '< 200'."""
            if not ref_str:
                return None, None
            try:
                s = str(ref_str).strip()
                if "-" in s:
                    parts = s.split("-", 1)
                    lo = float(parts[0].strip().replace("<", "").replace(">", ""))
                    hi = float(parts[1].strip().replace("<", "").replace(">", ""))
                    return lo, hi
                if "<" in s:
                    return None, float(s.replace("<", "").strip())
                if ">" in s:
                    return float(s.replace(">", "").strip()), None
            except (ValueError, TypeError, AttributeError):
                pass
            return None, None

        def _classify_status(value: float, ref_str: str | None, param_name: str) -> str:
            """Numeric-aware status classifier: Normal / Elevated / Critical / Low."""
            lo, hi = _parse_ref_range(ref_str)
            metric = _analyzer.PARAMETER_METRICS.get(param_name, {})
            if lo is None and metric.get("normal_min") is not None:
                lo = float(metric["normal_min"])
            if hi is None and metric.get("normal_max") is not None:
                hi = float(metric["normal_max"])

            try:
                v = float(value)
            except (ValueError, TypeError):
                return "Elevated"

            if lo is not None and hi is not None:
                span = max(1e-9, hi - lo)
                if v < lo - (0.5 * span):
                    return "Critical"
                if v > hi + (0.5 * span):
                    return "Critical"
                if lo <= v <= hi:
                    return "Normal"
                if v < lo:
                    return "Low"
                return "Elevated"
            if hi is not None and v > hi:
                return "Elevated"
            if lo is not None and v < lo:
                return "Low"
            return "Normal"

        # Group reports by (DATE, ASSOCIATED_MEDICATION) tuple — this is the core fix
        # that prevents multiple distinct reports on the same date from merging.
        report_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in lab_reports:
            try:
                d = str(r.get("date", "Unknown Date")).strip() or "Unknown Date"
                med = str(r.get("associated_medication", "Baseline / Unassigned")).strip() or "Baseline / Unassigned"
                key = (d, med)
                report_groups.setdefault(key, []).append(r)
            except Exception:
                continue

        chronological_results: list[dict[str, Any]] = []
        sorted_keys = sorted(report_groups.keys(), key=lambda k: (k[0], k[1]))
        for idx, (dt, med_name) in enumerate(sorted_keys):
            items = report_groups[(dt, med_name)]
            med_sanitized = "".join(ch if ch.isalnum() else "_" for ch in med_name[:16]).strip("_") or "MED"
            params: list[dict[str, Any]] = []
            for item in items:
                try:
                    v_float = float(item.get("value", 0.0))
                except (ValueError, TypeError):
                    v_float = 0.0
                pname = item.get("parameter") or "Biomarker"
                ref = item.get("reference_range") or "Normal reference interval"
                params.append(
                    {
                        "test_name": pname,
                        "value": round(v_float, 3),
                        "unit": item.get("unit") or "",
                        "reference_range": ref,
                        "status": _classify_status(v_float, ref, pname),
                    }
                )
            chronological_results.append(
                {
                    "report_id": f"REP_{idx + 1:03d}_{med_sanitized}",
                    "report_date": dt,
                    "active_medication": med_name,
                    "parameters": params,
                    "report_parameter_count": len(params),
                }
            )

        # Delta comparisons: Track ALL chronological phases per parameter (not just baseline vs one follow-up)
        delta_comparisons: list[dict[str, Any]] = []
        try:
            # Organize lab reports by parameter, in chronological order
            param_to_records: dict[str, list[dict[str, Any]]] = {}
            for r in sorted(lab_reports, key=lambda x: (str(x.get("date", "")), str(x.get("associated_medication", "")))):
                pname = r.get("parameter") or "Unknown"
                param_to_records.setdefault(pname, []).append(r)

            for pname, records in param_to_records.items():
                if not records:
                    continue
                metric_def = _analyzer.PARAMETER_METRICS.get(pname, {})
                unit = records[0].get("unit") or metric_def.get("unit", "")
                ref_range = records[0].get("reference_range") or (
                    f"{metric_def.get('normal_min', '')} - {metric_def.get('normal_max', '')}"
                )
                try:
                    baseline_val = float(records[0].get("value", 0.0))
                except (ValueError, TypeError):
                    baseline_val = 0.0

                phase_results: list[dict[str, Any]] = []
                # Phase 0 = Baseline
                phase_results.append(
                    {
                        "phase": "Baseline",
                        "date": str(records[0].get("date", "")),
                        "medication": str(records[0].get("associated_medication", "Baseline")),
                        "value": round(baseline_val, 3),
                        "absolute_change": 0.0,
                        "percentage_change": 0.0,
                        "trend_direction": "Baseline",
                    }
                )
                # Phases 1..N = Subsequent observations
                for i in range(1, len(records)):
                    prev = records[i - 1]
                    curr = records[i]
                    try:
                        prev_v = float(prev.get("value", 0.0))
                        curr_v = float(curr.get("value", 0.0))
                    except (ValueError, TypeError):
                        prev_v = curr_v = 0.0
                    abs_change = round(curr_v - prev_v, 3)
                    pct_change = round(((curr_v - prev_v) / prev_v) * 100, 2) if prev_v != 0 else 0.0

                    lower_better = metric_def.get("lower_is_better", True)
                    try:
                        _, hi = _parse_ref_range(ref_range)
                    except Exception:
                        hi = None

                    def _direction(
                        abs_ch: float,
                        pct_ch: float,
                        lb: bool = lower_better,
                        hi_lim: float | None = hi,
                        cur_val: float = curr_v,
                    ) -> str:
                        if abs(pct_ch) < 2.0:
                            return "Stable"
                        if lb:
                            if abs_ch < 0:
                                if hi_lim is not None and cur_val <= hi_lim:
                                    return "Improving"
                                return "Improving"
                            return "Worsening"
                        else:
                            if abs_ch > 0:
                                return "Improving"
                            return "Worsening"

                    phase_results.append(
                        {
                            "phase": f"Post-Medication Phase {i}",
                            "date": str(curr.get("date", "")),
                            "medication": str(curr.get("associated_medication", "Active Therapy")),
                            "value": round(curr_v, 3),
                            "absolute_change": abs_change,
                            "percentage_change": pct_change,
                            "trend_direction": _direction(abs_change, pct_change),
                        }
                    )

                delta_comparisons.append(
                    {
                        "test_name": pname,
                        "unit": unit,
                        "reference_range": ref_range,
                        "baseline_value": round(baseline_val, 3),
                        "phase_results": phase_results,
                    }
                )
        except Exception:
            delta_comparisons = []

        # ── Weighted Medication Efficiency Evaluation ──────────────────────
        # Score each medication by the fraction of parameters trending toward normal,
        # weighted by the percentage-magnitude of improvement.  This is purely
        # descriptive and NOT a causal claim (clinical_disclaimer reflects this).
        medication_scores: dict[str, float] = {}
        medication_counts: dict[str, int] = {}
        try:
            for comp in comparison_records:
                try:
                    pct_raw = str(comp.get("raw_pct") or comp.get("Percentage Change", "0")).replace("%", "").strip()
                    pct_v = float(pct_raw) if pct_raw else 0.0
                except (ValueError, TypeError):
                    pct_v = 0.0
                status = str(comp.get("Status Indicator", "")).lower()
                med = str(comp.get("Associated Medication", "Unassigned Therapy"))

                # Score magnitude = absolute percentage change toward the target direction
                is_favorable = any(k in status for k in ["improve", "in range", "downward", "better", "normal"])
                score_magnitude = abs(pct_v) if is_favorable else -abs(pct_v) * 0.5

                medication_scores[med] = medication_scores.get(med, 0.0) + score_magnitude
                medication_counts[med] = medication_counts.get(med, 0) + 1
        except Exception:
            medication_scores = {}
            medication_counts = {}

        # Normalize scores by observation count to avoid bias toward regimens with more data points
        normalized_scores: dict[str, float] = {}
        for med, total in medication_scores.items():
            cnt = max(1, medication_counts.get(med, 1))
            normalized_scores[med] = round(total / cnt, 3)

        best_med: str
        evidence_lines: list[str]
        if normalized_scores:
            best_med = max(normalized_scores, key=lambda k: normalized_scores[k])
            # Build evidence-based reasoning narrative
            evidence_lines = [
                f"Descriptive longitudinal trend analysis reviewed {len(report_groups)} distinct diagnostic "
                f"reports (each uniquely identified by date + associated medication) across "
                f"{len(param_to_records)} monitored biomarkers."
            ]
            ranked = sorted(normalized_scores.items(), key=lambda kv: kv[1], reverse=True)
            for i, (med, score) in enumerate(ranked[:3]):
                adj = "strongest aggregate therapeutic signal" if i == 0 else "next-strongest signal"
                direction = "favorable trends" if score >= 0 else "net unfavorable trends"
                evidence_lines.append(
                    f"• {med}: Weighted therapeutic-score = {score:+.2f} per observation "
                    f"({medication_counts.get(med, 0)} linked readings) → {adj} with {direction}."
                )
            # Parameter-level narrative for the best-scoring medication
            best_med_deltas = [d for d in delta_comparisons if any(ph["medication"] == best_med for ph in d["phase_results"])]
            if best_med_deltas:
                narratives = []
                for d in best_med_deltas[:4]:
                    final_ph = d["phase_results"][-1]
                    narratives.append(
                        f"{d['test_name']} Δ{final_ph['percentage_change']:+.1f}% ({final_ph['trend_direction']})"
                    )
                if narratives:
                    evidence_lines.append(
                        f"Under {best_med}: " + "; ".join(narratives) + "."
                    )
            evidence_lines.append(
                "Scores summarize direction-consistent movement toward reference ranges; "
                "they do NOT establish pharmacologic causality."
            )
        else:
            best_med = "Insufficient comparative data"
            evidence_lines = [
                "Insufficient paired before/after laboratory records exist to rank regimen efficiency. "
                "Add sequential reports in the Reports Comparision tab with distinct medication labels to populate this analysis."
            ]

        evidence_text = "\n".join(evidence_lines)

        longitudinal_analysis = {
            "report_summary_metadata": {
                "total_distinct_reports": len(chronological_results),
                "total_unique_report_ids": [r["report_id"] for r in chronological_results],
                "reports_grouping_method": "Grouped by (report_date, associated_medication) to prevent merging of distinct same-day reports.",
            },
            "chronological_lab_results": chronological_results,
            "delta_comparisons": delta_comparisons,
            "medication_efficiency_evaluation": {
                "most_effective_medication": best_med,
                "weighted_therapeutic_scores": normalized_scores,
                "evidence_based_reasoning": evidence_text,
                "clinical_disclaimer": (
                    "Trends describe observed laboratory data and do not substitute direct medical advice. "
                    "Efficiency scoring is a descriptive aggregation of movement toward reference ranges; "
                    "it is not a causal proof of medication action."
                ),
            },
        }

        return {
            "sidebar_patient_profile": sidebar_profile,
            "prescription_extraction": prescription_extraction,
            "longitudinal_report_analysis": longitudinal_analysis,
        }
