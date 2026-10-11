import os
from pathlib import Path
from typing import Any

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from google import genai
except ImportError:
    genai = None


class PatientHistorySynthesisAgent:
    """
    Patient History Synthesis Agent.
    Aggregates multi-document uploads (reports, past prescriptions, discharge summaries)
    into a structured longitudinal timeline, detects biomarker progression trends,
    highlights clinical gaps in care, and generates an executive physician summary.
    """

    def __init__(self):
        self.groq_client = self._init_groq()
        self.gemini_client = self._init_gemini()

    def _init_gemini(self) -> Any | None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            env_file = Path(__file__).resolve().parents[2] / "config" / ".env"
            if env_file.exists():
                try:
                    with open(env_file) as f:
                        for line in f:
                            if line.startswith("GEMINI_API_KEY="):
                                api_key = line.split("=", 1)[1].strip()
                                break
                except Exception:
                    pass
        if api_key and genai is not None:
            try:
                return genai.Client(api_key=api_key)
            except Exception as e:
                print(f"[HistoryAgent] Gemini init warning: {e}")
        return None

    def _init_groq(self) -> Any | None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            env_file = Path(__file__).resolve().parents[2] / "config" / ".env"
            if env_file.exists():
                try:
                    with open(env_file) as f:
                        for line in f:
                            if line.startswith("GROQ_API_KEY="):
                                api_key = line.split("=", 1)[1].strip()
                                break
                except Exception:
                    pass

        if api_key and Groq is not None:
            try:
                import httpx

                return Groq(api_key=api_key, http_client=httpx.Client())
            except Exception:
                try:
                    return Groq(api_key=api_key)
                except Exception as e:
                    print(f"[HistoryAgent] Groq init warning: {e}")
        return None

    def synthesize_history(self, parsed_docs: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Takes parsed documents, extracts chronological journey, biomarker trends,
        care gaps, and synthesizes an executive summary.
        """
        if not parsed_docs:
            return {
                "timeline": [],
                "biomarker_trends": [],
                "care_gaps": [],
                "executive_summary": "No historical documents provided.",
                "all_diagnoses": [],
                "all_medications": [],
                "all_allergies": [],
            }

        # Build chronological timeline
        timeline = []
        all_diagnoses = set()
        all_medications = set()
        all_allergies = set()
        biomarker_history: dict[str, list[dict[str, Any]]] = {}

        for doc in parsed_docs:
            doc_name = doc.get("document_name", "Report")
            date_str = doc.get("date", "Undated")
            vitals = doc.get("vitals", {})
            biomarkers = doc.get("biomarkers", {})
            diagnoses = doc.get("diagnoses", [])
            medications = doc.get("medications", [])
            allergies = doc.get("allergies", [])

            all_diagnoses.update(diagnoses)
            all_medications.update(medications)
            all_allergies.update(allergies)

            # Record biomarkers for trend analysis
            for b_name, b_val in biomarkers.items():
                if b_name not in biomarker_history:
                    biomarker_history[b_name] = []
                biomarker_history[b_name].append({"date": date_str, "value": b_val, "doc": doc_name})

            # Create timeline entry
            timeline.append(
                {
                    "date": date_str,
                    "document": doc_name,
                    "diagnoses": diagnoses,
                    "medications": medications,
                    "key_biomarkers": biomarkers,
                    "vitals": vitals,
                }
            )

        # Sort timeline if dates are present
        timeline.sort(key=lambda item: item["date"])

        # Compute Biomarker Trajectories (Trending up, down, stable)
        biomarker_trends = self._compute_trends(biomarker_history)

        # Detect Missing Care Gaps
        care_gaps = self._detect_care_gaps(all_diagnoses, biomarker_history, all_medications)

        # Generate Executive Narrative
        executive_summary = self._generate_executive_narrative(
            timeline, list(all_diagnoses), list(all_medications), list(all_allergies), biomarker_trends, care_gaps
        )

        return {
            "timeline": timeline,
            "biomarker_trends": biomarker_trends,
            "care_gaps": care_gaps,
            "executive_summary": executive_summary,
            "all_diagnoses": sorted(all_diagnoses),
            "all_medications": sorted(all_medications),
            "all_allergies": sorted(all_allergies),
        }

    def _compute_trends(self, history: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        marker_names = {
            "hba1c": "HbA1c",
            "fasting_blood_glucose": "Fasting Blood Glucose",
            "postprandial_blood_glucose": "Postprandial Blood Glucose",
            "serum_creatinine": "Serum Creatinine",
            "blood_urea": "Blood Urea Nitrogen",
            "total_cholesterol": "Total Cholesterol",
            "ldl_cholesterol": "LDL Cholesterol",
            "hdl_cholesterol": "HDL Cholesterol",
            "triglycerides": "Triglycerides",
            "hemoglobin": "Hemoglobin",
            "alt_sgpt": "ALT (SGPT)",
            "ast_sgot": "AST (SGOT)",
            "platelets": "Platelet Count",
        }

        trends = []
        for marker, records in history.items():
            if len(records) >= 2:
                first_val = records[0]["value"]
                last_val = records[-1]["value"]
                diff = round(last_val - first_val, 2)
                percent_change = round((diff / first_val) * 100, 1) if first_val else 0

                if abs(percent_change) < 5:
                    direction = "STABLE"
                elif diff > 0:
                    direction = (
                        "WORSENING"
                        if marker in ["hba1c", "fasting_blood_glucose", "serum_creatinine", "ldl_cholesterol"]
                        else "INCREASING"
                    )
                else:
                    direction = (
                        "IMPROVING"
                        if marker in ["hba1c", "fasting_blood_glucose", "serum_creatinine", "ldl_cholesterol"]
                        else "DECREASING"
                    )

                clean_name = marker_names.get(marker, marker.replace("_", " ").title())
                trends.append(
                    {
                        "marker": clean_name,
                        "initial_value": first_val,
                        "latest_value": last_val,
                        "change": diff,
                        "percent_change": percent_change,
                        "trajectory": direction,
                    }
                )
        return trends

    def _detect_care_gaps(self, diagnoses: set, biomarker_history: dict, medications: set) -> list[str]:
        gaps = []
        diag_lower = {d.lower() for d in diagnoses}
        med_lower = {m.lower() for m in medications}

        # Check Diabetes monitoring
        if any("diabetes" in d for d in diag_lower):
            if "hba1c" not in biomarker_history:
                gaps.append("Missing annual HbA1c glycemic monitoring for known Diabetic profile.")
            if "serum_creatinine" not in biomarker_history:
                gaps.append("Diabetic nephropathy screening gap: Serum Creatinine / eGFR not recorded.")

        # Check Hypertension monitoring
        if (
            any("hypertension" in d for d in diag_lower)
            and "serum_creatinine" not in biomarker_history
            and "blood_urea" not in biomarker_history
        ):
            gaps.append("Renal clearance gap: Hypertensive patient requires baseline BUN/Creatinine.")

        # Check Statin / Lipid therapy monitoring
        if any("statin" in m or "atorvastatin" in m for m in med_lower):
            if "alt_sgpt" not in biomarker_history:
                gaps.append("Hepatic monitoring gap: Baseline ALT/SGPT recommended for ongoing statin therapy.")
            if "total_cholesterol" not in biomarker_history:
                gaps.append("Lipid panel follow-up missing to verify Statin efficacy target.")

        if not gaps:
            gaps.append("No critical monitoring gaps identified in the uploaded history.")

        return gaps

    def _generate_executive_narrative(
        self,
        timeline: list[dict[str, Any]],
        diagnoses: list[str],
        medications: list[str],
        allergies: list[str],
        trends: list[dict[str, Any]],
        gaps: list[str],
    ) -> str:
        prompt = (
            f"You are a clinical history synthesis agent.\n"
            f"Patient Records: {len(timeline)} documents uploaded.\n"
            f"Documented Diagnoses: {', '.join(diagnoses) if diagnoses else 'None noted'}\n"
            f"Medications: {', '.join(medications) if medications else 'None noted'}\n"
            f"Allergies: {', '.join(allergies) if allergies else 'None noted'}\n"
            f"Biomarker Trends: {trends!s}\n"
            f"Care Gaps: {gaps!s}\n\n"
            f"Provide an executive 4-sentence clinical narrative: "
            f"1. Summarize primary comorbidities and chronological trajectory. "
            f"2. Note any worsening or improving biomarker values. "
            f"3. Highlight key care gaps requiring immediate physician attention."
        )

        # 1. Try Gemini first if available
        if self.gemini_client:
            for model_name in ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest"]:
                try:
                    resp = self.gemini_client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    if resp.text:
                        return resp.text.strip()
                except Exception:
                    continue

        # 2. Try Groq as fallback
        if self.groq_client:
            try:
                completion = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are MedAgent History Synthesizer. Provide succinct clinical narrative summaries.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=250,
                )
                return completion.choices[0].message.content.strip()
            except Exception:
                pass

        # Fallback offline deterministic clinical narrative
        diag_text = ", ".join(diagnoses) if diagnoses else "no chronic conditions"
        med_text = ", ".join(medications[:4]) if medications else "no documented active medications"
        allergy_text = ", ".join(allergies) if allergies else "No known drug allergies"

        summary = (
            f"Patient history synthesized from {len(timeline)} document(s). "
            f"Active clinical profile includes {diag_text}, with current therapy consisting of {med_text}. "
            f"Allergy status: {allergy_text}. "
        )
        if trends:
            t_str = "; ".join(
                [f"{t['marker']} is {t['trajectory'].lower()} (change: {t['change']:+})" for t in trends[:2]]
            )
            summary += f"Biomarker tracking indicates {t_str}. "
        if gaps and "No critical" not in gaps[0]:
            summary += f"Attention needed for care gap: {gaps[0]}"
        return summary
