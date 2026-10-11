import math
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


class DiseasePredictionAgent:
    """
    Disease Prediction & Biomarker Risk Stratification Agent.
    Evaluates extracted laboratory biomarkers, vitals, and patient clinical history
    to calculate disease likelihood percentages, generate differential diagnoses,
    and recommend confirmatory clinical workups.
    """

    # Reference clinical ranges (Adult standard)
    REFERENCE_RANGES = {
        "fasting_blood_glucose": {"min": 70.0, "max": 99.0, "unit": "mg/dL", "name": "Fasting Blood Glucose"},
        "postprandial_blood_glucose": {
            "min": 90.0,
            "max": 140.0,
            "unit": "mg/dL",
            "name": "Postprandial Blood Glucose",
        },
        "hba1c": {"min": 4.0, "max": 5.6, "unit": "%", "name": "HbA1c (Glycated Hemoglobin)"},
        "systolic_bp": {"min": 90, "max": 120, "unit": "mmHg", "name": "Systolic Blood Pressure"},
        "diastolic_bp": {"min": 60, "max": 80, "unit": "mmHg", "name": "Diastolic Blood Pressure"},
        "serum_creatinine": {"min": 0.6, "max": 1.2, "unit": "mg/dL", "name": "Serum Creatinine"},
        "blood_urea": {"min": 7.0, "max": 20.0, "unit": "mg/dL", "name": "Blood Urea Nitrogen (BUN)"},
        "total_cholesterol": {"min": 125.0, "max": 200.0, "unit": "mg/dL", "name": "Total Cholesterol"},
        "ldl_cholesterol": {"min": 50.0, "max": 100.0, "unit": "mg/dL", "name": "LDL Cholesterol"},
        "hdl_cholesterol": {"min": 40.0, "max": 60.0, "unit": "mg/dL", "name": "HDL Cholesterol"},
        "triglycerides": {"min": 50.0, "max": 150.0, "unit": "mg/dL", "name": "Serum Triglycerides"},
        "hemoglobin": {"min": 12.0, "max": 16.5, "unit": "g/dL", "name": "Hemoglobin"},
        "alt_sgpt": {"min": 7.0, "max": 45.0, "unit": "U/L", "name": "Alanine Aminotransferase (ALT/SGPT)"},
        "ast_sgot": {"min": 8.0, "max": 40.0, "unit": "U/L", "name": "Aspartate Aminotransferase (AST/SGOT)"},
        "platelets": {"min": 1.5, "max": 4.5, "unit": "Lakhs/mcL", "name": "Platelet Count"},
        "bmi": {"min": 18.5, "max": 24.9, "unit": "kg/m²", "name": "Body Mass Index (BMI)"},
    }

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
                print(f"[PredictionAgent] Gemini init warning: {e}")
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
                    print(f"[PredictionAgent] Groq init warning: {e}")
        return None

    def analyze_biomarkers(self, biomarkers: dict[str, float], vitals: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Flags each biomarker as Normal, Elevated, Low, or Critical based on clinical reference thresholds.
        """
        combined = dict(biomarkers)
        if vitals.get("systolic_bp") is not None:
            combined["systolic_bp"] = vitals["systolic_bp"]
        if vitals.get("diastolic_bp") is not None:
            combined["diastolic_bp"] = vitals["diastolic_bp"]
        if vitals.get("bmi") is not None:
            combined["bmi"] = vitals["bmi"]

        table = []
        for key, val in combined.items():
            if val is None or key not in self.REFERENCE_RANGES:
                continue

            ref = self.REFERENCE_RANGES[key]
            min_val = ref["min"]
            max_val = ref["max"]

            # Compute clinical status
            if val < min_val:
                status = "LOW"
                severity = "WARNING" if (val < min_val * 0.8) else "INFO"
            elif val > max_val * 1.5:
                status = "CRITICAL HIGH"
                severity = "CRITICAL"
            elif val > max_val:
                status = "ELEVATED"
                severity = "WARNING"
            else:
                status = "NORMAL"
                severity = "SAFE"

            table.append(
                {
                    "key": key,
                    "biomarker": ref["name"],
                    "value": val,
                    "reference_range": f"{min_val} - {max_val} {ref['unit']}",
                    "unit": ref["unit"],
                    "status": status,
                    "severity": severity,
                }
            )

        return table

    def predict_diseases(
        self,
        biomarkers: dict[str, float],
        vitals: dict[str, Any],
        patient_info: dict[str, Any],
        diagnoses_history: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Computes disease likelihood percentages across 5 major non-communicable conditions.
        Returns differential diagnosis, risk tiers, and follow-up test recommendations.
        """
        diagnoses_history = [d.lower() for d in (diagnoses_history or [])]
        fbg = biomarkers.get("fasting_blood_glucose")
        ppbg = biomarkers.get("postprandial_blood_glucose")
        hba1c = biomarkers.get("hba1c")
        sbp = vitals.get("systolic_bp")
        dbp = vitals.get("diastolic_bp")
        cr = biomarkers.get("serum_creatinine")
        bun = biomarkers.get("blood_urea")
        tc = biomarkers.get("total_cholesterol")
        ldl = biomarkers.get("ldl_cholesterol")
        hdl = biomarkers.get("hdl_cholesterol")
        tg = biomarkers.get("triglycerides")
        alt = biomarkers.get("alt_sgpt")
        ast = biomarkers.get("ast_sgot")
        hb = biomarkers.get("hemoglobin")
        age = patient_info.get("age") or 50

        predictions = []

        # 1. Type 2 Diabetes Mellitus
        t2dm_score = self._compute_diabetes_risk(fbg, ppbg, hba1c, diagnoses_history)
        if t2dm_score["evaluated"]:
            predictions.append(t2dm_score)

        # 2. Hypertension
        htn_score = self._compute_hypertension_risk(sbp, dbp, diagnoses_history)
        if htn_score["evaluated"]:
            predictions.append(htn_score)

        # 3. Cardiovascular Disease (Atherosclerotic CVD)
        cvd_score = self._compute_cvd_risk(tc, ldl, hdl, tg, sbp, age, diagnoses_history)
        if cvd_score["evaluated"]:
            predictions.append(cvd_score)

        # 4. Chronic Kidney Disease (CKD)
        ckd_score = self._compute_ckd_risk(cr, bun, fbg, sbp, diagnoses_history)
        if ckd_score["evaluated"]:
            predictions.append(ckd_score)

        # 5. Hepatic Dysfunction / Fatty Liver
        liver_score = self._compute_liver_risk(alt, ast, tg, diagnoses_history)
        if liver_score["evaluated"]:
            predictions.append(liver_score)

        # 6. Anemia
        if hb is not None:
            anemia_score = self._compute_anemia_risk(hb, patient_info.get("gender"))
            predictions.append(anemia_score)

        # Sort predictions by percentage descending
        predictions.sort(key=lambda p: p["probability_percent"], reverse=True)

        # Generate biomarker table
        biomarker_table = self.analyze_biomarkers(biomarkers, vitals)

        # Collect recommended confirmatory tests
        recommended_tests = []
        for p in predictions:
            if p["risk_tier"] in ["High Risk", "Moderate Risk"]:
                recommended_tests.extend(p["follow_up_tests"])
        recommended_tests = list(dict.fromkeys(recommended_tests))

        # Groq Clinical Narrative
        clinical_narrative = None
        if self.groq_client and predictions:
            clinical_narrative = self._generate_clinical_differential(predictions, biomarker_table, patient_info)

        return {
            "predictions": predictions,
            "biomarker_table": biomarker_table,
            "recommended_tests": recommended_tests,
            "clinical_narrative": clinical_narrative,
            "high_risk_count": sum(1 for p in predictions if p["risk_tier"] == "High Risk"),
        }

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(min(x, 15), -15)))

    def _compute_diabetes_risk(self, fbg, ppbg, hba1c, diagnoses) -> dict[str, Any]:
        evaluated = any(v is not None for v in [fbg, ppbg, hba1c]) or any("diabetes" in d for d in diagnoses)
        if not evaluated:
            return {"evaluated": False}

        points = 0.0
        reasons = []

        if "type 2 diabetes" in diagnoses or "diabetes mellitus" in diagnoses:
            points += 5.0
            reasons.append("Documented medical history of Diabetes Mellitus")

        if hba1c is not None:
            if hba1c >= 6.5:
                points += 6.0
                reasons.append(f"HbA1c of {hba1c}% exceeds diagnostic threshold (≥6.5%)")
            elif hba1c >= 5.7:
                points += 3.0
                reasons.append(f"HbA1c of {hba1c}% in prediabetic range (5.7 - 6.4%)")
            else:
                points -= 2.0

        if fbg is not None:
            if fbg >= 126.0:
                points += 5.0
                reasons.append(f"Fasting Glucose {fbg} mg/dL is in diabetic range (≥126 mg/dL)")
            elif fbg >= 100.0:
                points += 2.5
                reasons.append(f"Fasting Glucose {fbg} mg/dL is impaired (100-125 mg/dL)")
            else:
                points -= 1.5

        if ppbg is not None and ppbg >= 200.0:
            points += 4.5
            reasons.append(f"Postprandial Glucose {ppbg} mg/dL exceeds 200 mg/dL")

        # Map to probability 0-100%
        prob = round(self._sigmoid(points - 1.5) * 100, 1)
        risk_tier = "High Risk" if prob >= 70.0 else ("Moderate Risk" if prob >= 40.0 else "Low Risk")

        return {
            "disease": "Type 2 Diabetes Mellitus",
            "probability_percent": prob,
            "risk_tier": risk_tier,
            "evaluated": True,
            "contributing_factors": reasons or ["Biomarkers within normal glycemic limits"],
            "follow_up_tests": [
                "Oral Glucose Tolerance Test (OGTT)",
                "Fasting Lipid Profile",
                "Urine Albumin-to-Creatinine Ratio (uACR)",
            ],
        }

    def _compute_hypertension_risk(self, sbp, dbp, diagnoses) -> dict[str, Any]:
        evaluated = (sbp is not None or dbp is not None) or any("hypertension" in d for d in diagnoses)
        if not evaluated:
            return {"evaluated": False}

        points = 0.0
        reasons = []

        if any("hypertension" in d for d in diagnoses):
            points += 4.5
            reasons.append("Documented medical history of Hypertension")

        if sbp is not None:
            if sbp >= 160:
                points += 6.0
                reasons.append(f"Systolic BP {sbp} mmHg (Stage 2 Severe Hypertension)")
            elif sbp >= 140:
                points += 4.5
                reasons.append(f"Systolic BP {sbp} mmHg (Stage 2 Hypertension)")
            elif sbp >= 130:
                points += 2.5
                reasons.append(f"Systolic BP {sbp} mmHg (Stage 1 Hypertension)")
            elif sbp >= 120:
                points += 1.0
                reasons.append(f"Systolic BP {sbp} mmHg (Elevated BP)")
            else:
                points -= 1.5

        if dbp is not None:
            if dbp >= 100:
                points += 5.5
                reasons.append(f"Diastolic BP {dbp} mmHg (Stage 2 Severe Hypertension)")
            elif dbp >= 90:
                points += 4.0
                reasons.append(f"Diastolic BP {dbp} mmHg (Stage 2 Hypertension)")
            elif dbp >= 80:
                points += 2.0
                reasons.append(f"Diastolic BP {dbp} mmHg (Stage 1 Hypertension)")

        prob = round(self._sigmoid(points - 1.0) * 100, 1)
        risk_tier = "High Risk" if prob >= 70.0 else ("Moderate Risk" if prob >= 40.0 else "Low Risk")

        return {
            "disease": "Essential Hypertension",
            "probability_percent": prob,
            "risk_tier": risk_tier,
            "evaluated": True,
            "contributing_factors": reasons or ["Blood pressure within optimal range (<120/80 mmHg)"],
            "follow_up_tests": [
                "24-Hour Ambulatory Blood Pressure Monitoring (ABPM)",
                "Electrocardiogram (ECG)",
                "Serum Electrolytes & Renal Function Test",
            ],
        }

    def _compute_cvd_risk(self, tc, ldl, hdl, tg, sbp, age, diagnoses) -> dict[str, Any]:
        evaluated = any(v is not None for v in [tc, ldl, hdl, tg])
        if not evaluated:
            return {"evaluated": False}

        points = 0.0
        reasons = []

        if age > 55:
            points += 1.5
            reasons.append(f"Age {age} years is an independent cardiovascular risk factor")

        if tc is not None and tc >= 200:
            points += 2.0
            reasons.append(f"Elevated Total Cholesterol ({tc} mg/dL)")

        if ldl is not None:
            if ldl >= 160:
                points += 4.5
                reasons.append(f"Very high atherogenic LDL Cholesterol ({ldl} mg/dL)")
            elif ldl >= 130:
                points += 2.5
                reasons.append(f"Borderline high LDL Cholesterol ({ldl} mg/dL)")

        if hdl is not None and hdl < 40:
            points += 2.5
            reasons.append(f"Low protective HDL Cholesterol ({hdl} mg/dL)")

        if tg is not None and tg >= 150:
            points += 1.5
            reasons.append(f"Hypertriglyceridemia ({tg} mg/dL)")

        if sbp is not None and sbp >= 140:
            points += 2.0
            reasons.append("Uncontrolled systolic BP compounding vascular shear stress")

        prob = round(self._sigmoid(points - 2.0) * 100, 1)
        risk_tier = "High Risk" if prob >= 65.0 else ("Moderate Risk" if prob >= 35.0 else "Low Risk")

        return {
            "disease": "Atherosclerotic Cardiovascular Disease (ASCVD)",
            "probability_percent": prob,
            "risk_tier": risk_tier,
            "evaluated": True,
            "contributing_factors": reasons or ["Lipid profile within low atherogenic risk boundaries"],
            "follow_up_tests": [
                "12-Lead Electrocardiogram (ECG)",
                "Coronary Artery Calcium (CAC) Scan",
                "High-Sensitivity C-Reactive Protein (hs-CRP)",
            ],
        }

    def _compute_ckd_risk(self, cr, bun, fbg, sbp, diagnoses) -> dict[str, Any]:
        evaluated = cr is not None or bun is not None or any("kidney" in d for d in diagnoses)
        if not evaluated:
            return {"evaluated": False}

        points = 0.0
        reasons = []

        if cr is not None:
            if cr >= 2.0:
                points += 7.0
                reasons.append(f"Critically elevated Serum Creatinine ({cr} mg/dL)")
            elif cr >= 1.3:
                points += 4.5
                reasons.append(f"Elevated Serum Creatinine ({cr} mg/dL) indicating nephron loss")
            else:
                points -= 2.0

        if bun is not None and bun > 20.0:
            points += 2.0
            reasons.append(f"Elevated Blood Urea Nitrogen ({bun} mg/dL)")

        if fbg is not None and fbg > 140:
            points += 1.5
            reasons.append("Diabetic milieu increases microvascular nephropathy susceptibility")

        prob = round(self._sigmoid(points - 2.0) * 100, 1)
        risk_tier = "High Risk" if prob >= 65.0 else ("Moderate Risk" if prob >= 35.0 else "Low Risk")

        return {
            "disease": "Chronic Kidney Disease (CKD) / Renal Impairment",
            "probability_percent": prob,
            "risk_tier": risk_tier,
            "evaluated": True,
            "contributing_factors": reasons or ["Renal function markers within normal clearance limits"],
            "follow_up_tests": [
                "Estimated Glomerular Filtration Rate (eGFR)",
                "Urine Routine & Albumin-to-Creatinine Ratio (uACR)",
                "Renal Ultrasonography",
            ],
        }

    def _compute_liver_risk(self, alt, ast, tg, diagnoses) -> dict[str, Any]:
        evaluated = alt is not None or ast is not None or any("liver" in d for d in diagnoses)
        if not evaluated:
            return {"evaluated": False}

        points = 0.0
        reasons = []

        if alt is not None:
            if alt >= 100:
                points += 5.5
                reasons.append(f"Marked ALT/SGPT transaminase leak ({alt} U/L)")
            elif alt > 45:
                points += 3.0
                reasons.append(f"Mildly elevated ALT/SGPT ({alt} U/L)")

        if ast is not None and ast > 40:
            points += 2.5
            reasons.append(f"Elevated AST/SGOT ({ast} U/L)")

        if tg is not None and tg > 200:
            points += 1.5
            reasons.append("Elevated triglycerides supporting hepatic steatosis risk")

        prob = round(self._sigmoid(points - 1.5) * 100, 1)
        risk_tier = "High Risk" if prob >= 65.0 else ("Moderate Risk" if prob >= 35.0 else "Low Risk")

        return {
            "disease": "Hepatic Dysfunction / Non-Alcoholic Fatty Liver (NAFLD)",
            "probability_percent": prob,
            "risk_tier": risk_tier,
            "evaluated": True,
            "contributing_factors": reasons or ["Hepatic transaminases within normal ranges"],
            "follow_up_tests": [
                "Complete Liver Function Test (Total/Direct Bilirubin, ALP, Albumin)",
                "Ultrasound Whole Abdomen",
                "Viral Hepatitis Serology (HBsAg, Anti-HCV)",
            ],
        }

    def _compute_anemia_risk(self, hb: float, gender: str | None) -> dict[str, Any]:
        is_female = gender and gender.lower().startswith("f")
        cutoff = 12.0 if is_female else 13.0

        points = 0.0
        reasons = []

        if hb < 8.0:
            points += 7.0
            reasons.append(f"Severe anemia with critical Hemoglobin of {hb} g/dL")
        elif hb < 10.0:
            points += 5.0
            reasons.append(f"Moderate anemia with Hemoglobin of {hb} g/dL")
        elif hb < cutoff:
            points += 3.0
            reasons.append(f"Mild anemia with Hemoglobin of {hb} g/dL (Cutoff: {cutoff} g/dL)")
        else:
            points -= 3.0

        prob = round(self._sigmoid(points - 1.0) * 100, 1)
        risk_tier = "High Risk" if prob >= 70.0 else ("Moderate Risk" if prob >= 40.0 else "Low Risk")

        return {
            "disease": "Anemia / Hemoglobin Deficiency",
            "probability_percent": prob,
            "risk_tier": risk_tier,
            "evaluated": True,
            "contributing_factors": reasons or ["Hemoglobin levels are physiologically normal"],
            "follow_up_tests": [
                "Serum Ferritin & Iron Studies (TIBC, Transferrin Saturation)",
                "Peripheral Blood Smear (PBS)",
                "Vitamin B12 & Folate Levels",
            ],
        }

    def _generate_clinical_differential(
        self,
        predictions: list[dict[str, Any]],
        biomarker_table: list[dict[str, Any]],
        patient_info: dict[str, Any],
    ) -> str | None:
        summary_preds = "\n".join(
            [f"- {p['disease']}: {p['probability_percent']}% ({p['risk_tier']})" for p in predictions]
        )
        abnormal_biomarkers = [
            f"{b['biomarker']}: {b['value']} {b['unit']} ({b['status']})"
            for b in biomarker_table
            if b["status"] != "NORMAL"
        ]

        prompt = (
            f"You are an expert internal medicine diagnostic AI agent.\n"
            f"Patient Demographics: Age {patient_info.get('age', 'N/A')}, Gender {patient_info.get('gender', 'N/A')}\n"
            f"Abnormal Biomarkers Detected:\n{chr(10).join(abnormal_biomarkers) if abnormal_biomarkers else 'None'}\n\n"
            f"Calculated Disease Likelihood Scores:\n{summary_preds}\n\n"
            f"Synthesize a concise 4-sentence clinical differential diagnosis summary: "
            f"1. Highlight the primary metabolic/cardiovascular/renal risks. "
            f"2. Explain the pathophysiological link between the abnormal biomarkers. "
            f"3. Note immediate clinical action priorities for the physician."
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
                            "content": "You are MedAgent Diagnostic Specialist. Write concise, clinical-grade medical insights.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=300,
                )
                return completion.choices[0].message.content.strip()
            except Exception:
                pass

        return None
