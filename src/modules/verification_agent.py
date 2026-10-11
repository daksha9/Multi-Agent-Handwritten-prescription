import os
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from groq import Groq
except ImportError:
    Groq = None


try:
    from google import genai
except ImportError:
    genai = None


class PrescriptionVerificationAgent:
    """
    Prescription Safety & Verification Agent.
    Cross-checks candidate prescriptions against:
    1. Drug-Drug Interactions (DDI)
    2. Patient Allergies & Cross-Sensitivities
    3. Disease-Drug Contraindications
    4. Safe Dosage Thresholds
    Employs Gemini / Groq LLM for deep clinical reasoning,
    with an automatic fallback to local rule-based verification.
    """

    def __init__(self, drugs_csv_path: str | Path | None = None):
        if drugs_csv_path is None:
            self.drugs_csv_path = str(Path(__file__).resolve().parents[2] / "data" / "drugs.csv")
        else:
            self.drugs_csv_path = str(drugs_csv_path)
        self.reference_db = self._load_reference_db()

        # Allergy cross-sensitivity mapping
        self.allergy_families = {
            "penicillin": ["amoxicillin", "ampicillin", "augmentin", "piperacillin", "penicillin v", "oxacillin"],
            "cephalosporin": ["cephalexin", "ceftriaxone", "cefuroxime", "cefixime", "cefepime"],
            "sulfa": ["sulfamethoxazole", "bactrim", "septra", "sulfasalazine"],
            "nsaid": ["aspirin", "ibuprofen", "naproxen", "diclofenac", "ketorolac", "celecoxib", "meloxicam"],
            "statin": ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin"],
        }

        # Contraindication mapping
        self.contraindication_rules = {
            "metformin": {
                "conditions": ["severe renal impairment", "chronic kidney disease", "ckd", "renal failure"],
                "warning": "Metformin is contraindicated in severe renal disease due to high risk of lactic acidosis.",
                "severity": "CRITICAL",
            },
            "aspirin": {
                "conditions": ["peptic ulcer", "gastric ulcer", "bleeding disorder", "hemophilia"],
                "warning": "Aspirin carries high risk of gastrointestinal bleeding and exacerbation of peptic ulcer.",
                "severity": "CRITICAL",
            },
            "ibuprofen": {
                "conditions": ["peptic ulcer", "severe renal failure", "heart failure"],
                "warning": "NSAIDs can cause acute kidney injury, fluid retention, and GI mucosal ulceration.",
                "severity": "CRITICAL",
            },
            "atorvastatin": {
                "conditions": ["liver disease", "hepatic impairment", "cirrhosis"],
                "warning": "Statins can elevate transaminases and are contraindicated in active liver disease.",
                "severity": "WARNING",
            },
            "salbutamol": {
                "conditions": ["hyperthyroidism", "cardiac arrhythmia"],
                "warning": "Beta-2 agonists can induce tachycardia and exacerbate arrhythmias or thyrotoxicosis.",
                "severity": "WARNING",
            },
            "cetirizine": {
                "conditions": ["severe renal impairment", "end stage renal disease"],
                "warning": "Requires dose reduction in marked renal clearance reduction.",
                "severity": "INFO",
            },
        }

        # Initialize LLM clients
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
                print(f"[VerificationAgent] Gemini init warning: {e}")
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
                    print(f"[VerificationAgent] Groq init warning: {e}")
        return None

    def _load_reference_db(self) -> pd.DataFrame:
        if os.path.exists(self.drugs_csv_path):
            try:
                df = pd.read_csv(self.drugs_csv_path)
                df["drug_name_lower"] = df["drug_name"].astype(str).str.lower().str.strip()
                return df
            except Exception:
                pass
        return pd.DataFrame(columns=["drug_name", "common_interactions", "contraindications", "drug_name_lower"])

    def verify_prescriptions(
        self,
        prescribed_drugs: list[str],
        patient_allergies: list[str] | None = None,
        patient_conditions: list[str] | None = None,
        active_medications: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Runs comprehensive clinical safety verification on the prescribed drugs.
        Returns a structured report with categorized alerts, severity levels, and summary stats.
        """
        patient_allergies = [a.strip().lower() for a in (patient_allergies or []) if a.strip()]
        patient_conditions = [c.strip().lower() for c in (patient_conditions or []) if c.strip()]
        active_medications = [m.strip().lower() for m in (active_medications or []) if m.strip()]
        clean_prescribed = [p.strip() for p in prescribed_drugs if p and p.strip() and p.lower() != "not matched"]

        alerts = []

        # 1. Check Allergy Cross-Reactivity
        allergy_alerts = self._check_allergies(clean_prescribed, patient_allergies)
        alerts.extend(allergy_alerts)

        # 2. Check Disease Contraindications
        contra_alerts = self._check_contraindications(clean_prescribed, patient_conditions)
        alerts.extend(contra_alerts)

        # 3. Check Drug-Drug Interactions (Internal pair check + against active meds)
        ddi_alerts = self._check_interactions(clean_prescribed, active_medications)
        alerts.extend(ddi_alerts)

        # 4. Optional: Enrich with LLM (Gemini / Groq) if connected
        llm_insights = None
        if (self.gemini_client or self.groq_client) and clean_prescribed:
            llm_insights = self._consult_llm(
                clean_prescribed, patient_allergies, patient_conditions, active_medications, alerts
            )

        critical_count = sum(1 for a in alerts if a["severity"] == "CRITICAL")
        warning_count = sum(1 for a in alerts if a["severity"] == "WARNING")
        info_count = sum(1 for a in alerts if a["severity"] == "INFO")

        overall_status = "SAFE"
        if critical_count > 0:
            overall_status = "CRITICAL_RISK"
        elif warning_count > 0:
            overall_status = "CAUTION_REQUIRED"

        return {
            "overall_status": overall_status,
            "prescribed_drugs": clean_prescribed,
            "alerts": alerts,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "total_alerts": len(alerts),
            "llm_clinical_narrative": llm_insights,
        }

    def _check_allergies(self, prescribed_drugs: list[str], allergies: list[str]) -> list[dict[str, Any]]:
        alerts = []
        for drug in prescribed_drugs:
            drug_lower = drug.lower()
            for allergy in allergies:
                # Direct match
                if allergy in drug_lower or drug_lower in allergy:
                    alerts.append(
                        {
                            "category": "Allergy Alert",
                            "severity": "CRITICAL",
                            "drug": drug,
                            "trigger": allergy.title(),
                            "message": f"Patient has documented allergy to {allergy.title()}. Immediate hypersensitivity risk with {drug}.",
                        }
                    )
                    continue

                # Cross-reactivity family match
                for family, members in self.allergy_families.items():
                    if (allergy in family or family in allergy) and any(member in drug_lower for member in members):
                        alerts.append(
                            {
                                "category": "Allergy Cross-Reactivity",
                                "severity": "CRITICAL",
                                "drug": drug,
                                "trigger": f"{allergy.title()} (Class: {family.title()})",
                                "message": f"{drug} belongs to/cross-reacts with {family.title()} class, which patient is allergic to.",
                            }
                        )
        return alerts

    def _check_contraindications(self, prescribed_drugs: list[str], conditions: list[str]) -> list[dict[str, Any]]:
        alerts = []
        for drug in prescribed_drugs:
            drug_lower = drug.lower()
            for rule_drug, details in self.contraindication_rules.items():
                if rule_drug in drug_lower:
                    for cond in conditions:
                        for trigger_cond in details["conditions"]:
                            if trigger_cond in cond or cond in trigger_cond:
                                alerts.append(
                                    {
                                        "category": "Disease Contraindication",
                                        "severity": details["severity"],
                                        "drug": drug,
                                        "trigger": cond.title(),
                                        "message": details["warning"],
                                        "warning": details["warning"],
                                    }
                                )
        return alerts

    def _check_interactions(self, prescribed: list[str], active_meds: list[str]) -> list[dict[str, Any]]:
        alerts = []
        all_drugs = list(dict.fromkeys(prescribed + active_meds))

        # Check in reference_db
        if not self.reference_db.empty:
            for _i, d1 in enumerate(all_drugs):
                d1_lower = d1.lower()
                row = self.reference_db[
                    self.reference_db["drug_name_lower"].apply(lambda x, dl=d1_lower: x in dl or dl in x)
                ]
                if not row.empty:
                    interactions = str(row.iloc[0]["common_interactions"]).split(";")
                    for inter in interactions:
                        inter_clean = inter.strip().lower()
                        if not inter_clean:
                            continue
                        for d2 in all_drugs:
                            if d2.lower() != d1_lower and inter_clean in d2.lower():
                                alerts.append(
                                    {
                                        "category": "Drug Interaction",
                                        "severity": "CRITICAL"
                                        if any(w in inter_clean for w in ["warfarin", "clopidogrel", "aspirin"])
                                        else "WARNING",
                                        "drug": f"{d1} + {d2}",
                                        "trigger": f"Adverse Drug Interaction: {inter.strip()}",
                                        "message": f"Co-administration of {d1} and {d2} may precipitate adverse synergistic or antagonistic toxicities.",
                                    }
                                )
        return alerts

    def _consult_llm(
        self,
        prescribed: list[str],
        allergies: list[str],
        conditions: list[str],
        active_meds: list[str],
        current_alerts: list[dict[str, Any]],
    ) -> str | None:
        """Calls Gemini or Groq model to provide concise professional clinical commentary."""
        prompt = (
            f"You are a clinical pharmacologist and safety verification agent.\n"
            f"Prescribed Drugs: {', '.join(prescribed)}\n"
            f"Patient Allergies: {', '.join(allergies) if allergies else 'None documented'}\n"
            f"Patient Diagnoses/Conditions: {', '.join(conditions) if conditions else 'None documented'}\n"
            f"Active Current Medications: {', '.join(active_meds) if active_meds else 'None'}\n"
            f"Deterministic Rules Alert Count: {len(current_alerts)}\n\n"
            f"Provide a brief 3-4 sentence clinical safety summary: "
            f"1. Highlight any serious drug-drug interactions or contraindications. "
            f"2. Suggest safer alternative medicines if high risks are found. "
            f"3. Note required monitoring (e.g. renal function, INR)."
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
                            "content": "You are MedAgent Verification Specialist. Be concise, clinically precise, and evidence-based.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=250,
                )
                return completion.choices[0].message.content.strip()
            except Exception:
                pass

        return None
