import json
import os
import re
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
            "penicillin": [
                "amoxicillin",
                "ampicillin",
                "augmentin",
                "piperacillin",
                "penicillin v",
                "oxacillin",
                "cefixime",
                "cephalexin",
                "ceftriaxone",
                "cefuroxime",
            ],
            "cephalosporin": ["cephalexin", "ceftriaxone", "cefuroxime", "cefixime", "cefepime", "amoxicillin"],
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
            "naproxen": {
                "conditions": ["chronic kidney disease", "ckd", "renal impairment", "peptic ulcer"],
                "warning": "NSAIDs impair renal prostaglandins and aggravate preexisting renal impairment or ulceration.",
                "severity": "CRITICAL",
            },
            "doxycycline": {
                "conditions": ["pregnancy", "pregnant", "lactation"],
                "warning": "Tetracyclines are contraindicated in pregnancy due to risk of fetal tooth discoloration and skeletal impairment.",
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

        # High-risk drug-drug interaction pairs
        self.high_risk_interactions = [
            ("aspirin", "clopidogrel", "Dual antiplatelet therapy significantly increases risk of major gastrointestinal and systemic bleeding.", "WARNING"),
            ("clopidogrel", "aspirin", "Dual antiplatelet therapy significantly increases risk of major gastrointestinal and systemic bleeding.", "WARNING"),
            ("aspirin", "warfarin", "Combined anticoagulant and antiplatelet therapy markedly increases severe hemorrhagic risk.", "CRITICAL"),
            ("warfarin", "aspirin", "Combined anticoagulant and antiplatelet therapy markedly increases severe hemorrhagic risk.", "CRITICAL"),
            ("simvastatin", "amiodarone", "Amiodarone inhibits CYP3A4, markedly increasing simvastatin concentration and rhabdomyolysis risk.", "CRITICAL"),
            ("ciprofloxacin", "theophylline", "Ciprofloxacin inhibits theophylline clearance, risking life-threatening theophylline neurotoxicity.", "CRITICAL"),
            ("lisinopril", "spironolactone", "Combined ACE inhibitor and potassium-sparing diuretic creates severe hyperkalemia risk.", "CRITICAL"),
            ("fluconazole", "warfarin", "Fluconazole strongly inhibits CYP2C9, significantly increasing INR and life-threatening bleeding.", "CRITICAL"),
            ("sertraline", "tramadol", "Concurrent serotonergic agents markedly increase the risk of Serotonin Syndrome.", "CRITICAL"),
            ("methotrexate", "ibuprofen", "NSAIDs reduce renal tubular clearance of methotrexate, causing severe hematologic toxicity.", "CRITICAL"),
            ("clarithromycin", "simvastatin", "Macrolide inhibition of CYP3A4 elevates simvastatin levels, precipitating severe myopathy.", "CRITICAL"),
        ]

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
                df["drug_name_lower"] = df["drug_name"].astype(str).apply(lambda x: x.lower().strip())
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
        consult_llm: bool = True,
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
        if consult_llm and (self.gemini_client or self.groq_client) and clean_prescribed:
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

        # Check high-risk interactions matrix
        seen_pairs = set()
        for d1 in all_drugs:
            d1_l = d1.lower()
            for d2 in all_drugs:
                if d1 == d2:
                    continue
                d2_l = d2.lower()
                pair_key = tuple(sorted([d1_l, d2_l]))
                if pair_key in seen_pairs:
                    continue
                for drug_a, drug_b, warning_msg, severity in self.high_risk_interactions:
                    if (drug_a in d1_l and drug_b in d2_l) or (drug_b in d1_l and drug_a in d2_l):
                        seen_pairs.add(pair_key)
                        alerts.append(
                            {
                                "category": "Drug Interaction",
                                "severity": severity,
                                "drug": f"{d1} + {d2}",
                                "trigger": f"High-Risk Drug Interaction: {drug_a.title()} + {drug_b.title()}",
                                "message": warning_msg,
                            }
                        )

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
                                pair_key = tuple(sorted([d1_lower, d2.lower()]))
                                if pair_key not in seen_pairs:
                                    seen_pairs.add(pair_key)
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

    # ============================================================
    # STAGE 4 — Clinical Reasoning & Doctor Safety Agent
    # ============================================================

    # Severity mapping from internal rules to Stage 4 schema
    _SEVERITY_MAP = {"CRITICAL": "HIGH", "WARNING": "MEDIUM", "INFO": "LOW"}

    # Safe alternative suggestions for common conflict types
    _SAFE_ALTERNATIVES: dict[str, list[str]] = {
        "penicillin": ["Cefuroxime", "Azithromycin", "Doxycycline"],
        "amoxicillin": ["Cefuroxime", "Azithromycin", "Doxycycline"],
        "augmentin": ["Cefuroxime", "Azithromycin", "Doxycycline"],
        "clavulanic": ["Cefuroxime", "Azithromycin", "Doxycycline"],
        "cephalosporin": ["Azithromycin", "Levofloxacin"],
        "cefuroxime": ["Azithromycin", "Levofloxacin"],
        "cefpodoxime": ["Azithromycin", "Levofloxacin"],
        "nsaid": ["Paracetamol (Acetaminophen)", "Tramadol"],
        "ibuprofen": ["Paracetamol (Acetaminophen)", "Tramadol"],
        "diclofenac": ["Paracetamol (Acetaminophen)", "Tramadol"],
        "aceclofenac": ["Paracetamol (Acetaminophen)", "Tramadol"],
        "aspirin": ["Clopidogrel (if antiplatelet needed)", "Paracetamol for pain"],
        "warfarin": ["Apixaban", "Rivaroxaban"],
        "metformin": ["Glipizide", "Insulin", "Dapagliflozin"],
        "statin": ["Ezetimibe", "Bempedoic acid"],
        "atorvastatin": ["Ezetimibe", "Bempedoic acid"],
        "salbutamol": ["Ipratropium bromide"],
    }

    def run_stage4(self, stage2_output: dict, stage3_output: dict) -> dict:
        """
        Stage 4: Clinical Reasoning & Doctor Safety Agent.

        Cross-references the newly prescribed medications (Stage 2) against
        the patient's reported allergies, existing medications, and past
        side effects (Stage 3). Produces the final Doctor Dashboard JSON.
        """
        # ── Extract inputs ──
        prescribed = stage2_output.get("prescribed_medications", []) or []
        voice_history = stage3_output.get("patient_reported_history", {}) or {}

        patient_allergies = [a.strip().lower() for a in (voice_history.get("known_allergies") or []) if a.strip()]
        active_meds = [m.strip().lower() for m in (voice_history.get("current_active_medications") or []) if m.strip()]
        past_side_effects = voice_history.get("past_reported_side_effects") or []

        # Build lists of prescribed drug compounds
        prescribed_compounds = [
            (m.get("active_chemical_compound", "")).strip()
            for m in prescribed
        ]

        # ── 1. Prescription summary ──
        digitized = []
        for m in prescribed:
            raw = m.get("extracted_raw_text", "")
            compound = m.get("active_chemical_compound", "")
            dosage = m.get("dosage", "")
            freq = m.get("frequency", "")
            duration = m.get("duration", "")
            parts = [p for p in [dosage, freq, f"for {duration}" if duration and duration != "Not specified" else ""] if p]
            digitized.append({
                "medication": raw or compound,
                "chemical_compound": compound,
                "dosage_instructions": " ".join(parts) or "Not specified",
            })

        # ── 2. Patient voice summary ──
        reported_se_strings = []
        for se in past_side_effects:
            if isinstance(se, dict):
                med = se.get("medication_or_class", "")
                reaction = se.get("reported_reaction", "")
                if med or reaction:
                    reported_se_strings.append(f"{med}: {reaction}".strip(": "))

        patient_voice_summary = {
            "allergies": voice_history.get("known_allergies") or [],
            "reported_side_effects": reported_se_strings,
        }

        # ── 3. Clinical risk analysis ──
        risks: list[dict] = []

        # 3a. Drug-Allergy Cross-Reactivity
        for compound in prescribed_compounds:
            if not compound:
                continue
            compound_lower = compound.lower()
            for allergy in patient_allergies:
                if allergy in compound_lower or compound_lower in allergy:
                    risks.append({
                        "prescribed_drug": compound,
                        "conflict_type": "ALLERGY_HAZARD",
                        "risk_description": (
                            f"Patient has a documented allergy to '{allergy.title()}'. "
                            f"Prescribing '{compound}' carries a high risk of immediate hypersensitivity reaction."
                        ),
                        "severity": "HIGH",
                    })
                    continue
                for family, members in self.allergy_families.items():
                    if (allergy in family or family in allergy) and any(m in compound_lower for m in members):
                        risks.append({
                            "prescribed_drug": compound,
                            "conflict_type": "ALLERGY_HAZARD",
                            "risk_description": (
                                f"'{compound}' belongs to/cross-reacts with the {family.title()} class, "
                                f"which the patient is allergic to (allergy: '{allergy.title()}'). "
                                f"Risk of cross-sensitivity reaction."
                            ),
                            "severity": "HIGH",
                        })

        # 3b. Drug-Drug Interactions (prescribed vs active meds)
        all_drug_lower = [c.lower() for c in prescribed_compounds if c] + active_meds
        if not self.reference_db.empty:
            for comp in prescribed_compounds:
                if not comp:
                    continue
                row = self.reference_db[
                    self.reference_db["drug_name_lower"].apply(lambda x, c=comp: x in c.lower() or c.lower() in x)
                ]
                if not row.empty:
                    interactions = str(row.iloc[0]["common_interactions"]).split(";")
                    for inter in interactions:
                        inter_clean = inter.strip().lower()
                        if not inter_clean:
                            continue
                        for other in all_drug_lower:
                            if other and other != comp.lower() and inter_clean in other:
                                severity = "HIGH" if any(w in inter_clean for w in ["warfarin", "clopidogrel", "aspirin", "heparin"]) else "MEDIUM"
                                risks.append({
                                    "prescribed_drug": comp,
                                    "conflict_type": "DRUG_INTERACTION",
                                    "risk_description": (
                                        f"Co-administration of '{comp}' with '{other.title()}' may precipitate "
                                        f"a clinically significant drug-drug interaction ({inter.strip()})."
                                    ),
                                    "severity": severity,
                                })

        # 3c. Past Side Effect recurrence check
        for se in past_side_effects:
            if not isinstance(se, dict):
                continue
            past_med = (se.get("medication_or_class") or "").strip().lower()
            past_reaction = se.get("reported_reaction", "")
            if not past_med:
                continue
            for comp in prescribed_compounds:
                if not comp:
                    continue
                same_family = False
                for family, members in self.allergy_families.items():
                    if (past_med in family or family in past_med or past_med in members) and any(m in comp.lower() for m in members):
                        same_family = True
                        break
                if past_med in comp.lower() or comp.lower() in past_med or same_family:
                    risks.append({
                        "prescribed_drug": comp,
                        "conflict_type": "PAST_SIDE_EFFECT",
                        "risk_description": (
                            f"Patient previously reported '{past_reaction}' with '{past_med.title()}'. "
                            f"'{comp}' is related and may cause a similar adverse reaction."
                        ),
                        "severity": "MEDIUM",
                    })

        # 3d. Duplicate therapy check (same drug prescribed + already active)
        for comp in prescribed_compounds:
            if not comp:
                continue
            comp_lower = comp.lower()
            for active in active_meds:
                if not active:
                    continue
                if comp_lower in active or active in comp_lower:
                    risks.append({
                        "prescribed_drug": comp,
                        "conflict_type": "DRUG_INTERACTION",
                        "risk_description": (
                            f"Duplicate therapy: '{comp}' is already being taken by the patient "
                            f"('{active.title()}'). Risk of dose accumulation and toxicity."
                        ),
                        "severity": "HIGH",
                    })

        # ── 4. Overall safety alert level ──
        severities = [r["severity"] for r in risks]
        if "HIGH" in severities:
            alert_level = "RED"
        elif "MEDIUM" in severities:
            alert_level = "AMBER"
        else:
            alert_level = "GREEN"

        # ── 5. Doctor recommendation ──
        if alert_level == "RED":
            action = "REPLACE_MEDICATION"
        elif alert_level == "AMBER":
            action = "ADJUST_DOSAGE"
        else:
            action = "CONTINUE"

        suggested: list[str] = []
        conflicted_drugs = {r["prescribed_drug"].lower() for r in risks}
        for drug in conflicted_drugs:
            for key, alts in self._SAFE_ALTERNATIVES.items():
                if key in drug:
                    for alt in alts:
                        if alt not in suggested:
                            suggested.append(alt)
                    break
        if not suggested and alert_level != "GREEN":
            suggested = ["Consider reviewing the prescription with a clinical pharmacist."]

        return {
            "overall_safety_alert_level": alert_level,
            "prescription_summary": {"digitized_medications": digitized},
            "patient_voice_summary": patient_voice_summary,
            "clinical_risk_analysis": risks,
            "doctor_recommendation": {
                "action_required": action,
                "suggested_alternatives": suggested,
            },
            "clinical_disclaimer": (
                "This is an AI-generated clinical safety advisory. It does not substitute "
                "professional medical judgment. Verify all alerts with a licensed physician "
                "before altering therapy."
            ),
        }

    def analyze_post_medication_feedback(
        self,
        prescribed_medications: list[dict[str, Any]] | list[str],
        patient_allergies: list[str] | None = None,
        patient_conditions: list[str] | None = None,
        feedback_text: str = "",
    ) -> dict[str, Any]:
        """
        Analyzes post-medication patient feedback (symptoms/issues reported after taking meds)
        and performs Adverse Drug Reaction (ADR) causality assessment and clinical doctor advice.
        """
        allergies = [str(a) for a in (patient_allergies or [])]
        conditions = [str(c) for c in (patient_conditions or [])]

        # Extract clean drug names
        med_names: list[str] = []
        for m in prescribed_medications:
            if isinstance(m, dict):
                name = m.get("active_chemical_compound") or m.get("medicine_name") or m.get("raw_text") or ""
                if name:
                    med_names.append(str(name))
            elif isinstance(m, str) and m.strip():
                med_names.append(m.strip())

        clean_text = (feedback_text or "").strip()
        if not clean_text:
            return {
                "status": "NO_FEEDBACK",
                "reported_symptoms": [],
                "suspected_medication": "None",
                "causality_assessment": "No symptoms reported",
                "alert_level": "GREEN",
                "severity": "LOW",
                "pharmacological_mechanism": "Patient has not documented any adverse symptoms.",
                "doctor_recommendations": ["Routine clinical observation. Patient reports no active discomfort."],
                "suggested_alternatives": [],
                "symptomatic_relief": [],
                "emergency_red_flags": [],
                "doctor_clinical_summary": "No post-medication adverse events recorded.",
            }

        # ── LLM Assessment via Gemini ──
        if self.gemini_client is not None and genai is not None:
            prompt = f"""You are an expert Clinical Pharmacologist and Physician Decision Support Specialist.

The patient visited a doctor, was prescribed medications, and took them. The patient is now reporting symptoms/discomfort:

PATIENT'S POST-MEDICATION REPORT:
"{clean_text}"

PRESCRIBED MEDICATIONS:
{json.dumps(med_names)}

PATIENT DOCUMENTED ALLERGIES:
{json.dumps(allergies)}

PATIENT CHRONIC CONDITIONS:
{json.dumps(conditions)}

TASK: Perform a rigorous Adverse Drug Reaction (ADR) causality assessment and produce physician decision support.
Return ONLY valid JSON (no markdown formatting, no preamble) with this exact schema:
{{
  "reported_symptoms": ["list", "of", "extracted", "symptoms"],
  "suspected_medication": "Name of the prescribed medicine most likely responsible, or 'None / Unclear'",
  "causality_assessment": "One of: Definite ADR / Probable ADR / Possible ADR / Unlikely / General Condition",
  "alert_level": "RED or AMBER or GREEN",
  "severity": "HIGH or MEDIUM or LOW",
  "pharmacological_mechanism": "Clear explanation of the pharmacological or immunological cause",
  "doctor_recommendations": [
    "Immediate physician action 1 (e.g., Stop medication, adjust dose)",
    "Action 2"
  ],
  "suggested_alternatives": ["Safe substitute medication 1", "Safe substitute medication 2"],
  "symptomatic_relief": ["Supportive treatment e.g., Cetirizine 10mg, Pantoprazole 40mg, ORS"],
  "emergency_red_flags": ["Critical signs that warrant immediate emergency room transfer"],
  "doctor_clinical_summary": "Concise 2-3 sentence clinical briefing for the doctor summarizing the event and next step"
}}"""
            candidate_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
            for mod in candidate_models:
                try:
                    resp = self.gemini_client.models.generate_content(
                        model=mod,
                        contents=prompt,
                    )
                    txt = resp.text.strip() if resp and resp.text else ""
                    if txt:
                        if txt.startswith("```"):
                            txt = re.sub(r"^```(?:json)?\s*", "", txt)
                            txt = re.sub(r"\s*```$", "", txt)
                        parsed = json.loads(txt)
                        parsed["status"] = "SUCCESS"
                        parsed["raw_feedback"] = clean_text
                        return parsed
                except Exception as e:
                    print(f"[VerificationAgent] Gemini ADR analysis error ({mod}): {e}")
                    continue

        # ── Deterministic Clinical Rule Engine Fallback ──
        lower_feed = clean_text.lower()
        symptoms_found: list[str] = []
        for sym in ["rash", "itching", "hives", "swelling", "stomach pain", "burning", "acidity", "nausea", "vomiting", "diarrhea", "bloating", "cough", "dizziness", "headache", "fever", "cramp"]:
            if sym in lower_feed:
                symptoms_found.append(sym.title())

        # Check for allergy / hypersensitivity (rash, hives, itching)
        is_allergic = any(s in lower_feed for s in ["rash", "itch", "hive", "swell", "urticaria", "allergy"])
        # Check for gastric irritation (stomach, burning, pain, acidity, ulcer)
        is_gastric = any(s in lower_feed for s in ["stomach", "burn", "acid", "gastric", "cramp", "ulcer"])
        # Check for GI intolerance (nausea, diarrhea, vomit, bloat)
        is_gi = any(s in lower_feed for s in ["diarrhea", "vomit", "nausea", "bloat", "loose motion"])

        suspected = med_names[0] if med_names else "Prescribed Regimen"
        alternatives: list[str] = []
        recs: list[str] = []
        relief: list[str] = []
        red_flags: list[str] = [
            "Facial, lip, or tongue swelling (angioedema)",
            "Difficulty breathing, shortness of breath, or wheezing",
            "Dizziness, lightheadedness, or sudden drop in blood pressure",
        ]

        if is_allergic:
            alert_level = "RED"
            severity = "HIGH"
            causality = "Probable Cutaneous Hypersensitivity Reaction (ADR)"
            # Find beta lactam or antibiotic
            for m in med_names:
                if any(k in m.lower() for k in ["amox", "augmentin", "penicillin", "cefix", "clav"]):
                    suspected = m
                    break
            mechanism = (
                f"IgE-mediated hypersensitivity or drug eruption triggered by '{suspected}'. "
                "Cutaneous manifestation with acute pruritus/rash."
            )
            recs = [
                f"Immediately discontinue '{suspected}'. Do NOT re-administer.",
                "Prescribe oral non-sedating H1 antihistamine for pruritus.",
                "Switch to a non-cross-reactive drug class (e.g., Macrolide or Fluoroquinolone).",
                "Log documented drug allergy to patient permanent medical chart.",
            ]
            alternatives = ["Azithromycin 500mg once daily", "Clarithromycin 500mg twice daily", "Doxycycline 100mg"]
            relief = ["Cetirizine 10mg once daily", "Calamine lotion topically", "Fexofenadine 180mg once daily"]

        elif is_gastric:
            alert_level = "AMBER"
            severity = "MEDIUM"
            causality = "Probable Drug-Induced Gastric Mucosal Irritation"
            for m in med_names:
                if any(k in m.lower() for k in ["ibuprofen", "diclofenac", "aspirin", "naproxen", "aceclo", "nsaid"]):
                    suspected = m
                    break
            mechanism = (
                f"COX-1 inhibition by '{suspected}' causing depletion of gastroprotective prostaglandins, "
                "leading to hyperacidity and epigastric mucosal distress."
            )
            recs = [
                f"Discontinue or temporarily withhold '{suspected}'.",
                "Co-prescribe a Proton Pump Inhibitor (PPI) taken 30 minutes before meals.",
                "Switch analgesic to a stomach-sparing alternative (e.g., Paracetamol).",
            ]
            alternatives = ["Paracetamol 650mg SOS", "Tramadol 37.5mg + Paracetamol", "Celecoxib 100mg (COX-2 selective)"]
            relief = ["Pantoprazole 40mg once daily before breakfast", "Antacid oral gel (Magaldrate + Simethicone)"]

        elif is_gi:
            alert_level = "AMBER"
            severity = "MEDIUM"
            causality = "Medication-Associated Gastrointestinal Intolerance"
            for m in med_names:
                if any(k in m.lower() for k in ["metformin", "augmentin", "antibiotic", "iron"]):
                    suspected = m
                    break
            mechanism = f"Local mucosal transit acceleration or osmotic irritation associated with '{suspected}'."
            recs = [
                f"Ensure '{suspected}' is taken strictly in the middle of or immediately following a substantial meal.",
                "Consider dose reduction or titrating upwards slowly.",
                "Maintain adequate oral hydration to prevent electrolyte loss.",
            ]
            alternatives = ["Extended-Release (ER/SR) formulation", "Vildagliptin 50mg BD", "Linagliptin 5mg OD"]
            relief = ["Oral Rehydration Solution (ORS)", "Probiotics (Lactobacillus sporogenes)"]

        else:
            alert_level = "GREEN"
            severity = "LOW"
            causality = "Mild Non-Specific Discomfort / Possible Adaptation"
            mechanism = "Symptoms do not display severe drug toxicity patterns; consistent with mild adaptation."
            recs = [
                "Continue prescribed regimen under routine monitoring.",
                "Ensure proper hydration and take with meals if mild nausea occurs.",
            ]
            relief = ["Rest and oral fluid hydration"]

        return {
            "status": "SUCCESS",
            "raw_feedback": clean_text,
            "reported_symptoms": symptoms_found if symptoms_found else ["Non-specific discomfort"],
            "suspected_medication": suspected,
            "causality_assessment": causality,
            "alert_level": alert_level,
            "severity": severity,
            "pharmacological_mechanism": mechanism,
            "doctor_recommendations": recs,
            "suggested_alternatives": alternatives,
            "symptomatic_relief": relief,
            "emergency_red_flags": red_flags,
            "doctor_clinical_summary": (
                f"Patient reported '{clean_text}'. Evaluated as {causality} secondary to {suspected}. "
                f"Physician action: {recs[0] if recs else 'Monitor patient'}"
            ),
        }
