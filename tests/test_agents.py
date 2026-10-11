import os

from src.modules.history_agent import PatientHistorySynthesisAgent
from src.modules.orchestrator import ClinicalAgentOrchestrator
from src.modules.prediction_agent import DiseasePredictionAgent
from src.modules.verification_agent import PrescriptionVerificationAgent


def test_verification_agent_allergy():
    agent = PrescriptionVerificationAgent()
    result = agent.verify_prescriptions(
        prescribed_drugs=["Amoxicillin 500mg"],
        patient_allergies=["Penicillin"],
    )
    assert result["overall_status"] == "CRITICAL_RISK"
    assert result["critical_count"] >= 1
    assert any("Allergy" in a["category"] for a in result["alerts"])


def test_verification_agent_contraindication():
    agent = PrescriptionVerificationAgent()
    result = agent.verify_prescriptions(
        prescribed_drugs=["Metformin 500mg"],
        patient_conditions=["Severe Renal Impairment"],
    )
    assert result["overall_status"] == "CRITICAL_RISK"
    assert any("Contraindication" in a["category"] for a in result["alerts"])


def test_verification_agent_interaction():
    agent = PrescriptionVerificationAgent()
    result = agent.verify_prescriptions(
        prescribed_drugs=["Aspirin 75mg"],
        active_medications=["Warfarin 5mg"],
    )
    assert result["critical_count"] >= 1
    assert any("Interaction" in a["category"] for a in result["alerts"])


def test_prediction_agent_diabetes_and_htn():
    agent = DiseasePredictionAgent()
    biomarkers = {
        "fasting_blood_glucose": 160.0,
        "hba1c": 8.1,
    }
    vitals = {
        "systolic_bp": 150,
        "diastolic_bp": 94,
    }
    patient = {"age": 60, "gender": "Male"}

    res = agent.predict_diseases(biomarkers, vitals, patient)
    preds = {p["disease"]: p for p in res["predictions"]}

    # Diabetes should be high risk
    assert "Type 2 Diabetes Mellitus" in preds
    assert preds["Type 2 Diabetes Mellitus"]["probability_percent"] > 75.0
    assert preds["Type 2 Diabetes Mellitus"]["risk_tier"] == "High Risk"

    # Hypertension should be high risk
    assert "Essential Hypertension" in preds
    assert preds["Essential Hypertension"]["probability_percent"] > 75.0


def test_history_agent_synthesis():
    agent = PatientHistorySynthesisAgent()
    doc1 = {
        "document_name": "Visit 2025",
        "date": "2025-06-10",
        "biomarkers": {"hba1c": 7.1, "fasting_blood_glucose": 130.0},
        "diagnoses": ["Type 2 Diabetes"],
        "medications": ["Metformin 500mg"],
        "allergies": ["Penicillin"],
    }
    doc2 = {
        "document_name": "Visit 2026",
        "date": "2026-03-15",
        "biomarkers": {"hba1c": 7.9, "fasting_blood_glucose": 155.0},
        "diagnoses": ["Type 2 Diabetes", "Hypertension"],
        "medications": ["Metformin 500mg", "Atorvastatin 20mg"],
        "allergies": ["Penicillin"],
    }

    res = agent.synthesize_history([doc1, doc2])
    assert len(res["timeline"]) == 2
    assert "Type 2 Diabetes" in res["all_diagnoses"]
    assert len(res["biomarker_trends"]) >= 1

    # Check trend for hba1c is worsening
    hba1c_trend = next(t for t in res["biomarker_trends"] if "hba1c" in t["marker"].lower())
    assert hba1c_trend["trajectory"] == "WORSENING"


def test_orchestrator_powerbi_export(tmp_path):
    orchestrator = ClinicalAgentOrchestrator(export_dir=str(tmp_path))

    # Run prescription verification
    orchestrator.process_prescription_verification(
        extracted_drugs=["Metformin 500mg", "Atorvastatin 20mg"],
        ocr_confidence_scores=[92.0, 88.5],
        patient_allergies=["Penicillin"],
    )

    # Export datasets
    paths = orchestrator.export_powerbi_datasets()
    for filepath in paths.values():
        assert os.path.exists(filepath), f"Export file {filepath} not found"
