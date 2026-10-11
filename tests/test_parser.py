from pathlib import Path

from src.modules.parser import ClinicalDocumentParser


def test_parser_biomarker_extraction():
    parser = ClinicalDocumentParser()
    sample_path = Path(__file__).resolve().parents[1] / "data" / "reports" / "sample_patient_lab_report.txt"
    assert sample_path.exists(), "Sample report file missing"

    with open(sample_path, encoding="utf-8") as f:
        text = f.read()

    parsed = parser.parse_document(text, doc_name="LabReport_2026")

    # Biomarkers
    biomarkers = parsed["biomarkers"]
    assert biomarkers.get("fasting_blood_glucose") == 154.0
    assert biomarkers.get("postprandial_blood_glucose") == 215.0
    assert biomarkers.get("hba1c") == 7.8
    assert biomarkers.get("serum_creatinine") == 1.45
    assert biomarkers.get("total_cholesterol") == 228.0
    assert biomarkers.get("ldl_cholesterol") == 142.0

    # Vitals
    vitals = parsed["vitals"]
    assert vitals["systolic_bp"] == 148
    assert vitals["diastolic_bp"] == 92
    assert vitals["bmi"] == 28.4

    # Demographics
    assert parsed["patient"]["age"] == 58
    assert parsed["patient"]["gender"] == "Male"

    # Allergies & Diagnoses
    assert any("Penicillin" in a for a in parsed["allergies"])
    assert any("Diabetes" in d for d in parsed["diagnoses"])
    assert any("Hypertension" in d for d in parsed["diagnoses"])
