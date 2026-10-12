from src.modules.impact_analyzer import MedicationImpactAnalyzer
from src.modules.patient_database import PatientMedicalDatabase


def test_patient_database_initialization():
    db = PatientMedicalDatabase()
    patients = db.list_patients()
    assert len(patients) >= 3

    # Check John Doe
    john = db.get_patient("P001")
    assert john is not None
    assert john["name"] in ["John Doe", "Santu Ghorai"]
    assert "Type 2 Diabetes Mellitus" in john["known_conditions"]
    assert len(john["lab_reports"]) >= 8
    assert len(john["medication_timeline"]) >= 3



def test_medication_impact_comparison():
    db = PatientMedicalDatabase()
    analyzer = MedicationImpactAnalyzer()

    john = db.get_patient("P001")
    assert john is not None
    comparison = analyzer.compute_parameter_comparison_table(
        lab_reports=john["lab_reports"], medication_timeline=john["medication_timeline"]
    )

    assert len(comparison) > 0

    # Verify required Objective 2 fields in each row
    required_keys = [
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
    for row in comparison:
        for k in required_keys:
            assert k in row, f"Missing key {k} in comparison row"

    # Check HbA1c trajectory specifically
    hba1c_rows = [r for r in comparison if r["Parameter"] == "HbA1c"]
    assert len(hba1c_rows) >= 3
    # Check that later values decreased
    latest_hba1c = hba1c_rows[0]  # sorted descending by date
    assert float(latest_hba1c["Current Value"]) < 7.0
