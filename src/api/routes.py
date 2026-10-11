"""FastAPI backend routes for Multi-Intelligence Agent for Handwritten Prescription & Predictive Health Analytics.

Serves multimodal vision extraction, report comparison, and unified clinical analytics JSON.
"""

from __future__ import annotations

import io
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from ..modules.impact_analyzer import MedicationImpactAnalyzer
from ..modules.orchestrator import ClinicalAgentOrchestrator
from ..modules.patient_database import PatientMedicalDatabase
from ..modules.prescription_vision import PrescriptionVisionExtractor
from ..modules.verification_agent import PrescriptionVerificationAgent

router = APIRouter(prefix="/api", tags=["Clinical Intelligence"])

db = PatientMedicalDatabase()
vision_extractor = PrescriptionVisionExtractor()
verification_agent = PrescriptionVerificationAgent()
impact_analyzer = MedicationImpactAnalyzer()
orchestrator = ClinicalAgentOrchestrator()


@router.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint confirming engine status and multimodal model availability."""
    return {
        "status": "healthy",
        "vision_engine": "Gemini 3.8 Flash Multimodal Vision",
        "vision_available": str(vision_extractor.is_available),
    }


@router.post("/prescription/extract")
async def extract_prescription(file: UploadFile = File(...)) -> dict[str, Any]:
    """Extracts structured prescription data from uploaded doctor prescription image (JPG/PNG/WEBP)."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}") from e

    rx_res = vision_extractor.extract(image)
    if rx_res.get("status") == "error":
        raise HTTPException(status_code=503, detail=rx_res.get("error", "Vision extraction failed"))

    # Convert to standard prescription_extraction schema
    meds = rx_res.get("medicines", [])
    prescribed_meds = []
    clinical_explanations = []

    for m in meds:
        name = m.get("medicine_name", "Unknown Medication")
        prescribed_meds.append(
            {
                "medicine_name": name,
                "dosage": m.get("dosage", "Standard dose"),
                "frequency": m.get("frequency", "As directed"),
                "duration": m.get("duration", "As directed"),
                "instructions": m.get("instructions", "Take as directed by physician"),
                "confidence_level": m.get("certainty", "High"),
            }
        )
        clinical_explanations.append(
            {
                "medicine_name": name,
                "therapeutic_use": m.get("verification_notes", "Therapeutic regimen for diagnosed condition."),
                "mechanism_of_action": f"Pharmacological agent targeting pathophysiological pathways of {rx_res.get('diagnosed_condition', 'patient diagnosis')}.",
            }
        )

    return {
        "patient_info": {
            "name": rx_res.get("patient_name"),
            "age": rx_res.get("patient_age"),
            "gender": rx_res.get("patient_gender"),
            "date": rx_res.get("date"),
        },
        "doctor_info": {
            "name": rx_res.get("doctor_name"),
            "qualification": rx_res.get("doctor_registration") or "Consultant Physician",
        },
        "diagnosed_condition": rx_res.get("diagnosed_condition"),
        "prescribed_medications": prescribed_meds,
        "unreadable_fields": rx_res.get("unreadable_fields", []),
        "clinical_explanations": clinical_explanations,
        "safety_and_disclaimers": {
            "warnings_and_interactions": [
                "Verify compatibility with existing chronic medications before starting new pharmacotherapy.",
                "Maintain periodic laboratory monitoring of hepatic, renal, and electrolyte parameters.",
            ],
            "potential_adverse_effects": [
                "Mild transient nausea, gastrointestinal discomfort, or headache during initiation.",
                "Seek immediate emergency care for symptoms of hypersensitivity (urticaria, dyspnea, edema).",
            ],
            "general_disclaimer": "AI-generated extraction powered by Gemini 3.8 Flash Vision. Verify with a licensed healthcare provider before administration.",
        },
        "patient_care_tips": [
            "Take all doses strictly according to the schedule prescribed by the physician.",
            "Store medications in a secure, cool, dry location away from moisture and direct sunlight.",
            "Record daily blood pressure and glycemic logs for presentation at follow-up consultations.",
        ],
    }


@router.get("/patient/{patient_id}/unified")
def get_unified_patient_analytics(patient_id: str) -> dict[str, Any]:
    """Returns the complete unified JSON covering sidebar profile, prescription, and longitudinal report analysis."""
    patient = db.get_patient_profile(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    lab_reports = db.get_lab_reports(patient_id)
    prescriptions = patient.get("prescriptions", [])
    comparison_records = impact_analyzer.compute_parameter_comparison_table(lab_reports)

    return orchestrator.generate_unified_clinical_json(
        patient_info=patient,
        comparison_records=comparison_records,
        prescriptions=prescriptions,
        lab_reports=lab_reports,
    )
