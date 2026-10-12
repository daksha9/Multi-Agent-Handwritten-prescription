"""
Patient Medical Database Module for MedAgent.
Stores and manages longitudinal patient profiles, prescriptions, lab reports,
medication timelines, and allergy/safety records.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

DATABASE_FILE = Path(__file__).resolve().parents[2] / "data" / "patient_database.json"


def get_default_database() -> dict[str, Any]:
    """Generates pre-seeded high-fidelity longitudinal clinical records."""
    return {
        "patients": {
            "P001": {
                "patient_id": "P001",
                "name": "John Doe",
                "age": 52,
                "gender": "Male",
                "known_conditions": ["Type 2 Diabetes Mellitus", "Essential Hypertension", "Dyslipidemia"],
                "known_allergies": ["Penicillin", "Sulfa drugs"],
                "adverse_reactions": ["Cough with ACE Inhibitors (Enalapril)"],
                "medical_history": "Diagnosed with T2DM in Nov 2024. Hypertension since 2022. Non-smoker.",
                "current_medications": [
                    "Metformin 1000mg BD",
                    "Glimepiride 1mg OD",
                    "Telmisartan 40mg OD",
                    "Atorvastatin 20mg HS",
                ],
                "medication_timeline": [
                    {
                        "period_id": "MED_P1",
                        "period_name": "Period 1: Baseline (Pre-treatment)",
                        "start_date": "2024-11-01",
                        "end_date": "2024-11-15",
                        "medications": ["None (Lifestyle observation)"],
                        "clinical_target": "Initial diagnostic workup",
                    },
                    {
                        "period_id": "MED_P2",
                        "period_name": "Period 2: Metformin 500mg + Telmisartan 40mg",
                        "start_date": "2024-11-16",
                        "end_date": "2025-02-14",
                        "medications": ["Metformin 500mg BD", "Telmisartan 40mg OD"],
                        "clinical_target": "Glycemic and BP initiation",
                    },
                    {
                        "period_id": "MED_P3",
                        "period_name": "Period 3: Metformin 1000mg + Glimepiride 1mg + Atorvastatin 20mg",
                        "start_date": "2025-02-15",
                        "end_date": "2025-09-30",
                        "medications": [
                            "Metformin 1000mg BD",
                            "Glimepiride 1mg OD",
                            "Telmisartan 40mg OD",
                            "Atorvastatin 20mg HS",
                        ],
                        "clinical_target": "Intensive glycemic & lipid control",
                    },
                ],
                "prescriptions": [
                    {
                        "prescription_id": "RX-2024-11-15",
                        "date": "2024-11-15",
                        "doctor_name": "Dr. Robert Smith, MD (Gen. Med)",
                        "clinic_name": "City Health Clinic",
                        "diagnosed_condition": "Type 2 Diabetes Mellitus, Essential Hypertension",
                        "medicines": [
                            {
                                "medicine_name": "Metformin 500mg",
                                "form": "Tablet",
                                "dosage": "500mg",
                                "frequency": "1-0-1 (twice daily)",
                                "duration": "3 months",
                                "instructions": "Take after meals",
                                "certainty": "High",
                            },
                            {
                                "medicine_name": "Telmisartan 40mg",
                                "form": "Tablet",
                                "dosage": "40mg",
                                "frequency": "1-0-0 (once daily morning)",
                                "duration": "3 months",
                                "instructions": "Take in morning before breakfast",
                                "certainty": "High",
                            },
                        ],
                    },
                    {
                        "prescription_id": "RX-2025-02-14",
                        "date": "2025-02-14",
                        "doctor_name": "Dr. Robert Smith, MD (Gen. Med)",
                        "clinic_name": "City Health Clinic",
                        "diagnosed_condition": "Suboptimal Glycemic Control, Mixed Dyslipidemia",
                        "medicines": [
                            {
                                "medicine_name": "Metformin 1000mg",
                                "form": "Tablet",
                                "dosage": "1000mg",
                                "frequency": "1-0-1 (twice daily)",
                                "duration": "6 months",
                                "instructions": "Take with breakfast and dinner",
                                "certainty": "High",
                            },
                            {
                                "medicine_name": "Glimepiride 1mg",
                                "form": "Tablet",
                                "dosage": "1mg",
                                "frequency": "1-0-0 (once daily morning)",
                                "duration": "6 months",
                                "instructions": "Take 15 minutes before breakfast",
                                "certainty": "High",
                            },
                            {
                                "medicine_name": "Atorvastatin 20mg",
                                "form": "Tablet",
                                "dosage": "20mg",
                                "frequency": "0-0-1 (once daily night)",
                                "duration": "6 months",
                                "instructions": "Take at bedtime",
                                "certainty": "High",
                            },
                        ],
                    },
                ],
                "lab_reports": [
                    # Visit 1: Baseline (Pre-Medication) - 2024-11-10
                    {
                        "date": "2024-11-10",
                        "parameter": "HbA1c",
                        "value": 8.9,
                        "unit": "%",
                        "reference_range": "4.0 - 5.6",
                        "condition": "Type 2 Diabetes",
                        "associated_medication": "None (Pre-treatment)",
                    },
                    {
                        "date": "2024-11-10",
                        "parameter": "Fasting Blood Glucose",
                        "value": 188.0,
                        "unit": "mg/dL",
                        "reference_range": "70 - 99",
                        "condition": "Type 2 Diabetes",
                        "associated_medication": "None (Pre-treatment)",
                    },
                    {
                        "date": "2024-11-10",
                        "parameter": "Systolic Blood Pressure",
                        "value": 154.0,
                        "unit": "mmHg",
                        "reference_range": "90 - 120",
                        "condition": "Hypertension",
                        "associated_medication": "None (Pre-treatment)",
                    },
                    {
                        "date": "2024-11-10",
                        "parameter": "Diastolic Blood Pressure",
                        "value": 96.0,
                        "unit": "mmHg",
                        "reference_range": "60 - 80",
                        "condition": "Hypertension",
                        "associated_medication": "None (Pre-treatment)",
                    },
                    {
                        "date": "2024-11-10",
                        "parameter": "Total Cholesterol",
                        "value": 242.0,
                        "unit": "mg/dL",
                        "reference_range": "< 200",
                        "condition": "Dyslipidemia",
                        "associated_medication": "None (Pre-treatment)",
                    },
                    {
                        "date": "2024-11-10",
                        "parameter": "LDL Cholesterol",
                        "value": 158.0,
                        "unit": "mg/dL",
                        "reference_range": "< 100",
                        "condition": "Dyslipidemia",
                        "associated_medication": "None (Pre-treatment)",
                    },
                    {
                        "date": "2024-11-10",
                        "parameter": "Serum Creatinine",
                        "value": 1.05,
                        "unit": "mg/dL",
                        "reference_range": "0.7 - 1.3",
                        "condition": "Renal Function",
                        "associated_medication": "None (Pre-treatment)",
                    },
                    # Visit 2: Follow-up on Medicine A (Metformin 500mg + Telmisartan 40mg) - 2025-01-20
                    {
                        "date": "2025-01-20",
                        "parameter": "HbA1c",
                        "value": 7.8,
                        "unit": "%",
                        "reference_range": "4.0 - 5.6",
                        "condition": "Type 2 Diabetes",
                        "associated_medication": "Metformin 500mg BD",
                    },
                    {
                        "date": "2025-01-20",
                        "parameter": "Fasting Blood Glucose",
                        "value": 146.0,
                        "unit": "mg/dL",
                        "reference_range": "70 - 99",
                        "condition": "Type 2 Diabetes",
                        "associated_medication": "Metformin 500mg BD",
                    },
                    {
                        "date": "2025-01-20",
                        "parameter": "Systolic Blood Pressure",
                        "value": 138.0,
                        "unit": "mmHg",
                        "reference_range": "90 - 120",
                        "condition": "Hypertension",
                        "associated_medication": "Telmisartan 40mg OD",
                    },
                    {
                        "date": "2025-01-20",
                        "parameter": "Diastolic Blood Pressure",
                        "value": 86.0,
                        "unit": "mmHg",
                        "reference_range": "60 - 80",
                        "condition": "Hypertension",
                        "associated_medication": "Telmisartan 40mg OD",
                    },
                    {
                        "date": "2025-01-20",
                        "parameter": "Total Cholesterol",
                        "value": 235.0,
                        "unit": "mg/dL",
                        "reference_range": "< 200",
                        "condition": "Dyslipidemia",
                        "associated_medication": "Dietary modification",
                    },
                    {
                        "date": "2025-01-20",
                        "parameter": "LDL Cholesterol",
                        "value": 150.0,
                        "unit": "mg/dL",
                        "reference_range": "< 100",
                        "condition": "Dyslipidemia",
                        "associated_medication": "Dietary modification",
                    },
                    {
                        "date": "2025-01-20",
                        "parameter": "Serum Creatinine",
                        "value": 1.02,
                        "unit": "mg/dL",
                        "reference_range": "0.7 - 1.3",
                        "condition": "Renal Function",
                        "associated_medication": "Metformin 500mg BD",
                    },
                    # Visit 3: After Regimen Switch to Medicine B (Metformin 1000mg + Glimepiride 1mg + Atorvastatin) - 2025-05-15
                    {
                        "date": "2025-05-15",
                        "parameter": "HbA1c",
                        "value": 6.7,
                        "unit": "%",
                        "reference_range": "4.0 - 5.6",
                        "condition": "Type 2 Diabetes",
                        "associated_medication": "Metformin 1000mg + Glimepiride 1mg",
                    },
                    {
                        "date": "2025-05-15",
                        "parameter": "Fasting Blood Glucose",
                        "value": 114.0,
                        "unit": "mg/dL",
                        "reference_range": "70 - 99",
                        "condition": "Type 2 Diabetes",
                        "associated_medication": "Metformin 1000mg + Glimepiride 1mg",
                    },
                    {
                        "date": "2025-05-15",
                        "parameter": "Systolic Blood Pressure",
                        "value": 124.0,
                        "unit": "mmHg",
                        "reference_range": "90 - 120",
                        "condition": "Hypertension",
                        "associated_medication": "Telmisartan 40mg OD",
                    },
                    {
                        "date": "2025-05-15",
                        "parameter": "Diastolic Blood Pressure",
                        "value": 78.0,
                        "unit": "mmHg",
                        "reference_range": "60 - 80",
                        "condition": "Hypertension",
                        "associated_medication": "Telmisartan 40mg OD",
                    },
                    {
                        "date": "2025-05-15",
                        "parameter": "Total Cholesterol",
                        "value": 178.0,
                        "unit": "mg/dL",
                        "reference_range": "< 200",
                        "condition": "Dyslipidemia",
                        "associated_medication": "Atorvastatin 20mg HS",
                    },
                    {
                        "date": "2025-05-15",
                        "parameter": "LDL Cholesterol",
                        "value": 92.0,
                        "unit": "mg/dL",
                        "reference_range": "< 100",
                        "condition": "Dyslipidemia",
                        "associated_medication": "Atorvastatin 20mg HS",
                    },
                    {
                        "date": "2025-05-15",
                        "parameter": "Serum Creatinine",
                        "value": 0.98,
                        "unit": "mg/dL",
                        "reference_range": "0.7 - 1.3",
                        "condition": "Renal Function",
                        "associated_medication": "Metformin 1000mg",
                    },
                    # Visit 4: Maintenance Monitoring - 2025-09-10
                    {
                        "date": "2025-09-10",
                        "parameter": "HbA1c",
                        "value": 6.4,
                        "unit": "%",
                        "reference_range": "4.0 - 5.6",
                        "condition": "Type 2 Diabetes",
                        "associated_medication": "Metformin 1000mg + Glimepiride 1mg",
                    },
                    {
                        "date": "2025-09-10",
                        "parameter": "Fasting Blood Glucose",
                        "value": 104.0,
                        "unit": "mg/dL",
                        "reference_range": "70 - 99",
                        "condition": "Type 2 Diabetes",
                        "associated_medication": "Metformin 1000mg + Glimepiride 1mg",
                    },
                    {
                        "date": "2025-09-10",
                        "parameter": "Systolic Blood Pressure",
                        "value": 120.0,
                        "unit": "mmHg",
                        "reference_range": "90 - 120",
                        "condition": "Hypertension",
                        "associated_medication": "Telmisartan 40mg OD",
                    },
                    {
                        "date": "2025-09-10",
                        "parameter": "Diastolic Blood Pressure",
                        "value": 76.0,
                        "unit": "mmHg",
                        "reference_range": "60 - 80",
                        "condition": "Hypertension",
                        "associated_medication": "Telmisartan 40mg OD",
                    },
                    {
                        "date": "2025-09-10",
                        "parameter": "Total Cholesterol",
                        "value": 165.0,
                        "unit": "mg/dL",
                        "reference_range": "< 200",
                        "condition": "Dyslipidemia",
                        "associated_medication": "Atorvastatin 20mg HS",
                    },
                    {
                        "date": "2025-09-10",
                        "parameter": "LDL Cholesterol",
                        "value": 85.0,
                        "unit": "mg/dL",
                        "reference_range": "< 100",
                        "condition": "Dyslipidemia",
                        "associated_medication": "Atorvastatin 20mg HS",
                    },
                    {
                        "date": "2025-09-10",
                        "parameter": "Serum Creatinine",
                        "value": 0.96,
                        "unit": "mg/dL",
                        "reference_range": "0.7 - 1.3",
                        "condition": "Renal Function",
                        "associated_medication": "Metformin 1000mg",
                    },
                ],
            },
            "P002": {
                "patient_id": "P002",
                "name": "Mr. Srinivas",
                "age": 41,
                "gender": "Male",
                "known_conditions": [
                    "Chronic Paranoid Schizophrenia",
                    "Type 2 Diabetes",
                    "Hypertension",
                    "Hypercholesterolemia",
                ],
                "known_allergies": ["None documented"],
                "adverse_reactions": ["Mild sedation with antipsychotics"],
                "medical_history": "Diagnosed with Paranoid Schizophrenia in 2021. Managed on maintenance neuroleptic therapy. Mother expired 3 months ago (psychological trigger).",
                "current_medications": [
                    "Sizodon Plus (Risperidone + Trihexyphenidyl) 1-0-1",
                    "Qutipin 200mg (Quetiapine) 0-0-1",
                    "Ativan 2mg (Lorazepam) 0-0-1",
                    "Rivotril 0.5mg (Clonazepam) 0-0-1",
                    "Serta 50mg (Sertraline) 0-0-1",
                ],
                "medication_timeline": [
                    {
                        "period_id": "SRINIVAS_P1",
                        "period_name": "Period 1: Acute Stabilization",
                        "start_date": "2023-10-01",
                        "end_date": "2024-03-14",
                        "medications": ["Sizodon Plus 1-0-1", "Ativan 2mg 0-0-1"],
                        "clinical_target": "Acute psychosis and anxiety reduction",
                    },
                    {
                        "period_id": "SRINIVAS_P2",
                        "period_name": "Period 2: Full Neuro-Psychiatric Maintenance (Dr. Nagendar Rao)",
                        "start_date": "2024-03-15",
                        "end_date": "2024-09-30",
                        "medications": [
                            "Sizodon Plus 1-0-1",
                            "Qutipin 200mg 0-0-1",
                            "Ativan 2mg 0-0-1",
                            "Rivotril 0.5mg 0-0-1",
                            "Serta 50mg 0-0-1",
                        ],
                        "clinical_target": "Paranoid delusions, sleep disruption, and depression maintenance",
                    },
                ],
                "prescriptions": [
                    {
                        "prescription_id": "RX-2024-03-15",
                        "date": "2024-03-15",
                        "doctor_name": "Dr. Y. Nagendar Rao, MBBS, MD (Psy), FIPS",
                        "clinic_name": "Consultant Neuro - Psychiatrist Clinic (Secunderabad)",
                        "diagnosed_condition": "Chr. Schizophrenia (Paranoid), DM, HTN, Hypercholesterolemia",
                        "medicines": [
                            {
                                "medicine_name": "Sizodon Plus",
                                "form": "Tablet",
                                "dosage": "Risperidone 2mg + Trihexyphenidyl 2mg",
                                "frequency": "1-0-1",
                                "duration": "6 months",
                                "instructions": "Take after food",
                                "certainty": "High",
                            },
                            {
                                "medicine_name": "Qutipin 200mg",
                                "form": "Tablet",
                                "dosage": "200mg",
                                "frequency": "0-0-1",
                                "duration": "6 months",
                                "instructions": "Take at night",
                                "certainty": "High",
                            },
                            {
                                "medicine_name": "Ativan 2mg",
                                "form": "Tablet",
                                "dosage": "2mg",
                                "frequency": "0-0-1",
                                "duration": "6 months",
                                "instructions": "Take at night",
                                "certainty": "High",
                            },
                            {
                                "medicine_name": "Rivotril 0.5mg",
                                "form": "Tablet",
                                "dosage": "0.5mg",
                                "frequency": "0-0-1",
                                "duration": "6 months",
                                "instructions": "Take at night",
                                "certainty": "High",
                            },
                            {
                                "medicine_name": "Serta 50mg",
                                "form": "Tablet",
                                "dosage": "50mg",
                                "frequency": "0-0-1",
                                "duration": "6 months",
                                "instructions": "Take at night",
                                "certainty": "High",
                            },
                        ],
                    }
                ],
                "lab_reports": [
                    {
                        "date": "2024-02-10",
                        "parameter": "Fasting Blood Glucose",
                        "value": 152.0,
                        "unit": "mg/dL",
                        "reference_range": "70 - 99",
                        "condition": "Metabolic Monitoring",
                        "associated_medication": "Sizodon Plus",
                    },
                    {
                        "date": "2024-02-10",
                        "parameter": "Total Cholesterol",
                        "value": 260.0,
                        "unit": "mg/dL",
                        "reference_range": "< 200",
                        "condition": "Lipid Profile",
                        "associated_medication": "None",
                    },
                    {
                        "date": "2024-02-10",
                        "parameter": "Serum Triglycerides",
                        "value": 230.0,
                        "unit": "mg/dL",
                        "reference_range": "< 150",
                        "condition": "Lipid Profile",
                        "associated_medication": "None",
                    },
                    {
                        "date": "2024-06-20",
                        "parameter": "Fasting Blood Glucose",
                        "value": 140.0,
                        "unit": "mg/dL",
                        "reference_range": "70 - 99",
                        "condition": "Metabolic Monitoring",
                        "associated_medication": "Sizodon Plus + Qutipin 200mg",
                    },
                    {
                        "date": "2024-06-20",
                        "parameter": "Total Cholesterol",
                        "value": 240.0,
                        "unit": "mg/dL",
                        "reference_range": "< 200",
                        "condition": "Lipid Profile",
                        "associated_medication": "Dietary modification",
                    },
                    {
                        "date": "2024-06-20",
                        "parameter": "Serum Triglycerides",
                        "value": 205.0,
                        "unit": "mg/dL",
                        "reference_range": "< 150",
                        "condition": "Lipid Profile",
                        "associated_medication": "Dietary modification",
                    },
                ],
            },
            "P003": {
                "patient_id": "P003",
                "name": "Mr. Aman",
                "age": 19,
                "gender": "Male",
                "known_conditions": ["Acute Upper Respiratory Tract Infection", "Fever"],
                "known_allergies": ["No known drug allergies"],
                "adverse_reactions": ["None"],
                "medical_history": "Presented to Sai Ram Clinic with acute complaints of cold, throat irritation, and high-grade fever for 1 day.",
                "current_medications": [
                    "Oprox-CV 200mg 1-0-1",
                    "Altinose-SP 1-1-1",
                    "Breezy 0-0-1",
                    "Chipen-D 40mg 1-0-1 (B/F)",
                ],
                "medication_timeline": [
                    {
                        "period_id": "AMAN_P1",
                        "period_name": "Acute Therapy (Sai Ram Clinic)",
                        "start_date": "2025-05-22",
                        "end_date": "2025-05-28",
                        "medications": [
                            "Oprox-CV 200mg (Cefpodoxime + Clavulanate)",
                            "Altinose-SP (Aceclofenac + Paracetamol + Serratiopeptidase)",
                            "Breezy (Levocetirizine / Montelukast)",
                            "Chipen-D 40mg (Pantoprazole + Domperidone)",
                        ],
                        "clinical_target": "Bacterial infection eradication, fever control, and decongestion",
                    }
                ],
                "prescriptions": [
                    {
                        "prescription_id": "RX-2025-05-22",
                        "date": "2025-05-22",
                        "doctor_name": "Dr. Sachin Patil, Consultant Family Physician",
                        "clinic_name": "Sai Ram Clinic",
                        "diagnosed_condition": "c/o cold, throat irritation, fever x 1 day. Vitals: BP 120/70, HR 116/min, Temp 102.2°F, SpO2 98%",
                        "medicines": [
                            {
                                "medicine_name": "Oprox-CV 200mg",
                                "form": "Tablet",
                                "dosage": "200mg",
                                "frequency": "1-0-1",
                                "duration": "6 days",
                                "instructions": "Take after food",
                                "certainty": "High",
                            },
                            {
                                "medicine_name": "Altinose-SP",
                                "form": "Tablet",
                                "dosage": "Fixed combination",
                                "frequency": "1-1-1",
                                "duration": "8 days",
                                "instructions": "Take after meals for fever/pain",
                                "certainty": "High",
                            },
                            {
                                "medicine_name": "Breezy",
                                "form": "Tablet",
                                "dosage": "Anti-allergic",
                                "frequency": "0-0-1",
                                "duration": "5 days",
                                "instructions": "Take at bedtime",
                                "certainty": "High",
                            },
                            {
                                "medicine_name": "Chipen-D 40mg",
                                "form": "Tablet",
                                "dosage": "40mg",
                                "frequency": "1-0-1",
                                "duration": "10 days",
                                "instructions": "Take before food (B/F)",
                                "certainty": "High",
                            },
                        ],
                    }
                ],
                "lab_reports": [
                    {
                        "date": "2025-05-22",
                        "parameter": "Body Temperature",
                        "value": 102.2,
                        "unit": "°F",
                        "reference_range": "97.8 - 99.1",
                        "condition": "Infection/Fever",
                        "associated_medication": "Pre-treatment",
                    },
                    {
                        "date": "2025-05-22",
                        "parameter": "Heart Rate",
                        "value": 116.0,
                        "unit": "bpm",
                        "reference_range": "60 - 100",
                        "condition": "Vitals",
                        "associated_medication": "Pre-treatment",
                    },
                    {
                        "date": "2025-05-22",
                        "parameter": "SpO2",
                        "value": 98.0,
                        "unit": "%",
                        "reference_range": "95 - 100",
                        "condition": "Respiratory",
                        "associated_medication": "Pre-treatment",
                    },
                    {
                        "date": "2025-05-25",
                        "parameter": "Body Temperature",
                        "value": 98.6,
                        "unit": "°F",
                        "reference_range": "97.8 - 99.1",
                        "condition": "Infection/Fever",
                        "associated_medication": "Altinose-SP + Oprox-CV 200mg",
                    },
                    {
                        "date": "2025-05-25",
                        "parameter": "Heart Rate",
                        "value": 78.0,
                        "unit": "bpm",
                        "reference_range": "60 - 100",
                        "condition": "Vitals",
                        "associated_medication": "Altinose-SP + Oprox-CV 200mg",
                    },
                ],
            },
        }
    }


class PatientMedicalDatabase:
    """Manages reading and updating patient records, prescriptions, and lab telemetry."""

    def __init__(self, db_path: Path = DATABASE_FILE):
        self.db_path = db_path
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if self.db_path.exists():
            try:
                with open(self.db_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        default_data = get_default_database()
        self._save(default_data)
        return default_data

    def _save(self, data: dict[str, Any] | None = None):
        if data is None:
            data = self.data
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def list_patients(self) -> list[dict[str, Any]]:
        """Returns summary list of all patients."""
        res = []
        for pid, p in self.data.get("patients", {}).items():
            res.append(
                {
                    "patient_id": pid,
                    "name": p.get("name", "Unknown"),
                    "age": p.get("age", "N/A"),
                    "gender": p.get("gender", "N/A"),
                    "conditions": ", ".join(p.get("known_conditions", [])),
                    "active_medications_count": len(p.get("current_medications", [])),
                    "lab_reports_count": len(p.get("lab_reports", [])),
                }
            )
        return res

    def get_patient(self, patient_id: str) -> dict[str, Any] | None:
        """Returns full clinical record of a patient."""
        return self.data.get("patients", {}).get(patient_id)

    def get_patient_profile(self, patient_id: str) -> dict[str, Any] | None:
        """Alias for get_patient returning full clinical record."""
        return self.get_patient(patient_id)

    def add_or_update_patient(self, patient_data: dict[str, Any]) -> str:
        """Adds a new patient or updates existing."""
        pid = patient_data.get("patient_id") or f"P{len(self.data.get('patients', {})) + 1:03d}"
        patient_data["patient_id"] = pid
        self.data.setdefault("patients", {})[pid] = patient_data
        self._save()
        return pid

    def add_prescription(self, patient_id: str, prescription_dict: dict[str, Any]) -> bool:
        """Attaches an extracted prescription to the patient profile."""
        patient = self.get_patient(patient_id)
        if not patient:
            return False
        rx_list = patient.setdefault("prescriptions", [])
        if "prescription_id" not in prescription_dict:
            prescription_dict["prescription_id"] = f"RX-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        rx_list.append(prescription_dict)

        # Update active medications list
        new_meds = [m.get("medicine_name") for m in prescription_dict.get("medicines", []) if m.get("medicine_name")]
        if new_meds:
            curr_meds = set(patient.get("current_medications", []))
            curr_meds.update(new_meds)
            patient["current_medications"] = list(curr_meds)

        self._save()
        return True

    def get_lab_reports(self, patient_id: str) -> list[dict[str, Any]]:
        """Returns patient lab records as a raw list of dictionaries."""
        patient = self.get_patient(patient_id)
        if not patient or not patient.get("lab_reports"):
            return []
        return list(patient["lab_reports"])

    def add_lab_report(self, patient_id: str, lab_entry: dict[str, Any]) -> bool:
        """Appends a new lab parameter record."""
        patient = self.get_patient(patient_id)
        if not patient:
            return False
        patient.setdefault("lab_reports", []).append(lab_entry)
        self._save()
        return True

    def save_structured_report(
        self,
        patient_id: str,
        report_id: str,
        report_name: str,
        test_date: str,
        associated_medication: str,
        parameters: list[dict[str, Any]],
    ) -> bool:
        """
        Saves or updates a distinct laboratory report for a patient.
        Replaces any prior entries with matching report_id so reports are
        never merged, duplicated, or corrupted by repeated saves.
        """
        patient = self.get_patient(patient_id)
        if not patient:
            return False

        existing_reports = patient.setdefault("lab_reports", [])
        # Filter out old entries for this report_id or matching date & medication
        cleaned_reports = [
            r
            for r in existing_reports
            if r.get("report_id") != report_id
            and not (
                r.get("date") == test_date
                and r.get("associated_medication") == associated_medication
            )
        ]

        new_entries = []
        for p in parameters:
            entry = {
                "report_id": report_id,
                "report_name": report_name,
                "date": test_date,
                "parameter": p["parameter"],
                "value": float(p["value"]),
                "unit": p.get("unit", ""),
                "reference_range": p.get("reference_range", ""),
                "condition": p.get("condition", "General Health"),
                "associated_medication": associated_medication,
            }
            new_entries.append(entry)

        patient["lab_reports"] = cleaned_reports + new_entries
        self._save()
        return True

    def get_distinct_reports(self, patient_id: str) -> list[dict[str, Any]]:
        """
        Returns a list of distinct grouped reports for the patient.
        Each item summarizes report_id, report_name, date, associated_medication,
        and its list of parameter observations.
        """
        patient = self.get_patient(patient_id)
        if not patient or not patient.get("lab_reports"):
            return []

        reports_map: dict[str, dict[str, Any]] = {}
        for r in patient["lab_reports"]:
            rep_id = r.get("report_id") or f"Report on {r.get('date', 'Unknown')}"
            key = f"{rep_id}_{r.get('date')}_{r.get('associated_medication')}"
            if key not in reports_map:
                reports_map[key] = {
                    "report_id": rep_id,
                    "report_name": r.get("report_name") or rep_id,
                    "date": r.get("date", "Unknown"),
                    "associated_medication": r.get("associated_medication", "Unassigned"),
                    "parameters": [],
                }
            reports_map[key]["parameters"].append(
                {
                    "parameter": r.get("parameter"),
                    "value": r.get("value"),
                    "unit": r.get("unit"),
                    "reference_range": r.get("reference_range"),
                    "condition": r.get("condition"),
                }
            )

        # Return sorted chronologically by date
        sorted_reports = sorted(
            reports_map.values(), key=lambda x: str(x.get("date", ""))
        )
        return sorted_reports

    def get_lab_reports_df(self, patient_id: str) -> pd.DataFrame:
        """Returns patient lab records as a cleaned pandas DataFrame."""
        patient = self.get_patient(patient_id)
        if not patient or not patient.get("lab_reports"):
            return pd.DataFrame()
        df = pd.DataFrame(patient["lab_reports"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values(by="date").reset_index(drop=True)
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df

    def get_medication_timeline_df(self, patient_id: str) -> pd.DataFrame:
        """Returns medication periods dataframe."""
        patient = self.get_patient(patient_id)
        if not patient or not patient.get("medication_timeline"):
            return pd.DataFrame()
        return pd.DataFrame(patient["medication_timeline"])

    def update_patient_conditions_and_allergies(
        self,
        patient_id: str,
        known_conditions: list[str],
        known_allergies: list[str],
        adverse_reactions: list[str] | None = None,
    ) -> bool:
        """Updates documented medical conditions, allergies, and adverse reactions."""
        patient = self.get_patient(patient_id)
        if not patient:
            return False
        patient["known_conditions"] = self._deduplicate_items(known_conditions)
        patient["known_allergies"] = self._deduplicate_items(known_allergies)
        if adverse_reactions is not None:
            patient["adverse_reactions"] = self._deduplicate_items(adverse_reactions)
        self._save()
        return True

    @staticmethod
    def _deduplicate_items(items: list[str]) -> list[str]:
        """Deduplicates strings case-insensitively while preserving original casing and order."""
        seen = set()
        deduped: list[str] = []
        for it in items:
            cleaned = str(it).strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                deduped.append(cleaned)
        return deduped

    def update_patient_profile(
        self,
        patient_id: str,
        name: str | None = None,
        age: int | None = None,
        gender: str | None = None,
        known_conditions: list[str] | None = None,
        known_allergies: list[str] | None = None,
        current_medications: list[str] | None = None,
        adverse_reactions: list[str] | None = None,
        merge_lists: bool = True,
    ) -> bool:
        """
        Updates patient demographic and clinical fields.
        If merge_lists is True, merges new conditions/allergies/medications
        with existing values, strictly deduplicating so values are never repeated.
        """
        patient = self.get_patient(patient_id)
        if not patient:
            return False

        if name and name.strip():
            patient["name"] = name.strip()
        if age is not None and age > 0:
            patient["age"] = age
        if gender and gender.strip():
            patient["gender"] = gender.strip()

        if known_conditions is not None:
            if merge_lists:
                combined = list(patient.get("known_conditions", [])) + list(known_conditions)
                patient["known_conditions"] = self._deduplicate_items(combined)
            else:
                patient["known_conditions"] = self._deduplicate_items(known_conditions)

        if known_allergies is not None:
            if merge_lists:
                combined = list(patient.get("known_allergies", [])) + list(known_allergies)
                patient["known_allergies"] = self._deduplicate_items(combined)
            else:
                patient["known_allergies"] = self._deduplicate_items(known_allergies)

        if current_medications is not None:
            if merge_lists:
                combined = list(patient.get("current_medications", [])) + list(current_medications)
                patient["current_medications"] = self._deduplicate_items(combined)
            else:
                patient["current_medications"] = self._deduplicate_items(current_medications)

        if adverse_reactions is not None:
            if merge_lists:
                combined = list(patient.get("adverse_reactions", [])) + list(adverse_reactions)
                patient["adverse_reactions"] = self._deduplicate_items(combined)
            else:
                patient["adverse_reactions"] = self._deduplicate_items(adverse_reactions)

        self._save()
        return True

    def add_medication_period(
        self,
        patient_id: str,
        period_name: str,
        start_date: str,
        end_date: str,
        medications: list[str],
        clinical_target: str = "Clinical control",
    ) -> bool:
        """Appends a new chronological medication period to patient timeline."""
        patient = self.get_patient(patient_id)
        if not patient:
            return False
        timeline = patient.setdefault("medication_timeline", [])
        period_id = f"MED_P{len(timeline) + 1}"
        timeline.append(
            {
                "period_id": period_id,
                "period_name": period_name,
                "start_date": start_date,
                "end_date": end_date,
                "medications": medications,
                "clinical_target": clinical_target,
            }
        )
        self._save()
        return True
