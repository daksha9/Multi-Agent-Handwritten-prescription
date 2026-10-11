Implementation Plan
Implementation Plan: End-to-End Handwritten Prescription Detection & Clinical Decision Support
This plan details the full implementation of the MedAgent project: extending the prescription OCR pipeline, implementing specialized clinical AI agents, integrating lab biomarker disease prediction (% likelihood), document parsing, Streamlit UI integration, and Power BI telemetry export.

User Review Required
IMPORTANT

The system is designed to use Groq API (llama-3.3-70b-versatile) with an automatic offline fallback to rule-based clinical logic (from data/drugs.csv and medical reference guidelines). An existing key is detected in config/.env. If an API call fails or is rate-limited, the system seamlessly transitions to deterministic rule-based algorithms so execution never breaks.

NOTE

For Disease Prediction, we implement a clinically calibrated probabilistic scoring engine based on established medical risk algorithms (Framingham Risk, ADA Diabetes Risk Score, KDIGO Kidney Disease staging) that computes exact percentage likelihoods for conditions based on extracted lab biomarkers (HbA1c, Glucose, Blood Pressure, Cholesterol, Creatinine, etc.).

Proposed Architecture & File Changes

med_agent/
├── data/
│   ├── reports/
│   │   └── sample_patient_lab_report.txt  <-- [NEW] Realistic multi-visit patient test record
├── outputs/
│   └── exports/                           <-- Telemetry folder for Power BI CSVs
├── src/
│   ├── app.py                             <-- [MODIFY] Wire Tab 1 (verification), Tab 2 (history & prediction), Tab 3 (Power BI export)
│   └── modules/
│       ├── parser.py                      <-- [NEW] PDF & clinical text/table extractor for reports
│       ├── verification_agent.py          <-- [NEW] Drug interaction, contraindication, allergy & safety checker
│       ├── prediction_agent.py            <-- [NEW] Disease likelihood (%) predictor & biomarker analyzer
│       ├── history_agent.py               <-- [NEW] Longitudinal timeline & clinical narrative synthesizer
│       └── orchestrator.py                <-- [NEW] Multi-agent coordinator & telemetry logger
└── tests/
    ├── test_parser.py                     <-- [NEW] Unit tests for document parser
    └── test_agents.py                     <-- [NEW] Unit tests for verification, prediction, and history agents
Component Breakdown
1. src/modules/parser.py [NEW]
Extracts text and tabular biomarker data from uploaded PDFs (via pdfplumber), text files, or report scans.
Implements robust regular expressions and clinical entity extraction for:
Vitals: Blood Pressure (Systolic / Diastolic), Heart Rate, BMI.
Glycemic markers: Fasting Blood Glucose, Postprandial Glucose, HbA1c.
Renal markers: Serum Creatinine, Blood Urea Nitrogen, eGFR.
Lipid profile: Total Cholesterol, LDL, HDL, Triglycerides.
Liver panel: ALT (SGPT), AST (SGOT), Bilirubin.
Medications, Diagnoses, Allergies, and Chronological dates.
2. src/modules/verification_agent.py [NEW]
Verifies extracted prescription drugs against patient safety profiles:
Drug-Drug Interactions (DDI): Detects dangerous pairs (e.g., Warfarin + Aspirin, Metformin + Contrast, Pantoprazole + Clopidogrel).
Allergy Flags: Checks for cross-sensitivities (e.g., Penicillin allergy vs. Amoxicillin).
Contraindications: Validates drugs against patient conditions (e.g., Metformin in severe renal disease, NSAIDs in peptic ulcers).
Dosage Range Checks: Verifies dosage against standard clinical dosage ceilings.
Powered by Groq LLM with offline fallback to data/drugs.csv and local clinical rules.
3. src/modules/prediction_agent.py [NEW]
Computes probabilistic disease likelihood percentages based on biomarkers:
Type-2 Diabetes: Evaluates Fasting Glucose and HbA1c against ADA clinical cutoffs (e.g., 
HbA1c
≥
6.5
%
HbA1c≥6.5% yields 
>
85
%
>85% likelihood).
Hypertension: Evaluates Systolic/Diastolic BP according to AHA/ACC staging.
Cardiovascular Disease Risk: Computes lipid ratio (
Total Cholesterol
/
HDL
Total Cholesterol/HDL) and age/BP-adjusted risk score.
Chronic Kidney Disease (CKD): Evaluates Serum Creatinine and estimated GFR.
Liver Dysfunction: Assesses transaminase elevations (
ALT
/
AST
ALT/AST).
Generates differential diagnoses, clinical confidence ratings, biomarker status flags (Normal, Elevated, Critical), and suggested follow-up tests.
4. src/modules/history_agent.py [NEW]
Ingests parsed documents and builds a chronological clinical timeline.
Tracks biomarker trajectories over time (e.g., whether HbA1c is improving or worsening).
Identifies clinical gaps (e.g., missing annual kidney screen or missing diabetic foot exam).
Synthesizes a concise executive clinical summary for the physician.
5. src/modules/orchestrator.py [NEW]
Coordinates the multi-agent execution pipeline.
Maintains session state and aggregates performance telemetry:
Execution runtimes per agent
Total prescriptions processed, error rates, and safety alerts triggered
Formats and writes export datasets for Power BI.
6. src/app.py [MODIFY]
Tab 1 (Prescription Scanner):
Adds interactive inputs for Patient Allergies and Active Medications.
Adds the Prescription Verification Agent widget displaying real-time safety alerts (DDI, Contraindications, Dosage warnings) next to the OCR result.
Tab 2 (Patient History & Disease Prediction):
Wires document upload to parser.py, history_agent.py, and prediction_agent.py.
Displays:
Disease Likelihood Percentage Cards with color-coded risk levels (High, Moderate, Low).
Biomarker Analysis Table (Value, Reference Range, Status).
Longitudinal Patient Timeline and Clinical Summary.
Recommended Confirmatory Diagnostic Tests.
Tab 3 (Project Dashboard & Power BI Export):
Live session metrics (Prescriptions scanned, safety alerts flagged, disease risk distribution).
Interactive charts (Biomarker distributions and risk breakdown).
One-click "Export Data for Power BI" button which writes cleaned relational CSV files to outputs/exports/ and presents direct Power BI integration instructions.
Verification Plan
Automated Unit Tests
Run pytest on tests/test_parser.py to verify biomarker extraction and PDF/text parsing.
Run pytest on tests/test_agents.py to verify:
Drug interaction and allergy detection in VerificationAgent.
Disease percentage calculations and boundary checks in PredictionAgent.
Timeline generation and clinical summary synthesis in HistoryAgent.
Manual & End-to-End Verification
Launch Streamlit app (streamlit run src/app.py).
Test Tab 1 using an existing sample image from data/prescriptions/rx_01.jpeg and verify OCR + Drug Suggestion + Safety Verification alerts.
Test Tab 2 using data/reports/sample_patient_lab_report.txt and verify document parsing, patient timeline generation, and disease prediction percentages.
Test Tab 3: Click "Export Data for Power BI" and verify that all export CSV files (prescriptions.csv, disease_predictions.csv, patient_timeline.csv, kpi_metrics.csv) are generated with correct schemas in outputs/exports/.I am developing an AI-powered healthcare project focused on handwritten prescription extraction, patient report comparison, medication impact analysis, and healthcare analytics.

The project has three main objectives:

OBJECTIVE 1 – HANDWRITTEN PRESCRIPTION DATA EXTRACTION

The system should extract structured medical information from handwritten doctor prescriptions.

Use multimodal Vision AI, specifically Gemini Vision, as the primary model for understanding and extracting information from handwritten prescriptions, including cursive, messy, and scribbled doctor handwriting.

The system should identify and extract information such as:
- Patient name
- Age and gender, if available
- Doctor details, if available
- Medicine names
- Dosage
- Frequency
- Duration
- Instructions
- Diagnosed condition
- Other relevant medical information

The extracted information should be converted into a clean, structured format such as JSON or a database record.

Do NOT use a local TrOCR or offline OCR fallback. Gemini Vision should be the primary handwriting-understanding mechanism.

The system should also validate the extracted information where possible and clearly mark uncertain or unreadable fields rather than hallucinating medical information.

OBJECTIVE 2 – PATIENT REPORT COMPARISON AND MEDICATION IMPACT ANALYSIS

The second objective is to compare a patient's medical reports over different time periods and analyze how their condition changes after taking a particular medication.

For example:

A patient has diabetes.

1. The patient's initial medical report contains a blood glucose/HbA1c value before starting a particular medicine.
2. The patient starts Medicine A.
3. A follow-up report is generated after using Medicine A.
4. The system compares the before-medication and after-medication values.
5. If the patient's condition worsens or improves, the system should identify the change.
6. If the doctor changes the medication to Medicine B, the system should compare subsequent reports again.
7. The system should track whether the relevant medical parameters increased, decreased, or remained relatively stable.

The system should extract relevant medical parameters from laboratory reports and organize them chronologically.

For each parameter, display:
- Test/parameter name
- Previous value
- Current value
- Unit
- Reference/normal range
- Date of test
- Difference/change
- Percentage change where appropriate
- Associated medication
- Condition being monitored
- Interpretation of the trend

The changes should be represented graphically using time-series charts.

For example:
- X-axis → Date
- Y-axis → Medical parameter value
- Different medication periods should be clearly identifiable
- The graph should show whether the value increased or decreased over time

The system should also provide a tabular representation of the patient's condition and medication history.

Important: The system should describe observed trends and changes in the data rather than independently diagnosing the patient or claiming that a medication definitively caused a medical improvement or deterioration.

OBJECTIVE 3 – POWER BI HEALTHCARE ANALYTICS DASHBOARD

The third objective is to present all the extracted and analyzed information through an interactive Power BI dashboard.

The dashboard should provide a comprehensive view of the patient's medical information, including:

1. Patient Overview
   - Patient demographics
   - Current medical conditions
   - Current medications
   - Relevant medical history

2. Prescription Information
   - Extracted medicines
   - Dosage
   - Frequency
   - Duration
   - Prescription dates

3. Medical Report Trends
   - Blood glucose
   - HbA1c
   - Blood pressure
   - Cholesterol
   - Other relevant parameters depending on the patient

4. Medication Impact Timeline
   - Medication start date
   - Medication change date
   - Medical report dates
   - Parameter changes before and after medication changes

5. Allergies and Medication Safety
   - Known allergies
   - Reported adverse reactions, if available
   - Potential medication-allergy conflicts
   - Relevant medication warnings

6. Condition Monitoring
   - Condition being monitored
   - Current values
   - Historical values
   - Trend direction
   - Reference ranges

7. Visual Analytics
   - Line charts for medical parameter trends
   - Bar charts for comparisons
   - Tables for detailed medical information
   - KPI cards for important current values
   - Timeline showing medications and report dates
   - Filters for patient, date, condition, medication, and medical parameter

OVERALL SYSTEM WORKFLOW

The complete system should follow this workflow:

Handwritten Prescription/Image
        ↓
Gemini Vision AI
        ↓
Structured Prescription Data
        ↓
Patient Medical Database
        ↓
Medical/Lab Reports
        ↓
Data Extraction & Normalization
        ↓
Chronological Patient Timeline
        ↓
Before/After Medication Comparison
        ↓
Trend and Change Analysis
        ↓
Power BI Dashboard

The final system should combine AI-based prescription understanding, medical report data extraction, longitudinal patient data analysis, medication timeline comparison, allergy information, and interactive Power BI visualization into one integrated healthcare analytics system.

The goal is to help users and healthcare professionals understand a patient's historical medical data and visualize changes over time in a structured and understandable manner, while keeping the analysis evidence-based and avoiding unsupported medical diagnoses or treatment recommendations.