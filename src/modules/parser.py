import io
import os
import re
from typing import Any

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


class ClinicalDocumentParser:
    """
    Parses unstructured and semi-structured patient medical documents
    (PDF reports, clinical text notes, discharge summaries).
    Extracts patient metadata, vitals, lab biomarkers, medications, allergies, and diagnoses.
    """

    def __init__(self):
        # Biomarker extraction regex patterns
        self.biomarker_patterns = {
            "fasting_blood_glucose": re.compile(
                r"(?:fasting\s+(?:blood\s+)?(?:glucose|sugar)|fbs|fbg)(?:\s*\([^)]*\))?[\s:=-]+(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            "postprandial_blood_glucose": re.compile(
                r"(?:post\s*prandial\s+(?:blood\s+)?(?:glucose|sugar)|ppbs|ppbg)(?:\s*\([^)]*\))?[\s:=-]+(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            "hba1c": re.compile(
                r"(?:hba1c|glycated\s+hemoglobin|glycosylated\s+hemoglobin)(?:\s*\([^)]*\))?[\s:=-]+(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            "serum_creatinine": re.compile(
                r"(?:serum\s+creatinine|creatinine)(?:\s*\([^)]*\))?[\s:=-]+(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            "blood_urea": re.compile(
                r"(?:blood\s+urea(?:\s+nitrogen)?|bun|urea)(?:\s*\([^)]*\))?[\s:=-]+(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            "total_cholesterol": re.compile(
                r"(?:total\s+cholesterol|cholesterol\s+total)(?:\s*\([^)]*\))?[\s:=-]+(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            "ldl_cholesterol": re.compile(
                r"(?:ldl(?:\s+cholesterol)?|low\s+density\s+lipoprotein)(?:\s*\([^)]*\))?[\s:=-]+(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            "hdl_cholesterol": re.compile(
                r"(?:hdl(?:\s+cholesterol)?|high\s+density\s+lipoprotein)(?:\s*\([^)]*\))?[\s:=-]+(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            "triglycerides": re.compile(
                r"(?:triglycerides|tg)(?:\s*\([^)]*\))?[\s:=-]+(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            "hemoglobin": re.compile(
                r"(?<!glycated\s)(?<!glycosylated\s)\b(?:hemoglobin|hb)\b(?:\s*\([^)]*\))?[\s:=-]+(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            "alt_sgpt": re.compile(
                r"(?:alt|sgpt|alanine\s+aminotransferase)(?:\s*\([^)]*\))?[\s:=-]+(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            "ast_sgot": re.compile(
                r"(?:ast|sgot|aspartate\s+aminotransferase)(?:\s*\([^)]*\))?[\s:=-]+(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            "platelets": re.compile(
                r"(?:platelet\s+count|platelets)(?:\s*\([^)]*\))?[\s:=-]+(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
            "bmi": re.compile(
                r"(?:body\s+mass\s+index|bmi)(?:\s*\([^)]*\))?[\s:=-]+(\d+(?:\.\d+)?)",
                re.IGNORECASE,
            ),
        }

        self.bp_pattern = re.compile(
            r"(?:bp|blood\s+pressure)\s*[:=-]?\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mm\s*hg)?",
            re.IGNORECASE,
        )

        self.age_pattern = re.compile(
            r"(?:age|years\s+old|yrs\s+old)\s*[:=-]?\s*(\d{1,3})\s*(?:years|yrs|y)?",
            re.IGNORECASE,
        )

        self.gender_pattern = re.compile(
            r"(?:gender|sex)\s*[:=-]?\s*(male|female|other|m|f)\b",
            re.IGNORECASE,
        )

        self.date_pattern = re.compile(
            r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
            re.IGNORECASE,
        )

        self.common_diagnoses = [
            "type 2 diabetes",
            "type 1 diabetes",
            "diabetes mellitus",
            "prediabetes",
            "hypertension",
            "coronary artery disease",
            "myocardial infarction",
            "heart failure",
            "asthma",
            "copd",
            "chronic kidney disease",
            "renal impairment",
            "fatty liver",
            "cirrhosis",
            "peptic ulcer",
            "hypothyroidism",
            "hyperthyroidism",
            "dyslipidemia",
            "hypercholesterolemia",
        ]

    def extract_text_from_file(self, file_source: str | bytes | io.BytesIO, filename: str = "") -> str:
        """Extracts plain text from a filepath, raw bytes, or BytesIO buffer."""
        # Handle file paths or direct text content
        if isinstance(file_source, str):
            lower = file_source.lower()
            if lower.endswith(".pdf"):
                return self._extract_text_from_pdf(file_source)
            elif os.path.exists(file_source):
                with open(file_source, encoding="utf-8", errors="ignore") as f:
                    return f.read()
            else:
                return file_source

        # Handle uploaded BytesIO or bytes
        if isinstance(file_source, (bytes, io.BytesIO)):
            buffer = file_source if isinstance(file_source, io.BytesIO) else io.BytesIO(file_source)
            if filename.lower().endswith(".pdf") or buffer.getvalue().startswith(b"%PDF"):
                return self._extract_text_from_pdf(buffer)
            else:
                try:
                    return buffer.getvalue().decode("utf-8")
                except UnicodeDecodeError:
                    return buffer.getvalue().decode("latin-1", errors="ignore")

        return ""

    def _extract_text_from_pdf(self, pdf_input: str | io.BytesIO) -> str:
        text_parts = []
        if pdfplumber is not None:
            try:
                with pdfplumber.open(pdf_input) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text_parts.append(extracted)
                        # Also extract tabular data if present
                        tables = page.extract_tables()
                        for table in tables:
                            for row in table:
                                if row:
                                    clean_row = [str(cell).strip() for cell in row if cell]
                                    if clean_row:
                                        text_parts.append(" | ".join(clean_row))
                return "\n".join(text_parts)
            except Exception as e:
                print(f"[Parser] pdfplumber extraction failed: {e}")
        return "\n".join(text_parts)

    def parse_document(self, text: str, doc_name: str = "Document") -> dict[str, Any]:
        """
        Extracts structured biomarkers, vitals, patient metadata, and clinical entities
        from unstructured clinical text.
        """
        biomarkers = {}
        for key, pattern in self.biomarker_patterns.items():
            match = pattern.search(text)
            if match:
                try:
                    biomarkers[key] = float(match.group(1))
                except (ValueError, IndexError):
                    pass

        # Blood pressure
        bp_match = self.bp_pattern.search(text)
        systolic_bp = None
        diastolic_bp = None
        if bp_match:
            try:
                systolic_bp = int(bp_match.group(1))
                diastolic_bp = int(bp_match.group(2))
            except (ValueError, IndexError):
                pass

        # Demographics
        age_match = self.age_pattern.search(text)
        try:
            age = int(age_match.group(1)) if age_match else None
        except (ValueError, TypeError):
            age = None

        gender_match = self.gender_pattern.search(text)
        gender = gender_match.group(1).capitalize() if gender_match else "Unknown"
        if gender.upper() == "M":
            gender = "Male"
        elif gender.upper() == "F":
            gender = "Female"

        # Dates
        dates = self.date_pattern.findall(text)
        primary_date = dates[0] if dates else "Undated"

        # Known diagnoses in text
        detected_diagnoses = []
        lower_text = text.lower()
        for diag in self.common_diagnoses:
            if re.search(rf"\b{re.escape(diag)}\b", lower_text):
                detected_diagnoses.append(diag.title())

        # Allergies detection
        allergies = self._extract_allergies(text)

        # Current Medications detection
        medications = self._extract_medications(text)

        return {
            "document_name": doc_name,
            "date": primary_date,
            "patient": {
                "age": age,
                "gender": gender,
            },
            "vitals": {
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "bmi": biomarkers.get("bmi"),
            },
            "biomarkers": biomarkers,
            "diagnoses": detected_diagnoses,
            "allergies": allergies,
            "medications": medications,
            "raw_text": text,
        }

    def _extract_allergies(self, text: str) -> list[str]:
        allergies = []
        allergy_match = re.search(
            r"(?:allerg(?:y|ies)|allergic\s+to)\s*[:=-]?\s*([^\n.;]+)",
            text,
            re.IGNORECASE,
        )
        if allergy_match:
            raw_allergies = allergy_match.group(1)
            parts = re.split(r"[,;/]\s*", raw_allergies)
            for part in parts:
                cleaned = part.strip().title()
                if cleaned and cleaned.lower() not in ["none", "nil", "n/a", "no known allergies", "nkda"]:
                    allergies.append(cleaned)
        return allergies

    def _extract_medications(self, text: str) -> list[str]:
        meds = []
        med_section = re.search(
            r"(?:current\s+medications?|medications?|rx|prescriptions?)\s*[:=-]?\s*([\s\S]*?)(?=\n\s*(?:allergies|diagnosis|lab|history|plan|tests|impression)|\Z)",
            text,
            re.IGNORECASE,
        )
        if med_section:
            lines = med_section.group(1).strip().splitlines()
            for line in lines:
                line = re.sub(r"^[-*•\d.)\s]+", "", line).strip()
                if 2 <= len(line) <= 60:
                    meds.append(line)
        return meds[:8]

    def parse_multiple(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Parses a list of uploaded documents [{"content": ..., "filename": ...}]."""
        results = []
        for doc in docs:
            content = doc.get("content", "")
            filename = doc.get("filename", "Record")
            text = self.extract_text_from_file(content, filename)
            parsed = self.parse_document(text, doc_name=filename)
            results.append(parsed)
        return results
