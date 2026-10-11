import os
from pathlib import Path

import pytest
from PIL import Image

from src.modules.prescription_vision import PrescriptionVisionExtractor


def test_vision_extractor_availability():
    extractor = PrescriptionVisionExtractor()
    # If API key is present in environment or config/.env, is_available must be True
    has_key = bool(os.getenv("GEMINI_API_KEY"))
    if not has_key:
        env_path = Path(__file__).resolve().parents[1] / "config" / ".env"
        if env_path.exists():
            with open(env_path) as f:
                has_key = any(
                    line.startswith("GEMINI_API_KEY=") and len(line.split("=", 1)[1].strip()) > 5 for line in f
                )

    if has_key:
        assert extractor.is_available is True


def test_vision_extractor_on_sample_prescription():
    extractor = PrescriptionVisionExtractor()
    if not extractor.is_available:
        pytest.skip("Gemini API key not configured, skipping live vision test.")

    sample_path = Path(__file__).resolve().parents[1] / "data" / "prescriptions" / "sample_doctor_rx.png"
    if not sample_path.exists():
        pytest.skip("sample_doctor_rx.png not found")

    image = Image.open(sample_path)
    result = extractor.extract(image)

    assert result["status"] == "success"
    assert len(result["medicines"]) >= 3

    # Check for presence of key drugs from the sample image
    med_names = " ".join([m["medicine_name"].lower() for m in result["medicines"]])
    assert any(x in med_names for x in ["augmentin", "amoxicillin"])
    assert any(x in med_names for x in ["dolo", "paracetamol"])

    # Verify that dosage, frequency, and duration fields are populated
    for med in result["medicines"]:
        assert "dosage" in med
        assert "frequency" in med
        assert "duration" in med
