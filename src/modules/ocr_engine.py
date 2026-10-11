import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Import our preprocessing class.
#
# This also keeps preprocessing and OCR separate.
from .preprocessor import PrescriptionPreprocessor


class HandwrittenOCREngine:
    """
    Handwritten OCR engine using Microsoft's TrOCR model.

    Model:
        microsoft/trocr-base-handwritten

    Pipeline:
        PIL image
            ↓
        preprocessing
            ↓
        line segmentation
            ↓
        TrOCR
            ↓
        text
    """

    def __init__(self, model_name="microsoft/trocr-base-handwritten", device=None):

        print("[OCR] Initializing handwritten OCR engine...")

        # --------------------------------------------------------
        # DEVICE
        # --------------------------------------------------------

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"[OCR] Device: {self.device}")

        # --------------------------------------------------------
        # LOAD PROCESSOR
        # --------------------------------------------------------

        print(f"[OCR] Loading processor: {model_name}")

        from typing import Any

        self.processor: Any = TrOCRProcessor.from_pretrained(model_name)

        # --------------------------------------------------------
        # LOAD MODEL
        # --------------------------------------------------------

        print(f"[OCR] Loading model: {model_name}")

        self.model: Any = VisionEncoderDecoderModel.from_pretrained(model_name)

        self.model.to(self.device)

        self.model.eval()

        # --------------------------------------------------------
        # PREPROCESSOR
        # --------------------------------------------------------

        self.preprocessor = PrescriptionPreprocessor()

        print("[OCR] Handwritten OCR engine ready.")

    # ============================================================
    # CLEAN OCR TEXT
    # ============================================================

    def clean_text(self, text):
        """
        Basic cleanup of OCR output.

        Does NOT try to guess medicine names.
        """

        if text is None:
            return ""

        text = str(text)

        # Remove unnecessary whitespace.
        text = " ".join(text.split())

        return text.strip()

    # ============================================================
    # RECOGNIZE ONE LINE
    # ============================================================

    def recognize_line(self, line_image):
        """
        Run TrOCR on a single handwritten line.
        """

        if line_image is None:
            return ""

        if not isinstance(line_image, Image.Image):
            line_image = Image.fromarray(line_image)

        line_image = line_image.convert("RGB")

        # --------------------------------------------------------
        # TrOCR PROCESSOR
        # --------------------------------------------------------

        pixel_values = self.processor(images=line_image, return_tensors="pt").pixel_values

        pixel_values = pixel_values.to(self.device)

        # --------------------------------------------------------
        # GENERATE TEXT
        # --------------------------------------------------------

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=64, num_beams=4, early_stopping=True)

        # --------------------------------------------------------
        # DECODE
        # --------------------------------------------------------

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return self.clean_text(text)

    # ============================================================
    # RECOGNIZE MULTIPLE LINES
    # ============================================================

    def recognize_lines(self, line_images, debug=False):
        """
        OCR every detected handwritten line.

        Returns:
            list of dictionaries
        """

        results = []

        for index, line_image in enumerate(line_images, start=1):
            if debug:
                print(f"[OCR] Processing line {index}/{len(line_images)}")

            try:
                text = self.recognize_line(line_image)

            except Exception as exc:
                print(f"[OCR] Line {index} failed: {exc}")

                text = ""

            results.append({"line": index, "text": text})

            if debug:
                print(f"[OCR] Line {index}: {text}")

        return results

    # ============================================================
    # COMPLETE PRESCRIPTION OCR
    # ============================================================

    def process(self, pil_image, debug=False):
        """
        Complete prescription OCR pipeline.

        Returns:

            {
                "preprocessed_image": PIL.Image,
                "lines": [...],
                "text": "..."
            }
        """

        if pil_image is None:
            raise ValueError("Prescription image is None.")

        # --------------------------------------------------------
        # STEP 1
        # PREPROCESS
        # --------------------------------------------------------

        if debug:
            print("[OCR] Step 1: preprocessing")

        cleaned_image = self.preprocessor.preprocess(pil_image)

        # --------------------------------------------------------
        # STEP 2
        # LINE SEGMENTATION
        # --------------------------------------------------------

        if debug:
            print("[OCR] Step 2: detecting handwritten lines")

        line_images = self.preprocessor.extract_text_lines(cleaned_image, debug=debug)

        if debug:
            print(f"[OCR] Total detected lines: {len(line_images)}")

        # --------------------------------------------------------
        # STEP 3
        # OCR
        # --------------------------------------------------------

        if debug:
            print("[OCR] Step 3: handwritten OCR")

        line_results = self.recognize_lines(line_images, debug=debug)

        # --------------------------------------------------------
        # STEP 4
        # COMBINE TEXT
        # --------------------------------------------------------

        valid_text = []

        for result in line_results:
            text = result.get("text", "").strip()

            if text:
                valid_text.append(text)

        full_text = "\n".join(valid_text)

        # --------------------------------------------------------
        # RESULT
        # --------------------------------------------------------

        return {
            "preprocessed_image": cleaned_image,
            "line_images": line_images,
            "lines": line_results,
            "text": full_text,
        }

    # ============================================================
    # ALIAS: extract_text
    # ============================================================

    def extract_text(self, pil_image, debug=False):
        """
        Convenience method.

        Returns only the complete OCR text.
        """

        result = self.process(pil_image, debug=debug)

        return result["text"]

    # ============================================================
    # ALIAS: recognize
    # ============================================================

    def recognize(self, pil_image, debug=False):
        """
        Alias for extract_text().
        """

        return self.extract_text(pil_image, debug=debug)

    # ============================================================
    # ALIAS: read
    # ============================================================

    def read(self, line_image):
        """
        Compatibility wrapper used by the Streamlit app.

        Expects a single processed line image and returns the OCR result text.
        """

        return self.recognize_line(line_image)


# ================================================================
# BACKWARD COMPATIBILITY
# ================================================================

# Older imports/usage in the app expect these names to exist.
OCREngine = HandwrittenOCREngine
__all__ = ["HandwrittenOCREngine", "OCREngine", "PrescriptionPreprocessor"]
