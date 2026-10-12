"""Stage 3: Patient Voice History & Speech Agent.

Processes an audio recording of a patient orally describing their health
background. Supports fast Groq Whisper transcription and Gemini Multimodal
Audio. Handles code-mixed inputs (English, Hindi, Telugu, etc.), transcribes speech,
and runs Named Entity Recognition (NER) to extract structured medical background.

Output schema (strict):
{
  "raw_transcript": "string",
  "patient_reported_history": {
    "known_allergies": ["string"],
    "chronic_conditions": ["string"],
    "current_active_medications": ["string"],
    "past_reported_side_effects": [
      {"medication_or_class": "string", "reported_reaction": "string"}
    ]
  }
}
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    requests = None

try:
    from google import genai
except ImportError:
    genai = None

try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parents[2] / "config" / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except (ImportError, OSError):
    pass


class PatientVoiceHistoryAgent:
    """
    Stage 3 Agent: Audio transcription + medical NER.

    Transcribes patient's spoken history (Groq Whisper or Gemini Audio)
    and extracts structured entities (Allergies, Conditions, Medications, ADRs).
    """

    CANDIDATE_MODELS = [
        "gemini-2.5-flash",
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    def __init__(self, api_key: str | None = None, groq_api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        if not self.api_key or genai is None:
            return
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            print(f"[VoiceAgent] Gemini init warning: {e}")
            self.client = None

    @property
    def is_available(self) -> bool:
        return bool(self.groq_api_key or self.client is not None)

    # ----------------------------------------------------------------
    # Audio loading & MIME helper
    # ----------------------------------------------------------------

    @staticmethod
    def _load_audio(audio_path: str | Path) -> bytes:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return path.read_bytes()

    @staticmethod
    def _mime_type(audio_path: str | Path | None = None, audio_bytes: bytes | None = None) -> str:
        if audio_bytes and len(audio_bytes) >= 4:
            if audio_bytes[:4] == b"RIFF":
                return "audio/wav"
            if audio_bytes[:4] == b"\x1aE\xdf\xa3":
                return "audio/webm"
            if audio_bytes[:4] == b"OggS":
                return "audio/ogg"
            if audio_bytes[:3] == b"ID3" or audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
                return "audio/mpeg"
            if b"ftyp" in audio_bytes[:16]:
                return "audio/mp4"

        if audio_path:
            suffix = Path(audio_path).suffix.lower().lstrip(".")
            mapping = {
                "wav": "audio/wav",
                "mp3": "audio/mpeg",
                "m4a": "audio/mp4",
                "aac": "audio/aac",
                "ogg": "audio/ogg",
                "flac": "audio/flac",
                "webm": "audio/webm",
            }
            if suffix in mapping:
                return mapping[suffix]

        return "audio/wav"

    # ----------------------------------------------------------------
    # Transcription Engines (Groq Whisper + Gemini Audio)
    # ----------------------------------------------------------------

    def _transcribe_groq(self, audio_bytes: bytes, mime: str | None = None) -> str:
        """Fast transcription via Groq Whisper API (whisper-large-v3)."""
        key = self.groq_api_key or os.getenv("GROQ_API_KEY")
        if not key or requests is None or not audio_bytes:
            return ""

        ext = "wav"
        if mime == "audio/webm":
            ext = "webm"
        elif mime == "audio/mpeg":
            ext = "mp3"
        elif mime == "audio/ogg":
            ext = "ogg"
        elif mime == "audio/mp4":
            ext = "m4a"

        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (f"recording.{ext}", audio_bytes, mime or "audio/wav")},
                data={"model": "whisper-large-v3", "response_format": "json"},
                timeout=30,
            )
            if resp.status_code == 200:
                text = resp.json().get("text", "").strip()
                if text:
                    return text
            else:
                print(f"[VoiceAgent] Groq Whisper status {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            print(f"[VoiceAgent] Groq transcription error: {e}")
        return ""

    def transcribe_audio_bytes(self, audio_bytes: bytes, mime: str | None = None) -> str:
        """Transcribe raw audio bytes using Groq Whisper or Gemini Audio API."""
        if not audio_bytes:
            return ""

        resolved_mime = mime or self._mime_type(audio_bytes=audio_bytes)

        # 1. Try Groq Whisper (ultra-fast, zero-latency, high quota)
        groq_result = self._transcribe_groq(audio_bytes, mime=resolved_mime)
        if groq_result:
            return groq_result

        # 2. Try Gemini Audio Multimodal API
        if self.client is not None and genai is not None:
            prompt = (
                "You are a medical speech-to-text expert. "
                "Transcribe this spoken audio verbatim into clean, accurate English text. "
                "Preserve all medical terms, drug names, and reported symptoms accurately. "
                "Return ONLY the verbatim transcript text without preamble, quotes, or markdown formatting."
            )
            for model_name in self.CANDIDATE_MODELS:
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=[
                            genai.types.Part.from_bytes(data=audio_bytes, mime_type=resolved_mime),
                            prompt,
                        ],
                    )
                    if response and response.text:
                        text = response.text.strip()
                        if text:
                            return text
                except Exception as e:
                    print(f"[VoiceAgent] Gemini transcription error with {model_name}: {str(e)[:100]}")
                    continue

        return ""

    def transcribe_audio(self, audio_path: str | Path) -> str:
        """Transcribe an audio file path directly into text."""
        try:
            audio_bytes = self._load_audio(audio_path)
            mime = self._mime_type(audio_path)
            return self.transcribe_audio_bytes(audio_bytes, mime=mime)
        except Exception:
            return ""

    # ----------------------------------------------------------------
    # Main Entry Point: Transcribe + Clinical NER Extraction
    # ----------------------------------------------------------------

    def transcribe_and_extract(self, audio_source: str | Path | bytes) -> dict[str, Any]:
        """
        Stage 3 pipeline:
        1. Load audio bytes.
        2. Transcribe using Groq Whisper or Gemini.
        3. Extract structured medical background (Allergies, Conditions, Meds, Reactions).
        4. Return strict Stage 3 JSON.
        """
        try:
            if isinstance(audio_source, (str, Path)):
                audio_bytes = self._load_audio(audio_source)
                mime = self._mime_type(audio_source, audio_bytes=audio_bytes)
            else:
                audio_bytes = audio_source
                mime = self._mime_type(audio_bytes=audio_bytes)
        except Exception:
            return self._offline_fallback("")

        if not audio_bytes:
            return self._offline_fallback("")

        # 1. Transcribe speech
        transcript = self.transcribe_audio_bytes(audio_bytes, mime=mime)

        if not transcript:
            return self._offline_fallback("")

        # 2. Extract structured clinical history from transcript
        extracted_data = self._extract_entities_from_transcript(transcript)
        extracted_data["raw_transcript"] = transcript
        return self._validate_schema(extracted_data)

    def _extract_entities_from_transcript(self, transcript: str) -> dict[str, Any]:
        """Extract structured medical entities using Gemini or rule engine."""
        prompt = f"""You are a multilingual medical transcription and clinical NER specialist.

PATIENT SPOKEN TRANSCRIPT:
"{transcript}"

TASK: Extract structured medical background entities from the transcript.
Return ONLY a valid JSON object (no markdown fences, no preamble) with this EXACT schema:
{{
  "raw_transcript": "{transcript}",
  "patient_reported_history": {{
    "known_allergies": ["list of reported drug/food allergies, e.g. 'Penicillin', 'Peanuts'"],
    "chronic_conditions": ["list of chronic diseases, e.g. 'Type 2 Diabetes', 'Hypertension'"],
    "current_active_medications": ["list of medicines the patient says they are currently taking, e.g. 'Metformin 500mg'"],
    "past_reported_side_effects": [
      {{"medication_or_class": "drug name or class, e.g. 'Aspirin'", "reported_reaction": "e.g. 'Stomach bleeding, gastric ulcer'"}}
    ]
  }}
}}

Rules:
- If the patient did NOT mention a category, return an empty list [].
- Do NOT invent or hallucinate medical information.
- Normalize drug names when possible."""

        if self.client is not None and genai is not None:
            for model_name in self.CANDIDATE_MODELS:
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    raw_text = response.text.strip() if response and response.text else ""
                    if raw_text:
                        data = self._parse_json_response(raw_text)
                        if data and "patient_reported_history" in data:
                            return data
                except Exception as e:
                    print(f"[VoiceAgent] NER extraction error with {model_name}: {str(e)[:100]}")
                    continue

        # Rule-based NER fallback
        return self._extract_entities_rule_based(transcript)

    @staticmethod
    def _extract_entities_rule_based(transcript: str) -> dict[str, Any]:
        """Rule-based clinical entity extractor for patient voice transcripts."""
        text_lower = transcript.lower()

        allergies: list[str] = []
        allergy_keywords = [
            ("penicillin", "Penicillin"),
            ("amoxicillin", "Amoxicillin"),
            ("sulfa", "Sulfa Drugs"),
            ("sulfonamide", "Sulfa Drugs"),
            ("aspirin", "Aspirin"),
            ("nsaid", "NSAIDs"),
            ("ibuprofen", "Ibuprofen"),
            ("peanut", "Peanuts"),
            ("latex", "Latex"),
            ("codeine", "Codeine"),
            ("paracetamol", "Paracetamol"),
        ]
        for kw, standard_name in allergy_keywords:
            if (
                kw in text_lower
                and any(term in text_lower for term in ["allergic", "allergy", "reaction", "hives", "rash", "itching"])
                and standard_name not in allergies
            ):
                allergies.append(standard_name)

        conditions: list[str] = []
        condition_keywords = [
            ("diabetes", "Type 2 Diabetes"),
            ("sugar", "Diabetes"),
            ("bp", "Hypertension"),
            ("blood pressure", "Hypertension"),
            ("hypertension", "Hypertension"),
            ("asthma", "Bronchial Asthma"),
            ("thyroid", "Hypothyroidism"),
            ("heart", "Ischemic Heart Disease"),
            ("kidney", "Chronic Kidney Disease"),
            ("gastric", "GERD / Gastritis"),
            ("ulcer", "Peptic Ulcer Disease"),
            ("arthritis", "Osteoarthritis"),
        ]
        for kw, standard_name in condition_keywords:
            if kw in text_lower and standard_name not in conditions:
                conditions.append(standard_name)

        medications: list[str] = []
        med_keywords = [
            "metformin",
            "glimepiride",
            "atorvastatin",
            "amlodipine",
            "telmisartan",
            "losartan",
            "pantoprazole",
            "omeprazole",
            "levothyroxine",
            "paracetamol",
            "augmentin",
            "azithromycin",
            "aspirin",
            "ibuprofen",
        ]
        for m in med_keywords:
            if m in text_lower:
                medications.append(m.title())

        side_effects: list[dict[str, str]] = []
        if "stomach" in text_lower and any(k in text_lower for k in ["pain", "burning", "acid", "ulcer"]):
            side_effects.append({"medication_or_class": "NSAIDs / Aspirin", "reported_reaction": "Gastric irritation"})
        if "rash" in text_lower or "itching" in text_lower:
            side_effects.append({"medication_or_class": "Antibiotics", "reported_reaction": "Cutaneous rash"})

        return {
            "raw_transcript": transcript,
            "patient_reported_history": {
                "known_allergies": allergies,
                "chronic_conditions": conditions,
                "current_active_medications": medications,
                "past_reported_side_effects": side_effects,
            },
        }

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _parse_json_response(raw_text: str) -> dict[str, Any] | None:
        """Extract JSON from model response (tolerates markdown fences)."""
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
            return None

    @staticmethod
    def _validate_schema(data: dict[str, Any]) -> dict[str, Any]:
        """Ensure the output matches the strict Stage 3 schema."""
        history = data.get("patient_reported_history", {}) or {}
        return {
            "raw_transcript": str(data.get("raw_transcript", "")),
            "patient_reported_history": {
                "known_allergies": [str(a) for a in (history.get("known_allergies") or [])],
                "chronic_conditions": [str(c) for c in (history.get("chronic_conditions") or [])],
                "current_active_medications": [str(m) for m in (history.get("current_active_medications") or [])],
                "past_reported_side_effects": [
                    {
                        "medication_or_class": str(se.get("medication_or_class", "")),
                        "reported_reaction": str(se.get("reported_reaction", "")),
                    }
                    for se in (history.get("past_reported_side_effects") or [])
                    if isinstance(se, dict)
                ],
            },
        }

    @staticmethod
    def _offline_fallback(audio_source: str | Path | bytes) -> dict[str, Any]:
        """Return a safe empty structure when audio transcription is unavailable."""
        return {
            "raw_transcript": "",
            "patient_reported_history": {
                "known_allergies": [],
                "chronic_conditions": [],
                "current_active_medications": [],
                "past_reported_side_effects": [],
            },
        }
