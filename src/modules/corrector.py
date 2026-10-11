import re

from rapidfuzz import fuzz, process


class DrugNameCorrector:
    def __init__(self, drug_names, top_k: int = 3, min_score_to_show: float = 45.0):
        """
        drug_names: list[str] of candidate medicine names (e.g. from
                    drug_dictionary.load_drug_table()["name"].tolist())
        top_k: how many ranked suggestions to return per line
        min_score_to_show: below this, don't bother suggesting anything -
                    it's indistinguishable from noise (see docstring above)
        """
        self.drug_names = list(drug_names)
        self.top_k = top_k
        self.min_score_to_show = min_score_to_show

    def suggest(self, ocr_text: str):
        """Returns [(candidate_name, score), ...] sorted best-first, or [] if
        the OCR text is empty/too short or nothing clears min_score_to_show."""
        cleaned = (ocr_text or "").strip()
        if len(cleaned) < 2 or not self.drug_names:
            return []

        matches = process.extract(cleaned, self.drug_names, scorer=fuzz.token_sort_ratio, limit=self.top_k)
        return [(name, round(score, 1)) for name, score, _ in matches if score >= self.min_score_to_show]


# --- Structured field extraction (dosage / frequency / duration) ---------------
# Same patterns you already had in app.py, pulled out here so both the scanner
# tab and any future batch/report processing can reuse them.

DOSAGE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|tablet|tab|capsule|cap)\b", re.IGNORECASE)
FREQUENCY_PATTERN = re.compile(
    r"\b\d[-/]\d[-/]\d\b|once daily|twice daily|three times daily|\bod\b|\bbd\b|\btid\b|\bhs\b",
    re.IGNORECASE,
)
DURATION_PATTERN = re.compile(r"\b\d+\s?(?:day|days|week|weeks|month|months)\b", re.IGNORECASE)


def looks_like_name_candidate(ocr_text: str, min_remaining_letter_ratio: float = 0.4) -> bool:
    """
    Rough filter so we don't fuzzy-match instruction lines against the drug
    dictionary. A line like "1 tablet twice daily for 5 days" isn't a medicine
    name, and searching for it in the dictionary only produces noise (tested:
    it matched real-looking but meaningless drug names at 55-60% confidence).

    Strips out dosage/frequency/duration wording first; if most of the line's
    remaining letters are something else, it's probably a name line worth
    fuzzy-matching. This is an imperfect heuristic, not a classifier - it will
    occasionally let a pure instruction line through, but it cuts down on the
    obvious cases for free.
    """
    text = ocr_text or ""
    stripped = text
    for pattern in (DOSAGE_PATTERN, FREQUENCY_PATTERN, DURATION_PATTERN):
        stripped = pattern.sub("", stripped)

    total_letters = sum(c.isalpha() for c in text)
    if total_letters == 0:
        return False

    remaining_letters = sum(c.isalpha() for c in stripped)
    return (remaining_letters / total_letters) > min_remaining_letter_ratio


def extract_fields(ocr_text: str) -> dict:
    text = ocr_text or ""
    dosage_match = DOSAGE_PATTERN.search(text)
    frequency_match = FREQUENCY_PATTERN.search(text)
    duration_match = DURATION_PATTERN.search(text)

    return {
        "Possible Dosage": dosage_match.group(0) if dosage_match else "Not detected",
        "Possible Frequency": frequency_match.group(0) if frequency_match else "Not detected",
        "Possible Duration": duration_match.group(0) if duration_match else "Not detected",
    }
