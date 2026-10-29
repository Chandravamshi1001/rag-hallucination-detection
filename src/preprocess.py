import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Apply safe normalization for hallucination detection.

    Important: Keep capitalization, punctuation, numbers, dates,
    names, and negation because they can determine whether a claim
    is supported by the source evidence.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()