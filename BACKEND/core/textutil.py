import unicodedata

def norm(text: str) -> str:
    """Minuscules, sans accents, apostrophes droites."""
    text = (text or "").lower().replace("’", "'")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))
