import json
import re
from typing import Any

FALLBACK = {
    "status": "VALID",
    "score": 7,
    "errors": ["parsing_error"],
    "correction_instruction": "aucune"
}

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_json_parse(text: str) -> Any:
    """Parse JSON retourné par LLM même s'il est mal formaté."""
    if not text:
        return FALLBACK

    text = re.sub(r"```(?:json)?", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

    text_upper = text.upper()
    if "INVALID" in text_upper:
        return {**FALLBACK, "status": "INVALID", "score": 3}
    if "VALID" in text_upper:
        return {**FALLBACK, "status": "VALID", "score": 7}

    return FALLBACK


def format_context(docs) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


def truncate_text(text: str, max_tokens: int = 2000) -> str:
    return text[:max_tokens]


def safe_llm_call(llm, prompt: str):
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Erreur LLM: {str(e)}"