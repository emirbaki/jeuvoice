"""Engine text pipeline (port of the CLI): script -> render parts.

Behavior = `tts_lab/scripts/try_omnivoice.py` with `--by-paragraph`:
`split_paragraphs`, `split_by_language`, `clean` (trnorm normalize,
apostrophe rule, I→ı, İ→i, lowercase, quote terms), then respell
`OLD=NEW` on the cleaned text. Do not import tts_lab at runtime.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Part:
    """One model input. `index` is 1-based, as in the CLI."""

    index: int
    paragraph: int
    language: str
    text: str


def split_paragraphs(text: str) -> list[tuple[int, str]]:
    """[(paragraph index, whole paragraph)]: one model input per paragraph."""
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    return list(enumerate(paragraphs))


def split_by_language(group: str, en_terms: list[str]) -> list[tuple[str, str]]:
    """Yield (language, text). Sentences with an English term stay apart."""
    parts: list[tuple[str, str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", group.strip()):
        lang = "en" if any(term.lower() in sentence.lower() for term in en_terms) else "tr"
        if parts and parts[-1][0] == lang:
            parts[-1] = (lang, f"{parts[-1][1]} {sentence}")
        else:
            parts.append((lang, sentence))
    return parts


def clean(text: str, lang: str, quote_terms: tuple[str, ...] = ()) -> str:
    """Lowercase Turkish-safe cleaning, exactly as in the CLI."""
    from trnorm import normalize

    text = normalize(text)
    if lang == "tr":  # an English title keeps its apostrophe: "us'ta" must not become "usta"
        text = re.sub(r"(?<=\w)['’](?=\w)", "", text)
    text = text.replace("I", "ı").replace("İ", "i").lower()
    for term in quote_terms:
        if lang == "en":
            text = text.replace(term.lower(), f'"{term}"')
    return text


def plan_parts(
    script: str,
    en_terms: list[str],
    quote_terms: list[str],
    respell: list[tuple[str, str]],
) -> list[Part]:
    """Split `script` into render parts (paragraph mode)."""
    parts: list[Part] = []
    index = 1
    for para, group in split_paragraphs(script):
        for lang, text in split_by_language(group, en_terms):
            text = clean(text, lang, tuple(quote_terms))
            for old, new in respell:
                text = text.replace(old, new)
            parts.append(Part(index=index, paragraph=para, language=lang, text=text))
            index += 1
    return parts
