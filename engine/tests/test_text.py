"""Text pipeline tests: engine output equals the CLI functions (read-only reference)."""

import sys
from pathlib import Path

import pytest

from jeuvoice_engine.text import clean, plan_parts

FIXTURES = Path(__file__).parent / "fixtures"
TTS_SCRIPTS = Path(r"D:/MLProjelerim/tts_lab/scripts")


def cli_functions():
    """Import the CLI functions read-only. Skip when unavailable (no folder or no deps)."""
    if not TTS_SCRIPTS.exists():
        pytest.skip(f"reference folder missing: {TTS_SCRIPTS}")
    # render_script_gguf pulls tts_lab-only imports (normalizer_tr, librosa) that
    # the split/clean functions never call. Stub them so the reference stays importable.
    import sys as _sys
    import types as _types

    if "normalizer_tr" not in _sys.modules:
        try:
            import normalizer_tr  # noqa: F401
        except ImportError:
            _stub = _types.ModuleType("normalizer_tr")

            class _Normalizer:
                def __init__(self, *a, **k):
                    raise RuntimeError("stub")

            _stub.Normalizer = _Normalizer
            _sys.modules["normalizer_tr"] = _stub
    if "librosa" not in _sys.modules:
        try:
            import librosa  # noqa: F401
        except ImportError:
            _sys.modules["librosa"] = _types.ModuleType("librosa")
    sys.path.insert(0, str(TTS_SCRIPTS))
    try:
        import try_omnivoice
        from render_script_gguf import split_paragraphs
    except ImportError as e:
        pytest.skip(f"reference deps missing: {e}")
    return try_omnivoice, split_paragraphs


def cli_parts(script, en_terms, quote_terms, respell):
    try_omnivoice, split_paragraphs = cli_functions()
    out = []
    index = 1
    for para, group in split_paragraphs(script):
        for lang, text in try_omnivoice.split_by_language(group, en_terms):
            text = try_omnivoice.clean(text, lang, quote_terms)
            for old, new in respell:
                text = text.replace(old, new)
            out.append((index, para, lang, text))
            index += 1
    return out


@pytest.mark.parametrize("name,en_terms,quote_terms,respell", [
    ("script_tr.txt", [], [], []),
    ("script_mixed.txt", ["tünel"], ["The Last of Us"], []),
    ("script_two_para.txt", [], [], [("ellie", "elif")]),
])
def test_plan_parts_match_cli(name, en_terms, quote_terms, respell):
    script = (FIXTURES / name).read_text(encoding="utf-8")
    expected = cli_parts(script, en_terms, quote_terms, respell)
    got = [(p.index, p.paragraph, p.language, p.text) for p in plan_parts(script, en_terms, quote_terms, respell)]
    assert got == expected


def test_quote_term_stays_quoted_in_en_part():
    parts = plan_parts(
        "Joel sabah tünele giriyor. The Last of Us'ta bu sahne herkesin aklında kaldı.",
        ["the last of us"],
        ["The Last of Us"],
        [],
    )
    en = [p for p in parts if p.language == "en"]
    assert en, "the quoted sentence is an en part"
    assert '"The Last of Us"' in en[0].text


def test_turkish_dotted_capitals():
    assert clean("İSTANBUL IŞIK", "tr") == "istanbul ışık"


def test_respell_key_is_cleaned_like_the_text():
    # The app sends the script word as written. The text is lowercase and has no
    # apostrophe inside a word, so the key must get the same cleaning.
    parts = plan_parts("Dün Invincible'da yeni sezon.", [], [], [("Invincible'da", "invinsıbılda")])
    assert "invinsıbılda" in parts[0].text
    assert "invincible" not in parts[0].text


def test_respell_key_with_turkish_capitals():
    parts = plan_parts("İstanbul çok güzel.", [], [], [("İstanbul", "istambul")])
    assert parts[0].text.startswith("istambul")


def test_respell_changes_whole_words_only():
    parts = plan_parts("Bir gün birlikte gittik.", [], [], [("bir", "bi")])
    assert parts[0].text == "bi gün birlikte gittik."
