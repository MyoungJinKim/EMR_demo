from __future__ import annotations

from mediqa_qwen_eval.postprocess import postprocess_note


def test_postprocess_only_removes_wrappers() -> None:
    raw = "```text\nClinical note: **HISTORY OF PRESENT ILLNESS**\n\nPatient has cough.\n```"
    assert postprocess_note(raw) == "HISTORY OF PRESENT ILLNESS\n\nPatient has cough."


def test_postprocess_trims_repeated_sentence_tail() -> None:
    context = " ".join(f"Clinical context {index}." for index in range(300)) + " "
    raw = (
        ("ASSESSMENT AND PLAN\n\n" + context)
        + "The patient has seborrheic dermatitis. "
        "Ketoconazole shampoo was prescribed. "
        "Use it twice weekly. "
        "Use regular shampoo on other days. "
        "Use it twice weekly. "
        "Use regular shampoo on other days. "
        "Use it twice weekly. "
        "Use regular shampoo on other days."
    )
    assert postprocess_note(raw) == (
        ("ASSESSMENT AND PLAN\n\n" + context)
        + "The patient has seborrheic dermatitis. "
        "Ketoconazole shampoo was prescribed. "
        "Use it twice weekly. "
        "Use regular shampoo on other days."
    )
    assert postprocess_note(raw, trim_repetition=False) == raw


def test_postprocess_does_not_trim_short_repeated_text() -> None:
    raw = (
        "Use it twice weekly. "
        "Use regular shampoo on other days. "
        "Use it twice weekly. "
        "Use regular shampoo on other days. "
        "Use it twice weekly. "
        "Use regular shampoo on other days."
    )
    assert postprocess_note(raw) == raw
