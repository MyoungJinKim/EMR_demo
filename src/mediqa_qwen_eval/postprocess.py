from __future__ import annotations

import re


LEADING_LABEL = re.compile(
    r"^\s*(?:final\s+answer|possible\s+clinical\s+note|clinical\s+note|emr\s+note)\s*:\s*",
    re.IGNORECASE,
)
MARKDOWN_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s+(?=[A-Z][A-Z &_/()-]*\s*$)")
BOLD_HEADING = re.compile(
    r"(?m)^[ \t]*\*\*([A-Z][A-Z &_/()-]*)\*\*[ \t]*:?[ \t]*$"
)
SENTENCE_OR_LINE = re.compile(r"[^\n.!?]+(?:[.!?]+|$)")


def _normalized_segment(segment: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", segment.lower()).strip()


def _repeat_key(segment: str) -> str:
    return segment[:80]


def _trim_repetitive_tail(text: str) -> str:
    if len(text) < 5000:
        return text
    segments: list[tuple[int, int, str]] = []
    for match in SENTENCE_OR_LINE.finditer(text):
        raw_segment = match.group(0)
        normalized = _normalized_segment(raw_segment)
        if normalized:
            segments.append((match.start(), match.end(), normalized))
    if len(segments) < 6:
        return text

    max_block = 32
    min_average_segment_chars = 24
    for index in range(len(segments)):
        for block_size in range(1, max_block + 1):
            end = index + (block_size * 3)
            if end > len(segments):
                continue
            first = [item[2] for item in segments[index : index + block_size]]
            if sum(len(item) for item in first) / len(first) < min_average_segment_chars:
                continue
            first_keys = [_repeat_key(item) for item in first]
            second = [
                _repeat_key(item[2])
                for item in segments[index + block_size : index + (block_size * 2)]
            ]
            third = [
                _repeat_key(item[2])
                for item in segments[index + (block_size * 2) : end]
            ]
            if first_keys == second == third:
                return text[: segments[index + block_size][0]].rstrip()
    return text


def postprocess_note(raw_output: str, *, trim_repetition: bool = True) -> str:
    """Remove wrappers; legacy repetition trimming remains enabled for evaluation."""
    text = raw_output.replace("__lf1__", "\n").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    text = LEADING_LABEL.sub("", text, count=1)
    text = MARKDOWN_HEADING.sub("", text)
    text = BOLD_HEADING.sub(lambda match: match.group(1), text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if trim_repetition:
        text = _trim_repetitive_tail(text)
    return text.strip()
