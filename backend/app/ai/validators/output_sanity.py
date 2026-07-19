"""Basic malformed-output and language checks."""

import re

from ..schemas import AnswerDraft

CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
REPEATED_PATTERN = re.compile(r"(.)\1{20,}", re.DOTALL)
MOJIBAKE_MARKERS = ("�", "锟斤拷", "Ã", "â€")


def validate_output_sanity(draft: AnswerDraft, original_question: str) -> list[str]:
    errors: list[str] = []
    answer = draft.answer.strip()
    if not answer:
        errors.append("answer is blank")
    if REPEATED_PATTERN.search(answer):
        errors.append("answer contains pathological repeated characters")
    if any(marker in answer for marker in MOJIBAKE_MARKERS):
        errors.append("answer contains malformed or mojibake text")
    if CJK_PATTERN.search(original_question) and not CJK_PATTERN.search(answer):
        errors.append("answer language does not match the Chinese question")
    return errors
