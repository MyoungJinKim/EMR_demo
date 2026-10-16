from __future__ import annotations

import pytest

from mediqa_qwen_eval.soap import SOAP_HEADINGS, SOAP_MISSING, normalize_soap_note, soap_messages, soap_translation_messages, validate_source_numbers, place_explicit_clinician_statements, format_korean_soap_style


def test_current_clinician_cause_uncertainty_moves_to_assessment_verbatim():
    statement = "Clinician states cause is not confirmed."
    source = "S: Nausea.\n- " + statement + "\nO: BP 118/74.\nA: Not documented in the conversation.\nP: Recheck tomorrow."
    # An inline/multiline S body is deliberately left alone rather than splitting a continued fact.
    normalized = normalize_soap_note(source, "en")
    assert place_explicit_clinician_statements(source) == normalized
    source = source.replace("S: Nausea.", "S · Subjective\n- Nausea.")
    result = place_explicit_clinician_statements(source)
    assert result.startswith("S · Subjective\n- Nausea.\n\nO")
    assert "A · Assessment\n- " + statement + "\n\nP" in result
    assert result.count(statement) == 1
    assert "BP 118/74." in result and "Recheck tomorrow." in result


@pytest.mark.parametrize("content", [
    "Patient states cause is not confirmed.",
    "Patient reports the doctor said the cause is unknown.",
    "Clinician previously stated the cause is unknown.",
    "Clinician states cause was unknown.",
    "Clinician states drug allergy status is unknown.",
    "Clinician states the temperature is unknown.",
    "Clinician states the diagnosis is unknown.",
    "Clinician states the cause is unknown; tests planned.",
    "Clinician states the cause of nausea was previously unclear but the patient reports the cause of dizziness is unknown.",
    "Clinician states the cause of past pain was unknown and the current drug allergy status is unknown.",
])
def test_uncertainty_classifier_does_not_move_history_allergies_exams_or_mixed_statements(content):
    source = "S: - " + content + "\nO: Not documented.\nA: Not documented in the conversation.\nP: Not documented."
    assert place_explicit_clinician_statements(source) == normalize_soap_note(source, "en")


def test_uncertainty_classifier_preserves_existing_assessment_and_only_removes_exact_duplicates():
    statement = "Doctor says the cause is currently unknown."
    source = "S: - " + statement + "\nO: Not documented.\nA: - Possible nerve disorder.\n- " + statement + "\nP: Test next week."
    result = place_explicit_clinician_statements(source)
    assert result.startswith("S · Subjective\n- Not documented in the conversation.")
    assert "Possible nerve disorder." in result
    assert result.count(statement) == 1


def test_uncertainty_classifier_collapses_only_identical_moved_bullets():
    statement = "Clinician states cause is not confirmed."
    source = "S: - " + statement + "\n\n- " + statement + "\nO: Not documented.\nA: Not documented in the conversation.\nP: Test tomorrow."
    result = place_explicit_clinician_statements(source)
    assert result.count(statement) == 1
    assert "S · Subjective\n- Not documented in the conversation." in result


def test_uncertainty_classifier_keeps_nonidentical_existing_assessment_unchanged():
    source = "S: - Clinician states cause is not confirmed.\nO: Not documented.\nA: - Cause unknown.\nP: Test tomorrow."
    assert place_explicit_clinician_statements(source) == normalize_soap_note(source, "en")


@pytest.mark.parametrize("plan", [
    "- Not documented in the conversation.",
    "- Continue 2.5 mg daily.\n- Recheck in 3 days.",
    "- Medication dosage adjustment is not to be done arbitrarily.\n- Recheck in 3 days.",
])
def test_exact_medication_prohibition_moves_verbatim_to_plan_without_losing_existing_items(plan):
    statement = "Medication dosage adjustment is not to be done arbitrarily."
    source = f"S: - {statement}\n- {statement}\nO: Not documented.\nA: Not documented.\nP: {plan}"
    result = place_explicit_clinician_statements(source)
    assert result.startswith("S · Subjective\n- Not documented in the conversation.")
    assert statement not in result.split("P · Plan")[0]
    assert result.count(statement) == 1
    for line in plan.splitlines():
        if line != "- " + SOAP_MISSING["en"]:
            assert line in result.split("P · Plan\n")[1]


@pytest.mark.parametrize("statement", [
    '"Medication dosage adjustment is not to be done arbitrarily."',
    "Patient says Medication dosage adjustment is not to be done arbitrarily.",
    "Medication dosage adjustment was not to be done arbitrarily.",
    "Medication dosage adjustment is not to be done arbitrarily, according to the patient.",
    "Medication dosage adjustment is not to be done arbitrarily. Patient disagrees.",
    "Currently taking 2.5 mg daily without adjusting the dose.",
])
def test_medication_prohibition_move_excludes_quotes_patient_history_and_larger_clauses(statement):
    source = f"S: - {statement}\nO: Not documented.\nA: Not documented.\nP: Not documented."
    assert place_explicit_clinician_statements(source) == normalize_soap_note(source, "en")


def test_medication_prohibition_move_rejects_mixed_paragraph_and_preserves_other_soap_facts():
    statement = "Medication dosage adjustment is not to be done arbitrarily."
    source = f"S: Takes 2.5 mg daily.\n- {statement}\nO: BP 118/74.\nA: Cause unknown.\nP: Recheck tomorrow."
    assert place_explicit_clinician_statements(source) == normalize_soap_note(source, "en")
    source = source.replace("S: Takes", "S: - Takes")
    result = place_explicit_clinician_statements(source)
    assert result.startswith("S · Subjective\n- Takes 2.5 mg daily.")
    assert "O · Objective\nBP 118/74." in result
    assert "A · Assessment\nCause unknown." in result
    assert "P · Plan\nRecheck tomorrow.\n- " + statement in result


@pytest.mark.parametrize("language", ["ko", "en"])
def test_soap_prompt_preserves_source_and_requires_missing_distinction(language):
    source = "환자: 열감이 있어요. 의사: 체온은 36.7도입니다. 내일 검사하겠습니다."
    messages = soap_messages(source, language)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert source in messages[-1]["content"]
    assert messages[-1]["content"].endswith("- [내용]" if language == "ko" else "- [content]")
    assert all(heading in messages[-1]["content"] for heading in SOAP_HEADINGS[language])
    assert SOAP_MISSING[language] in messages[0]["content"]
    assert "확인되지" in messages[0]["content"]
    assert source not in messages[0]["content"]
    assert ("예시 대화" if language == "ko" else "Example conversation") not in messages[-1]["content"]


def test_soap_normalization_keeps_clinical_content_unchanged():
    source = """SOAP note
## **S (Subjective):**
- Takes Drug-X 2.5 mg; no fever reported.
**O: Objective**
- Temperature 36.7 °C.
### A - Assessment:
- Possible infection, not confirmed.
P. Plan
- Test tomorrow only if symptoms persist.
"""
    result = normalize_soap_note(source, "en")
    assert result == (
        "S · Subjective\n- Takes Drug-X 2.5 mg; no fever reported.\n\n"
        "O · Objective\n- Temperature 36.7 °C.\n\n"
        "A · Assessment\n- Possible infection, not confirmed.\n\n"
        "P · Plan\n- Test tomorrow only if symptoms persist."
    )


def test_korean_note_allows_explicit_absence_of_source_information():
    source = "\n\n".join(f"{heading}\n- {SOAP_MISSING['ko']}" for heading in SOAP_HEADINGS["ko"])
    assert normalize_soap_note(source, "ko") == source


@pytest.mark.parametrize("source,expected", [
    ("환자는 구토를 부정합니다.", "환자는 구토를 부정함."),
    ("환자는 구토를 부정했습니다.", "환자는 구토를 부정했음."),
    ("증상은 어제부터 시작되었습니다.", "증상은 어제부터 시작되었음."),
    ("추후 확인됩니다.", "추후 확인됨."),
    ("메스꺼움은 있습니다. 구토는 없습니다.", "메스꺼움은 있음. 구토는 없음."),
    ("오늘 새 약은 처방하지 않습니다.", "오늘 새 약은 처방하지 않음."),
    ("아직 확인되지 않았습니다.", "아직 확인되지 않았음."),
    ("보호자는 아이의 알레르기를 모릅니다.", "보호자는 아이의 알레르기를 모름."),
    ("내일 확인할 예정입니다", "내일 확인할 예정임"),
    ("1 주일 이상 지속되면 재진합니다.", "1주일 이상 지속되면 재진함."),
    ("다른 약물 알레르기는 확인되지 않았습니다라고 진술함.", "다른 약물 알레르기는 확인되지 않았다고 진술함."),
    ("확인되지 않았습니다고 보고함.", "확인되지 않았다고 보고함."),
    ("보호자는 본인에게 알레르기가 있습니다라고 함.", "보호자는 본인에게 알레르기가 있다고 함."),
    ("환자는 구토가 없습니다라고 함.", "환자는 구토가 없다고 함."),
    ("환자가 목이 안 아파한다고 함.", "환자가 목이 아프지 않다고 함."),
])
def test_known_korean_record_endings_preserve_negation_uncertainty_and_timing(source, expected):
    assert format_korean_soap_style(source) == expected
    assert format_korean_soap_style(expected) == expected


def test_korean_style_normalization_preserves_clinical_facts_sections_and_values():
    source = (
        "S: - 환자가 와파린 2.5mg 을 하루 한 번 복용한다고 보고함.\n"
        "- 환자는 0.5 mg 을 저녁에 복용하며 아침에는 복용하지 않습니다.\n"
        "- 보호자는 아이의 콧물이 3 일 전부터 났다고 보고함.\n"
        "- 보호자의 페니실린 알레르기는 확인되었으나 아이의 약물 알레르기는 모릅니다.\n"
        "- 귀 통증은 없고 3 일 뒤 재확인한다고 함.\n"
        "O: - 혈압 118/74.\n- 맥박 분당 80 회.\n"
        "A: - 의사는 원인을 확정하지 않았다고 합니다.\n"
        "P: - 오늘 새 약은 처방하지 않습니다.\n"
        "- 내일부터 2.5mg 으로 줄임.\n- 오늘 검사 후 결과를 확인한 뒤 결정할 예정입니다."
    )
    expected = (
        "S · 주관적 정보\n- 환자가 와파린 2.5mg을 하루 한 번 복용한다고 보고함.\n"
        "- 환자는 0.5 mg을 저녁에 복용하며 아침에는 복용하지 않음.\n"
        "- 보호자는 아이의 콧물이 3일 전부터 났다고 보고함.\n"
        "- 보호자의 페니실린 알레르기는 확인되었으나 아이의 약물 알레르기는 모름.\n"
        "- 귀 통증은 없고 3일 뒤 재확인한다고 함.\n\n"
        "O · 객관적 소견\n- 혈압 118/74.\n- 맥박 분당 80회.\n\n"
        "A · 평가\n- 의사는 원인을 확정하지 않았다고 함.\n\n"
        "P · 계획\n- 오늘 새 약은 처방하지 않음.\n"
        "- 내일부터 2.5mg으로 줄임.\n- 오늘 검사 후 결과를 확인한 뒤 결정할 예정임."
    )
    assert normalize_soap_note(source, "ko") == expected
    assert normalize_soap_note(expected, "ko") == expected


@pytest.mark.parametrize("source", [
    '- 환자는 "약이 없습니다. 2.5mg 을 먹지 않습니다."라고 함.',
    "- 보호자는 '아이의 알레르기를 모릅니다'라고 함.",
    "- 환자는 “귀는 안 아파한다고 합니다.”라고 인용함.",
    "- 있습니다라는 말의 의미는 미확인.",
    "- 할 수 없습니다만 재확인은 예정되어 있음.",
    "- 환자는 약을 먹었습니다.",
    "- 3 일단이라는 이름은 미확인.",
    "- 2.5mg 의존성은 확인되지 않음.",
    "- 귀 통증 없음. 뒤에서 들리는 소리 있음.",
])
def test_style_leaves_quotes_other_inflections_and_clinical_words_untouched(source):
    assert format_korean_soap_style(source) == source


def test_korean_formatter_does_not_change_english_note_or_transcript_prompt():
    source = "S: - Patient said 구토는 없습니다.\nO: Not documented.\nA: Not documented.\nP: Test in 3 days."
    normalized = normalize_soap_note(source, "en")
    assert "구토는 없습니다." in normalized
    transcript = '환자입니다. 약은 2.5mg 을 먹습니다. 귀는 안 아파한다고 했습니다.'
    messages = soap_translation_messages(source, transcript)
    assert transcript in messages[1]["content"]
    assert normalized in messages[1]["content"]


def test_test_order_wording_preserves_timing_and_leaves_supplies_medications_and_quotes_untouched():
    source = (
        "- 오늘 혈액 검사 발주.\n- 내일 혈액 검사 발주 예정.\n"
        "- 검사 발주 계획은 없음.\n- 약물 발주.\n- 검사용 물품 발주.\n"
        "- 검사 발주서 확인.\n- 환자가 '검사 발주'라고 말함."
    )
    expected = source.replace("검사 발주.", "검사 의뢰.").replace("검사 발주 예정", "검사 의뢰 예정").replace("검사 발주 계획", "검사 의뢰 계획")
    assert format_korean_soap_style(source) == expected
    assert format_korean_soap_style(expected) == expected


@pytest.mark.parametrize("language", ["ko", "en"])
@pytest.mark.parametrize("prefix", ["", "- ", "* ", "+ ", "• "])
def test_redundant_missing_placeholder_is_removed_only_beside_actual_content(language, prefix):
    missing = SOAP_MISSING[language]
    unknown = "다른 약물 알레르기는 미확인." if language == "ko" else "Other drug allergies unknown."
    related = "가족력: " + missing if language == "ko" else "Family history: " + missing
    plans = "- 내일 검사 예정.\n- 2주 뒤 재진." if language == "ko" else "- Test tomorrow.\n- Follow-up in 2 weeks."
    expected_bodies = [f"- {unknown}\n- {related}", f"- {missing}", f"- {missing}", plans]
    source_bodies = [expected_bodies[0] + f"\n{prefix}{missing}", expected_bodies[1], expected_bodies[2], plans + f"\n{prefix}{missing}"]
    source = "\n\n".join(f"{heading}\n{body}" for heading, body in zip(SOAP_HEADINGS[language], source_bodies))
    expected = "\n\n".join(f"{heading}\n{body}" for heading, body in zip(SOAP_HEADINGS[language], expected_bodies))
    assert normalize_soap_note(source, language) == expected


@pytest.mark.parametrize("language", ["ko", "en"])
def test_empty_bullet_does_not_remove_the_only_missing_placeholder(language):
    source = "\n\n".join(f"{heading}\n- \n- {SOAP_MISSING[language]}" for heading in SOAP_HEADINGS[language])
    assert normalize_soap_note(source, language) == source


def test_inline_soap_preserves_content_and_normalizes_headings():
    source = "S: 보호자가 환자의 가슴 통증을 보고함.\nO: 대화에 언급되지 않음.\nA: 대화에 언급되지 않음.\nP: 대화에 언급되지 않음."
    result = normalize_soap_note(source, "ko")
    assert result.startswith("S · 주관적 정보\n보호자가 환자의 가슴 통증을 보고함.")
    assert result.endswith("P · 계획\n대화에 언급되지 않음.")


def test_ordered_section_letters_fix_label_typo_without_moving_facts():
    source = (
        "S · Subjective\n- Patient suspects carpal tunnel syndrome.\n\n"
        "O · Objective\n- Not documented in the conversation.\n\n"
        "A · Subjective\n- Clinician has not confirmed the cause.\n\n"
        "P · Plan\n- Test next week."
    )
    assert normalize_soap_note(source, "en") == source.replace("A · Subjective", "A · Assessment")


def test_middle_dot_inline_missing_section_preserves_the_documented_placeholder():
    source = "S: Limited activity.\nO: Not documented.\nA · Not documented in the conversation.\nP: Avoid overexertion."
    result = normalize_soap_note(source, "en")
    assert "A · Assessment\nNot documented in the conversation." in result
    assert "Limited activity." in result and "Avoid overexertion." in result
    with pytest.raises(ValueError, match="섹션"):
        normalize_soap_note(source.replace("A ·", "S ·"), "en")


def test_translation_uses_complete_existing_note_and_keeps_section_identity():
    source = (
        "S · Subjective\n- Hives after penicillin; other drug allergies unknown.\n"
        "- Metformin 500 mg twice daily.\n\nO · Objective\n- BP 140/80.\n\n"
        "A · Assessment\n- Not documented in the conversation.\n\nP · Plan\n- Follow-up in 2 weeks."
    )
    messages = soap_translation_messages(source)
    assert source in messages[1]["content"]
    assert all(heading in messages[1]["content"] for heading in SOAP_HEADINGS["ko"])
    assert "S/O/A/P 소속을 그대로 유지" in messages[1]["content"]
    assert "unknown" in messages[1]["content"]
    assert "반응·증상·용량·단위·횟수·기간·수치" in messages[1]["content"]


def test_translation_rejects_incomplete_source_soap():
    with pytest.raises(ValueError, match="SOAP"):
        soap_translation_messages("S · Subjective\n- Chest pain.")


@pytest.mark.parametrize("verb,report", [("denies", "reports"), ("denied", "reported")])
@pytest.mark.parametrize("symptom", ["sore throat", "skin rash", "black stools", "hematuria", "black stools or hematuria"])
def test_translation_canonicalizes_only_verified_standalone_denials(verb, report, symptom):
    source = f"S: - Patient {verb} {symptom}.\nO: Not documented.\nA: Not documented.\nP: Recheck in 3 days."
    normalized = normalize_soap_note(source, "en")
    messages = soap_translation_messages(source)
    assert normalized.replace(f"Patient {verb} {symptom}.", f"Patient {report} no {symptom}.") in messages[1]["content"]
    assert f"Patient {verb} {symptom}." in source


@pytest.mark.parametrize("statement", [
    "Patient reports sore throat.",
    "Patient's sore throat status is unknown.",
    "Patient denies no sore throat.",
    "Patient denies sore throat and fever.",
    "Patient denies sore throat, but reports skin rash.",
    "Caregiver reports that Patient denies sore throat.",
    'Patient quotes: "Patient denies sore throat."',
    "Patient previously denied sore throat.",
    "Patient denies taking Lamotrigine.",
    "Patient reports black stools.",
    "Patient's hematuria status is unknown.",
    "Patient denies no hematuria.",
    "Patient denies black stools, but reports hematuria.",
    "Patient denies hematuria but reports black stools.",
    "Patient denies black stools or hematuria, but reports dizziness.",
    "Caregiver reports that Patient denies hematuria.",
    "Patient previously denied black stools or hematuria.",
])
def test_translation_leaves_positive_unknown_reported_and_compound_statements_unchanged(statement):
    source = f"S: - {statement}\nO: Not documented.\nA: Not documented.\nP: Not documented."
    assert normalize_soap_note(source, "en") in soap_translation_messages(source)[1]["content"]


def test_translation_keeps_original_vocabulary_reference_separate_from_english_note():
    english = "S: Runny nose.\nO: Not documented.\nA: Not documented.\nP: Not documented."
    transcript = "환자: 콧물이 나요."
    messages = soap_translation_messages(english, transcript)
    assert transcript in messages[1]["content"]
    assert transcript not in messages[0]["content"]
    assert normalize_soap_note(english, "en") in messages[1]["content"]
    assert "영문 기록에 없는 사실을 대화에서 추가" in messages[0]["content"]
    assert "분자/분모" in messages[1]["content"]


def test_english_soap_preserves_source_medication_identifiers_without_translation():
    source = "의사: 복용 중인 덱시부프로펜은 중단하세요."
    messages = soap_messages(source, "en")
    assert source in messages[1]["content"]
    assert "exact original drug name" in messages[0]["content"]
    assert "Never translate" in messages[0]["content"]


@pytest.mark.parametrize("source", [
    "S\n- Cough.\nO\n- Not documented.\nP\n- Rest.",
    "S\n- Cough.\nO\n- Not documented.\nA\n- Not documented.\nA\n- Flu.\nP\n- Rest.",
    "S\n- Cough.\nA\n- Flu.\nO\n- Not documented.\nP\n- Rest.",
    "S\n- Cough.\nO\n\nA\n- Not documented.\nP\n- Rest.",
    "S\n- Cough.\nO\n- Not documented.\nA\n- Not documented.\nP\n- ",
    "S: 가슴 통증.\nO: -\nA: -\nP: 대화에 언급되지 않음.",
    "Here is my reasoning.\nS\n- Cough.\nO\n- None.\nA\n- None.\nP\n- None.",
    "진료기록\n- 기침이 있습니다.",
])
def test_invalid_soap_is_rejected_instead_of_inventing_sections(source):
    with pytest.raises(ValueError, match="SOAP"):
        normalize_soap_note(source, "en")


@pytest.mark.parametrize("operation,args", [(soap_messages, ("source", "fr")), (normalize_soap_note, ("note", "fr"))])
def test_unsupported_note_language_is_rejected(operation, args):
    with pytest.raises(ValueError, match="언어"):
        operation(*args)


@pytest.mark.parametrize("english,korean", [
    ("Metformin 500 mg twice daily.", "Metformin 500 mg 하루 2회."),
    ("1,000.00 mg and .5 mg; repeat 1,000 mg.", "1000 mg 및 0.50 mg; 1000.0 mg 반복."),
    ("Two tablets daily.", "하루 2정."),
    ("Dose 5 mg morning and 5 mg evening.", "아침 5.0 mg, 저녁 5 mg."),
    ("Platelets 150 × 10^9/L.", "혈소판 150 x10^9/L."),
    ("Temperature 36.8°C.", "체온 T36.8°C."),
    ("SpO2 98%.", "산소포화도 98%."),
    ("COVID-19 infection.", "COVID-19 감염."),
])
def test_source_numbers_allow_equivalent_formatting_and_word_number_translation(english, korean):
    notes = []
    for language, content in [("en", english), ("ko", korean)]:
        notes.append("\n\n".join(heading + "\n- " + (content if index == 0 else SOAP_MISSING[language])
                                 for index, heading in enumerate(SOAP_HEADINGS[language])))
    validate_source_numbers(*notes)


@pytest.mark.parametrize("english,korean", [
    ("Metformin 500 mg.", "Metformin 50 mg."),
    ("Follow up in 2 weeks.", "3주 뒤 재진."),
    ("Follow up in 2 weeks.", "두 주 뒤 재진."),
    ("Dose 5 mg morning and 5 mg evening.", "아침 5 mg 복용."),
    ("Change -2.5.", "변화 2.5."),
])
def test_source_numbers_reject_changed_missing_or_spelled_out_values(english, korean):
    source = f"S: {english}\nO: Not documented.\nA: Not documented.\nP: Not documented."
    translated = f"S: {korean}\nO: 미기재.\nA: 미기재.\nP: 미기재."
    with pytest.raises(ValueError, match="S 섹션.*숫자"):
        validate_source_numbers(source, translated)


def test_source_numbers_reject_values_moved_to_another_section():
    source = "S: Cough.\nO: Blood pressure 140/80.\nA: Not documented.\nP: Rest."
    translated = "S: 기침.\nO: 미기재.\nA: 미기재.\nP: 혈압 140/80. 휴식."
    with pytest.raises(ValueError, match="O 섹션.*숫자"):
        validate_source_numbers(source, translated)
