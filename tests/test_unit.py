import pytest
import os
from app.ingestion import validate_file, chunk_pages_with_overlap, compute_sha256
from app.safety_engine import classify_safety_category, redact_dosage
from app.tools import tool_explain_medical_term, ExplainMedicalTermInput, tool_format_citations, FormatCitationsInput

def test_file_validation_valid_txt():
    content = b"Sample medical text content"
    ext = validate_file(content, "test.txt")
    assert ext == ".txt"

def test_file_validation_empty_file():
    with pytest.raises(ValueError, match="empty"):
        validate_file(b"", "empty.txt")

def test_file_validation_invalid_extension():
    with pytest.raises(ValueError, match="Unsupported file format"):
        validate_file(b"some content", "file.exe")

def test_file_validation_corrupt_pdf():
    with pytest.raises(ValueError, match="Invalid PDF magic bytes"):
        validate_file(b"NOT_A_PDF_HEADER", "corrupt.pdf")

def test_chunking_with_overlap():
    pages = [(1, "Page 1 content " * 50), (2, "Page 2 content " * 50)]
    chunks = chunk_pages_with_overlap(pages, max_chars=200, overlap_chars=50)
    assert len(chunks) > 0
    assert chunks[0]["page_number"] in [1, 2]
    assert "token_count" in chunks[0]

def test_sha256_computation():
    c1 = b"Medical text A"
    c2 = b"Medical text A"
    c3 = b"Medical text B"
    assert compute_sha256(c1) == compute_sha256(c2)
    assert compute_sha256(c1) != compute_sha256(c3)

def test_safety_classification_emergency():
    res = classify_safety_category("Help, I am having severe chest pain and cannot breathe!")
    assert res["category"] == "A_EMERGENCY"
    assert res["is_emergency"] is True
    assert "🚨" in res["template_response"]

def test_safety_classification_negation():
    res = classify_safety_category("The patient has no chest pain and no shortness of breath.")
    assert res["category"] != "A_EMERGENCY"
    assert res["is_emergency"] is False

def test_safety_classification_definitional():
    res = classify_safety_category("What is a stroke and what causes it?")
    assert res["category"] != "A_EMERGENCY"

def test_safety_classification_professional_review():
    res = classify_safety_category("How many mg of amlodipine should I take daily for my blood pressure?")
    assert res["category"] == "B_PROFESSIONAL_REVIEW"
    assert res["requires_escalation"] is True

def test_dosage_redaction():
    text = "Take 10 mg of amlodipine twice a day with 500 mg metformin."
    redacted = redact_dosage(text)
    assert "[REDACTED DOSE]" in redacted
    assert "10 mg" not in redacted
    assert "500 mg" not in redacted

def test_medical_term_explanation():
    res = tool_explain_medical_term(ExplainMedicalTermInput(term="hypertension"))
    assert "High blood pressure" in res["plain_explanation"]

def test_is_health_related_query():
    from app.agent import is_health_related_query
    assert is_health_related_query("I have a severe migraine headache and nausea") is True
    assert is_health_related_query("What are the symptoms of high blood pressure?") is True
    assert is_health_related_query("My lower back and knee hurt after lifting") is True
    assert is_health_related_query("What is 2+2?") is False
    assert is_health_related_query("How do I write a Python function?") is False
    assert is_health_related_query("Who won the cricket world cup?") is False
    assert is_health_related_query("Tell me a joke") is False

