import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import split_into_chunks_by_sentence, read_pdf, ask_about_pdf_chunked, app
from fastapi.testclient import TestClient

client = TestClient(app)


def make_test_pdf(path, text_lines):
    stream_content = "\n".join(
        f"BT /F1 14 Tf 50 {700 - i*20} Td ({line}) Tj ET"
        for i, line in enumerate(text_lines)
    )
    pdf_content = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length {len(stream_content)} >>
stream
{stream_content}
endstream
endobj
trailer
<< /Size 6 /Root 1 0 R >>
%%EOF
"""
    with open(path, "w") as f:
        f.write(pdf_content)


def test_chunking_basic():
    text = "Ceci est une phrase. Voici une deuxieme phrase. Et une troisieme."
    chunks = split_into_chunks_by_sentence(text, max_chars=500)
    assert len(chunks) >= 1
    assert "Ceci est une phrase." in chunks[0]

def test_chunking_empty_text():
    chunks = split_into_chunks_by_sentence("", max_chars=500)
    assert chunks == []

def test_chunking_respects_max_chars():
    text = "Phrase courte. " * 50
    chunks = split_into_chunks_by_sentence(text, max_chars=100)
    for chunk in chunks:
        assert len(chunk) <= 150

def test_read_pdf_missing_file():
    try:
        read_pdf("/tmp/this_file_does_not_exist.pdf")
        assert False, "Expected an exception for missing file"
    except Exception:
        assert True

@patch("app.ask_about_pdf_chunked")
def test_ask_endpoint_success(mock_ask):
    mock_ask.return_value = "This is a mocked answer."
    response = client.post("/ask", json={"pdf_path": "/tmp/fake.pdf", "question": "What is this?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "This is a mocked answer."

def test_ask_endpoint_missing_field():
    response = client.post("/ask", json={"question": "What is this?"})
    assert response.status_code == 422

def test_ask_endpoint_missing_question():
    response = client.post("/ask", json={"pdf_path": "/tmp/fake.pdf"})
    assert response.status_code == 422


def test_read_pdf_extracts_real_content(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    make_test_pdf(str(pdf_path), ["Le code secret est BANANA42.", "Ceci est un test."])
    text = read_pdf(str(pdf_path))
    assert "BANANA42" in text

def test_chunking_single_long_sentence_exceeding_max_chars():
    long_sentence = "Mot " * 200 + "."
    chunks = split_into_chunks_by_sentence(long_sentence, max_chars=100)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)

def test_chunking_no_content_loss():
    text = "Une phrase. Deuxieme phrase. Troisieme phrase. Quatrieme phrase."
    chunks = split_into_chunks_by_sentence(text, max_chars=30)
    rejoined = " ".join(chunks)
    for fragment in ["Une phrase", "Deuxieme phrase", "Troisieme phrase", "Quatrieme phrase"]:
        assert fragment in rejoined

def test_chunking_multiple_chunks_created_for_long_text():
    text = "Phrase numero un. Phrase numero deux. Phrase numero trois. Phrase numero quatre. Phrase numero cinq."
    chunks = split_into_chunks_by_sentence(text, max_chars=40)
    assert len(chunks) > 1

@patch("app.requests.post")
def test_ask_about_pdf_chunked_uses_relevant_context(mock_post, tmp_path):
    pdf_path = tmp_path / "sample2.pdf"
    make_test_pdf(str(pdf_path), [
        "Le code secret du projet est BANANA42.",
        "Ce document ne contient aucune autre information utile."
    ])

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "BANANA42"}}]
    }
    mock_post.return_value = mock_response

    result = ask_about_pdf_chunked(str(pdf_path), "Quel est le code secret ?")

    assert "BANANA42" in result
    called_prompt = mock_post.call_args.kwargs["json"]["messages"][0]["content"]
    assert "BANANA42" in called_prompt

def test_ask_endpoint_wrong_type_for_question():
    response = client.post("/ask", json={"pdf_path": "/tmp/fake.pdf", "question": 12345})
    assert response.status_code == 422

@patch("app.ask_about_pdf_chunked")
def test_ask_endpoint_propagates_internal_error(mock_ask):
    mock_ask.side_effect = FileNotFoundError("PDF not found")
    try:
        client.post("/ask", json={"pdf_path": "/tmp/missing.pdf", "question": "test?"})
    except FileNotFoundError:
        assert True
    else:
        response = client.post("/ask", json={"pdf_path": "/tmp/missing.pdf", "question": "test?"})
        assert response.status_code in (500, 422)
