from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from learning_studio.main import app, extract_entity_from_chat, synthesize_fast_two_paragraphs


def test_extract_entity_from_chat():
    assert extract_entity_from_chat("Ask from [Delhi Government]") == "Delhi Government"
    assert extract_entity_from_chat("Ask from Delhi Government") == "Delhi Government"
    assert extract_entity_from_chat("Ask about ISRO") == "ISRO"
    assert extract_entity_from_chat("What is Nuclear Energy?", "Fallback") == "Nuclear Energy"
    assert extract_entity_from_chat("Random message", "Fallback") == "Fallback"


def test_synthesize_fast_two_paragraphs():
    p = synthesize_fast_two_paragraphs(
        "ISRO",
        "Science.pdf",
        [{"text": "ISRO was founded in 1969. It leads space research in India."}],
        [{"entity_type": "Agency", "entity_name": "DOS", "relationship": "REPORTS_TO"}],
        [],
    )
    assert "\n\n" in p
    paragraphs = p.strip().split("\n\n")
    assert len(paragraphs) == 2


def test_chat_change_question_and_app_aware_response():
    client = TestClient(app)
    with patch("learning_studio.main.KnowledgeGraph") as mock_kg_class:
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg
        mock_kg.session_context.return_value = {"textbook": "Courseware.pdf", "group": "Group 1"}
        mock_kg.recent_messages.return_value = []
        mock_kg.last_screening_entity.return_value = "Delhi Government"
        mock_kg.search.return_value = [{"concept": "Delhi Government", "description": "State government", "page": 5}]
        mock_kg.textbook_entity_evidence.return_value = [
            {"entity_type": "Govt", "entity_name": "Secretariat", "relationship": "ADMINISTERS"}
        ]
        mock_kg.textbook_passages.return_value = [{"text": "Delhi Government implemented solar policies.", "page": 5}]

        # Test 1: Change question trigger
        resp1 = client.post("/api/v1/sessions/test-session/chat", json={"message": "change question"})
        assert resp1.status_code == 200
        assert resp1.json()["action"] == "change_question"

        # Test 2: Ask from [entity]
        resp2 = client.post(
            "/api/v1/sessions/test-session/chat",
            json={"message": "Ask from [Delhi Government]", "topic": "Delhi Government"},
        )
        assert resp2.status_code == 200
        paragraphs = resp2.json()["answer"].strip().split("\n\n")
        assert len(paragraphs) == 2
