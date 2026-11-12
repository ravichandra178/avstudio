from unittest.mock import MagicMock, call, patch

from fastapi.testclient import TestClient
from neo4j.exceptions import Neo4jError

from learning_studio.ai import AIExtractionError, AIProvider
from learning_studio.config import Settings
from learning_studio.graph import KnowledgeGraph
from learning_studio.main import (
    app,
    extract_entity_from_chat,
    synthesize_fast_two_paragraphs,
    textbook_topic_excerpt,
)


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


def test_textbook_topic_excerpt_starts_at_topic_and_stays_within_word_limit():
    passage = (
        "Earlier textbook content. Har Gobind Khorana (1922–2011) "
        "was an Indo-American biochemist. ● He received the Nobel Prize "
        "for research on nucleotides and the genetic code. ● "
        "Unrelated following topic starts here."
    )

    excerpt = textbook_topic_excerpt(passage, "Har Gobind Khorana", max_words=25)

    assert excerpt.startswith("Har Gobind Khorana")
    assert "Earlier textbook content" not in excerpt
    assert "Unrelated following topic" not in excerpt
    assert len(excerpt.split()) <= 25


def test_textbook_passages_reads_chunk_and_legacy_document_text():
    graph = KnowledgeGraph.__new__(KnowledgeGraph)
    session = MagicMock()
    session.__enter__.return_value = session
    graph.session = MagicMock(return_value=session)
    record = MagicMock()
    record.data.return_value = {
        "text": "Har Gobind Khorana was an Indo-American biochemist.",
        "page": 33,
    }
    session.run.return_value = [record]

    passages = graph.textbook_passages(
        "Courseware on Science and Technology.pdf",
        topic="Har Gobind Khorana",
        limit=2,
    )

    query = session.run.call_args.args[0]
    assert "OPTIONAL MATCH (d)-[:HAS_CHUNK]->(chunk:Chunk)" in query
    assert "document_properties.text" in query
    assert "document_properties.page_content" in query
    assert "topic_match_count DESC, page" in query
    assert passages == [
        {"text": "Har Gobind Khorana was an Indo-American biochemist.", "page": 33}
    ]


def test_learning_plan_uses_textbook_passage_instead_of_graph_relationships():
    client = TestClient(app)
    with patch("learning_studio.main.KnowledgeGraph") as mock_kg_class:
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg
        mock_kg.session_context.return_value = {
            "textbook": "Courseware on Science and Technology.pdf",
            "group": "Group 1",
        }
        mock_kg.entity_question.return_value = {
            "question_id": "q-khorana",
            "question": "What was Har Gobind Khorana's field?",
            "options": [
                "Physics",
                "Civil engineering",
                "Chemistry",
                "Biochemistry",
            ],
            "correct_option": 3,
            "explanation": "He was an Indo-American biochemist.",
            "citations": [],
            "entity_name": "Har Gobind Khorana",
        }
        mock_kg.textbook_passages.return_value = [
            {
                "text": (
                    "1.7.15 Har Gobind Khorana (1922–2011) "
                    "● Har Gobind Khorana was an Indo-American biochemist. "
                    "● He received the Nobel Prize in 1968 for research on "
                    "nucleotides and the genetic code that controls protein synthesis."
                ),
                "page": 33,
            }
        ]
        mock_kg.related_learning_question.return_value = None

        response = client.post(
            "/api/v1/sessions/test-session/learning-plan",
            json={"mode": "guided", "topic": "Har Gobind Khorana"},
        )

    assert response.status_code == 200
    result = response.json()
    assert "Indo-American biochemist" in result["etext"]
    assert "nucleotides and the genetic code" in result["etext"]
    assert "Indo-American biochemist" in result["summary"]
    assert "Nobel Prize" in result["summary"]
    assert "studied in relation to" not in result["etext"]
    mock_kg.textbook_passages.assert_called_with(
        "Courseware on Science and Technology.pdf",
        topic="Har Gobind Khorana",
        limit=2,
    )


def test_learning_plan_does_not_present_mentions_as_a_textbook_summary():
    client = TestClient(app)
    with patch("learning_studio.main.KnowledgeGraph") as mock_kg_class:
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg
        mock_kg.session_context.return_value = {
            "textbook": "Courseware on Science and Technology.pdf",
            "group": "Group 1",
        }
        mock_kg.entity_question.return_value = {
            "question_id": "q-khorana",
            "question": "Which scientist is mentioned?",
            "options": ["Person A", "Har Gobind Khorana", "Person C", "Person D"],
            "correct_option": 1,
            "explanation": "The entity is mentioned.",
            "citations": [],
            "entity_name": "Har Gobind Khorana",
        }
        mock_kg.textbook_passages.return_value = []
        mock_kg.textbook_entity_evidence.return_value = [
            {
                "entity_type": "Person",
                "entity_name": "Har Gobind Khorana",
                "relationship": "MENTIONS",
            }
        ]
        mock_kg.related_learning_question.return_value = None

        response = client.post(
            "/api/v1/sessions/test-session/learning-plan",
            json={"mode": "guided", "topic": "Har Gobind Khorana"},
        )

    assert response.status_code == 200
    result = response.json()
    assert "cannot provide a textbook-grounded summary" in result["etext"]
    assert "studied in relation to" not in result["etext"]
    assert result["summary"] == "No descriptive textbook passage is indexed for this topic."


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


def test_chat_uses_gemini_search_when_graph_only_mentions_entity():
    client = TestClient(app)
    with (
        patch("learning_studio.main.KnowledgeGraph") as mock_kg_class,
        patch.object(
            AIProvider,
            "gemini_search_answer",
            return_value="Baudhayan was an ancient Indian mathematician.\n\n"
            "His work is associated with early geometry and mathematical rules.",
        ) as gemini_search,
    ):
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg
        mock_kg.session_context.return_value = {
            "textbook": "Courseware on Science and Technology.pdf",
            "group": "Group 1",
        }
        mock_kg.search.return_value = [
            {"concept": "Baudhayan", "description": None, "page": 12}
        ]
        mock_kg.textbook_entity_evidence.return_value = [
            {
                "entity_type": "Person",
                "entity_name": "Baudhayan",
                "relationship": "MENTIONS",
            }
        ]
        mock_kg.textbook_passages.return_value = []
        mock_kg.recent_messages.return_value = []

        response = client.post(
            "/api/v1/sessions/test-session/chat",
            json={
                "message": "Summarize Baudhayan in 2 paragraphs",
                "topic": "Baudhayan",
            },
        )

        assert response.status_code == 200
        assert len(response.json()["answer"].split("\n\n")) == 2
        gemini_search.assert_called_once()
        assert gemini_search.call_args.kwargs["timeout_seconds"] == 12
        assert "Google Search grounding" in gemini_search.call_args.args[0]


def test_chat_does_not_hide_gemini_search_failure_as_a_graph_answer():
    client = TestClient(app)
    with (
        patch("learning_studio.main.KnowledgeGraph") as mock_kg_class,
        patch.object(
            AIProvider,
            "gemini_search_answer",
            side_effect=AIExtractionError("Gemini request timed out"),
        ),
    ):
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg
        mock_kg.session_context.return_value = {
            "textbook": "Courseware on Science and Technology.pdf",
            "group": "Group 1",
        }
        mock_kg.search.return_value = [
            {"concept": "Public Key Technique", "description": None, "page": 12}
        ]
        mock_kg.textbook_entity_evidence.return_value = [
            {
                "entity_type": "Concept",
                "entity_name": "Public Key Technique",
                "relationship": "MENTIONS",
            }
        ]
        mock_kg.textbook_passages.return_value = []
        mock_kg.recent_messages.return_value = []

        response = client.post(
            "/api/v1/sessions/test-session/chat",
            json={
                "message": "Summarize Public Key Technique in 2 paragraphs",
                "topic": "Public Key Technique",
            },
        )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert len(answer.split("\n\n")) == 2
    assert "couldn't get an internet-grounded summary" in answer
    assert "identifies Public Key Technique as" not in answer


def test_gemini_search_answer_enables_google_search_grounding():
    provider = AIProvider(
        Settings(gemini_api_key="test-key", gemini_model="models/gemini-3.8-flash")
    )
    response = MagicMock()
    response.json.return_value = {
        "steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "A grounded answer."}],
            }
        ]
    }

    with patch("learning_studio.ai.httpx.post", return_value=response) as post:
        answer = provider.gemini_search_answer("Summarize Baudhayan", "Textbook: Science.pdf", 4)

    assert answer == "A grounded answer."
    assert post.call_args.args[0] == (
        "https://generativelanguage.googleapis.com/v1beta/interactions"
    )
    assert post.call_args.kwargs["headers"]["x-goog-api-key"] == "test-key"
    assert post.call_args.kwargs["json"]["tools"] == [{"type": "google_search"}]
    assert post.call_args.kwargs["json"]["model"] == "gemini-3.8-flash"


def test_screening_answer_hidden_until_two_incorrect_attempts():
    client = TestClient(app)
    with patch("learning_studio.main.KnowledgeGraph") as mock_kg_class:
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg

        # First incorrect attempt
        mock_kg.screening_question.return_value = {
            "question": "What is ISRO?",
            "options": ["Agency", "Club"],
            "correct_option": 0,
            "explanation": "ISRO is India space agency.",
            "attempts_used": 0,
            "resolved": False,
        }
        mock_kg.record_screening_attempt.return_value = {
            "question": "What is ISRO?",
            "options": ["Agency", "Club"],
            "correct_option": 0,
            "explanation": "ISRO is India space agency.",
            "attempts_used": 1,
            "resolved": False,
        }
        resp1 = client.post("/api/v1/sessions/test-session/screening/q1/attempt", json={"option_index": 1})
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["correct"] is False
        assert data1["answer_revealed"] is False
        assert data1["correct_option_index"] is None
        assert data1["attempts_remaining"] == 1
        assert "1 attempt remaining" in data1["feedback"]

        # Second incorrect attempt -> answer is revealed!
        mock_kg.screening_question.return_value["attempts_used"] = 1
        mock_kg.record_screening_attempt.return_value = {
            "question": "What is ISRO?",
            "options": ["Agency", "Club"],
            "correct_option": 0,
            "explanation": "ISRO is India space agency.",
            "attempts_used": 2,
            "resolved": True,
        }
        resp2 = client.post("/api/v1/sessions/test-session/screening/q1/attempt", json={"option_index": 1})
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["correct"] is False
        assert data2["answer_revealed"] is True
        assert data2["correct_option_index"] == 0
        assert data2["attempts_remaining"] == 0
        assert "Two incorrect attempts" in data2["feedback"]


def test_learning_plan_ask_from_loads_a_different_entity_question():
    client = TestClient(app)
    with patch("learning_studio.main.KnowledgeGraph") as mock_kg_class:
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg
        mock_kg.session_context.return_value = {
            "textbook": "Courseware.pdf",
            "group": "Group 1",
        }
        mock_kg.related_learning_question.side_effect = [{
            "question_id": "q-related",
            "question": "What is the composition of the atmosphere?",
            "options": ["Nitrogen and oxygen", "Hydrogen only"],
            "correct_option": 0,
            "answer": "Nitrogen and oxygen",
            "explanation": "The atmosphere is mainly nitrogen and oxygen.",
            "citations": [],
            "entity_name": "Atmosphere",
        }, {
            "question_id": "q-next",
            "question": "Which bodies govern the region?",
            "options": ["Government", "Council"],
            "correct_option": 0,
            "answer": "Government",
            "explanation": "Administrative structure.",
            "citations": [],
            "entity_name": "Delhi Government",
        }]
        mock_kg.textbook_passages.return_value = [
            {"text": "The atmosphere contains mostly nitrogen and oxygen.", "page": 3}
        ]

        response = client.post(
            "/api/v1/sessions/test-session/learning-plan",
            json={"mode": "guided", "related_from": "Indus Valley"},
        )

        assert response.status_code == 200
        assert response.json()["topic"] == "Atmosphere"
        assert response.json()["next_topic"] == "Delhi Government"
        mock_kg.entity_question.assert_not_called()
        assert mock_kg.related_learning_question.call_args_list == [
            call("test-session", "Courseware.pdf", "Indus Valley"),
            call("test-session", "Courseware.pdf", "Atmosphere"),
        ]


def test_learning_plan_ask_from_does_not_fall_back_to_same_entity():
    client = TestClient(app)
    with patch("learning_studio.main.KnowledgeGraph") as mock_kg_class:
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg
        mock_kg.session_context.return_value = {
            "textbook": "Courseware.pdf",
            "group": "Group 1",
        }
        mock_kg.related_learning_question.return_value = None

        response = client.post(
            "/api/v1/sessions/test-session/learning-plan",
            json={"mode": "guided", "related_from": "Indus Valley"},
        )

        assert response.status_code == 404
        assert "other than Indus Valley" in response.json()["detail"]


def test_learning_plan_survives_failure_to_load_optional_next_topic():
    client = TestClient(app)
    with patch("learning_studio.main.KnowledgeGraph") as mock_kg_class:
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg
        mock_kg.session_context.return_value = {
            "textbook": "Courseware.pdf",
            "group": "Group 1",
        }
        mock_kg.related_learning_question.side_effect = [
            {
                "question_id": "q-related",
                "question": "What is the composition of the atmosphere?",
                "options": ["Nitrogen and oxygen", "Hydrogen only"],
                "correct_option": 0,
                "answer": "Nitrogen and oxygen",
                "explanation": "The atmosphere is mainly nitrogen and oxygen.",
                "citations": [],
                "entity_name": "Atmosphere",
            },
            Neo4jError("Could not retrieve another related entity"),
        ]
        mock_kg.textbook_passages.return_value = [
            {"text": "The atmosphere contains mostly nitrogen and oxygen.", "page": 3}
        ]

        response = client.post(
            "/api/v1/sessions/test-session/learning-plan",
            json={"mode": "guided", "related_from": "Indus Valley"},
        )

        assert response.status_code == 200
        assert response.json()["topic"] == "Atmosphere"
        assert response.json()["next_topic"] is None


def test_related_learning_question_uses_nearest_other_entity_hop():
    graph = KnowledgeGraph.__new__(KnowledgeGraph)
    session = MagicMock()
    session.__enter__.return_value = session
    graph.session = MagicMock(return_value=session)
    result = MagicMock()
    record = MagicMock()
    record.data.return_value = {
        "question_id": "q-nearest",
        "question": "Which department administers the policy?",
        "options": ["Delhi Government", "Local Council"],
        "correct_option": 0,
        "entity_name": "Department",
    }
    result.single.return_value = record
    session.run.return_value = result

    question = graph.related_learning_question(
        "session-1", "Courseware.pdf", "Delhi Government"
    )

    assert question["entity_name"] == "Department"
    query = session.run.call_args.args[0]
    assert "MATCH path = (anchor)-[*..5]-(candidate)" in query
    assert "ORDER BY hops ASC, rand()" in query
    assert "toLower(q.entity_name) <> toLower($anchor_entity)" in query
    assert session.run.call_args.kwargs["anchor_entity"] == "Delhi Government"
