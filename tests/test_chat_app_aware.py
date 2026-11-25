from unittest.mock import MagicMock, call, patch

from fastapi.testclient import TestClient
from neo4j.exceptions import Neo4jError

from learning_studio.ai import AIExtractionError, AIProvider
from learning_studio.config import Settings
from learning_studio.graph import KnowledgeGraph
from learning_studio.main import (
    app,
    bilingual_text_fields,
    extract_entity_from_chat,
    generate_textbook_lesson,
    lesson_paragraphs,
    passage_summary_fallback,
    synthesize_fast_two_paragraphs,
    textbook_topic_excerpt,
)

LESSON_EN = (
    "The atmosphere is a mixture of gases surrounding Earth. Nitrogen and oxygen "
    "form its largest components and support essential natural processes.\n\n"
    "The composition of the atmosphere is important for climate, weather, and life. "
    "Remember the dominant gases and connect them to the atmosphere's role in "
    "maintaining conditions on Earth."
)
LESSON_TE = (
    "భూమిని చుట్టుముట్టి ఉండే వాయువుల మిశ్రమమే వాతావరణం. నత్రజని మరియు ఆక్సిజన్ "
    "దాని ప్రధాన భాగాలు; ఇవి ముఖ్యమైన సహజ ప్రక్రియలకు తోడ్పడతాయి.\n\n"
    "వాతావరణ కూర్పు శీతోష్ణస్థితి, వాతావరణ పరిస్థితులు మరియు జీవానికి ముఖ్యమైనది. "
    "ప్రధాన వాయువులను గుర్తుంచుకొని, భూమిపై జీవన పరిస్థితులను నిలబెట్టడంలో "
    "వాతావరణం పాత్రతో వాటిని అనుసంధానించండి."
)


def set_mock_lesson_provider(mock_provider_class):
    mock_provider_class.return_value.structured_answer.side_effect = [
        {"etext": LESSON_EN.split("\n\n")[0], "summary": LESSON_EN},
        {"etext_te": LESSON_TE.split("\n\n")[0], "summary_te": LESSON_TE},
    ]


def test_textbook_lesson_uses_source_passages_when_model_summary_is_invalid():
    provider = MagicMock()
    provider.structured_answer.return_value = {
        "etext": "Too short.",
        "summary": "A single paragraph does not meet the requested format.",
    }
    passages = [{
        "page": 12,
        "text": (
            "Baudhayan described geometric rules for constructing shapes. "
            "His work includes an approximation related to the square root of two. "
            "These rules are associated with the Sulba Sutras. "
            "They show the development of mathematical knowledge in ancient India."
        ),
    }]

    etext, summary, etext_te, summary_te = generate_textbook_lesson(
        provider,
        "Baudhayan",
        "Courseware.pdf",
        passages,
    )

    assert etext.startswith("Baudhayan")
    assert len(lesson_paragraphs(summary)) == 2
    assert "Sulba Sutras" in summary
    assert etext_te == ""
    assert summary_te == ""


def test_textbook_lesson_uses_source_passages_when_provider_fails():
    provider = MagicMock()
    provider.structured_answer.side_effect = AIExtractionError("invalid JSON")
    passages = [{
        "page": 7,
        "text": (
            "Public key cryptography uses a pair of related keys. "
            "One key can be shared publicly while the other is kept private. "
            "The public key can encrypt information for its owner. "
            "The private key is used to decrypt that information."
        ),
    }]

    etext, summary, etext_te, summary_te = generate_textbook_lesson(
        provider,
        "Public Key Technique",
        "Courseware.pdf",
        passages,
    )

    assert etext.startswith("Public key cryptography")
    assert len(lesson_paragraphs(summary)) == 2
    assert "private key" in summary
    assert etext_te == ""
    assert summary_te == ""


def test_textbook_lesson_returns_translations_from_same_generation():
    provider = MagicMock()
    etext_en = LESSON_EN.split("\n\n")[0]
    etext_te = LESSON_TE.split("\n\n")[0]
    provider.structured_answer.return_value = {
        "etext": etext_en,
        "summary": LESSON_EN,
        "etext_te": etext_te,
        "summary_te": LESSON_TE,
    }

    result = generate_textbook_lesson(
        provider,
        "Atmosphere",
        "Courseware.pdf",
        [{"page": 12, "text": LESSON_EN}],
    )

    assert result == (etext_en, LESSON_EN, etext_te, LESSON_TE)
    assert provider.structured_answer.call_count == 1
    assert "exactly four string keys" in provider.structured_answer.call_args.args[0]


def test_passage_fallback_prefers_topic_alias_and_related_facts():
    etext, summary = passage_summary_fallback(
        "Swachh Bharat Mission-Urban (SBM-U)",
        [
            {
                "page": "5-6",
                "text": (
                    "The article first explains broad sanitation goals. "
                    "Similarly, under SBM-U, the focus is not only on waste processing. "
                    "It focusses on making waste segregation at source a public habit and "
                    "achieving total urban sanitation and public cleanliness. "
                    "The article later discusses another topic."
                ),
            }
        ],
        [
            "urban sanitation, waste management, and clean public spaces.",
            "waste segregation at source a public habit.",
        ],
    )

    assert "SBM-U" in etext
    assert "article first explains broad sanitation goals" not in summary
    assert "waste segregation at source" in summary
    assert len(lesson_paragraphs(summary)) == 2


def test_telugu_lesson_summary_is_kept_when_intro_translation_is_missing():
    provider = MagicMock()
    provider.structured_answer.return_value = {
        "etext_te": "",
        "summary_te": LESSON_TE,
    }

    translated = bilingual_text_fields(provider, LESSON_EN.split("\n\n")[0], LESSON_EN)

    assert translated["etext_te"] == ""
    assert translated["summary_te"] == LESSON_TE


def test_learning_plan_keeps_working_when_generated_summary_has_one_paragraph():
    client = TestClient(app)
    passages = [{
        "page": 12,
        "text": (
            "Public key cryptography uses a pair of related keys. "
            "One key can be shared publicly while the other is kept private. "
            "The public key can encrypt information for its owner. "
            "The private key is used to decrypt that information."
        ),
    }]
    generated_question = {
        "question": "How many keys are used in public key cryptography?",
        "options": ["One", "Two"],
        "correct_option": 1,
        "explanation": "It uses a public and a private key.",
        "etext": "Public key cryptography uses two related keys.",
        "summary": "Public key cryptography uses two related keys.",
        "question_te": "పబ్లిక్ కీ క్రిప్టోగ్రఫీలో ఎన్ని కీలను ఉపయోగిస్తారు?",
        "options_te": ["ఒకటి", "రెండు"],
        "answer_te": "రెండు",
        "explanation_te": "ఇది పబ్లిక్ మరియు ప్రైవేట్ కీలను ఉపయోగిస్తుంది.",
    }

    with (
        patch("learning_studio.main.KnowledgeGraph") as mock_kg_class,
        patch("learning_studio.main.AIProvider") as mock_provider_class,
    ):
        graph = mock_kg_class.return_value
        graph.session_context.return_value = {
            "textbook": "Courseware.pdf",
            "group": "Group 1",
        }
        graph.entity_question.return_value = None
        graph.related_learning_question.return_value = None
        graph.search.return_value = []
        graph.textbook_passages.return_value = passages
        graph.textbook_entity_evidence.return_value = []
        mock_provider_class.return_value.structured_answer.side_effect = [
            generated_question,
            {"etext_te": "", "summary_te": ""},
        ]

        response = client.post(
            "/api/v1/sessions/test-session/learning-plan",
            json={"mode": "guided", "topic": "Public Key Technique"},
        )

    assert response.status_code == 200
    result = response.json()
    assert len(lesson_paragraphs(result["summary_en"])) == 2
    assert "private key" in result["summary_en"]


def test_ui_translation_endpoint_preserves_translation_order():
    client = TestClient(app)
    with (
        patch("learning_studio.main.AIProvider") as mock_provider,
        patch("learning_studio.main.settings.grok_api_key", ""),
    ):
        mock_provider.return_value.structured_answer.return_value = {
            "translations": ["ప్రారంభించండి", "తెలుగు"]
        }
        response = client.post(
            "/api/v1/translate",
            json={
                "texts": ["Start learning", "English"],
                "target_language": "te",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "translations": ["ప్రారంభించండి", "తెలుగు"]
    }


def test_ui_translation_endpoint_accepts_english_passthrough():
    client = TestClient(app)
    response = client.post(
        "/api/v1/translate",
        json={
            "texts": ["Start learning", "English"],
            "target_language": "en",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "translations": ["Start learning", "English"]
    }


def test_previous_question_api_uses_the_requested_language():
    client = TestClient(app)
    with patch("learning_studio.main.KnowledgeGraph") as mock_graph_class:
        graph = mock_graph_class.return_value
        graph.question_history.return_value = []
        graph.session_context.return_value = {"textbook": "Courseware.pdf"}
        graph.question_bank.return_value = [{
            "question_id": "previous-1",
            "question": "ఈ ప్రశ్న తెలుగులో ఉందా?",
            "question_en": "Is this question in Telugu?",
            "question_te": "ఈ ప్రశ్న తెలుగులో ఉందా?",
            "options": ["అవును", "కాదు"],
            "options_en": ["Yes", "No"],
            "options_te": ["అవును", "కాదు"],
            "answer": "అవును",
            "answer_en": "Yes",
            "answer_te": "అవును",
            "citations": [{"source": "paper.pdf", "page": 1}],
            "liked": False,
        }]

        response = client.post(
            "/api/v1/sessions/test-session/questions",
            json={"mode": "previous", "language": "te"},
        )

    assert response.status_code == 200
    assert response.json()["question"] == "ఈ ప్రశ్న తెలుగులో ఉందా?"
    assert response.json()["question_en"] == "Is this question in Telugu?"
    assert response.json()["options_en"] == ["Yes", "No"]
    assert graph.question_bank.call_args.kwargs["language"] == "te"


def test_textbook_player_stores_missing_telugu_translation():
    client = TestClient(app)
    with (
        patch("learning_studio.main.KnowledgeGraph") as mock_graph_class,
        patch("learning_studio.main.AIProvider") as mock_provider_class,
    ):
        graph = mock_graph_class.return_value
        graph.question_history.return_value = []
        graph.session_context.return_value = {"textbook": "Courseware.pdf"}
        graph.question_bank.return_value = [{
            "question_id": "study-q1",
            "question": "Which field did Khorana study?",
            "question_en": "Which field did Khorana study?",
            "question_te": None,
            "options": ["Biochemistry", "Physics", "Astronomy", "Geology"],
            "options_en": ["Biochemistry", "Physics", "Astronomy", "Geology"],
            "options_te": None,
            "answer": "Biochemistry. He was a biochemist.",
            "answer_en": "Biochemistry. He was a biochemist.",
            "answer_te": None,
            "correct_option": 0,
            "citations": [],
            "liked": False,
        }]
        mock_provider_class.return_value.structured_answer.return_value = {
            "question_te": "ఖోరానా ఏ రంగాన్ని అధ్యయనం చేశారు?",
            "options_te": ["జీవరసాయన శాస్త్రం", "భౌతిక శాస్త్రం", "ఖగోళ శాస్త్రం", "భూగర్భ శాస్త్రం"],
            "answer_te": "జీవరసాయన శాస్త్రం. ఆయన జీవరసాయన శాస్త్రవేత్త.",
            "explanation_te": "",
        }

        response = client.post(
            "/api/v1/sessions/test-session/questions",
            json={"mode": "textbook", "language": "te"},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["question"] == "ఖోరానా ఏ రంగాన్ని అధ్యయనం చేశారు?"
    assert result["options"] == ["జీవరసాయన శాస్త్రం", "భౌతిక శాస్త్రం", "ఖగోళ శాస్త్రం", "భూగర్భ శాస్త్రం"]
    graph.save_study_question_translation.assert_called_once_with(
        "study-q1",
        result["question_te"],
        result["options_te"],
        result["answer_te"],
        "",
    )


def test_learning_plan_survives_telugu_translation_provider_failure():
    client = TestClient(app)
    with (
        patch("learning_studio.main.KnowledgeGraph") as mock_kg_class,
        patch("learning_studio.main.AIProvider") as mock_provider_class,
    ):
        mock_kg = mock_kg_class.return_value
        mock_kg.session_context.return_value = {
            "textbook": "Courseware.pdf",
            "group": "Group 1",
        }
        mock_kg.entity_question.return_value = {
            "question_id": "q-legacy",
            "question": "Which scientist is mentioned?",
            "options": ["Person A", "Har Gobind Khorana", "Person C", "Person D"],
            "correct_option": 1,
            "answer": "Har Gobind Khorana",
            "explanation": "The entity is mentioned.",
            "citations": [],
            "entity_name": "Har Gobind Khorana",
        }
        mock_kg.related_learning_question.return_value = None
        mock_kg.textbook_passages.return_value = [
            {"text": "Har Gobind Khorana was a biochemist.", "page": 33}
        ]
        mock_provider_class.return_value.structured_answer.side_effect = [
            AIExtractionError("Telugu question translation provider unavailable"),
            {"etext": LESSON_EN.split("\n\n")[0], "summary": LESSON_EN},
            AIExtractionError("Telugu lesson translation provider unavailable"),
        ]

        response = client.post(
            "/api/v1/sessions/test-session/learning-plan",
            json={"mode": "guided", "topic": "Har Gobind Khorana", "language": "te"},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["question"]["question"] == "Which scientist is mentioned?"
    assert result["question"]["question_te"] == ""
    assert result["question"]["options_te"] == []
    mock_kg.save_study_question_translation.assert_not_called()


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


def test_learning_plan_generates_a_related_multi_paragraph_summary_from_textbook():
    client = TestClient(app)
    generated_etext = "Har Gobind Khorana was an Indo-American biochemist."
    generated_summary = (
        "Har Gobind Khorana was an Indo-American biochemist whose research focused on "
        "nucleotides and the genetic code. His work helped explain how biological "
        "information guides protein synthesis.\n\n"
        "Khorana received the Nobel Prize in 1968 for this research. For exam preparation, "
        "connect his contribution to molecular biology with the role of the genetic code "
        "in directing protein formation."
    )
    telugu_etext = "హర్ గోవింద్ ఖొరానా ఇండో-అమెరికన్ జీవరసాయన శాస్త్రవేత్త."
    telugu_summary = (
        "హర్ గోవింద్ ఖొరానా న్యూక్లియోటైడ్లు మరియు జన్యు సంకేతంపై పరిశోధన చేసిన "
        "ఇండో-అమెరికన్ జీవరసాయన శాస్త్రవేత్త. ఆయన పని జీవసంబంధ సమాచారం ప్రోటీన్ "
        "సంశ్లేషణకు ఎలా మార్గనిర్దేశం చేస్తుందో వివరించింది.\n\n"
        "ఈ పరిశోధనకు ఖొరానాకు 1968లో నోబెల్ బహుమతి లభించింది. పరీక్షల కోసం ఆయన "
        "కృషిని అణు జీవశాస్త్రంతో, అలాగే ప్రోటీన్ నిర్మాణంలో జన్యు సంకేతం పాత్రతో "
        "సంబంధించి గుర్తుంచుకోండి."
    )
    with (
        patch("learning_studio.main.KnowledgeGraph") as mock_kg_class,
        patch("learning_studio.main.AIProvider") as mock_provider_class,
    ):
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
            "question_te": "హర్ గోవింద్ ఖొరానా ఏ రంగానికి చెందినవారు?",
            "options_te": ["భౌతిక శాస్త్రం", "పౌర ఇంజనీరింగ్", "రసాయన శాస్త్రం", "జీవరసాయన శాస్త్రం"],
            "answer_te": "జీవరసాయన శాస్త్రం",
            "explanation_te": "ఆయన ఇండో-అమెరికన్ జీవరసాయన శాస్త్రవేత్త.",
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
        mock_provider_class.return_value.structured_answer.side_effect = [
            {"etext": generated_etext, "summary": generated_summary},
            {"etext_te": telugu_etext, "summary_te": telugu_summary},
        ]

        response = client.post(
            "/api/v1/sessions/test-session/learning-plan",
            json={"mode": "guided", "topic": "Har Gobind Khorana", "language": "te"},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["etext_te"] == telugu_etext
    assert result["summary_te"] == telugu_summary
    assert result["summary"] == generated_summary
    assert len(result["summary_en"].split("\n\n")) == 2
    assert len(result["summary_te"].split("\n\n")) == 2
    assert "studied in relation to" not in result["etext"]
    assert result["question"]["question_te"] == "హర్ గోవింద్ ఖొరానా ఏ రంగానికి చెందినవారు?"
    assert result["summary_en"]
    assert result["etext_en"]
    generation_call = mock_provider_class.return_value.structured_answer.call_args_list[0]
    assert "exactly 2 or 3 coherent paragraphs" in generation_call.args[0]
    assert "nucleotides and the genetic code" in generation_call.args[1]
    mock_kg.save_study_question_translation.assert_called_once()
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
            "question_te": "ఏ శాస్త్రవేత్త ప్రస్తావించబడ్డారు?",
            "options_te": ["వ్యక్తి ఎ", "హర్ గోవింద్ ఖొరానా", "వ్యక్తి సి", "వ్యక్తి డి"],
            "answer_te": "హర్ గోవింద్ ఖొరానా",
            "explanation_te": "ఈ అంశం ప్రస్తావించబడింది.",
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


def test_learning_plan_summarizes_related_indexed_facts_when_passage_is_not_matched():
    client = TestClient(app)
    summary = (
        "Swachh Bharat Mission-Urban focuses on urban sanitation, waste management, "
        "and clean public spaces. These priorities connect cleanliness with the "
        "management of public environments.\n\n"
        "The indexed facts also describe waste segregation at source as a public habit. "
        "Together, these details show how the mission links urban cleanliness with "
        "everyday community practices."
    )
    summary_te = (
        "స్వచ్ఛ భారత్ మిషన్-అర్బన్ పట్టణ పారిశుధ్యం, వ్యర్థాల నిర్వహణ మరియు శుభ్రమైన "
        "బహిరంగ ప్రదేశాలపై దృష్టి పెడుతుంది. ఈ ప్రాధాన్యతలు పరిశుభ్రతను బహిరంగ "
        "పరిసరాల నిర్వహణతో అనుసంధానిస్తాయి.\n\n"
        "మూలంలో ఉన్న వాస్తవాలు, వ్యర్థాలను వాటి ఉత్పత్తి స్థలంలోనే వేరు చేయడం ఒక "
        "ప్రజా అలవాటుగా పేర్కొంటాయి. ఈ వివరాలు పట్టణ పరిశుభ్రతను రోజువారీ సమాజ "
        "ఆచారాలతో మిషన్ ఎలా కలుపుతుందో చూపిస్తాయి."
    )
    with (
        patch("learning_studio.main.KnowledgeGraph") as mock_kg_class,
        patch("learning_studio.main.AIProvider") as mock_provider_class,
    ):
        mock_kg = MagicMock()
        mock_kg_class.return_value = mock_kg
        mock_kg.session_context.return_value = {
            "textbook": "pib_prism.pdf",
            "group": "Group 1",
        }
        mock_kg.entity_question.return_value = {
            "question_id": "q-sbm-u",
            "question": "Which focus is associated with SBM-U?",
            "options": [
                "Urban sanitation and waste management",
                "Space exploration",
                "Mountain forestry",
                "Marine navigation",
            ],
            "correct_option": 0,
            "explanation": "The indexed facts identify urban sanitation and waste management.",
            "question_te": "స్వచ్ఛ భారత్ మిషన్‌తో ఏ దృష్టి సంబంధించింది?",
            "options_te": [
                "పట్టణ పారిశుధ్యం మరియు వ్యర్థాల నిర్వహణ",
                "అంతరిక్ష అన్వేషణ",
                "పర్వత అటవీ సంరక్షణ",
                "సముద్ర నావిగేషన్",
            ],
            "answer_te": "పట్టణ పారిశుధ్యం మరియు వ్యర్థాల నిర్వహణ",
            "explanation_te": "మూలంలోని వాస్తవాలు పట్టణ పారిశుధ్యం మరియు వ్యర్థాల నిర్వహణను పేర్కొంటాయి.",
            "citations": [
                {
                    "related_concepts": [
                        {
                            "relationship": "MENTIONS",
                            "name": "A document title and unrelated front matter.",
                        },
                        {
                            "relationship": "RELATED_FACT",
                            "name": "urban sanitation, waste management, and clean public spaces",
                        },
                        {
                            "relationship": "RELATED_FACT",
                            "name": "waste segregation at source a public habit",
                        },
                    ]
                }
            ],
            "entity_name": "Swachh Bharat Mission-Urban (SBM-U)",
        }
        mock_kg.textbook_passages.return_value = []
        mock_kg.related_learning_question.return_value = None
        mock_provider_class.return_value.structured_answer.side_effect = [
            {
                "etext": "SBM-U focuses on urban sanitation and waste management.",
                "summary": summary,
            },
            {
                "etext_te": "SBM-U పట్టణ పారిశుధ్యం మరియు వ్యర్థాల నిర్వహణపై దృష్టి పెడుతుంది.",
                "summary_te": summary_te,
            },
        ]

        response = client.post(
            "/api/v1/sessions/test-session/learning-plan",
            json={
                "mode": "topic_focused",
                "topic": "Swachh Bharat Mission-Urban (SBM-U)",
            },
        )

    assert response.status_code == 200
    result = response.json()
    assert result["summary_en"] == summary
    assert result["summary_te"] == summary_te
    assert len(result["summary_en"].split("\n\n")) == 2
    assert len(result["summary_te"].split("\n\n")) == 2
    lesson_source = mock_provider_class.return_value.structured_answer.call_args_list[0].args[1]
    assert "urban sanitation, waste management, and clean public spaces" in lesson_source
    assert "waste segregation at source a public habit" in lesson_source
    assert "unrelated front matter" not in lesson_source
    assert "Page not recorded" in lesson_source
    mock_kg.textbook_passages.assert_any_call(
        "pib_prism.pdf",
        topic="urban sanitation, waste management, and clean public spaces.",
        limit=2,
    )
    lesson_instruction = mock_provider_class.return_value.structured_answer.call_args_list[0].args[0]
    assert "Every factual statement must be directly supported" in lesson_instruction


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


def test_screening_attempt_returns_telugu_feedback_when_requested():
    client = TestClient(app)
    with patch("learning_studio.main.KnowledgeGraph") as mock_kg_class:
        graph = mock_kg_class.return_value
        graph.screening_question.return_value = {
            "question": "What is ISRO?",
            "options": ["Agency", "Club"],
            "correct_option": 0,
            "explanation": "ISRO is India's space agency.",
            "explanation_te": "ఇస్రో భారత అంతరిక్ష సంస్థ.",
            "attempts_used": 0,
            "resolved": False,
        }
        graph.record_screening_attempt.return_value = {
            "question": "What is ISRO?",
            "options": ["Agency", "Club"],
            "correct_option": 0,
            "explanation": "ISRO is India's space agency.",
            "explanation_te": "ఇస్రో భారత అంతరిక్ష సంస్థ.",
            "attempts_used": 2,
            "resolved": True,
        }

        response = client.post(
            "/api/v1/sessions/test-session/screening/q1/attempt",
            json={"option_index": 1, "language": "te"},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["feedback"] == result["feedback_te"]
    assert "ఇస్రో భారత అంతరిక్ష సంస్థ" in result["feedback"]
    assert "ISRO is India's space agency" not in result["feedback"]


def test_learning_plan_ask_from_loads_a_different_entity_question():
    client = TestClient(app)
    with (
        patch("learning_studio.main.KnowledgeGraph") as mock_kg_class,
        patch("learning_studio.main.AIProvider") as mock_provider_class,
    ):
        set_mock_lesson_provider(mock_provider_class)
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
            "question_te": "వాతావరణంలో ప్రధానంగా ఉండే వాయువులు ఏవి?",
            "options_te": ["నత్రజని మరియు ఆక్సిజన్", "హైడ్రోజన్ మాత్రమే"],
            "answer_te": "నత్రజని మరియు ఆక్సిజన్",
            "explanation_te": "వాతావరణంలో ప్రధానంగా నత్రజని మరియు ఆక్సిజన్ ఉంటాయి.",
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
    with (
        patch("learning_studio.main.KnowledgeGraph") as mock_kg_class,
        patch("learning_studio.main.AIProvider") as mock_provider_class,
    ):
        set_mock_lesson_provider(mock_provider_class)
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
                "question_te": "వాతావరణంలో ప్రధానంగా ఉండే వాయువులు ఏవి?",
                "options_te": ["నత్రజని మరియు ఆక్సిజన్", "హైడ్రోజన్ మాత్రమే"],
                "answer_te": "నత్రజని మరియు ఆక్సిజన్",
                "explanation_te": "వాతావరణంలో ప్రధానంగా నత్రజని మరియు ఆక్సిజన్ ఉంటాయి.",
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
