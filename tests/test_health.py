from fastapi.testclient import TestClient

from learning_studio.main import app


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_question_parser() -> None:
    from learning_studio.main import _split_question

    assert _split_question("QUESTION: What is a graph?\nANSWER: A set of nodes and edges.") == (
        "What is a graph?",
        "A set of nodes and edges.",
    )
