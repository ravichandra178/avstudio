from datetime import datetime

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    status: str


class DocumentStatus(BaseModel):
    document_id: str
    filename: str
    status: str
    error: str | None = None
    created_at: datetime


class SearchResult(BaseModel):
    concept: str
    description: str | None = None
    source_document_id: str
    page: int | None = None
    related_concepts: list[str] = Field(default_factory=list)


class GraphResponse(BaseModel):
    nodes: list[dict]
    relationships: list[dict]


class StudySessionResponse(BaseModel):
    session_id: str
    status: str = "active"


class StudySessionRequest(BaseModel):
    group: str = Field(min_length=1, max_length=80)
    textbook: str = Field(min_length=1, max_length=160)


class StudyProgress(BaseModel):
    messages: int
    questions: int
    percent: int


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    document_id: str | None = None
    language: str = Field(default="en", pattern="^(en|te)$")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[dict] = Field(default_factory=list)


class QuestionRequest(BaseModel):
    mode: str = Field(default="textbook", pattern="^(textbook|previous|repeated)$")
    document_id: str | None = None
    concept: str | None = None
    language: str = Field(default="en", pattern="^(en|te)$")


class QuestionResponse(BaseModel):
    question_id: str
    question: str
    options: list[str] = Field(min_length=2, max_length=6)
    answer: str | None = None
    mode: str
    citations: list[dict] = Field(default_factory=list)
    liked: bool = False


class QuestionLikeRequest(BaseModel):
    liked: bool = True


class LearningPlanRequest(BaseModel):
    mode: str = Field(pattern="^(guided|topic_focused)$")
    topic: str | None = Field(default=None, max_length=200)
    language: str = Field(default="en", pattern="^(en|te)$")


class ScreeningQuestion(BaseModel):
    question_id: str
    question: str
    options: list[str] = Field(min_length=2, max_length=6)
    attempts_used: int = 0
    max_attempts: int = 3


class LearningPlanResponse(BaseModel):
    session_id: str
    mode: str
    topic: str
    etext: str
    summary: str
    question: ScreeningQuestion
    citations: list[dict] = Field(default_factory=list)
    source: str


class ScreeningAttemptRequest(BaseModel):
    option_index: int = Field(ge=0, le=5)


class ScreeningAttemptResponse(BaseModel):
    correct: bool
    attempts_used: int
    attempts_remaining: int
    answer_revealed: bool
    correct_option_index: int | None = None
    feedback: str
    progress: int
