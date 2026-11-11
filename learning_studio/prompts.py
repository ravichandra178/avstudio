"""Editable prompts used by Learning Studio AI providers."""


EXTRACTION_SYSTEM = (
    "You are an expert Knowledge Graph Information Extraction engine for Indian "
    "Administrative and Science and Technology exam content. Extract typed entities "
    "and relationships; do not label every entity as Concept."
)


def extraction_prompt(document_id: str, text: str) -> str:
    return (
        "Extract structured entities and relationships from the textbook passage. "
        "Use entity types such as Person, Organization, Event, Location, Topic, "
        "Concept, Document, and Technology when supported by the text. "
        "A person must be Person, an institution must be Organization, and an incident "
        "or disaster must be Event. Do not invent facts or types. "
        "Return JSON only as an array of objects with entity_type, name, description, "
        "page, and relationships fields. Each relationship must include relationship_type "
        "and target_name. Preserve only facts explicitly present in the passage. "
        f"Document id: {document_id}\n\n{text[:120000]}"
    )


TUTOR_SYSTEM = (
    "You are a context-aware textbook tutor. Use only the supplied textbook context, "
    "explain clearly, and cite document pages when available."
)
JSON_SUFFIX = "\nReturn valid JSON only. Do not wrap it in markdown fences."
FALLBACK_LABEL = " Clearly label this response as AI-generated fallback content."
FALLBACK_CONTEXT = (
    "No matching Neo4j content is currently indexed. Use only the configured fallback scope "
    "and clearly identify the response as AI-generated fallback content."
)


def learning_plan_prompt(
    mode: str,
    topic: str,
    group: str,
    level: str,
    language: str,
    passages: str = "",  # Made optional with a default so it never raises TypeError
    fallback: bool = False,
) -> str:
    scope = "the selected textbook overview" if not topic else f"the topic {topic!r}"
    fallback_rule = (
        " This is an explicitly configured AI fallback; label the summary as fallback."
        if fallback
        else ""
    )
    
    passage_section = (
        f"\n\n--- TEXTBOOK PASSAGE (GROUND TRUTH FROM NEO4J DATABASE) ---\n{passages}\n----------------------------------------------------------\n\n"
        if passages else 
        "\n\n--- NOTE: No direct text chunk found. Stick strictly to verified historical/scientific facts for this topic. ---\n\n"
    )

    return (
        f"Create a {mode} learning step for {scope}. "
        f"This is for {group} at the {level} level. "
        f"Respond in {language}. "
        f"{passage_section}"
        "CRITICAL EXAM STANDARD RULES:\n"
        "1. Question Format: Formulate rigorous AP/Telangana Group 1 Prelims style questions. Use multi-statement assertion formats ('Consider the following statements...') with combination choices (e.g., '1 and 2 only', 'All of the above') wherever applicable.\n"
        "2. Zero Hallucination: Do not invent biographical details, dates, or institutional roles.\n"
        "3. Distractors: Ensure all distractors are plausible civil-services exam traps.\n"
        "Return JSON with exactly these keys: etext, summary, question, options, correct_option, explanation. "
        "options must be an array of 4 plausible multiple-choice answers. "
        "correct_option must be a zero-based integer. "
        "Keep etext under 120 words, summary under 45 words, and explanation under 60 words. "
        "Do not add markdown, citations, or extra keys. "
        f"For topic mode, focus only on the requested topic.{fallback_rule}"
    )


def question_prompt(
    mode: str, 
    group: str, 
    level: str, 
    language: str,
    passages: str = ""  # Optional parameter to prevent argument mismatch
) -> str:
    passage_section = (
        f"\n\n--- TEXTBOOK PASSAGE (GROUND TRUTH) ---\n{passages}\n---------------------------------------\n\n"
        if passages else ""
    )
    return (
        f"Create one {mode} multiple-choice question at the {level} level for {group}. "
        f"Respond in {language}. "
        f"{passage_section}"
        "EXAM STANDARD: Write analytical multi-statement questions matching state public service commission standards. "
        "Return valid JSON only with exactly these keys: question, options, correct_option, explanation. "
        "options must contain exactly 4 plausible choices. correct_option must be a zero-based integer."
    )

def entity_question_ingestion_prompt(
    entity_type: str,
    entity_name: str,
    relationships: str,
    group: str,
    level: str,
) -> str:
    return (
        "Create exactly one textbook-grounded four-option MCQ for Indian competitive exams. "
        f"Exam group: {group}. Exam level: {level}. "
        f"Entity type: {entity_type}. Entity: {entity_name}. "
        f"Direct textbook relationships and facts: {relationships}. "
        "Use only these supplied facts. Never use model memory, web knowledge, or inference. "
        "Do not invent dates, roles, institutions, awards, locations, or affiliations. "
        "The correct answer must be directly supported by the supplied facts. "
        "Return JSON only with exactly: question, options, correct_option, explanation. "
        "options must contain exactly 4 strings and correct_option must be a zero-based integer. "
        "CRITICAL FORMAT RULE: If writing a statement-based question, put numbered statements (1, 2, 3) in the 'question' text itself, and make the 4 options combinations like '1 and 2 only', '2 and 3 only', or 'All of the above'. Do NOT put raw descriptive sentences as options unless it is a standard single-fact MCQ."
    )

def chat_prompt(message: str, language: str, passages: str = "") -> str:
    return (
        "Answer the learner's question using only the supplied textbook context from Neo4j. "
        f"Respond in {language}.\n\n"
        f"--- TEXTBOOK PASSAGE ---\n{passages}\n------------------------\n\n"
        f"Learner question: {message}"
    )