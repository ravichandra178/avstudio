import json

from neo4j import Driver, GraphDatabase

from .config import Settings


class KnowledgeGraph:
    def __init__(self, settings: Settings):
        self.driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self.database = settings.neo4j_database

    def session(self):
        return self.driver.session(database=self.database)

    def close(self) -> None:
        self.driver.close()

    def ensure_schema(self) -> None:
        with self.session() as session:
            for statement in (
                "CREATE INDEX concept_name_index IF NOT EXISTS "
                "FOR (c:Concept) ON (c.name)",
                "CREATE INDEX document_id_index IF NOT EXISTS "
                "FOR (d:Document) ON (d.id)",
                "CREATE INDEX document_name_index IF NOT EXISTS "
                "FOR (d:Document) ON (d.name)",
                "CREATE INDEX document_filename_index IF NOT EXISTS "
                "FOR (d:Document) ON (d.filename)",
            ):
                session.run(statement).consume()

    def upsert_document(self, document_id: str, filename: str) -> None:
        with self.session() as session:
            session.run(
                """
                MERGE (d:Document {id: $document_id})
                SET d.filename = $filename, d.updated_at = datetime()
                """,
                document_id=document_id,
                filename=filename,
            ).consume()

    def ensure_vector_index(self, dimensions: int) -> None:
        dimensions = int(dimensions)
        with self.session() as session:
            session.run(
                f"""
                CREATE VECTOR INDEX study_chunk_embedding IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {dimensions},
                  `vector.similarity_function`: 'cosine'
                }}}}
                """,
            ).consume()

    def save_chunks(
        self, document_id: str, chunks: list[dict], embeddings: list[list[float]]
    ) -> None:
        with self.session() as session:
            session.execute_write(self._save_chunks, document_id, chunks, embeddings)

    @staticmethod
    def _save_chunks(tx, document_id: str, chunks: list[dict], embeddings: list[list[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            tx.run(
                """
                MATCH (d:Document {id: $document_id})
                CREATE (c:Chunk {id: $chunk_id, text: $text, page: $page, embedding: $embedding})
                CREATE (d)-[:HAS_CHUNK]->(c)
                """,
                document_id=document_id,
                chunk_id=chunk["id"],
                text=chunk["text"],
                page=chunk["page"],
                embedding=embedding,
            ).consume()

    def vector_search(self, embedding: list[float], limit: int = 8) -> list[dict]:
        with self.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('study_chunk_embedding', $limit, $embedding)
                YIELD node, score
                MATCH (d:Document)-[:HAS_CHUNK]->(node)
                RETURN node.text AS text, node.page AS page, d.id AS source_document_id,
                       d.filename AS filename, score
                ORDER BY score DESC
                """,
                embedding=embedding,
                limit=limit,
            )
            return [record.data() for record in result]

    def progress(self, session_id: str) -> dict:
        with self.session() as session:
            result = session.run(
                """
                MATCH (s:StudySession {id: $session_id})
                OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:StudyMessage)
                OPTIONAL MATCH (s)-[:ASKED]->(q:StudyQuestion)
                OPTIONAL MATCH (s)-[:HAS_SCREENING_QUESTION]->(sq:ScreeningQuestion)
                RETURN count(DISTINCT m) AS messages,
                       count(DISTINCT q) + count(DISTINCT sq) AS questions
                """,
                session_id=session_id,
            ).single()
            values = result.data() if result else {"messages": 0, "questions": 0}
            total = values["messages"] + values["questions"]
            return {**values, "percent": min(100, total * 10)}

    def save_concepts(self, document_id: str, concepts: list[dict]) -> None:
        with self.session() as session:
            session.execute_write(self._save_concepts, document_id, concepts)

    @staticmethod
    def _save_concepts(tx, document_id: str, concepts: list[dict]) -> None:
        for item in concepts:
            tx.run(
                """
                MATCH (d:Document {id: $document_id})
                MERGE (c:Concept {name: $name})
                SET c.description = $description
                MERGE (d)-[:CONTAINS {page: $page}]->(c)
                """,
                document_id=document_id,
                name=item["name"],
                description=item.get("description"),
                page=item.get("page"),
            ).consume()
        for item in concepts:
            for related in item.get("related", []):
                tx.run(
                    """
                    MATCH (a:Concept {name: $name}), (b:Concept {name: $related})
                    MERGE (a)-[:RELATED_TO]->(b)
                    """,
                    name=item["name"],
                    related=related,
                ).consume()

    def search(self, query: str, limit: int = 20, textbook: str | None = None) -> list[dict]:
        with self.session() as session:
            result = session.run(
                """
                MATCH (d:Document)-[r]->(c)
                WITH d, c, properties(d) AS document_properties,
                     properties(c) AS concept_properties
                WITH c, document_properties,
                     coalesce(
                         concept_properties.name,
                         concept_properties.title,
                         concept_properties.id,
                         concept_properties.text
                     ) AS concept_name
                WHERE toLower(coalesce(concept_name, '')) CONTAINS toLower($search_text)
                  AND (
                    $textbook IS NULL
                    OR document_properties.filename = $textbook
                    OR document_properties.name = $textbook
                    OR document_properties.source = $textbook
                    OR document_properties.id = $textbook
                  )
                  AND concept_name IS NOT NULL
                  AND NOT concept_name =~ '[0-9a-fA-F]{32,}'
                  AND NOT any(label IN labels(c) WHERE label IN [
                      'StudyGroup', 'Textbook', 'Document', 'Chunk',
                      'StudySession', 'StudyMessage'
                  ])
                RETURN DISTINCT concept_name AS concept,
                       "" AS description,
                       $textbook AS source_document_id,
                       null AS page,
                       [] AS related_concepts
                LIMIT $limit
                """,
                search_text=query,
                textbook=textbook,
                limit=limit,
            )
            return [record.data() for record in result]

    def textbook_topics(self, textbook: str, limit: int = 8) -> list[dict]:
        with self.session() as session:
            result = session.run(
                """
                MATCH (d:Document)-[r]->(c)
                WITH d, c, properties(d) AS document_properties,
                     properties(c) AS concept_properties
                WITH c, document_properties,
                   coalesce(
                       concept_properties.name,
                       concept_properties.title,
                       concept_properties.id
                   ) AS concept_name
                WHERE (
                  document_properties.filename = $textbook
                  OR document_properties.name = $textbook
                  OR document_properties.source = $textbook
                  OR document_properties.id = $textbook
                )
                AND concept_name IS NOT NULL
                AND NOT concept_name =~ '[0-9a-fA-F]{32,}'
                AND NOT any(label IN labels(c) WHERE label IN [
                    'StudyGroup', 'Textbook', 'Document', 'Chunk',
                    'StudySession', 'StudyMessage'
                ])
                RETURN concept_name AS concept,
                       "" AS description,
                       $textbook AS source_document_id,
                       null AS page,
                       [] AS related_concepts
                ORDER BY concept
                LIMIT $limit
                """,
                textbook=textbook,
                limit=limit,
            )
            return [record.data() for record in result]

    def textbook_passages(
        self, textbook: str, topic: str | None = None, limit: int = 6
    ) -> list[dict]:
        with self.session() as session:
            result = session.run(
                """
                MATCH (d:Document)-[:MENTIONS]->(c)
                WITH properties(d) AS document_properties, properties(c) AS chunk_properties
                WHERE (
                    document_properties.filename = $textbook
                    OR document_properties.name = $textbook
                    OR document_properties.source = $textbook
                    OR document_properties.id = $textbook
                )
                  AND chunk_properties.text IS NOT NULL
                  AND ($topic IS NULL OR toLower(chunk_properties.text) CONTAINS toLower($topic))
                RETURN DISTINCT chunk_properties.text AS text,
                       chunk_properties.page AS page
                ORDER BY page
                LIMIT $limit
                """,
                textbook=textbook,
                topic=topic or None,
                limit=limit,
            )
            return [record.data() for record in result]

    def textbook_entity_evidence(
        self, textbook: str, topic: str | None = None, limit: int = 24
    ) -> list[dict]:
        with self.session() as session:
            result = session.run(
                """
                MATCH (d:Document)-[r:MENTIONS]->(entity)
                WITH entity, r, properties(d) AS document_properties,
                     coalesce(
                         properties(entity).name,
                         properties(entity).title,
                         properties(entity).text,
                         properties(entity).id
                     ) AS entity_name
                WHERE (
                    document_properties.filename = $textbook
                    OR document_properties.name = $textbook
                    OR document_properties.source = $textbook
                    OR document_properties.id = $textbook
                )
                  AND entity_name IS NOT NULL
                  AND NOT entity_name =~ '[0-9a-fA-F]{32,}'
                  AND ($topic IS NULL OR toLower(entity_name) CONTAINS toLower($topic))
                RETURN DISTINCT labels(entity)[0] AS entity_type,
                       entity_name, type(r) AS relationship
                ORDER BY entity_type, entity_name
                LIMIT $limit
                """,
                textbook=textbook,
                topic=topic or None,
                limit=limit,
            )
            return [record.data() for record in result]

    def graph(self, document_id: str | None = None) -> dict:
        with self.session() as session:
            result = session.run(
                """
                MATCH (n:Document)
                WHERE $document_id IS NULL OR (
                    n.id = $document_id
                    OR n.filename = $document_id
                    OR n.name = $document_id
                    OR n.source = $document_id
                )
                OPTIONAL MATCH (n)-[r]-(m)
                WITH collect(DISTINCT n) + collect(DISTINCT m) AS all_nodes,
                     collect(DISTINCT CASE
                         WHEN r IS NULL THEN null
                         ELSE {from: elementId(n), to: elementId(m), type: type(r)}
                     END) AS relationship_values
                UNWIND all_nodes AS node
                WITH collect(DISTINCT CASE
                         WHEN node IS NULL THEN null
                         ELSE {
                             id: elementId(node),
                             labels: labels(node),
                             name: coalesce(node.name, node.title, node.id, node.text),
                             filename: node.filename
                         }
                     END) AS node_values,
                     relationship_values
                RETURN [item IN node_values WHERE item IS NOT NULL] AS nodes,
                       [item IN relationship_values WHERE item IS NOT NULL] AS relationships
                """,
                document_id=document_id,
            ).single()
            return result.data() if result else {"nodes": [], "relationships": []}

    def create_session(self, session_id: str, group: str, textbook: str) -> None:
        with self.session() as session:
            session.run(
                """
                MERGE (s:StudySession {id: $session_id})
                SET s.group = $group, s.textbook = $textbook,
                    s.created_at = coalesce(s.created_at, datetime())
                """,
                session_id=session_id,
                group=group,
                textbook=textbook,
            ).consume()

    def session_context(self, session_id: str) -> dict:
        with self.session() as session:
            result = session.run(
                "MATCH (s:StudySession {id: $session_id}) RETURN s.group AS group, s.textbook AS textbook",
                session_id=session_id,
            ).single()
            return result.data() if result else {}

    def library(self) -> dict:
        with self.session() as session:
            result = session.run(
                """
                MATCH (d:Document)
                WITH d, properties(d) AS document_properties
                WITH d, document_properties,
                     CASE
                         WHEN toLower(coalesce(document_properties.filename, '')) ENDS WITH '.pdf'
                             THEN document_properties.filename
                         WHEN toLower(coalesce(document_properties.name, '')) ENDS WITH '.pdf'
                             THEN document_properties.name
                         WHEN toLower(coalesce(document_properties.source, '')) ENDS WITH '.pdf'
                             THEN document_properties.source
                         ELSE null
                     END AS textbook_name
                WHERE textbook_name IS NOT NULL
                MATCH (d)-[:MENTIONS]->(c)
                WITH textbook_name, labels(c)[0] AS entity_type,
                     coalesce(
                         properties(c).name,
                         properties(c).title,
                         properties(c).id,
                         properties(c).text
                     ) AS topic
                WHERE NOT any(label IN labels(c) WHERE label IN [
                    'StudyGroup', 'Textbook', 'Document', 'Chunk',
                    'StudySession', 'StudyMessage'
                ])
                  AND topic IS NOT NULL
                  AND NOT topic =~ '[0-9a-fA-F]{32,}'
                ORDER BY rand()
                WITH textbook_name, entity_type, collect(DISTINCT topic)[0] AS topic
                WITH textbook_name, collect(topic) AS topics
                RETURN collect({
                    id: textbook_name,
                    filename: textbook_name,
                    topics: [
                        topic IN topics
                        WHERE topic IS NOT NULL
                    ][..12]
                }) AS textbooks
                """
            ).single()
            return {
                "textbooks": list(
                    {
                        item["filename"]: item
                        for item in (result["textbooks"] if result else [])
                        if item.get("filename")
                    }.values()
                )
            }

    def question_bank(
        self, session_id: str, mode: str, level: str, textbook: str, limit: int = 20
    ) -> list[dict]:
        label = "PreviousQuestion" if mode == "previous" else "StudyQuestion"
        with self.session() as session:
            result = session.run(
                f"""
                MATCH (q:{label})
                WHERE q.level = $level
                  AND ($mode = 'previous' OR q.textbook = $textbook OR q.textbook IS NULL)
                RETURN q.id AS question_id, q.question AS question, q.options AS options,
                       q.correct_option AS correct_option, q.answer AS answer,
                       q.citations AS citations, coalesce(q.liked, false) AS liked
                ORDER BY coalesce(q.year, 0) DESC, coalesce(q.asked_at, q.created_at) DESC
                LIMIT $limit
                """,
                mode=mode,
                level=level,
                textbook=textbook,
                limit=limit,
            )
            questions = [record.data() for record in result]
            for question in questions:
                question["citations"] = json.loads(question["citations"] or "[]")
            return questions

    def liked_question(self, session_id: str, textbook: str) -> dict | None:
        with self.session() as session:
            result = session.run(
                """
                MATCH (q:StudyQuestion)
                WHERE q.textbook = $textbook AND coalesce(q.liked, false) = true
                RETURN q.id AS question_id, q.question AS question, q.options AS options,
                       q.answer AS answer, q.citations AS citations, true AS liked
                ORDER BY rand()
                LIMIT 1
                """,
                textbook=textbook,
            ).single()
            if not result:
                return None
            item = result.data()
            item["citations"] = json.loads(item["citations"] or "[]")
            return item

    def set_question_liked(self, session_id: str, question_id: str, liked: bool) -> bool:
        with self.session() as session:
            result = session.run(
                """
                MATCH (s:StudySession {id: $session_id})
                MATCH (q:StudyQuestion {id: $question_id})
                WHERE q.textbook = s.textbook
                SET q.liked = $liked
                RETURN q.liked AS liked
                """,
                session_id=session_id,
                question_id=question_id,
                liked=liked,
            ).single()
            return bool(result["liked"]) if result else False

    def save_learning_question(
        self,
        question_id: str,
        session_id: str,
        topic: str,
        mode: str,
        question: str,
        options: list[str],
        correct_option: int,
        explanation: str,
        level: str,
        textbook: str,
    ) -> None:
        with self.session() as session:
            session.run(
                """
                MATCH (s:StudySession {id: $session_id})
                CREATE (q:ScreeningQuestion {
                    id: $question_id, topic: $topic, mode: $mode,
                    question: $question, options: $options,
                    correct_option: $correct_option, explanation: $explanation,
                    level: $level, textbook: $textbook,
                    attempts_used: 0, resolved: false, created_at: datetime()
                })
                CREATE (s)-[:HAS_SCREENING_QUESTION]->(q)
                """,
                session_id=session_id,
                question_id=question_id,
                topic=topic,
                mode=mode,
                question=question,
                options=options,
                correct_option=correct_option,
                explanation=explanation,
                level=level,
                textbook=textbook,
            ).consume()

    def screening_question(self, session_id: str, question_id: str) -> dict | None:
        with self.session() as session:
            result = session.run(
                """
                MATCH (s:StudySession {id: $session_id})-[:HAS_SCREENING_QUESTION]->(q:ScreeningQuestion {id: $question_id})
                RETURN q { .* }
                """,
                session_id=session_id,
                question_id=question_id,
            ).single()
            return result["q"] if result else None

    def record_screening_attempt(
        self, session_id: str, question_id: str, option_index: int
    ) -> dict | None:
        with self.session() as session:
            result = session.run(
                """
                MATCH (s:StudySession {id: $session_id})-[:HAS_SCREENING_QUESTION]->(q:ScreeningQuestion {id: $question_id})
                WHERE coalesce(q.resolved, false) = false
                SET q.attempts_used = q.attempts_used + 1
                SET q.resolved = ($option_index = q.correct_option OR q.attempts_used >= 3)
                RETURN q { .* }
                """,
                session_id=session_id,
                question_id=question_id,
                option_index=option_index,
            ).single()
            return result["q"] if result else None

    def save_message(self, session_id: str, role: str, content: str) -> None:
        with self.session() as session:
            session.run(
                """
                MATCH (s:StudySession {id: $session_id})
                CREATE (m:StudyMessage {role: $role, content: $content, created_at: datetime()})
                CREATE (s)-[:HAS_MESSAGE]->(m)
                """,
                session_id=session_id,
                role=role,
                content=content,
            ).consume()

    def recent_messages(self, session_id: str, limit: int = 12) -> list[dict]:
        with self.session() as session:
            result = session.run(
                """
                MATCH (s:StudySession {id: $session_id})-[:HAS_MESSAGE]->(m:StudyMessage)
                RETURN m.role AS role, m.content AS content
                ORDER BY m.created_at DESC LIMIT $limit
                """,
                session_id=session_id,
                limit=limit,
            )
            return list(reversed([record.data() for record in result]))

    def save_question(
        self,
        question_id: str,
        session_id: str,
        question: str,
        answer: str,
        options: list[str],
        correct_option: int,
        level: str,
        textbook: str,
        mode: str,
        document_id: str | None,
        citations: list[dict],
    ) -> None:
        with self.session() as session:
            session.run(
                """
                MATCH (s:StudySession {id: $session_id})
                MERGE (q:StudyQuestion {id: $question_id})
                SET q.question = $question, q.answer = $answer, q.mode = $mode,
                    q.options = $options, q.correct_option = $correct_option,
                    q.level = $level, q.textbook = $textbook,
                    q.document_id = $document_id, q.citations = $citations,
                    q.liked = coalesce(q.liked, false),
                    q.asked_at = datetime()
                MERGE (s)-[:ASKED]->(q)
                """,
                question_id=question_id,
                session_id=session_id,
                question=question,
                answer=answer,
                options=options,
                correct_option=correct_option,
                level=level,
                textbook=textbook,
                mode=mode,
                document_id=document_id,
                citations=json.dumps(citations),
            ).consume()

    def question_history(
        self, session_id: str, mode: str | None, document_id: str | None, limit: int = 10
    ) -> list[dict]:
        with self.session() as session:
            result = session.run(
                """
                MATCH (s:StudySession {id: $session_id})-[:ASKED]->(q:StudyQuestion)
                WHERE ($mode IS NULL OR q.mode = $mode)
                  AND ($document_id IS NULL OR q.document_id = $document_id)
                RETURN q.question AS question, q.answer AS answer,
                       q.options AS options, q.citations AS citations
                ORDER BY q.asked_at DESC LIMIT $limit
                """,
                session_id=session_id,
                mode=mode,
                document_id=document_id,
                limit=limit,
            )
            questions = [record.data() for record in result]
            for question in questions:
                question["citations"] = json.loads(question["citations"] or "[]")
                question["options"] = question["options"] or []
            return questions
