from __future__ import annotations

from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

from edadvisor.config import settings
from edadvisor.generation.citations import _format_context, extract_citations
from edadvisor.generation.memory import SessionMemory
from edadvisor.generation.prompts import load_prompt
from edadvisor.retrieval.retriever import RetrievalResult, retrieve


ESCALATION_RESPONSE = (
    "I don't have reliable information in my knowledge base to answer that question accurately. "
    "I'd recommend checking the official university website directly, or speaking with a counsellor "
    "who can access the latest information."
)


@dataclass
class QueryResponse:
    answer: str
    sources: list[dict]
    confidence: float
    escalated: bool
    session_id: str
    prompt_version: str
    retrieval: RetrievalResult | None = None


class RAGChain:
    """
    The main RAG query pipeline.

    For each query:
      1. Retrieve relevant chunks (with MMR + confidence scoring)
      2. If confidence too low → return escalation message
      3. Inject history + context into versioned prompt
      4. Generate answer with Gemini
      5. Parse [Source N] citations from the answer
      6. Store exchange in session memory
    """

    def __init__(self, store: FAISS):
        self._store = store
        self._memory = SessionMemory()
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,
        )
        self._prompt_version = settings.active_prompt_version
        self._prompt = load_prompt(self._prompt_version)
        logger.info(f"RAGChain ready  model={settings.gemini_model}  prompt={self._prompt_version}")

    def query(self, question: str, session_id: str) -> QueryResponse:
        # step 1 — retrieve
        result = retrieve(question, self._store)

        # step 2 — escalate if low confidence
        if result.should_escalate or not result.docs:
            logger.info(f"escalating  session={session_id[:8]}  q='{question[:60]}'")
            self._memory.add_turn(session_id, question, ESCALATION_RESPONSE)
            return QueryResponse(
                answer=ESCALATION_RESPONSE,
                sources=[],
                confidence=result.confidence,
                escalated=True,
                session_id=session_id,
                prompt_version=self._prompt_version,
                retrieval=result,
            )

        # step 3 — build context string with [Source N] labels
        context = _format_context(result.docs)
        history_str = self._memory.format_history_for_prompt(session_id)
        if history_str:
            context = f"Conversation so far:\n{history_str}\n\n---\n\n{context}"

        # step 4 — generate
        chain = self._prompt | self._llm
        try:
            response = chain.invoke({"context": context, "question": question})
            answer = response.content.strip()
        except Exception as e:
            logger.exception(f"LLM call failed: {e}")
            answer = "Sorry, I ran into a problem generating a response. Please try again."

        # step 5 — parse citations
        sources = extract_citations(answer, result.docs)

        # step 6 — remember
        self._memory.add_turn(session_id, question, answer)

        logger.info(
            f"query done  session={session_id[:8]}  "
            f"confidence={result.confidence:.3f}  sources={len(sources)}"
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=result.confidence,
            escalated=False,
            session_id=session_id,
            prompt_version=self._prompt_version,
            retrieval=result,
        )
