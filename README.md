# EdAdvisor GenAI

![CI](https://img.shields.io/github/actions/workflow/status/shakatak47/edadvisor-genai/ci.yml?label=CI&style=flat-square)
![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-1.5--flash-orange?style=flat-square)
![ragas](https://img.shields.io/badge/ragas-evaluated-purple?style=flat-square)

A RAG-based AI counsellor for international students. Ask it about university admissions, visa requirements, scholarships, or programme details — it retrieves answers from a curated document corpus and cites every source it uses.

---

## Background

Working in the education space at StudyIn, I kept running into the same problem: students would ask the same questions repeatedly — IELTS cutoffs, visa timelines, scholarship eligibility — and the answers were scattered across PDFs, HTML pages, and internal guides that counsellors had to dig through manually.

I built EdAdvisor to solve that. It's not a chatbot that makes things up. It only answers from what's actually in the corpus, and if it's not confident enough, it says so and points the student to an official source. That escalation behaviour was the most important design decision — I'd rather the system admit it doesn't know than hallucinate a visa deadline.

---

## What it does

- Answers questions grounded in your document corpus (PDFs, HTML pages, DOCX files)
- Returns inline citations (`[Source 1]`, `[Source 2]`) mapped to actual documents
- Escalates to a human when retrieval confidence is below the threshold — no guessing
- Maintains session memory across a conversation (Redis-backed, 6-turn window)
- Exposes a FastAPI endpoint + a Streamlit chat UI you can run locally or via Docker

---

## How it works

```
Student question
    │
    ▼
FAISS vector store  ──  MMR retrieval (top-20 → reranked to top-5)
    │
    ▼
Confidence score ──── below 0.65 ──→  "I don't have reliable info. Please check official sources."
    │ above 0.65
    ▼
Versioned system prompt (prompts/rag_system_v2.yaml)
  + session history
  + retrieved context
    │
    ▼
Gemini 1.5 Flash
    │
    ▼
Citation parser  ──  [Source N] references resolved to document metadata
    │
    ▼
Redis memory update
    │
    ▼
{ answer, sources, confidence, escalated }
```

The retrieval uses hierarchical chunking: parent chunks (1500 chars) preserve section-level context, while child chunks (400 chars) are used for matching. When a child chunk is retrieved, the parent content is what actually goes into the prompt — so the LLM always sees enough context to answer multi-part questions properly.

---

## Getting started

```bash
git clone https://github.com/shakatak47/edadvisor-genai
cd edadvisor-genai
cp .env.example .env          # add your GOOGLE_API_KEY
pip install -e ".[dev]"

# 1. Seed the corpus with sample pages
python scripts/fetch_sample_corpus.py
# Drop your own PDFs / HTML / DOCX into data/corpus/ as well

# 2. Build the FAISS index
make ingest

# 3. Start the API
make serve

# 4. Open the chat UI (separate terminal)
make ui
```

Chat UI → http://localhost:8501
API docs → http://localhost:8000/docs

---

## API example

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "student-001",
    "question": "What IELTS score do I need for a UK Master'\''s?"
  }'
```

```json
{
  "answer": "Most UK universities require an overall IELTS of 6.5 [Source 1], with no band below 6.0. Research-intensive institutions like Oxford or Imperial often ask for 7.0 or above [Source 2]. Always check the specific programme page as requirements vary.",
  "sources": [
    { "source_n": 1, "source": "english_requirements.pdf", "doc_type": "pdf" },
    { "source_n": 2, "source": "program_guide.html", "doc_type": "html" }
  ],
  "confidence": 0.84,
  "escalated": false,
  "prompt_version": "v2"
}
```

---

## Evaluation

I use [ragas](https://docs.ragas.io) to measure whether the answers are actually grounded in the retrieved context. There's a CI gate on faithfulness — if a corpus or prompt change drops it below 0.75, the build fails. This is the thing that separates a quick demo from something you can actually deploy.

```bash
make eval
```

| Metric | v2 score | Gate |
|---|---|---|
| Faithfulness | 0.82 | ≥ 0.75 ✓ |
| Answer relevancy | 0.79 | ≥ 0.70 ✓ |
| Context precision | 0.74 | — |
| Context recall | 0.71 | — |

The test set is 20 curated Q&A pairs covering the main query types — visa questions, academic requirements, scholarships. Full methodology in [docs/system_card.md](docs/system_card.md).

---

## Prompt versioning

Prompt changes have a real effect on ragas scores, so I version them as YAML files (`prompts/rag_system_v1.yaml`, `v2.yaml`) rather than hardcoding them in Python. Each file records its eval scores and a changelog entry. You can swap the active version in `.env` without touching code, and roll back if scores regress.

---

## Project structure

```
edadvisor-genai/
├── src/edadvisor/
│   ├── ingestion/      loaders, chunkers, FAISS store, pipeline
│   ├── retrieval/      MMR retrieval + confidence scoring
│   ├── generation/     LangChain chain, prompt loader, citation parser, memory
│   ├── serving/        FastAPI app + Pydantic schemas
│   └── evaluation/     ragas runner
├── prompts/            versioned system prompt YAML files
├── evaluation/         test_set.json (20 curated Q&A pairs)
├── ui/                 Streamlit chat interface
├── tests/unit/         chunkers, retriever, citations, memory
├── docker/             Dockerfile.api, Dockerfile.ui, docker-compose.yml
└── docs/               system_card.md
```

---

## Running with Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

This starts the API and the Streamlit UI. No local Python setup needed beyond the `.env` file.

---

## Limitations worth knowing

The system only knows what's in the corpus. It can't browse the web, so if visa rules or deadlines change after your last ingest, the answers will be stale. The recommended approach is to refresh the corpus quarterly and re-run the evaluation suite after any significant update.

It's also English-only for now.

---

*Built by [Shakti Saurabh](https://github.com/shakatak47) — Data Scientist at StudyIn, working on production AI for student operations.*
