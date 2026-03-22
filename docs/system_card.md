# System Card — EdAdvisor GenAI

## Overview

EdAdvisor is a domain-specific Retrieval-Augmented Generation (RAG) system
for international student counselling. It answers questions about university
admissions, visa requirements, scholarships, and programme details by
retrieving relevant passages from a curated document corpus and generating
grounded, cited answers.

## Intended Use

**Primary users:** International students, education counsellors, admissions support teams.

**Supported query types:**
- Programme entry requirements (academic, language)
- Visa application guidance
- Scholarship eligibility and deadlines
- Application document requirements
- Programme structure and duration

**Out-of-scope:**
- Legal advice or visa decision-making
- Financial advice beyond general scholarship information
- Questions unrelated to international education

## Architecture

| Component | Implementation |
|-----------|---------------|
| Embedding | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store | FAISS IndexFlatIP (cosine on normalised vectors) |
| Chunking | Hierarchical (parent 1500 chars, child 400 chars) |
| Retrieval | MMR with confidence scoring |
| LLM | Google Gemini 1.5 Flash |
| Memory | Redis-backed sliding window (6 turns) |
| Evaluation | ragas (faithfulness, answer_relevancy, context_precision, context_recall) |

## Evaluation Results

| Metric | Score | Gate |
|--------|-------|------|
| Faithfulness | 0.82 | ≥ 0.75 ✓ |
| Answer relevancy | 0.79 | ≥ 0.70 ✓ |
| Context precision | 0.74 | — |
| Context recall | 0.71 | — |

*Scores reflect prompt v2 on the 20-question curated test set.*

## Failure Modes

| Failure mode | Mitigation |
|---|---|
| Answer not in corpus | Confidence threshold triggers escalation message |
| Stale visa / deadline information | System card recommends corpus refresh; answer always includes "verify with official source" instruction |
| Hallucinated citation | Only [Source N] labels that match retrieved docs are resolved; unreferenced numbers are ignored |
| Out-of-scope query | Low confidence → escalation; prompt explicitly instructs model to decline off-topic queries |

## Guardrails

1. **Retrieval gate:** Queries with confidence below 0.65 receive a canned escalation response — the LLM is never called.
2. **Prompt instruction:** The system prompt explicitly prohibits answering from outside the context.
3. **Citation extraction:** Only source numbers that appear in retrieved docs are surfaced to the user.
4. **Verification reminder:** All answers about visa or financial requirements include a reminder to verify with official sources.

## Limitations

- Accuracy depends entirely on the quality and freshness of the corpus.
- The system does not browse the web — it cannot answer about real-time changes to visa policies or deadlines.
- Language coverage is English-only in v1.
- Does not replace professional immigration or legal advice.

## Corpus Maintenance

| Activity | Recommended frequency |
|---|---|
| Corpus refresh (new programme guides, visa updates) | Quarterly |
| Re-run ragas evaluation after corpus update | On every significant update |
| Prompt version review | When ragas scores drop below gate thresholds |
