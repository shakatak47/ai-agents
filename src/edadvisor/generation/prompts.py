from __future__ import annotations

from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from loguru import logger

from edadvisor.config import settings


def load_prompt(version: str | None = None) -> ChatPromptTemplate:
    """
    Load a prompt template from prompts/<version>.yaml.
    Falls back to the inline default if the file isn't found.
    """
    ver = version or settings.active_prompt_version
    path = Path(settings.prompts_dir) / f"rag_system_{ver}.yaml"

    if path.exists():
        try:
            data = yaml.safe_load(path.read_text())
            system_tpl = data["system_template"]
            human_tpl  = data.get("human_template", "{context}\n\nQuestion: {question}")
            logger.debug(f"prompt loaded from {path.name}  eval_scores={data.get('eval_scores')}")
            return ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_tpl),
                HumanMessagePromptTemplate.from_template(human_tpl),
            ])
        except Exception as e:
            logger.warning(f"failed to load prompt file {path}: {e}  using default")

    # inline default
    system = (
        "You are EdAdvisor, a knowledgeable assistant helping international students "
        "with university admissions, visa requirements, scholarships, and programme details.\n\n"
        "Answer ONLY from the provided context. If the context does not contain enough "
        "information to answer fully, say so and recommend the student check the official "
        "university website or contact their counsellor.\n\n"
        "Cite your sources inline using [Source N] references. "
        "If you are uncertain about a specific detail (e.g. a deadline or score requirement), "
        "flag it explicitly rather than guessing.\n\n"
        "Context:\n{context}"
    )
    human = "Question: {question}"

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(human),
    ])
