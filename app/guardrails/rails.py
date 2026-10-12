import re
import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None

DEFAULT_OFF_TOPIC_REFUSAL = (
    "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking documentation. "
    "I cannot assist with general programming tasks or unrelated topics."
)

REFUSAL_TRIGGERS = [
    "can't help",
    "cannot help",
    "can't assist",
    "cannot assist",
    "not within the allowed topics",
    "not allowed to answer",
    "off-topic",
    "off topic",
    "unrelated topic",
    "sorry, but i can't",
    "sorry, but i cannot",
    "i maintain consistent guidelines",
    "enterprise it assistant focused on kubernetes",
    "cannot assist with general programming",
    "hello! i'm your enterprise it assistant",
    "goodbye! feel free to return",
    "i'm an enterprise ai assistant with deep expertise",
]


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses openai/gpt-oss-20b for fast intent classification at the gate.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="openai/gpt-oss-20b",
        temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised (openai/gpt-oss-20b).")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        raw_text = result.get("content", "") if isinstance(result, dict) else str(result)
        
        # Strip internal reasoning tags if present
        clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        norm_text = clean_text.replace("’", "'").replace("‘", "'").lower()

        # Check if rail or refusal fired
        fired = any(indicator.lower() in norm_text for indicator in RAIL_INDICATORS) or any(
            trigger in norm_text for trigger in REFUSAL_TRIGGERS
        )

        if fired:
            # If the response is an off-topic refusal or apology, return the clean authoritative refusal
            if any(term in norm_text for term in ["can't help", "cannot help", "cannot assist", "sorry", "not within"]):
                response_text = DEFAULT_OFF_TOPIC_REFUSAL
            else:
                response_text = clean_text

            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, response_text

        logfire.info("✅ Guardrails passed.")
        return False, None
