import logfire
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from groq import Groq

from app.config import settings


# Production gateway config:
# If PORTKEY_API_KEY is configured, route via Portkey.
# Otherwise, fall back to direct Groq client seamlessly.
if settings.PORTKEY_API_KEY:
    try:
        from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
        portkey_client = Portkey(
            api_key=settings.PORTKEY_API_KEY
        )
    except Exception as e:
        logfire.warning(f"Portkey init failed: {e}. Falling back to native Groq client.")
        portkey_client = Groq(api_key=settings.GROQ_API_KEY)
else:
    portkey_client = Groq(api_key=settings.GROQ_API_KEY)


def get_langchain_llm(feature: str = "rag"):
    """
    Returns a Portkey-backed ChatOpenAI if Portkey is configured,
    or direct ChatGroq otherwise.
    """
    if settings.PORTKEY_API_KEY:
        try:
            from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
            return ChatOpenAI(
                api_key=settings.PORTKEY_API_KEY,
                base_url=PORTKEY_GATEWAY_URL,
                model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
                temperature=0,
                default_headers=createHeaders(
                    api_key=settings.PORTKEY_API_KEY,
                    metadata={
                        "feature": feature,
                        "_user": "rag-system",
                        "environment": "production"
                    }
                )
            )
        except Exception:
            pass
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0
    )

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"