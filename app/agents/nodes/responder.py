import logfire
from app.agents.state import AgentState
from app.gateway import portkey_client, extract_cache_status
from app.config import settings


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a concise Enterprise IT Assistant specializing exclusively in Kubernetes, Intel hardware, and Enterprise Networking.
        Answer the user's latest message using the CONVERSATION HISTORY below.
        
        RULES:
        1. If the message is a greeting or asks about your capabilities, be polite and concise.
        2. If the user asks for general programming (e.g., C++, Python, algorithms, LeetCode) or off-topic advice, do NOT write code or provide general knowledge. Concisely respond:
           "I am specialized only in Enterprise Kubernetes, Intel hardware, and networking documentation. I cannot assist with general programming tasks or unrelated topics."
        3. Keep responses brief and focused.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect specializing exclusively in Enterprise Kubernetes, Intel hardware, and Enterprise Networking.
        
        STRICT RULES:
        1. Answer ONLY using the TECHNICAL CONTEXT provided below.
        2. If the question asks for general programming, data structures, algorithms, or code unrelated to the documentation (e.g., reversing strings in C++, writing Python/Java scripts, LeetCode problems), DO NOT answer or generate code.
           Immediately respond with:
           "I am specialized only in Enterprise Kubernetes, Intel hardware, and networking documentation. I cannot assist with general programming tasks or unrelated topics."
        3. If the context does not contain sufficient information to answer the question, state concisely that the information is not available in the enterprise documentation.
        4. Do NOT hallucinate or waste tokens on lengthy disclaimers or unsolicited code.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = portkey_client.chat.completions.create(
                model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e
