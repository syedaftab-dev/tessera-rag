import os
import re
import html
import time
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)

# Page configuration
st.set_page_config(
    page_title="Tessera - RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load external stylesheet
css_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# SVG Icons from design reference
LOGO_SVG = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3.5"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/></svg>'
USER_SVG = '<svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21c1-4 4-6 8-6s7 2 8 6"/></svg>'


def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:8]
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "questions_count" not in st.session_state:
        st.session_state.questions_count = 0
    if "total_latency" not in st.session_state:
        st.session_state.total_latency = 0.0
    if "pipe_state" not in st.session_state:
        st.session_state.pipe_state = {
            "guard": "",
            "plan": "",
            "retr": "",
            "rank": "",
            "resp": "",
        }


init_session()


def extract_file_chips(sources: list) -> list[str]:
    """Extract clean file names only from sources (no passage text)."""
    clean_files = []
    valid_exts = ("pdf", "html", "htm", "docx", "pptx", "txt")
    for s in sources:
        if not isinstance(s, str):
            continue
        s_clean = s.strip()
        base = os.path.basename(s_clean).strip()
        ext = base.lower().rsplit(".", 1)[-1] if "." in base else ""
        if ext in valid_exts and len(base) <= 100:
            if base not in clean_files:
                clean_files.append(base)
        else:
            # Check if text contains a referenced document name
            matches = re.findall(
                r"[\w\-\s()\[\]]+\.(?:pdf|html|htm|docx|pptx|txt)", s_clean, re.IGNORECASE
            )
            for m in matches:
                m_clean = os.path.basename(m.strip())
                if len(m_clean) <= 100 and m_clean not in clean_files:
                    clean_files.append(m_clean)
    return clean_files


from markdown_it import MarkdownIt

_md = MarkdownIt().disable("linkify")


def format_answer_html(text: str) -> str:
    """Format markdown answer text to semantic HTML with typography."""
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").strip()
    return _md.render(normalized)


def render_pipe_html(pipe_dict: dict) -> str:
    """Render the animated 5-stage progress pipeline HTML."""
    return (
        '<div class="side-section">'
        '<h3>Progress of your question</h3>'
        '<ol class="pipe">'
        f'<li class="{pipe_dict.get("guard", "")}">Safety check</li>'
        f'<li class="{pipe_dict.get("plan", "")}">Understanding</li>'
        f'<li class="{pipe_dict.get("retr", "")}">Searching documents</li>'
        f'<li class="{pipe_dict.get("rank", "")}">Picking best matches</li>'
        f'<li class="{pipe_dict.get("resp", "")}">Writing answer</li>'
        '</ol>'
        '</div>'
    )


# --- SIDEBAR (Fixed height, no scroll) ---
with st.sidebar:
    # 1. Brand
    st.markdown(
        f'<div class="brand">'
        f'<div class="logo">{LOGO_SVG}</div>'
        f'<div><b>Tessera - RAG</b><span>Enterprise IT assistant</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 2. System Health (Generic labels only)
    st.markdown(
        '<div class="side-section">'
        '<h3>System health</h3>'
        '<ul class="status">'
        '<li><span class="l"><i class="dot"></i>Assistant</span><span class="v">Online</span></li>'
        '<li><span class="l"><i class="dot"></i>Safety filter</span><span class="v">Active</span></li>'
        '<li><span class="l"><i class="dot"></i>Monitoring</span><span class="v">On</span></li>'
        '</ul>'
        '</div>',
        unsafe_allow_html=True,
    )

    # 3. This Session
    q_count = st.session_state.questions_count
    avg_lat = (
        f"{(st.session_state.total_latency / q_count):.1f}" if q_count > 0 else "0.0"
    )
    st.markdown(
        f'<div class="side-section">'
        f'<h3>This session</h3>'
        f'<div class="session-card">'
        f'<div class="sid" title="Session ID">{st.session_state.session_id}</div>'
        f'<div class="nums">'
        f'<div><b>{q_count}</b><span>Questions</span></div>'
        f'<div><b>{avg_lat}s</b><span>Avg response</span></div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 4. Progress of your question (dynamic slot for live animation)
    pipe_slot = st.empty()
    pipe_slot.markdown(render_pipe_html(st.session_state.pipe_state), unsafe_allow_html=True)

    # 5. Start new session (neutral outline button)
    if st.button("Start new session", key="start_new_session_btn"):
        st.session_state.session_id = uuid.uuid4().hex[:8]
        st.session_state.messages = []
        st.session_state.questions_count = 0
        st.session_state.total_latency = 0.0
        st.session_state.pipe_state = {
            "guard": "",
            "plan": "",
            "retr": "",
            "rank": "",
            "resp": "",
        }
        st.rerun()


# --- MAIN CONTENT AREA ---
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        # Empty State
        st.markdown(
            '<div class="hello">'
            '<h1>Ask your IT documentation. Get answers you can check.</h1>'
            '<p class="lede">Tessera is a private assistant for your enterprise IT knowledge base, already loaded and maintained by your team, so there is nothing to upload. It searches your Kubernetes, Intel hardware and networking documents and answers in plain language, naming the files it used. Use it to resolve a query fast, understand a configuration, and ask follow-ups.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # Chat Messages
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                safe_user_text = html.escape(msg["content"])
                st.markdown(
                    f'<div class="msg">'
                    f'<div class="who"><span class="me">{USER_SVG}</span>You</div>'
                    f'<div class="user-text">{safe_user_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                if msg.get("blocked"):
                    safe_blocked_text = html.escape(msg["content"])
                    st.markdown(
                        f'<div class="msg">'
                        f'<div class="who"><span class="logo">{LOGO_SVG}</span>Tessera</div>'
                        f'<div class="blocked">'
                        f'<b>This is outside what I can help with.</b>'
                        f'<span>{safe_blocked_text}</span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    ans_html = format_answer_html(msg["content"])
                    chips_html = ""
                    sources = msg.get("sources", [])
                    if sources:
                        clean_files = extract_file_chips(sources)
                        if clean_files:
                            chips = "".join(f"<i>{html.escape(f)}</i>" for f in clean_files)
                            chips_html = f'<div class="srcs"><span>Based on</span>{chips}</div>'

                    st.markdown(
                        f'<div class="msg">'
                        f'<div class="who"><span class="logo">{LOGO_SVG}</span>Tessera</div>'
                        f'<div class="answer">{ans_html}</div>'
                        f'{chips_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

import concurrent.futures

# Working placeholder for live query
working_slot = st.empty()

# --- CHAT INPUT ---
if prompt := st.chat_input("Ask about Kubernetes, Intel hardware or networking"):
    # Append user question
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "blocked": False, "sources": []}
    )

    # Show Working on it spinner
    working_slot.markdown(
        f'<div class="msg">'
        f'<div class="who"><span class="logo">{LOGO_SVG}</span>Tessera</div>'
        f'<div class="working"><i class="spin"></i>Working on it</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Call Backend API asynchronously in background thread so stages animate live
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    query_url = f"{backend_url.rstrip('/')}/query"
    payload = {"q": prompt, "thread_id": st.session_state.session_id}

    t0 = time.time()
    try:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(requests.post, query_url, json=payload, timeout=60)

        # Stage 1: Safety check active
        current_pipe = {"guard": "active", "plan": "", "retr": "", "rank": "", "resp": ""}
        pipe_slot.markdown(render_pipe_html(current_pipe), unsafe_allow_html=True)

        # Poll while NeMo Guardrails evaluates
        guard_ticks = 0
        while not future.done() and guard_ticks < 14:  # up to ~3.5s
            time.sleep(0.25)
            guard_ticks += 1

        data = None
        is_blocked = False
        is_conversational = False
        error_msg = None

        if future.done():
            res = future.result()
            if res.status_code == 200:
                data = res.json()
                is_blocked = (data.get("status") == "Blocked by guardrails.") or any(
                    "Guardrails Fired" in str(s) for s in data.get("thought_process", [])
                )
                tp_str = " ".join(str(s) for s in data.get("thought_process", []))
                is_conversational = "Retrieval: Skipped" in tp_str or "Conversational" in tp_str
            else:
                error_msg = "The assistant service is temporarily unavailable."

        if is_blocked:
            # Halt at safety check with danger red dot
            current_pipe = {"guard": "stop", "plan": "", "retr": "", "rank": "", "resp": ""}
            pipe_slot.markdown(render_pipe_html(current_pipe), unsafe_allow_html=True)
            time.sleep(0.4)
        else:
            # Stage 2: Understanding active (LLM Planner)
            current_pipe = {"guard": "done", "plan": "active", "retr": "", "rank": "", "resp": ""}
            pipe_slot.markdown(render_pipe_html(current_pipe), unsafe_allow_html=True)

            plan_ticks = 0
            while not future.done() and plan_ticks < 7:  # ~1.4s
                time.sleep(0.2)
                plan_ticks += 1

            if future.done() and data is None:
                res = future.result()
                if res.status_code == 200:
                    data = res.json()
                    is_blocked = (data.get("status") == "Blocked by guardrails.") or any(
                        "Guardrails Fired" in str(s) for s in data.get("thought_process", [])
                    )
                    tp_str = " ".join(str(s) for s in data.get("thought_process", []))
                    is_conversational = "Retrieval: Skipped" in tp_str or "Conversational" in tp_str
                else:
                    error_msg = "The assistant service is temporarily unavailable."

            if is_blocked:
                current_pipe = {"guard": "stop", "plan": "", "retr": "", "rank": "", "resp": ""}
                pipe_slot.markdown(render_pipe_html(current_pipe), unsafe_allow_html=True)
                time.sleep(0.4)
            elif is_conversational:
                # Conversational skips retrieval & ranking
                current_pipe = {"guard": "done", "plan": "done", "retr": "skip", "rank": "skip", "resp": "active"}
                pipe_slot.markdown(render_pipe_html(current_pipe), unsafe_allow_html=True)
                time.sleep(0.5)
            else:
                # Stage 3: Searching documents active (Qdrant Vector Retrieval)
                current_pipe = {"guard": "done", "plan": "done", "retr": "active", "rank": "", "resp": ""}
                pipe_slot.markdown(render_pipe_html(current_pipe), unsafe_allow_html=True)

                retr_ticks = 0
                while not future.done() and retr_ticks < 7:  # ~1.4s
                    time.sleep(0.2)
                    retr_ticks += 1

                # Stage 4: Picking best matches active (FlashRank Reranking)
                current_pipe = {"guard": "done", "plan": "done", "retr": "done", "rank": "active", "resp": ""}
                pipe_slot.markdown(render_pipe_html(current_pipe), unsafe_allow_html=True)

                rank_ticks = 0
                while not future.done() and rank_ticks < 7:  # ~1.4s
                    time.sleep(0.2)
                    rank_ticks += 1

                # Stage 5: Writing answer active (Groq Answer Synthesis)
                current_pipe = {"guard": "done", "plan": "done", "retr": "done", "rank": "done", "resp": "active"}
                pipe_slot.markdown(render_pipe_html(current_pipe), unsafe_allow_html=True)

        # Wait for backend completion if still synthesizing
        if data is None and error_msg is None:
            res = future.result()
            if res.status_code == 200:
                data = res.json()
                is_blocked = (data.get("status") == "Blocked by guardrails.") or any(
                    "Guardrails Fired" in str(s) for s in data.get("thought_process", [])
                )
                tp_str = " ".join(str(s) for s in data.get("thought_process", []))
                is_conversational = "Retrieval: Skipped" in tp_str or "Conversational" in tp_str
            else:
                error_msg = "The assistant service is temporarily unavailable."

        latency = time.time() - t0

        if data:
            answer_text = data.get("answer", "")
            sources = data.get("sources", [])

            if is_blocked:
                current_pipe = {"guard": "stop", "plan": "", "retr": "", "rank": "", "resp": ""}
            elif is_conversational:
                current_pipe = {"guard": "done", "plan": "done", "retr": "skip", "rank": "skip", "resp": "done"}
            else:
                current_pipe = {"guard": "done", "plan": "done", "retr": "done", "rank": "done", "resp": "done"}

            pipe_slot.markdown(render_pipe_html(current_pipe), unsafe_allow_html=True)
            time.sleep(0.2)
            st.session_state.pipe_state = current_pipe

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "blocked": is_blocked,
                    "sources": sources if not is_blocked else [],
                }
            )
            st.session_state.questions_count += 1
            st.session_state.total_latency += latency
        else:
            current_pipe = {"guard": "stop", "plan": "", "retr": "", "rank": "", "resp": ""}
            pipe_slot.markdown(render_pipe_html(current_pipe), unsafe_allow_html=True)
            st.session_state.pipe_state = current_pipe
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg or "The assistant service is temporarily unavailable.",
                    "blocked": True,
                    "sources": [],
                }
            )
    except Exception:
        current_pipe = {"guard": "stop", "plan": "", "retr": "", "rank": "", "resp": ""}
        pipe_slot.markdown(render_pipe_html(current_pipe), unsafe_allow_html=True)
        st.session_state.pipe_state = current_pipe
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Could not connect to the assistant service. Please check that the backend is running.",
                "blocked": True,
                "sources": [],
            }
        )

    working_slot.empty()
    st.rerun()
