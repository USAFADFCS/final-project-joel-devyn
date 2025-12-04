# streamlit_app.py

import os
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ─────────────────────────────
# Config
# ─────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()
PERSIST_DIR = PROJECT_ROOT / "policy_index"          # must match final_project.py
COLLECTION_NAME = "usafa_policy_rag"                 # must match final_project.py

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"                # same as SentenceTransformerEmbedder
CHAT_MODEL = "gpt-4.1-mini"                          # bump to gpt-4.1 / gpt-4o if you want

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────
# Backend init
# ─────────────────────────────

def ensure_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY not set.\n\n"
            "Export it in your shell or load it from a `.env` file."
        )
        st.stop()


def init_backends():
    """Initialize Chroma + SentenceTransformer once and keep in session_state."""
    # Ensure index directory exists
    if not PERSIST_DIR.exists():
        st.error(
            f"Policy index not found at `{PERSIST_DIR}`.\n\n"
            "Run this once from your project root:\n\n"
            "```bash\n"
            "python3 final_project.py --build-index-only\n"
            "```\n"
            "Then redeploy / rerun Streamlit."
        )
        st.stop()

    # Chroma client
    if "chroma_client" not in st.session_state:
        st.session_state.chroma_client = chromadb.PersistentClient(
            path=str(PERSIST_DIR)
        )

    # Collection
    if "policy_collection" not in st.session_state:
        client_chroma = st.session_state.chroma_client
        collections = {c.name for c in client_chroma.list_collections()}
        if COLLECTION_NAME not in collections:
            st.error(
                f"Chroma collection '{COLLECTION_NAME}' not found in index.\n\n"
                "Make sure your CLI used the same collection_name."
            )
            st.stop()
        st.session_state.policy_collection = client_chroma.get_collection(
            COLLECTION_NAME
        )

    # Embed model
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def get_collection_and_model():
    return st.session_state.policy_collection, st.session_state.embed_model


# ─────────────────────────────
# Core RAG helpers
# ─────────────────────────────

def retrieve_context(question: str, k: int = 8) -> List[Dict[str, Any]]:
    """Use prebuilt Chroma index + SentenceTransformer to get top-k chunks."""
    collection, embed_model = get_collection_and_model()
    query_vec = embed_model.encode([question])[0].tolist()

    res = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]

    out: List[Dict[str, Any]] = []
    for i, text in enumerate(docs):
        meta = metas[i] if metas and i < len(metas) else {}
        out.append(
            {
                "id": ids[i] if ids and i < len(ids) else f"chunk_{i}",
                "source": (meta or {}).get("source", "unknown"),
                "text": text,
            }
        )
    return out


def build_system_prompt(role_desc: str = "") -> str:
    base = (
        "You are a USAFA cadet standards, duties, and dress/appearance assistant.\n"
        "You must:\n"
        "  - Base your answers ONLY on the provided context (policy chunks and MFRs).\n"
        "  - Cite which document and, if possible, section/paragraph labels you are using.\n"
        "  - If something is not supported by the context, clearly say you are unsure.\n"
        "  - Never invent official policy.\n"
    )
    if role_desc:
        base += f"\nYou should answer from the perspective of: {role_desc}\n"
    return base


def run_chat(system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


# ─────────────────────────────
# High-level behaviors
# ─────────────────────────────

def answer_with_policies(question: str, role: str = "") -> Dict[str, Any]:
    ctx_chunks = retrieve_context(question, k=8)
    if not ctx_chunks:
        return {
            "answer": (
                "I couldn't retrieve any relevant policy chunks from the index. "
                "Make sure the index is built and non-empty."
            ),
            "context": [],
        }

    context_block = "\n\n".join(
        [f"[{c['source']}] {c['text']}" for c in ctx_chunks]
    )

    system_prompt = build_system_prompt(role)
    user_prompt = (
        f"Question:\n{question}\n\n"
        "Relevant context from policy corpus:\n"
        "-----------------\n"
        f"{context_block}\n"
        "-----------------\n\n"
        "Answer the question using only the context above. "
        "Cite documents like [Source: AFCWI 36-3501] or [Source: DAFI 36-2903]."
    )

    answer = run_chat(system_prompt, user_prompt)
    return {"answer": answer, "context": ctx_chunks}


def analyze_uploaded_doc(text: str, mode: str) -> Dict[str, Any]:
    ctx_chunks = retrieve_context(
        "overall cadet standards, duties, and dress & appearance", k=12
    )
    context_block = "\n\n".join(
        [f"[{c['source']}] {c['text']}" for c in ctx_chunks]
    )

    base_sys = build_system_prompt()
    truncated = text[:8000]

    if mode == "Compliance Review":
        user = (
            "You are reviewing a proposed event/training/document for compliance with USAFA policy.\n\n"
            "User document:\n"
            "```text\n"
            f"{truncated}\n"
            "```\n\n"
            "Policy context:\n"
            "-----------------\n"
            f"{context_block}\n"
            "-----------------\n\n"
            "Tasks:\n"
            "1. Identify any likely policy violations or risk areas (hazing, improper PT, uniform issues, etc.).\n"
            "2. Cite specific policy sources from the context.\n"
            "3. Suggest concrete fixes.\n"
        )
    elif mode == "Key Findings":
        user = (
            "Extract key actionable findings from the user's document.\n\n"
            "Document:\n"
            "```text\n"
            f"{truncated}\n"
            "```\n\n"
            "Policy context (for understanding roles/expectations):\n"
            "-----------------\n"
            f"{context_block}\n"
            "-----------------\n\n"
            "Output:\n"
            "- Mandatory tasks / actions\n"
            "- Roles and responsibilities\n"
            "- Deadlines/time windows\n"
            "- Hazards / safety concerns\n"
            "- Required approvals/authorities\n"
        )
    elif mode == "Style Check":
        user = (
            "You are a style and consistency checker for USAFA documents.\n\n"
            "Document:\n"
            "```text\n"
            f"{truncated}\n"
            "```\n\n"
            "Policy context (for terminology and examples):\n"
            "-----------------\n"
            f"{context_block}\n"
            "-----------------\n\n"
            "Tasks:\n"
            "1. Point out style/format/terminology issues (NOT deep policy violations).\n"
            "2. Group into categories: Terminology & Rank, Uniform Naming & Caps, Structure, Clarity & Tone.\n"
            "3. Suggest specific improvements.\n"
        )
    elif mode == "Regulation Format":
        user = (
            "Rewrite the user's content into a formal USAFA/Air Force-style memorandum.\n\n"
            "Original content:\n"
            "```text\n"
            f"{truncated}\n"
            "```\n\n"
            "Use policy context only for terminology; do NOT invent new policy.\n"
            "Policy context:\n"
            "-----------------\n"
            f"{context_block}\n"
            "-----------------\n\n"
            "Output:\n"
            "- Proper header (office symbol, date, etc.)\n"
            "- MEMORANDUM FOR / FROM / SUBJECT (if appropriate)\n"
            "- Structured body paragraphs\n"
            "- Signature block placeholder\n"
        )
    else:
        user = f"Document:\n```text\n{truncated}\n```\n\nGive a short summary."

    answer = run_chat(base_sys, user)
    return {"answer": answer, "context": ctx_chunks}


# ─────────────────────────────
# UI
# ─────────────────────────────

def main():
    ensure_openai_key()
    init_backends()

    st.set_page_config(
        page_title="USAFA Policy Assistant",
        page_icon="🇺🇸",
        layout="wide",
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### 🇺🇸 USAFA Policy Assistant")
        st.markdown(
            "- Uses a prebuilt Chroma index from your CLI tool\n"
            "- Answers questions over standards / duties / dress & appearance\n"
            "- Can review or reformat your documents\n"
        )
        collection, _ = get_collection_and_model()
        try:
            count = collection.count()
            st.caption(f"Indexed chunks: **{count}**")
        except Exception:
            st.caption("Indexed chunks: (unknown)")

        st.markdown("---")
        st.caption(
            "Index built with:\n\n"
            "`python3 final_project.py --build-index-only`"
        )

    st.title("USAFA Policy Assistant")

    tabs = st.tabs(["💬 Ask the Agent", "📄 Analyze a Document"])

    # ── Tab 1: Q&A ──
    with tabs[0]:
        st.subheader("Ask about USAFA cadet standards, duties, and dress/appearance")

        col1, col2 = st.columns([3, 2])
        with col1:
            question = st.text_area(
                "Your question",
                placeholder="Example: What paperwork should a C3C element leader use for repeated tardiness?",
                height=130,
            )
        with col2:
            role = st.text_input(
                "Optional perspective",
                placeholder="e.g., C4C, SQ/CC, First Sergeant",
            )

        show_context = st.checkbox("Show retrieved context", value=False)

        if st.button("Ask", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking with policy context…"):
                    result = answer_with_policies(question.strip(), role.strip())
                st.markdown("### Answer")
                st.markdown(result["answer"])

                if show_context:
                    with st.expander("View retrieved context"):
                        for c in result["context"]:
                            st.markdown(f"**Source:** `{c['source']}`")
                            st.write(c["text"])
                            st.markdown("---")

    # ── Tab 2: Document analysis ──
    with tabs[1]:
        st.subheader("Upload a document for policy-aware analysis")

        col1, col2 = st.columns([3, 2])
        with col1:
            uploaded = st.file_uploader(
                "Upload a text/markdown file",
                type=["txt", "md"],
            )
        with col2:
            mode = st.selectbox(
                "Analysis mode",
                [
                    "Compliance Review",
                    "Key Findings",
                    "Style Check",
                    "Regulation Format",
                ],
            )

        show_context_doc = st.checkbox(
            "Show retrieved context for document analysis",
            value=False,
        )

        if st.button("Analyze Document", type="primary"):
            if not uploaded:
                st.warning("Please upload a document first.")
            else:
                raw = uploaded.read().decode("utf-8", errors="ignore")
                if not raw.strip():
                    st.warning("Uploaded document is empty.")
                else:
                    with st.spinner(f"Running {mode}…"):
                        result = analyze_uploaded_doc(raw, mode)

                    st.markdown(f"### {mode} Result")
                    st.markdown(result["answer"])

                    if show_context_doc:
                        with st.expander("View retrieved context"):
                            for c in result["context"]:
                                st.markdown(f"**Source:** `{c['source']}`")
                                st.write(c["text"])
                                st.markdown("---")


if __name__ == "__main__":
    main()
