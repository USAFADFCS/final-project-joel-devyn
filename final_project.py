import asyncio
import logging
from pathlib import Path
import os
if "KMP_DUPLICATE_LIB_OK" not in os.environ:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PERSIST_DIR = "policy_index"  # on-disk Chroma DB for policies


import argparse
import urllib.request
from textwrap import shorten

# Optional dependencies for vector store and PDF/OCR
try:
    import chromadb
    CHROMADB_LOADED = True
except ImportError:
    print("chromadb not found. Run 'pip install chromadb'")
    chromadb = None
    CHROMADB_LOADED = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PdfReader = None
    PYPDF_AVAILABLE = False

# Optional OCR support for image-only PDFs (EXORD, etc.)
try:
    from pdf2image import convert_from_path
    import pytesseract
    # Point pytesseract directly to the Homebrew-installed tesseract binary
    # Adjust this if `which tesseract` gives a different path.
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# Optional: handle pdf2image's need for Poppler (pdfinfo, etc.)
try:
    from pdf2image.exceptions import PDFInfoNotInstalledError
except ImportError:
    PDFInfoNotInstalledError = Exception  # fallback

# Try to detect Poppler path (Homebrew on Apple Silicon/Intel)
POPPLER_PATH = None
if OCR_AVAILABLE:
    # If user explicitly sets POPPLER_PATH, trust that
    env_poppler = os.environ.get("POPPLER_PATH")
    if env_poppler:
        POPPLER_PATH = env_poppler
    else:
        # Common Homebrew location on macOS
        if Path("/opt/homebrew/bin/pdfinfo").exists():
            POPPLER_PATH = "/opt/homebrew/bin"
        elif Path("/usr/local/bin/pdfinfo").exists():
            POPPLER_PATH = "/usr/local/bin"


from fairlib.utils.document_processor import DocumentProcessor

from fairlib import (
    settings,
    OpenAIAdapter,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    LongTermMemory,
    ChromaDBVectorStore,
    ReActPlanner,
    SimpleAgent,
    SentenceTransformerEmbedder,
    SimpleRetriever,
    KnowledgeBaseQueryTool,
)

# ----------------- BASIC CONFIG -----------------

# Use your API keys from environment
settings.api_keys.openai_api_key = os.getenv("OPENAI_API_KEY")
settings.api_keys.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

logging.basicConfig(
    level=logging.INFO,  # you can change to WARNING or ERROR to be quieter
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------- POLICY CORPUS CONFIG -----------------

DAFI_URL = (
    "https://static.e-publishing.af.mil/production/1/af_a1/publication/"
    "dafi36-2903/dafi36-2903.pdf"
)
DAFI_LOCAL_PATH = Path("DAFI36-2903_Dress_and_Personal_Appearance.pdf")

POLICY_DOC_PATHS: list[Path] = [
    Path("CS34_Discipline_and_Reward_MFR.md"),
    Path("AFCWI 36-3501 Cadet Standards and Duties - 29 July 2025 (1).pdf"),
    Path("AFCW CD 2024 - What Does my Job Mean.pdf"),
    Path("EXORD 25-003 USAFA Dress and Appearance Standards.pdf"),
    Path("USAFA Dress & Appearance Standards.pdf"),
    Path("Hawg_Spins.md"),
    DAFI_LOCAL_PATH,
]

# --------------- HELPERS: TEXT SPLIT, PDF, OCR ---------------

def split_text(text: str, chunk_size: int = 3000, chunk_overlap: int = 200) -> list[str]:
    """Splits a long text into smaller, overlapping character-based chunks."""
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += max(1, chunk_size - chunk_overlap)
    return chunks


def extract_pdf_text_basic(path: Path) -> str:
    """Basic PDF text extraction using pypdf, if available."""
    if not PYPDF_AVAILABLE:
        logger.warning("pypdf not installed; cannot extract PDF text for %s", path)
        return ""
    try:
        reader = PdfReader(str(path))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        logger.error("pypdf failed on %s: %s", path, e, exc_info=True)
        return ""


try:
    from pdf2image.exceptions import PDFInfoNotInstalledError
except ImportError:
    PDFInfoNotInstalledError = Exception  # fallback


def extract_pdf_text_ocr(path: Path) -> str:
    """Try to OCR pages of a PDF if normal text extraction fails."""
    if not OCR_AVAILABLE:
        logger.warning(
            "OCR libraries (pdf2image + pytesseract) not installed; cannot OCR %s",
            path,
        )
        return ""

    if POPPLER_PATH is None:
        logger.warning(
            "POPPLER_PATH is not set and pdfinfo is not auto-detected; "
            "skipping OCR for %s.",
            path,
        )
        return ""

    try:
        # Tell pdf2image explicitly where Poppler lives
        images = convert_from_path(str(path), poppler_path=POPPLER_PATH)
        ocr_text_parts = []
        for idx, img in enumerate(images):
            try:
                page_text = pytesseract.image_to_string(img)
            except pytesseract.TesseractNotFoundError:
                logger.warning(
                    "Tesseract not accessible from pytesseract; skipping OCR for %s.",
                    path,
                )
                return ""
            if page_text.strip():
                ocr_text_parts.append(f"[OCR PAGE {idx+1}]\n{page_text}")
        full_ocr_text = "\n\n".join(ocr_text_parts)
        if full_ocr_text.strip():
            logger.info("OCR successfully extracted text from %s", path)
        else:
            logger.warning("OCR produced no text for %s", path)
            return ""
        return full_ocr_text
    except PDFInfoNotInstalledError:
        logger.warning(
            "Poppler/pdfinfo still not accessible for %s. "
            "Check that Poppler is installed and POPPLER_PATH is correct.",
            path,
        )
        return ""
    except Exception as e:
        logger.error("OCR failed on %s: %s", path, e, exc_info=True)
        return ""


def extract_pdf_text(path: Path) -> str:
    """
    Combined PDF extractor:
    1) Try built-in DocumentProcessor (handled elsewhere),
    2) Then pypdf,
    3) Then OCR as last resort.
    """
    # Here we only implement steps 2 and 3; step 1 is in main via DocumentProcessor.
    text = extract_pdf_text_basic(path)
    if text.strip():
        return text

    logger.info("No text from pypdf for %s, attempting OCR...", path)
    ocr_text = extract_pdf_text_ocr(path)
    return ocr_text


def load_text_file(path: Path) -> str:
    """Load a text-like file as best as we can (UTF-8 with errors ignored)."""
    return path.read_text(encoding="utf-8", errors="ignore")


# --------------- HIGH-LEVEL TOOL BEHAVIOR (PROMPT-BASED) ---------------

async def tool_policy_locator(agent: SimpleAgent, query: str) -> str:
    """
    Policy Locator:
    - Returns a list of relevant documents + sections/paragraphs for a query.
    """
    prompt = (
        "You are a POLICY LOCATOR for USAFA policy.\n"
        "You MUST use your knowledge base tool to locate the MOST relevant sections, paragraphs, "
        "or headings from the ingested corpus (CS34 MFR, AFCWI 36-3501, AFCW CD 2024, EXORD 25-003, "
        "USAFA Dress & Appearance Standards, and DAFI 36-2903) that relate to the following query.\n"
        "Do not guess; base everything on retrieved context.\n\n"
        f"Query: {query}\n\n"
        "Your task:\n"
        "1. Do NOT answer the policy question in normal prose.\n"
        "2. Only list the most relevant sources in a 'Sources' section.\n"
        "3. For each bullet, include:\n"
        "   • Document name\n"
        "   • Section/paragraph number or heading/title (if visible in context)\n"
        "   • VERY short hint (3–8 words) about what that section covers.\n\n"
        "Format:\n"
        "Sources:\n"
        "  • <Document>, <section/para>, <very short hint>\n"
        "  • ...\n"
    )
    result = await agent.arun(prompt)
    return result


async def tool_doc_summarizer(agent: SimpleAgent, text: str, filename: str) -> str:
    """Document summarizer, highlighting USAFA policy-relevant content."""
    truncated = shorten(text, width=8000, placeholder="\n\n...[TRUNCATED]")
    prompt = (
        "You are a DOCUMENT SUMMARIZER for USAFA-related content.\n"
        "Whenever relevant, you MUST consult your knowledge base tool so that your summary "
        "is grounded in the ingested policies rather than general knowledge.\n\n"
        f"The user has provided a document named '{filename}'.\n\n"
        "Document (truncated if very long):\n"
        "```text\n"
        f"{truncated}\n"
        "```\n\n"
        "Your task:\n"
        "1. Provide a concise executive summary (3–6 bullet points).\n"
        "2. Highlight any content that is clearly related to USAFA cadet standards, duties, "
        "   or dress/appearance policy.\n"
        "3. If relevant, mention which policies (by name) this document seems to interact with.\n"
        "4. End with a 'Sources' section listing only the policies/sections you used from your "
        "   knowledge base (if any).\n"
    )
    result = await agent.arun(prompt)
    return result


async def tool_rewrite_for_compliance(agent: SimpleAgent, text: str, filename: str) -> str:
    """Rewrite a user document to be as compliant as possible."""
    truncated = shorten(text, width=8000, placeholder="\n\n...[TRUNCATED]")
    prompt = (
        "You are a COMPLIANCE REWRITER for USAFA cadet standards, duties, and dress/appearance policy.\n"
        "You MUST consult your knowledge base tool and rely on actual retrieved policy text, "
        "not on generic assumptions.\n\n"
        "The user has provided a draft document (event plan, MFR, or similar). "
        "Your job is to rewrite it so that it is compliant with the policies you know "
        "(CS34 MFR, AFCWI 36-3501, AFCW CD 2024, EXORD 25-003, USAFA Dress & Appearance, DAFI 36-2903).\n\n"
        f"Document name: {filename}\n\n"
        "Original document (truncated if very long):\n"
        "```text\n"
        f"{truncated}\n"
        "```\n\n"
        "Your task:\n"
        "1. First, briefly state whether the ORIGINAL appears COMPLIANT or NON-COMPLIANT.\n"
        "2. Then provide a REWRITTEN version that is as compliant as possible while preserving the intent.\n"
        "3. Clearly label sections:\n"
        "   - 'Assessment of Original'\n"
        "   - 'Rewritten Compliant Version'\n"
        "4. At the end, add 'Sources' and list only the policies/sections you relied on.\n"
    )
    result = await agent.arun(prompt)
    return result


async def tool_risk_assessment(agent: SimpleAgent, text: str, filename: str) -> str:
    """ORM-style risk assessment generator for events/trainings."""
    truncated = shorten(text, width=8000, placeholder="\n\n...[TRUNCATED]")
    prompt = (
        "You are a RISK ASSESSMENT generator using USAFA and Air Force-style ORM thinking.\n"
        "Ground your recommendations in retrieved policy text wherever possible.\n\n"
        "The user has provided a description of an event/training plan.\n\n"
        f"Document name: {filename}\n\n"
        "Event description (truncated if very long):\n"
        "```text\n"
        f"{truncated}\n"
        "```\n\n"
        "Your task:\n"
        "1. Generate an ORM-style risk assessment with:\n"
        "   - List of primary hazards.\n"
        "   - Likely severity and probability (qualitative).\n"
        "   - Proposed controls/mitigations.\n"
        "   - Residual risk after controls.\n"
        "2. Present results in a structured bullet or table-like markdown.\n"
        "3. At the end, under 'Sources', list any relevant policy references you used.\n"
    )
    result = await agent.arun(prompt)
    return result


async def tool_deviations(agent: SimpleAgent, text: str, filename: str) -> str:
    """Deviation / violation detector for user documents."""
    truncated = shorten(text, width=8000, placeholder="\n\n...[TRUNCATED]")
    prompt = (
        "You are a DEVIATION DETECTOR for USAFA cadet standards, duties, and dress/appearance policy.\n"
        "You MUST use the knowledge base tool to identify where the document conflicts with policy.\n\n"
        "The user has provided a document and wants to know where it deviates from or violates policy.\n\n"
        f"Document name: {filename}\n\n"
        "Document (truncated if very long):\n"
        "```text\n"
        f"{truncated}\n"
        "```\n\n"
        "Your task:\n"
        "1. Identify specific statements or requirements in the document that appear NON-COMPLIANT.\n"
        "2. For each, explain:\n"
        "   - Why it is a problem (which concept it violates: hazing, improper uniform, unsafe PT, etc.).\n"
        "   - Which policy/section it conflicts with (as precisely as you can).\n"
        "3. Suggest how to fix or rewrite each problematic part.\n"
        "4. End with a 'Sources' section listing the policies/sections you used.\n"
    )
    result = await agent.arun(prompt)
    return result


async def tool_show_context(agent: SimpleAgent, last_question: str | None) -> str:
    """
    Show key context/sources used (approximate audit trail).
    If last_question is None, ask the user to supply a query instead.
    """
    if not last_question:
        return "No previous question in this session. Ask a question first, then use /show-context."

    prompt = (
        "You are providing an AUDIT TRAIL for your previous answer.\n"
        "You should re-run whatever retrieval you think is needed and explicitly surface the "
        "snippets that support your reasoning.\n\n"
        "The previous user question was:\n"
        f"\"{last_question}\"\n\n"
        "Your task:\n"
        "1. Re-run whatever retrieval you think is needed.\n"
        "2. List the top 3–6 most relevant snippets you used or would use, each starting with:\n"
        "   [SOURCE: <Document name>, approximate section/para or heading]\n"
        "   and then 1–3 sentences of excerpt/summary.\n"
        "3. Do NOT restate your entire answer, just the key supporting context.\n"
    )
    result = await agent.arun(prompt)
    return result


async def tool_explain_last(agent: SimpleAgent, last_question: str | None, last_answer: str | None) -> str:
    """Explain why the agent gave its last answer."""
    if not last_question or not last_answer:
        return "No previous Q&A in this session to explain. Ask something first."

    prompt = (
        "You are explaining your own reasoning from a previous answer.\n"
        "If you identify any part of your reasoning that was not well grounded in retrieved policy "
        "text, correct it.\n\n"
        "The previous user question was:\n"
        f"\"{last_question}\"\n\n"
        "Your previous answer was:\n"
        "```answer\n"
        f"{last_answer}\n"
        "```\n\n"
        "Your task:\n"
        "1. Explain, step by step, how you arrived at that answer.\n"
        "2. Make clear which parts relied on policy documents vs general reasoning.\n"
        "3. If you see any mistakes or things you would change, correct them and say why.\n"
        "4. End with a 'Sources' section listing which policies/sections you relied on.\n"
    )
    result = await agent.arun(prompt)
    return result


async def tool_stylecheck(agent: SimpleAgent, text: str, filename: str) -> str:
    """
    Style / consistency checker:
    - Looks for formatting, terminology, structure, and tone issues,
      NOT fundamental policy violations.
    """
    truncated = shorten(text, width=8000, placeholder="\n\n...[TRUNCATED]")
    prompt = (
        "You are a STYLE AND CONSISTENCY CHECKER for USAFA documents.\n"
        "You are NOT primarily looking for hazing/abuse/safety issues (those are handled by "
        "policy compliance tools), but instead for:\n"
        "  - Proper Air Force / USAFA terminology (ranks, cadet designations, uniforms).\n"
        "  - Reasonable formatting (headers, numbering, clear sections).\n"
        "  - Consistent tense, perspective, and tone.\n"
        "  - Clear, professional writing.\n\n"
        f"Document name: {filename}\n\n"
        "Document to review (truncated if very long):\n"
        "```text\n"
        f"{truncated}\n"
        "```\n\n"
        "Your task:\n"
        "1. List STYLE/FORMAT issues in bullet form. Group them into categories such as:\n"
        "   - Terminology and Rank\n"
        "   - Uniform Naming & Capitalization\n"
        "   - Section Headers and Structure\n"
        "   - Clarity and Tone\n"
        "2. Suggest concrete edits or patterns to fix each issue.\n"
        "3. Do NOT repeat the entire document; only quote small snippets as needed.\n"
        "4. At the end, add a 'Sources' section listing any relevant policy/guide references "
        "   you relied on (if any). If you are using general writing guidance, say so.\n"
    )
    return await agent.arun(prompt)


async def tool_keyfindings(agent: SimpleAgent, text: str, filename: str) -> str:
    """
    Extract key findings / action items / hazards / responsibilities from a document.
    More action-oriented than a summary.
    """
    truncated = shorten(text, width=8000, placeholder="\n\n...[TRUNCATED]")
    prompt = (
        "You are an ACTION-FOCUSED ANALYST for USAFA documents.\n"
        "The user wants to know the KEY THINGS they must pay attention to in this document.\n\n"
        f"Document name: {filename}\n\n"
        "Document (truncated if very long):\n"
        "```text\n"
        f"{truncated}\n"
        "```\n\n"
        "Your task:\n"
        "1. Identify and list the most important 'Key Findings', grouped as:\n"
        "   - Mandatory tasks / actions.\n"
        "   - Roles and responsibilities.\n"
        "   - Deadlines / time windows.\n"
        "   - Hazards / safety concerns.\n"
        "   - Required approvals / authorities.\n"
        "2. Each bullet should be short and actionable.\n"
        "3. At the end, add a 'Sources' section listing any relevant policies or sections "
        "   (from the ingested corpus) that this document interacts with, if clear.\n"
    )
    return await agent.arun(prompt)


async def tool_regformat(agent: SimpleAgent, text: str, filename: str) -> str:
    """
    Rewrite a document into a formal Air Force/USAFA-style memorandum format.
    Focus is on structure & formatting, not fundamentally changing intent.
    """
    truncated = shorten(text, width=8000, placeholder="\n\n...[TRUNCATED]")
    prompt = (
        "You are a FORMAL MEMORANDUM FORMATTER for USAFA documents.\n"
        "The user has provided text that should be turned into a proper Air Force or USAFA-style memo.\n\n"
        f"Original document name: {filename}\n\n"
        "Original content (truncated if very long):\n"
        "```text\n"
        f"{truncated}\n"
        "```\n\n"
        "Your task:\n"
        "1. Assume the content is roughly acceptable; focus on structure and formatting.\n"
        "2. Produce a rewritten version in a formal memorandum style, including typical elements such as:\n"
        "   - Header (unit, office symbol, date, etc.)\n"
        "   - MEMORANDUM FOR / FROM / SUBJECT lines if appropriate.\n"
        "   - References (if any).\n"
        "   - Body organized into paragraphs with clear topic sentences.\n"
        "   - Recommendation / conclusion if relevant.\n"
        "   - Signature block placeholder.\n"
        "3. Do NOT invent unrealistic names; use placeholders where needed (e.g., '//SIGNATURE//').\n"
        "4. Keep the final output in markdown so the user can copy-paste it into a document.\n"
        "5. At the end, under 'Sources', list any policy/style references you used from your knowledge base.\n"
    )
    return await agent.arun(prompt)


async def role_aware_answer(agent: SimpleAgent, question: str, role: str | None) -> str:
    """
    Wrap normal Q&A so that:
    - The agent MUST use the knowledge base tool.
    - The answer is tailored to the current user role.
    """
    if not role or role == "Default user":
        # Fallback: behave like the original agent
        return await agent.arun(question)

    prompt = (
        "You are answering a policy question for a user with the following role:\n"
        f"  {role}\n\n"
        "You MUST use your knowledge base tool to retrieve supporting passages from the ingested "
        "USAFA policies before you answer. When choosing which passages to rely on, give extra "
        "weight to any guidance that explicitly mentions this role or clearly applies to it "
        "(for example, duties, authorities, limitations, responsibilities, privileges, or "
        "training requirements for that role).\n\n"
        "After you have retrieved and considered the context, answer the question below, tailoring "
        "your explanation and recommendations to what a person in this role actually needs to know "
        "and do.\n\n"
        f"Question: {question}\n\n"
        "Remember to end your response with a 'Sources' section listing the specific documents and "
        "sections/paragraphs you used."
    )
    return await agent.arun(prompt)


# ----------------- MAIN RAG SETUP -----------------
async def main(check_doc: str | None = None, build_index_only: bool = False):

    """Main function to set up and run the RAG agent demonstration."""

    logger.info("Initializing RAG components...")

    if not CHROMADB_LOADED:
        logger.critical("ChromaDB is required for this demo but is not installed. Exiting.")
        return

    try:
        llm = OpenAIAdapter(
            api_key=settings.api_keys.openai_api_key,
            model_name=settings.models.get("openai_gpt4", {"model_name": "gpt-4o"}).model_name,
        )
        embedder = SentenceTransformerEmbedder()

        PERSIST_DIR = "policy_index"  # on-disk Chroma DB for policies

        # ✅ New-style Chroma client (no Settings object)
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

        vector_store = ChromaDBVectorStore(
            client=chroma_client,
            collection_name="usafa_policy_rag",
            embedder=embedder,
        )

        long_term_memory = LongTermMemory(vector_store)
        retriever = SimpleRetriever(vector_store)

    except Exception as e:
        logger.critical(f"Failed to initialize core components: {e}", exc_info=True)
        return

    # --------- Ensure DAFI locally ---------

    if not DAFI_LOCAL_PATH.exists():
        try:
            logger.info("Downloading DAFI 36-2903 from e-publishing.af.mil...")
            with urllib.request.urlopen(DAFI_URL) as r:
                data = r.read()
            DAFI_LOCAL_PATH.write_bytes(data)
            logger.info("✅ Downloaded DAFI 36-2903 to %s", DAFI_LOCAL_PATH)
        except Exception as e:
            logger.warning(
                "Could not download DAFI 36-2903 automatically (%s). "
                "If you want it in long-term memory, download it manually to %s.",
                e,
                DAFI_LOCAL_PATH,
            )

    # --------- Ingest all policy documents ---------

    files_dir = str(Path(".").resolve())
    logger.info("Using files directory: %s", files_dir)
    doc_proc = DocumentProcessor({"files_directory": files_dir})

    all_chunks: list[str] = []
    ingested_files: list[str] = []

    for path in POLICY_DOC_PATHS:
        if not path.exists():
            logger.warning("Policy document '%s' not found. Skipping.", path)
            continue

        try:
            logger.info("Processing policy document: %s", path)
            text = ""

            if path.name == "CS34_Discipline_and_Reward_MFR.md":
                # Manual read to ensure full content
                logger.info("Using manual text reader for CS34 MFR.")
                text = path.read_text(encoding="utf-8", errors="ignore")
            else:
                # Try DocumentProcessor first
                docs = doc_proc.process_file(str(path))
                if docs:
                    doc_obj = docs[0]
                    text = getattr(doc_obj, "page_content", "") or ""

                # If that failed and it's a PDF, fall back to pypdf/OCR
                if (not text.strip()) and path.suffix.lower() == ".pdf":
                    logger.info("Falling back to PDF extractor for %s", path)
                    text = extract_pdf_text(path)

            if not text.strip():
                logger.warning("No text extracted from %s; skipping.", path)
                continue

            # Smaller chunks for CS34; moderately sized chunks for big PDFs
            if path.name == "CS34_Discipline_and_Reward_MFR.md":
                chunk_size = 1200
                chunk_overlap = 200
            else:
                chunk_size = 2500  # slightly smaller than before for better granularity
                chunk_overlap = 200

            raw_chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Add a structured header with source + chunk index to every chunk
            chunks = []
            for idx, c in enumerate(raw_chunks):
                header = f"[SOURCE: {path.name} | CHUNK {idx+1}]\n\n"
                chunks.append(header + c)

            all_chunks.extend(chunks)
            ingested_files.append(path.name)
            logger.info("  → %s split into %d chunks.", path.name, len(chunks))

        except Exception as e:
            logger.error("Error processing %s: %s", path, e, exc_info=True)

    if not all_chunks:
        logger.error("No documents were successfully ingested; aborting.")
        return

    logger.info(
        "Ingesting %d chunks from %d documents into vector store.",
        len(all_chunks),
        len(ingested_files),
    )
    long_term_memory.vector_store.add_documents(all_chunks)
    logger.info("✅ All documents successfully ingested into Long-Term Memory.")

    # If we're only building the index (for reuse by Streamlit/App Runner), stop here.
    if build_index_only:
        logger.info("Build-index-only flag set; skipping agent construction and CLI loop.")
        return

    # --------- Build the Agent ---------

    knowledge_tool = KnowledgeBaseQueryTool(retriever)
    tool_registry = ToolRegistry()
    tool_registry.register_tool(knowledge_tool)

    planner = ReActPlanner(llm, tool_registry)
    executor = ToolExecutor(tool_registry)
    working_memory = WorkingMemory()

    rag_agent = SimpleAgent(llm, planner, executor, working_memory)

    base_role_description = (
        "You are a helpful AI assistant and an expert on USAFA cadet standards, duties, and "
        "dress/appearance policy. Your knowledge comes from the following ingested documents:\n"
        "  • CS34 Discipline and Reward MFR\n"
        "  • AFCWI 36-3501 Cadet Standards and Duties\n"
        "  • AFCW CD 2024 - What Does my Job Mean\n"
        "  • EXORD 25-003 USAFA Dress and Appearance Standards\n"
        "  • USAFA Dress & Appearance Standards\n"
        "  • DAFI 36-2903, Dress and Personal Appearance of Department of the Air Force Personnel.\n\n"
        "When answering policy, standards, duties, or dress/appearance questions:\n"
        "1. You should call your knowledge base tool to retrieve supporting passages before answering, "
        "   unless the question is purely about how to use the agent or its commands.\n"
        "2. Start with a short, direct answer.\n"
        "3. Explain your reasoning in clear, concise language grounded in the retrieved context.\n"
        "4. At the END of every answer, add a section titled 'Sources' and list ONLY the documents "
        "   and sections/paragraph numbers or titles you used, in bullet form, for example:\n"
        "      • DAFI 36-2903, para 3.1.2\n"
        "      • AFCWI 36-3501, Section 5.3 'Training Events'\n"
        "   If you cannot see a specific section number, use the closest heading or describe the "
        "   location (e.g., 'AFCW CD 2024, Squadron First Sergeant job description').\n"
        "5. Always base your reasoning on the ingested documents; if something is not clearly covered, "
        "   say so and avoid guessing.\n"
    )

    current_role = "Default user"
    rag_agent.role_description = (
        f"{base_role_description}\n\n"
        f"Current user role: {current_role}.\n"
        "When retrieving and selecting policy passages, prioritize guidance that is most relevant "
        "to this role's duties, authorities, limitations, responsibilities, and privileges."
    )

    logger.info("✅ RAG Agent created.")
    logger.info("\n--- Starting Interaction with RAG Agent ---\n")

    # --------- Optional: one-shot compliance review via CLI arg ---------

    if check_doc:
        doc_path = Path(check_doc)
        if not doc_path.exists():
            print(f"❌ Could not find document to check: {doc_path}")
            return

        raw_text = doc_path.read_text(encoding="utf-8", errors="ignore")
        truncated = shorten(raw_text, width=8000, placeholder="\n\n...[TRUNCATED]")

        compliance_prompt = (
            "You are a compliance assistant. The user has provided a document and wants to know "
            "whether it aligns with USAFA cadet standards, duties, and dress/appearance rules as "
            "defined in the ingested policies (CS34 MFR, AFCWI 36-3501, AFCW CD 2024, EXORD 25-003, "
            "USAFA Dress & Appearance Standards, and DAFI 36-2903).\n\n"
            "You should consult your knowledge base tool before deciding.\n\n"
            f"Document name: {doc_path.name}\n\n"
            "Document to review (truncated if very long):\n"
            "```text\n"
            f"{truncated}\n"
            "```\n\n"
            "Task:\n"
            "1. State clearly whether the document is COMPLIANT or NON-COMPLIANT.\n"
            "2. Explain briefly why.\n"
            "3. Recommend precise changes if it is not fully compliant.\n"
            "4. At the end, under 'Sources', list only the documents and sections/paragraphs you used.\n"
        )

        print("\n🤖 Compliance Review: thinking...\n")
        result = await rag_agent.arun(compliance_prompt)
        print("\n🤖 Compliance Review:\n")
        print(result)
        return

    # --------- Interactive Loop with Commands & Session State ---------

    loaded_doc_text: str | None = None
    loaded_doc_name: str | None = None
    last_question: str | None = None
    last_answer: str | None = None

    print("🎓 Agent is ready to work.")
    print("💬 Ask questions about USAFA cadet standards, duties, and dress/appearance.\n")
    print("Commands:")
    print("  /role <description>                      → set your role (e.g. 'C4C', 'SQ/CC')")
    print("  /locate <question>                       → list relevant documents/sections")
    print("  /summarize <path-to-file>               → summarize a document")
    print("  /rewrite <path-to-file>                 → rewrite a document for compliance")
    print("  /risk <path-to-file>                    → generate a risk assessment")
    print("  /deviations <path-to-file>              → find policy deviations/violations")
    print("  /load-doc <path-to-file>                → load a document into session")
    print("  /show-context                           → show key context/sources for last Q")
    print("  /why                                    → explain reasoning for last answer")
    print("  /stylecheck [path]                      → style & consistency check (or uses loaded doc)")
    print("  /keyfindings [path]                     → extract key tasks, hazards, responsibilities")
    print("  /regformat [path]                       → rewrite into formal memo/regulation format")
    print("  (or just type a normal question)\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("🤖 Agent: Goodbye! 👋")
                break

            # ---- /role ----
            if user_input.startswith("/role "):
                new_role = user_input[len("/role "):].strip()
                if not new_role:
                    print("⚠️ Usage: /role <description of your role>")
                    continue
                current_role = new_role
                rag_agent.role_description = (
                    f"{base_role_description}\n\n"
                    f"Current user role: {current_role}.\n"
                    "When retrieving and selecting policy passages, prioritize guidance that is most "
                    "relevant to this role's duties, authorities, limitations, responsibilities, and "
                    "privileges. Tailor tone and recommendations to this role."
                )
                print(f"✅ Role updated. Current role: {current_role}\n")
                continue

            # ---- /locate ----
            if user_input.startswith("/locate "):
                query = user_input[len("/locate "):].strip()
                if not query:
                    print("⚠️ Usage: /locate <question about policy>")
                    continue
                print("🤖 Agent (Policy Locator): thinking...\n")
                response = await tool_policy_locator(rag_agent, query)
                print(f"🤖 Agent (Policy Locator):\n{response}\n")
                last_question = query
                last_answer = response
                continue

            # ---- /summarize ----
            if user_input.startswith("/summarize "):
                path_str = user_input[len("/summarize "):].strip()
                path = Path(path_str)
                if not path.exists():
                    print(f"⚠️ File not found: {path}")
                    continue
                text = load_text_file(path)
                print("🤖 Agent (Summarizer): thinking...\n")
                response = await tool_doc_summarizer(rag_agent, text, path.name)
                print(f"🤖 Agent (Summarizer):\n{response}\n")
                last_question = f"Summarize document {path.name}"
                last_answer = response
                continue

            # ---- /rewrite ----
            if user_input.startswith("/rewrite "):
                path_str = user_input[len("/rewrite "):].strip()
                path = Path(path_str)
                if not path.exists():
                    print(f"⚠️ File not found: {path}")
                    continue
                text = load_text_file(path)
                print("🤖 Agent (Compliance Rewriter): thinking...\n")
                response = await tool_rewrite_for_compliance(rag_agent, text, path.name)
                print(f"🤖 Agent (Compliance Rewriter):\n{response}\n")
                last_question = f"Rewrite document {path.name} for compliance"
                last_answer = response
                continue

            # ---- /risk ----
            if user_input.startswith("/risk "):
                path_str = user_input[len("/risk "):].strip()
                path = Path(path_str)
                if not path.exists():
                    print(f"⚠️ File not found: {path}")
                    continue
                text = load_text_file(path)
                print("🤖 Agent (Risk Assessment): thinking...\n")
                response = await tool_risk_assessment(rag_agent, text, path.name)
                print(f"🤖 Agent (Risk Assessment):\n{response}\n")
                last_question = f"Risk assessment for {path.name}"
                last_answer = response
                continue

            # ---- /deviations ----
            if user_input.startswith("/deviations "):
                path_str = user_input[len("/deviations "):].strip()
                path = Path(path_str)
                if not path.exists():
                    print(f"⚠️ File not found: {path}")
                    continue
                text = load_text_file(path)
                print("🤖 Agent (Deviation Detector): thinking...\n")
                response = await tool_deviations(rag_agent, text, path.name)
                print(f"🤖 Agent (Deviation Detector):\n{response}\n")
                last_question = f"Find deviations in {path.name}"
                last_answer = response
                continue

            # ---- /load-doc ----
            if user_input.startswith("/load-doc "):
                path_str = user_input[len("/load-doc "):].strip()
                path = Path(path_str)
                if not path.exists():
                    print(f"⚠️ File not found: {path}")
                    continue
                loaded_doc_text = load_text_file(path)
                loaded_doc_name = path.name
                print(f"✅ Loaded document into session: {loaded_doc_name}\n")
                continue

            # ---- /stylecheck ----
            if user_input.startswith("/stylecheck"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 2:
                    path_str = parts[1].strip()
                    path = Path(path_str)
                    if not path.exists():
                        print(f"⚠️ File not found: {path}")
                        continue
                    text = load_text_file(path)
                    print("🤖 Agent (Style Checker): thinking...\n")
                    response = await tool_stylecheck(rag_agent, text, path.name)
                    print(f"🤖 Agent (Style Checker):\n{response}\n")
                    last_question = f"Style check for {path.name}"
                    last_answer = response
                else:
                    if loaded_doc_text is None or loaded_doc_name is None:
                        print("⚠️ Usage: /stylecheck <path-to-file> OR load a document with /load-doc first.")
                        continue
                    print("🤖 Agent (Style Checker): thinking...\n")
                    response = await tool_stylecheck(rag_agent, loaded_doc_text, loaded_doc_name)
                    print(f"🤖 Agent (Style Checker):\n{response}\n")
                    last_question = f"Style check for {loaded_doc_name}"
                    last_answer = response
                continue

            # ---- /keyfindings ----
            if user_input.startswith("/keyfindings"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 2:
                    path_str = parts[1].strip()
                    path = Path(path_str)
                    if not path.exists():
                        print(f"⚠️ File not found: {path}")
                        continue
                    text = load_text_file(path)
                    print("🤖 Agent (Key Findings): thinking...\n")
                    response = await tool_keyfindings(rag_agent, text, path.name)
                    print(f"🤖 Agent (Key Findings):\n{response}\n")
                    last_question = f"Key findings for {path.name}"
                    last_answer = response
                else:
                    if loaded_doc_text is None or loaded_doc_name is None:
                        print("⚠️ Usage: /keyfindings <path-to-file> OR load a document with /load-doc first.")
                        continue
                    print("🤖 Agent (Key Findings): thinking...\n")
                    response = await tool_keyfindings(rag_agent, loaded_doc_text, loaded_doc_name)
                    print(f"🤖 Agent (Key Findings):\n{response}\n")
                    last_question = f"Key findings for {loaded_doc_name}"
                    last_answer = response
                continue

            # ---- /regformat ----
            if user_input.startswith("/regformat"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 2:
                    path_str = parts[1].strip()
                    path = Path(path_str)
                    if not path.exists():
                        print(f"⚠️ File not found: {path}")
                        continue
                    text = load_text_file(path)
                    print("🤖 Agent (Regulation Format Rewriter): thinking...\n")
                    response = await tool_regformat(rag_agent, text, path.name)
                    print(f"🤖 Agent (Regulation Format Rewriter):\n{response}\n")
                    last_question = f"Regformat for {path.name}"
                    last_answer = response
                else:
                    if loaded_doc_text is None or loaded_doc_name is None:
                        print("⚠️ Usage: /regformat <path-to-file> OR load a document with /load-doc first.")
                        continue
                    print("🤖 Agent (Regulation Format Rewriter): thinking...\n")
                    response = await tool_regformat(rag_agent, loaded_doc_text, loaded_doc_name)
                    print(f"🤖 Agent (Regulation Format Rewriter):\n{response}\n")
                    last_question = f"Regformat for {loaded_doc_name}"
                    last_answer = response
                continue

            # ---- /show-context ----
            if user_input == "/show-context":
                print("🤖 Agent (Audit Trail): thinking...\n")
                response = await tool_show_context(rag_agent, last_question)
                print(f"🤖 Agent (Audit Trail):\n{response}\n")
                last_answer = response
                continue

            # ---- /why ----
            if user_input == "/why":
                print("🤖 Agent (Explanation): thinking...\n")
                response = await tool_explain_last(rag_agent, last_question, last_answer)
                print(f"🤖 Agent (Explanation):\n{response}\n")
                last_answer = response
                continue

            # ---- Normal Q&A (now role-aware) ----
            print("🤖 Agent: thinking...\n")
            agent_response = await role_aware_answer(rag_agent, user_input, current_role)
            print(f"🤖 Agent:\n{agent_response}\n")
            last_question = user_input
            last_answer = agent_response

        except KeyboardInterrupt:
            print("\n🤖 Agent: Session ended by user.")
            break
        except Exception as e:
            print(f"❌ Agent error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="USAFA Policy RAG Assistant"
    )
    parser.add_argument(
        "--check-doc",
        dest="check_doc",
        type=str,
        help="Path to a document to check for compliance against USAFA standards.",
    )
    parser.add_argument(
        "--build-index-only",
        dest="build_index_only",
        action="store_true",
        help="Ingest policies into the vector store and exit (no CLI interaction).",
    )
    args = parser.parse_args()

    # Ensure a dummy CS34 MFR exists so that the script always has at least one doc.
    if not Path("CS34_Discipline_and_Reward_MFR.md").exists():
        Path("CS34_Discipline_and_Reward_MFR.md").write_text(
            "APPLICABILITY: Applies to all Cadets or exchange personnel assigned to CS-34. "
            "In general, the cadet's immediate Supervisor should issue (if element leader or above) "
            "or be present for the issuing of all disciplinary paperwork, as determined by the CS-34 "
            "First Sergeant, CS-34 AFCW SQ/CC, and permanent party (Sq/CC and AMTs). "
            "PURPOSE: To provide subordinate leaders with options and recommendations for appropriate "
            "disciplinary actions or rewards for given situations. Ultimately, authority to issue "
            "adverse administrative action lies with the issuing/awarding authority for the relevant paperwork. "
        )

    asyncio.run(
        main(
            check_doc=args.check_doc,
            build_index_only=args.build_index_only,
        )
    )
