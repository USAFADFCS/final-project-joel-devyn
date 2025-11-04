import asyncio
import logging
from pathlib import Path
import os

try:
    import chromadb
    CHROMADB_LOADED=True
except ImportError:
    print("chromadb not found. To run this RAG demo, please run 'pip install chromadb'")
    chromadb = None
    CHROMADB_LOADED = False

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
    KnowledgeBaseQueryTool 
)

settings.api_keys.openai_api_key = os.getenv("OPENAI_API_KEY")
settings.api_keys.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Configure logging for the demo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# A simple text splitter for the demo. In a more complex application, this
# could be a more sophisticated utility, perhaps from a library like LangChain.
def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    """Splits a long text into smaller, overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

async def main():
    """Main function to set up and run the RAG agent demonstration."""

    # --- Step 2: Initialize Core RAG and Framework Components ---
    logger.info("Initializing RAG components...")

    # Add a check to ensure chromadb was imported successfully before proceeding
    if not CHROMADB_LOADED:
        logger.critical("ChromaDB library is required for this demo but is not installed. Exiting.")
        return

    try:
        llm = OpenAIAdapter(
            api_key=settings.api_keys.openai_api_key,
            model_name=settings.models.get("openai_gpt4", {"model_name": "gpt-4o"}).model_name
        )
        embedder = SentenceTransformerEmbedder()
        
        # Using an in-memory ChromaDB client for this demonstration.
        # For persistence, a server-based client would be used.
        vector_store = ChromaDBVectorStore(
            client=chromadb.Client(),
            collection_name="readme_rag",
            embedder=embedder
        )
        long_term_memory = LongTermMemory(vector_store)
        # retriever = SimpleRetriever(vector_store)
        retriever = SimpleRetriever(vector_store)

        
    except Exception as e:
        logger.critical(f"Failed to initialize fairlib.core.components: {e}", exc_info=True)
        return

    # --- Step 3: Load, Split, and Ingest the Document into LongTermMemory ---
    logger.info("Loading and ingesting document into Long-Term Memory...")
    
    # Use the robust DocumentLoader from our fairlib.utils.module.
    readme_path = Path("CS34_Discipline_and_Reward_MFR.md")
    if not readme_path.exists():
        logger.error(f"CS34_Discipline_and_Reward_MFR.md not found in the current directory. Please create one to run this demo.")
        return
        
    doc_proc = DocumentProcessor({"files_directory": str(readme_path.parent)})

    # Process a single file -> DP handles extraction + split_text_semantic internally
    # Important note: document processor now returns a Document object, instead of chunks and metadata
    document = doc_proc.process_file(str(readme_path))
    if not document:
        logger.error("DocumentProcessor returned no documents from CS34_Discipline_and_Reward_MFR.md.")
        return

    # Split the document into smaller chunks for effective retrieval.
    # chunks = split_text(document[0].page_content)
    chunks = split_text(document[0].page_content, chunk_size=3000, chunk_overlap=200)
    logger.info(f"Document split into {len(chunks)} chunks.")

    # Add the document chunks to the long-term memory (vector store).
    long_term_memory.vector_store.add_documents(chunks)
    logger.info("✅ Document successfully ingested into Long-Term Memory.")

    # --- Step 4: Create the RAG-Powered Agent ---
    logger.info("\nBuilding the RAG agent...")
    
    # The agent is given the official `KnowledgeBaseQueryTool` to access its new knowledge.
    # This is the same tool used by the `FactChecker` in our autograders.
    knowledge_tool = KnowledgeBaseQueryTool(retriever)
    
    tool_registry = ToolRegistry()
    tool_registry.register_tool(knowledge_tool)
    
    planner = ReActPlanner(llm, tool_registry)
    executor = ToolExecutor(tool_registry)
    working_memory = WorkingMemory()
    
    rag_agent = SimpleAgent(llm, planner, executor, working_memory)
    # This role description is a crucial part of the prompt, guiding the agent
    # to use its tool correctly.
    rag_agent.role_description = (
        "You are a helpful AI assistant and an expert on the CS34 Discipline and Rewards policy. "
        "You MUST use the 'course_knowledge_query' tool to answer questions about "
        "the document, its policies, or its rewards."
    )
    logger.info("✅ RAG Agent created.")

    # --- Step 5: Interact with the Agent ---
    logger.info("\n--- Starting Interaction with RAG Agent ---")

    print("🎓 Agent is ready to work.")
    print("💬 Ask questions about the CS34 Discipline and Rewards MFR")
    print("   Examples of supported queries:")
    print("    • 'What is the purpose of the CS34 Discipline and Rewards MFR'                   ← SUGGESTED Input")
    while True:
        try:
            user_input = input("👤 You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("🤖 Agent: Goodbye! 👋")
                break

            # Run the agent’s full Reason+Act cycle
            agent_response = await rag_agent.arun(user_input)
            print(f"🤖 Agent: {agent_response}")

        except KeyboardInterrupt:
            print("\n🤖 Agent: Session ended by user.")
            break
        except Exception as e:
            print(f"❌ Agent error: {e}")

if __name__ == "__main__":
    # Ensure a dummy README.md exists for the demo to run out-of-the-box.
    if not Path("CS34_Discipline_and_Reward_MFR.md").exists():
        Path("CS34_Discipline_and_Reward_MFR.md").write_text(
            "APPLICABILITY: Applies to all Cadets or exchange personnel assigned " \
            "to CS-34. In general, the cadet�s immediate Supervisor should issue (if element leader or above) "
            "or be present for the issuing of all disciplinary paperwork, as determined by the CS-34 First Sergeant, " \
            "CS-34 AFCW SQ/CC, and permanent party (Sq/CC and AMTs). The cadet should receive signatures on their paperwork " \
            "by all members of their squadron chain of command within 3 business days before delivering it to CS-34 PP. "
            "PURPOSE: To provide subordinate leaders with options and recommendations for appropriate disciplinary actions "
            "or rewards for given situations. Ultimately, authority to issue adverse administrative action lies with the issuing/awarding " \
            "authority for the relevant paperwork. Higher levels of command reserve the right to withhold authority and bring issues up to their " \
            "level in accordance with Article 37 of the UCMJ."
        )
    asyncio.run(main())


