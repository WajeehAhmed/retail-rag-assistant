# src/query.py
import os
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

db_dir = os.getenv("CHROMA_PATH")

def query_rag(query_text: str, category_filter: str = None, k: int = 3):

    if not os.path.exists(db_dir) or not os.listdir(db_dir):
        logger.error(f"ChromaDB not found at '{db_dir}'. Please run 'ingest' command first.")
        return 1

    logger.info(f"Loading ChromaDB from {db_dir}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    except Exception as e:
        logger.error(f"Failed to load ChromaDB: {e}")
        return 1

    logger.info(f"Received query: '{query_text}'")
    
    search_kwargs = {"k": k}
    if category_filter:
        search_kwargs["filter"] = {"category": category_filter}
        logger.info(f"Applying metadata filter: category = '{category_filter}'")

    try:
        results = db.similarity_search(query_text, **search_kwargs)
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        return 1

    if results:
        logger.info(f"Found {len(results)} relevant document chunks:")
        for i, doc in enumerate(results):
            logger.info(f"\n--- Result {i+1} ---")
            logger.info(f"Content: {doc.page_content[:300]}...")
            logger.info(f"Source: {doc.metadata.get('source', 'N/A')}")
            logger.info(f"Category: {doc.metadata.get('category', 'N/A')}")
            logger.info(f"Page: {doc.metadata.get('page', 'N/A')}")
            logger.info("---------------------")
    else:
        logger.warning("No relevant documents found for your query with the given filters.")

    return 0