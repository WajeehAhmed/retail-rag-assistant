import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()
# Define paths
datasource = os.getenv("DATA_PATH")
db_dir = os.getenv("CHROMA_PATH")

def ingest_documents():
    logger.info("Starting document ingestion process.")
    all_chunks = []
    
    file_metadata = {
        "acetaminophen.pdf": {"category": "drug_label"},
        "aspirine.pdf": {"category": "drug_label"},
        "headache_pain_management.pdf": {"category": "medicaid_policy"}
    }
    
    if os.path.exists(db_dir) and len(os.listdir(db_dir)) > 0:
        logger.info(f"ChromaDB already exists at '{db_dir}'. Skipping ingestion.")
        logger.info("If you want to re-ingest, please delete the 'chroma_db' directory first.")
        return 0

    for filename, metadata in file_metadata.items():
        file_path = os.path.join(datasource, filename)
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}. Skipping.")
            continue
            
        try:
            logger.info(f"Loading document: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            for doc in documents:
                doc.metadata.update(metadata)
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)
            all_chunks.extend(chunks)
            logger.info(f"Split document '{filename}' into {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Failed to load or process document {file_path}: {e}")
            return 1
        
    if not all_chunks:
        logger.error("No documents were successfully ingested. Exiting.")
        return 1

    logger.info(f"Completed initial loading and splitting. Total chunks: {len(all_chunks)}.")

    logger.info("Creating embeddings with a HuggingFace model.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    logger.info("Storing embeddings in ChromaDB...")
    try:
        db = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=db_dir
        )
        db.persist()
        logger.info(f"Embeddings successfully stored in ChromaDB at '{db_dir}'.")
    except Exception as e:
        logger.error(f"Failed to store embeddings in ChromaDB: {e}")
        return 1
    
    return 0      