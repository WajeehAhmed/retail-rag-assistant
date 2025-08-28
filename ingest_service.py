# ingest.py
import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DATA_PATH = "data/pharmacy"

def ingest_documents():
    """
    Loads documents from the data directory, splits them into chunks, and returns them.
    """
    logging.info("Starting document ingestion process.")
    all_chunks = []
    
    pdf_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    
    if not pdf_files:
        logging.warning("No PDF files found in the data directory. Please check the 'data/pharmacy' folder.")
        return []

    for file_path in pdf_files:
        try:
            logging.info(f"Loading document: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split documents into smaller chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)
            all_chunks.extend(chunks)
            logging.info(f"Split document into {len(chunks)} chunks.")
        except Exception as e:
            logging.error(f"Failed to load or process document {file_path}: {e}")

    logging.info(f"Completed ingestion. Loaded and split {len(pdf_files)} documents into a total of {len(all_chunks)} chunks.")
    
    return all_chunks

if __name__ == "__main__":
    chunks = ingest_documents()
    
    if chunks:
        logging.info("Successfully ingested documents.")
        logging.debug(f"First chunk content: {chunks[0].page_content}")
    else:
        logging.info("No chunks were created. Exiting.")