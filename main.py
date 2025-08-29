# main.py
import sys
import logging
import argparse
from service.ingest_service import ingest_documents
from service.query import query_rag

# Configure top-level logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    ingest_documents()
    query_rag("What is major cause of headache")

if __name__ == "__main__":
    main()