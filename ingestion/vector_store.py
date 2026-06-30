import os
import sys

# Add the parent directory to sys.path so we can import from ingestion
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.excel_loader import load_excel
from ingestion.chunk_documents import split_documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def create_vector_store(excel_path: str, persist_directory: str):
    print(f"Loading data from {excel_path}...")
    docs = load_excel(excel_path)
    
    print(f"Loaded {len(docs)} rows. Chunking documents...")
    chunked_docs = split_documents(docs)
    
    print(f"Created {len(chunked_docs)} chunks. Initializing HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Creating Chroma vector store...")
    vector_store = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vector store successfully created at {persist_directory}")
    return vector_store

if __name__ == "__main__":
    EXCEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Drive Failure Data ABB 880.xlsx")
    PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
    
    create_vector_store(EXCEL_PATH, PERSIST_DIR)
