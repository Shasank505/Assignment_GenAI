import os

# Directory configurations
DOCUMENT_STORAGE_DIR = "documents_storage"
VECTOR_DATABASE_DIR = os.path.abspath("embedding_storage")

# Model configurations
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-1.5-pro-002"

# Collection name for vector database
COLLECTION_NAME = "document_embeddings"

def initialize_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(DOCUMENT_STORAGE_DIR, exist_ok=True)
    os.makedirs(VECTOR_DATABASE_DIR, exist_ok=True)