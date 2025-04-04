import os
import chromadb
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from config.settings import VECTOR_DATABASE_DIR, EMBEDDING_MODEL_NAME, COLLECTION_NAME

# Initialize embedding model
text_embedder = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Initialize ChromaDB
try:
    # Try persistent storage first
    db_client = chromadb.PersistentClient(
        path=VECTOR_DATABASE_DIR,
        settings=chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    db_collection = db_client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"Successfully initialized persistent ChromaDB at {VECTOR_DATABASE_DIR}")
except Exception as e:
    # Fall back to in-memory if persistent fails
    print(f"Error with persistent ChromaDB: {e}")
    print("Using in-memory ChromaDB instead")
    db_client = chromadb.Client()
    db_collection = db_client.get_or_create_collection(name=COLLECTION_NAME)

def read_pdf_content(file_path):
    """Extract text content from a PDF file"""
    pdf = PdfReader(file_path)
    content = ""
    
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            content += page_text + "\n"
            
    return content

def chunk_document(document_text):
    """Split document text into manageable chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(document_text)

def process_document(file_path, document_title):
    """Process document and store embeddings in vector database"""
    # Extract text from document
    document_text = read_pdf_content(file_path)
    
    # Split text into chunks
    text_chunks = chunk_document(document_text)
    
    # Add chunks to vector database
    for idx, chunk in enumerate(text_chunks):
        try:
            chunk_id = f"{document_title}_chunk_{idx}"
            chunk_embedding = text_embedder.embed_documents([chunk])[0]
            
            db_collection.add(
                ids=[chunk_id],
                metadatas=[{"document": document_title}],
                documents=[chunk],
                embeddings=[chunk_embedding]
            )
        except Exception as e:
            print(f"Failed to add chunk {idx} to database: {e}")
    
    print(f"Indexed {len(text_chunks)} passages from document '{document_title}'")