import os
import chromadb
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Initialize SentenceTransformer model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformerEmbeddings(model_name=model_name)

# Initialize ChromaDB with proper error handling
VECTOR_DB_PATH = os.path.abspath("vector_db")
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

try:
    # Use client settings to prevent issues
    chroma_client = chromadb.PersistentClient(
        path=VECTOR_DB_PATH,
        settings=chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    collection = chroma_client.get_or_create_collection(name="pdf_embeddings")
except Exception as e:
    print(f"Error with ChromaDB persistent storage: {e}")
    print("Falling back to in-memory storage")
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="pdf_embeddings")

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def segment_text(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)

def process_pdf(pdf_path, pdf_name):
    """Process PDF file and store embeddings in ChromaDB"""
    text = extract_text_from_pdf(pdf_path)
    chunks = segment_text(text)
    
    # Add chunks to collection
    for idx, chunk in enumerate(chunks):
        try:
            collection.add(
                ids=[f"{pdf_name}_{idx}"],
                metadatas=[{"source": pdf_name}],
                documents=[chunk],
                embeddings=[embedding_model.embed_documents([chunk])[0]]
            )
        except Exception as e:
            print(f"Error adding chunk {idx} to collection: {e}")
    
    print(f" Successfully stored {len(chunks)} chunks for {pdf_name}.")