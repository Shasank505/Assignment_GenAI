import os
from fastapi import FastAPI, UploadFile, File, Form
import shutil
from utils.pdf_processor import process_pdf
from utils.qa_engine import search_chroma, get_answer
from dotenv import load_dotenv

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing! Set it in environment variables or .env file.")

# Initialize FastAPI app
app = FastAPI()

# Folder to store uploaded PDFs
UPLOAD_FOLDER = "pdf_uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...), pdf_name: str = Form(...)):
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_name)
    # Save PDF
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Process PDF
    process_pdf(pdf_path, pdf_name)
    return {"message": f"PDF '{pdf_name}' successfully processed and stored in ChromaDB."}

@app.post("/ask_question/")
async def ask_question(pdf_name: str, query: str):
    prompt = f"""
    You are an intelligent AI assistant designed to answer questions based on a specific document context.
    You will receive a set of relevant text excerpts from a pdf document and a user's question.
    Your task is to provide an accurate and concise answer using only the given context.
    
    Input:
    context
    {{context}}
    User's question
    {{query}}
    
    Instructions:
    1. Base your answers strictly on the provided context. Do not generate information that is not in the context
    2. If the context does not containing inough information respond with: "No relevant information was found for this question."
    3. Keep the answer concise and to the point.
    """
    
    relevant_chunks = search_chroma(query, pdf_name)
    answer = get_answer(query, relevant_chunks, prompt)
    return {"question": query, "answer": answer}


# import os
# import shutil
# import chromadb
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from pypdf import PdfReader
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from dotenv import load_dotenv
# # Load API Key
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY is missing! Set it in environment variables or .env file.")
# # Initialize FastAPI app
# app = FastAPI()
# # Initialize SentenceTransformer model
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# embedding_model = SentenceTransformerEmbeddings(model_name=model_name)
# # Initialize Google LLM
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", google_api_key=GOOGLE_API_KEY)
# # Initialize ChromaDB
# VECTOR_DB_PATH = "vector_db/chroma.sqlite3"
# chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
# collection = chroma_client.get_or_create_collection(name="pdf_embeddings")
# # Create retriever
# retriever = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_model).as_retriever()
# qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
# # Folder to store uploaded PDFs
# UPLOAD_FOLDER = "pdf_uploaded"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# @app.post("/upload_pdf/")
# async def upload_pdf(file: UploadFile = File(...), pdf_name: str = Form(...)):
#     pdf_path = os.path.join(UPLOAD_FOLDER, pdf_name)
#     # Save PDF
#     with open(pdf_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     # Process PDF
#     process_pdf(pdf_path, pdf_name)
#     return {"message": f"PDF '{pdf_name}' successfully processed and stored in ChromaDB."}
# def extract_text_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
#     return text
# def segment_text(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     return text_splitter.split_text(text)
# def process_pdf(pdf_path, pdf_name):
#     text = extract_text_from_pdf(pdf_path)
#     chunks = segment_text(text)
#     # Generate embeddings
#     chunk_embeddings = embedding_model.embed_documents(chunks)
#     for idx, chunk in enumerate(chunks):
#         collection.add(
#             ids=[f"{pdf_name}_{idx}"],
#             metadatas=[{"source": pdf_name}],
#             documents=[chunk]
#         )
#     print(f" Successfully stored {len(chunks)} chunks for {pdf_name}.")  # Debugging
# def search_chroma(question, pdf_name):
#     query_embedding = embedding_model.embed_query(question)
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=5,
#         where={"source": pdf_name}
#     )
#     # Extract relevant document chunks
#     retrieved_chunks = [doc for doc in results['documents'][0]] if results["documents"] else []
#     print(f" Retrieved {len(retrieved_chunks)} chunks from '{pdf_name}' for question: {question}")
#     return retrieved_chunks
# def get_answer(question, relevant_chunks):
#     if not relevant_chunks:
#         print(" No relevant information was found.")
#         return "No relevant information was found for this question."
#     context = "\n".join(relevant_chunks)
#     print(f" Using context:\n{context[:500]}")  # First 500 characters
#     prompt = f"""
#     You are an intelligent AI assistant. Answer the question strictly based on the provided context.
#     Context:
#     {context}
#     Question: {question}
#     If the answer is not found in the context, reply: 'No relevant information was found for this question.'
#     """
#     response = qa_chain.run(prompt)
#     return response
# @app.post("/ask_question/")
# async def ask_question(question: str, pdf_name: str):
#     relevant_chunks = search_chroma(question, pdf_name)
#     answer = get_answer(question, relevant_chunks)
#     return {"question": question, "answer": answer}