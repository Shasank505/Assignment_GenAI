import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
from utils.pdf_processor import collection, embedding_model

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Google LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", google_api_key=GOOGLE_API_KEY)

# Initialize vector store with error handling
VECTOR_DB_PATH = os.path.abspath("vector_db")
try:
    # Initialize Chroma for LangChain
    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH, 
        embedding_function=embedding_model,
        collection_name="pdf_embeddings"
    )
    retriever = vector_store.as_retriever()
except Exception as e:
    print(f"Error initializing Chroma for LangChain: {e}")
    # Fall back to in-memory
    vector_store = Chroma(
        embedding_function=embedding_model,
        collection_name="pdf_embeddings"
    )
    retriever = vector_store.as_retriever()

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

def search_chroma(question, pdf_name):
    """Search ChromaDB for relevant chunks"""
    try:
        query_embedding = embedding_model.embed_query(question)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"source": pdf_name}
        )
        
        # Extract relevant document chunks
        retrieved_chunks = [doc for doc in results['documents'][0]] if results["documents"] else []
        print(f" Retrieved {len(retrieved_chunks)} chunks from '{pdf_name}' for question: {question}")
        return retrieved_chunks
    except Exception as e:
        print(f"Error searching ChromaDB: {e}")
        return []

def get_answer(question, relevant_chunks, prompt_template=None):
    """Generate answer from relevant chunks"""
    if not relevant_chunks:
        print(" No relevant information was found.")
        return "No relevant information was found for this question."
    
    context = "\n".join(relevant_chunks)
    print(f" Using context:\n{context[:500]}")  # First 500 characters
    
    if prompt_template:
        # Use the provided prompt template
        prompt = prompt_template.format(context=context, query=question)
    else:
        # Default prompt
        prompt = f"""
        You are an intelligent AI assistant. Answer the question strictly based on the provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        If the answer is not found in the context, reply: 'No relevant information was found for this question.'
        """
    
    try:
        response = qa_chain.run(prompt)
        return response
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "An error occurred while generating the answer."