import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from config.settings import VECTOR_DATABASE_DIR, LLM_MODEL_NAME, COLLECTION_NAME
from services.document_processor import text_embedder, db_collection

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize language model
ai_model = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, google_api_key=GEMINI_API_KEY)

# Initialize vector retrieval system
try:
    # Setup Chroma for LangChain with persistent storage
    vector_database = Chroma(
        persist_directory=VECTOR_DATABASE_DIR,
        embedding_function=text_embedder,
        collection_name=COLLECTION_NAME
    )
    content_retriever = vector_database.as_retriever()
except Exception as e:
    print(f"Error setting up persistent Chroma for retrieval: {e}")
    # Use in-memory as fallback
    print("Using in-memory vector database for retrieval")
    vector_database = Chroma(
        embedding_function=text_embedder,
        collection_name=COLLECTION_NAME
    )
    content_retriever = vector_database.as_retriever()

# Create QA chain
qa_system = RetrievalQA.from_chain_type(ai_model, retriever=content_retriever)

def find_relevant_passages(question, document_title):
    """Find passages relevant to the question from the specified document"""
    try:
        # Create embedding for the question
        question_embedding = text_embedder.embed_query(question)
        
        # Query the database
        results = db_collection.query(
            query_embeddings=[question_embedding],
            n_results=5,
            where={"document": document_title}
        )
        
        # Extract content from results
        if results["documents"] and len(results["documents"]) > 0:
            found_passages = results["documents"][0]
            print(f"Found {len(found_passages)} relevant passages for question: '{question}'")
            return found_passages
        else:
            print(f"No relevant content found in document '{document_title}'")
            return []
            
    except Exception as e:
        print(f"Error searching for relevant passages: {e}")
        return []

def generate_response(question, relevant_passages, instruction_template=None):
    """Generate answer using LLM based on relevant passages"""
    if not relevant_passages:
        return "The document does not contain information to answer this question."
    
    # Combine passages into context
    document_context = "\n".join(relevant_passages)
    
    if instruction_template:
        # Use provided template
        prompt = instruction_template.replace("{{document_context}}", document_context).replace("{{user_question}}", question)
    else:
        # Default prompt format
        prompt = f"""
        Using only the following document content, answer the question.
        
        Document content:
        {document_context}
        
        Question: {question}
        
        If the content doesn't provide enough information, state that clearly.
        """
    
    try:
        response = qa_system.run(prompt)
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "An error occurred while processing your question."