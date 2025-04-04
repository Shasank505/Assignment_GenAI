import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form
from config.settings import DOCUMENT_STORAGE_DIR
from services.document_processor import process_document
from services.query_engine import find_relevant_passages, generate_response

def create_router():
    router = APIRouter(tags=["Document Operations"])
    
    @router.post("/document/upload/")
    async def upload_document(file: UploadFile = File(...), document_title: str = Form(...)):
        """Upload and process a PDF document"""
        document_path = os.path.join(DOCUMENT_STORAGE_DIR, document_title)
        
        # Save uploaded document
        with open(document_path, "wb") as destination:
            shutil.copyfileobj(file.file, destination)
        
        # Process the document
        process_document(document_path, document_title)
        
        return {
            "status": "success",
            "message": f"Document '{document_title}' has been processed and indexed successfully."
        }
    
    @router.post("/document/query/")
    async def query_document(document_title: str, question: str):
        """Query a specific document with a question"""
        instruction_template = """
        As an advanced document analysis system, your role is to answer questions about specific documents.
        
        Document context provided below:
        ------------
        {{document_context}}
        ------------
        
        User question: {{user_question}}
        
        Guidelines:
        1. Only use information explicitly stated in the provided document context
        2. If you cannot find sufficient information in the context, respond with: "The document does not contain information to answer this question."
        3. Provide direct, concise answers that address the specific question
        4. Do not reference information outside the provided context
        """
        
        # Find relevant content from the document
        relevant_passages = find_relevant_passages(question, document_title)
        
        # Generate answer based on the relevant content
        answer = generate_response(question, relevant_passages, instruction_template)
        
        return {
            "document": document_title,
            "question": question,
            "answer": answer
        }
    
    return router