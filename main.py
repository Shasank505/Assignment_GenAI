import os
from fastapi import FastAPI
from dotenv import load_dotenv
from routes.document_routes import create_router
from config.settings import initialize_directories

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found! Please add it to your environment variables or .env file.")

# Create necessary directories
initialize_directories()

# Initialize FastAPI application
app = FastAPI(title="Document QA System", 
              description="API for document upload and question answering")

# Include routers from separate files
document_router = create_router()
app.include_router(document_router)