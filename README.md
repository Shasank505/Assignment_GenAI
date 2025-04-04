# PDF Question-and-Answer Application

A FastAPI-based application that utilizes LangChain, Google LLM, and the Chroma Vector Database to extract information from private PDF documents.

## Features

- PDF upload and processing
- Document text splitting and embedding
- Question answering based on PDF content
- Vector similarity search

## Tech Stack

- **FastAPI**: Web framework for building the API
- **LangChain**: Framework for working with language models
- **Google Gemini**: LLM for generating answers
- **Chroma DB**: Vector database for storing embeddings
- **PyPDF**: Library for loading PDF documents

## Setup and Installation

1. Clone this repository

2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Google API key to the `.env` file

5. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

6. Access the application at http://localhost:8000
   - API documentation is available at http://localhost:8000/docs

## API Endpoints

### PDF Management

- **POST** `/api/upload_pdf/`: Upload a PDF file
  - Parameters: 
    - `pdf_file`: PDF file
    - `pdf_name`: Name for the PDF

### Question Answering

- **POST** `/api/ask_question/`: Ask a question about a PDF
  - Request body:
    ```json
    {
      "question": "What is the main topic?",
      "pdf_name": "document.pdf"
    }
    ```

## Project Structure

```
Assignment_GenAI/
├── main.py                       # FastAPI application entry point
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── utils/                        # Utility modules
│   ├── __init__.py               # Package initializer
│   ├── pdf_processor.py          # Handles PDF processing   and embedding
│   ├── qa_engine.py              # Handles question answering logic
```

