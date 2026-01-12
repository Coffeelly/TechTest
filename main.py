from rag import RagWorkFlow
from embedding import EmbeddingService
from document import QdrantDocumentStore, MemoryDocumentStore
from controller import Controller
from schemas import DocumentRequest, QuestionRequest
from fastapi import FastAPI, HTTPException
import os
from dotenv import load_dotenv

#Load environment variables from a .env file (if present)
load_dotenv()

# Load Qdrant Url from env
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Create the FastAPI application
app = FastAPI(title="Learning RAG Demo")

# -------------------------- COMPOSITION ROOT --------------------------

# Initialize the embedding service
embedder = EmbeddingService()

# Initialize the DocumentStore and Inject the embedder.
try:
    # Try Qdrant
    store = QdrantDocumentStore(embedder, url=QDRANT_URL)
    print("✅ Using Qdrant for storage.")
except Exception as e:
    # If fail, fallback to Memory Store
    print(f"⚠️  Qdrant error: {e}")
    print("⚠️  Falling back to In-Memory storage.")
    store = MemoryDocumentStore(embedder)

# Inject the store into the RagWorkFlow.
workflow = RagWorkFlow(store)

# Inject both workflow and store into the Controller.
controller = Controller(workflow, store)


# ---------------------------- API ENDPOINTS ----------------------------

# Register a POST endpoint at the URL path "/ask"
@app.post("/ask")
def ask(req: QuestionRequest):
    try:
        # Calls the controller's handle_ask method to process the question and returns the result.
        return controller.handle_ask(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Register a POST endpoint at the URL path "/add"
@app.post("/add")
def add(req: DocumentRequest):
    try:
       # Calls the controller's handle_add method to save the document and returns the result.
        return controller.handle_add(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Register a GET endpoint at the URL path "/status"
@app.get("/status")
def status():
    try:
        # Calls the controller's handle_status method to retrieve the state of database.
        return controller.handle_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

