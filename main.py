from rag import RagWorkFlow
from embedding import EmbeddingService
from document import DocumentStore
from controller import Controller
from schemas import DocumentRequest, QuestionRequest
from fastapi import FastAPI, HTTPException


embedder = EmbeddingService()
store = DocumentStore(embedder)
workflow = RagWorkFlow(store)
controller = Controller(workflow, store)
app = FastAPI(title="Learning RAG Demo")

@app.post("/ask")
def ask(req: QuestionRequest):
    try:
        return controller.handle_ask(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/add")
def add(req: DocumentRequest):
    try:
        return controller.handle_add(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/status")
def status():
    try:
        return controller.handle_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

