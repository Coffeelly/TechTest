from document import DocumentStore
from rag import RagWorkFlow
from schemas import DocumentRequest, QuestionRequest


class Controller:
    def __init__(self, rag_workflow: RagWorkFlow, document_store: DocumentStore):
        self.rag_workflow = rag_workflow
        self.document_store = document_store

    def handle_add(self, req: DocumentRequest):
        result = self.document_store.add_document(req.text)
        return result
    
    def handle_ask(self, req: QuestionRequest):
        result = self.rag_workflow.run_query(req.question)
        return result
    
    def handle_status(self):
        return {
            "qdrant_ready": self.document_store.USING_QDRANT,
            "in_memory_docs_count": len(self.document_store.docs_memory),
            "graph_ready": self.rag_workflow.chain is not None
        }
