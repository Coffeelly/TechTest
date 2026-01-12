from document import BaseDocumentStore
from rag import RagWorkFlow
from schemas import DocumentRequest, QuestionRequest
from typing import Dict, Any

class Controller:
    """
    Control the flow of data between the external API layer and internal business logic.
    """
    def __init__(self, rag_workflow: RagWorkFlow, document_store: BaseDocumentStore):
        """
        Initializes the controller with necessary dependencies.

        Args:
        - rag_workflow (RagWorkFlow) pipeline service.
        - document_store (BaseDocumentStore): could be Qdrant or Memory.
        """
        self.rag_workflow = rag_workflow
        self.document_store = document_store

    def handle_add(self, req: DocumentRequest) -> Dict[str, any]:
        """
        Processes a request to add a new document.

        Args    :req (DocumentRequest) to validate request object containing the text.
        Returns : Dict[str, any] containing the new document ID and operation status.
        """
        result = self.document_store.add_document(req.text)
        return result
    
    def handle_ask(self, req: QuestionRequest) -> Dict[str, any]:
        """
        Processes a user question through the RAG pipeline.

        Args    :req (QuestionRequest) to validate request object containing the question.
        Returns : Dict[str, any] containing The generated answer, context used, and latency metrics..
        """
        result = self.rag_workflow.run_query(req.question)
        return result
    
    def handle_status(self) -> Dict[str, Any]:
        """
        system status from various internal components.

        Returns: Dict[str, Any] Combined status of the storage backend and the RAG graph.
        """
        storage_status = self.document_store.get_status_info()
        
        # Combine Document status and Workflow status
        return {
            **storage_status, # Unpack dictionary (qdrant_ready & docs_count)
            "graph_ready": self.rag_workflow.chain is not None
        }
