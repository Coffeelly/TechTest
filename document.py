from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from embedding import EmbeddingService
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# Abstract Base Class
class BaseDocumentStore(ABC):
    """
    Abstract interface for document storage systems.
    
    This class defines the methods that any storage implementation 
    (whether in-memory, Qdrant, or others) must provide.
    """

    def __init__(self, embedder: EmbeddingService):
        """
        Initializes the document store with an embedding service.

        Args: embedder (EmbeddingService) to generate vector embeddings.
        """
        self.embedder = embedder

    @abstractmethod
    def add_document(self, text: str) -> Dict[str, any]:
        """
        Adds a new document text to the storage.

        Args    : text (str) to be stored.
        Returns : Dict[str, any] containing the document ID and status.
        """
        pass

    @abstractmethod
    def search_query(self, query: str) -> List[str]:
        """
        Searches for documents relevant to the given query string.

        Args    : query (str).
        Returns : List[str] of text contents from the most relevant documents.
        """
        pass

    @abstractmethod
    def get_status_info(self) -> Dict[str, Any]:
        """
        Retrieves the current status information of the storage.

        Returns: Dict[str, Any] containing status.
        """
        pass

# Qdrant
class QdrantDocumentStore(BaseDocumentStore):
    """
    Implementation of document storage using Qdrant Vector Database.
    """
    def __init__(self, embedder: EmbeddingService, url: str = "http://localhost:6333"):
        """
        Initializes the Qdrant client and ensures the collection exists.

        Args:
        - embedder (EmbeddingService) embedding generation.
        - url (str) of the Qdrant instance. Defaults to localhost.
        """
        super().__init__(embedder)
        self.qdrant = QdrantClient(url)
        self.collection_name = "demo_collection"
        
        # Setup collection saat inisialisasi
        self.qdrant.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE)
        )

    def add_document(self, text: str) -> Dict[str, any]:
        """
        Embeds the text and upserts it into the Qdrant collection.
        """
        emb = self.embedder.embed_text(text)
        doc_id = str(uuid.uuid4())
        payload = {"text": text}
        
        # Store point with ID, Vector, and Payload (original text)
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=doc_id, vector=emb, payload=payload)]
        )
        return {"id": doc_id, "status": "added (qdrant)"}

    def search_query(self, query: str) -> List[str]:
        """
        Converts query to vector and performs a similarity search in Qdrant.
        """
        emb = self.embedder.embed_text(query)
        results = []
        
        # Perform vector search
        hits = self.qdrant.query_points(collection_name="demo_collection", query=emb, limit=2)
        for hit in hits.points:
            results.append(hit.payload["text"])
        return results
    
    def get_status_info(self) -> Dict[str, Any]:
        """
        Returns Qdrant connection status.
        """
        return {
            "qdrant_ready": True,
            "in_memory_docs_count": 0  # 0 Because not using memory docs
        }
    
# Memory
class MemoryDocumentStore(BaseDocumentStore):
    """
    Implementation of document storage using an in-memory list.
    """
    def __init__(self, embedder: EmbeddingService):
        super().__init__(embedder)
        self.docs_memory = []

    def add_document(self, text: str) -> Dict[str, any]:
        """
        Appends text to the internal list
        """
        self.docs_memory.append(text)
        doc_id = str(uuid.uuid4())
        return {"id": doc_id, "status": "added (memory)"}
    
    def search_query(self, query: str) -> List[str]:
        """
        simple substring search on the in-memory list.
        """
        results = []
        for doc in self.docs_memory:
            if query.lower() in doc.lower():
                results.append(doc)
        
        # Fallback if no result, take the first data
        if not results and self.docs_memory:
            results = [self.docs_memory[0]]
        return results

    def get_status_info(self) -> Dict[str, Any]:
        """
        Returns the count of documents currently held in memory.
        """
        return {
            "qdrant_ready": False,
            "in_memory_docs_count": len(self.docs_memory)
        }




