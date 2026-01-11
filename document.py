from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from embedding import EmbeddingService
import uuid

class DocumentStore:
    def __init__(self, embedder: EmbeddingService):
        self.embedder = embedder
        self.docs_memory = []
        try:
            self.qdrant = QdrantClient("http://localhost:6333")
            self.qdrant.recreate_collection(
                collection_name="demo_collection",
                vectors_config=VectorParams(size=128, distance=Distance.COSINE)
            )
            self.USING_QDRANT = True
        except Exception as e:
            print("⚠️  Qdrant not available. Falling back to in-memory list.")
            self.USING_QDRANT = False
            

    def add_document(self, text: str):
        emb = self.embedder.embed_text(text)
        doc_id = str(uuid.uuid4())
        payload = {"text": text}
        if self.USING_QDRANT:
            self.qdrant.upsert(
                collection_name="demo_collection",
                points=[PointStruct(id=doc_id, vector=emb, payload=payload)]
            )
        else:
            self.docs_memory.append(text)
        return {"id": doc_id, "status": "added"}
    
    def search_query(self, query: str):
        emb = self.embedder.embed_text(query)
        results = []
        if self.USING_QDRANT:
            hits = self.qdrant.query_points(collection_name="demo_collection", query=emb, limit=2)
            for hit in hits.points:
                results.append(hit.payload["text"])
        else:
            for doc in self.docs_memory:
                if query.lower() in doc.lower():
                    results.append(doc)
            if not results and self.docs_memory:
                results = [self.docs_memory[0]]  # Just grab first
        return results




