import random

class EmbeddingService:
    # Pretend this is a real embedding model
    def embed_text(self, text: str):
        # Seed based on input so it's "deterministic"
        random.seed(abs(hash(text)) % 10000)
        return [random.random() for _ in range(128)]  # Small vector for demo