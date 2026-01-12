import random
from typing import List

class EmbeddingService:
    # Service responsible for transforming text into vector representations (embeddings).

    # Pretend this is a real embedding model
    def embed_text(self, text: str) -> List[float]:
        '''
        Args    : String to be embedded
        Returns : A list of 128 floating-point numbers as vector.
        '''
        # Seed based on input so it's "deterministic"
        random.seed(abs(hash(text)) % 10000)
        return [random.random() for _ in range(128)]  # Small vector for demo