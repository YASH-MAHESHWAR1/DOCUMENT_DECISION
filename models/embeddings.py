from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os
from config.settings import settings
import openai

class EmbeddingHandler:
    def __init__(self, provider=None):
        self.provider = provider or settings.DEFAULT_LLM
        self.chunks = []
        self.metadata = []
        self.index = None
        
        if self.provider == "openai":
            openai.api_key = settings.OPENAI_API_KEY
            self.embedding_model = settings.EMBEDDING_MODEL_OPENAI
        else:
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL_OS)
    
    def create_embeddings(self, texts, metadata=None):
        """Create embeddings for text chunks"""
        if self.provider == "openai":
            embeddings = []
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = openai.Embedding.create(
                    input=batch,
                    model=self.embedding_model
                )
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
            return np.array(embeddings)
        else:
            texts_with_prefix = [f"passage: {text}" for text in texts]
            embeddings = self.model.encode(texts_with_prefix, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings
    
    def build_index(self, embeddings):
        """Build FAISS index for similarity search"""
        dimension = embeddings.shape[1]
        if self.provider == "openai":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        else:
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.index.add(embeddings.astype('float32'))
        
    def search(self, query, k=settings.TOP_K_RESULTS):
        """Search for similar chunks"""
        if self.provider == "openai":
            response = openai.Embedding.create(
                input=[query],
                model=self.embedding_model
            )
            query_embedding = np.array([response['data'][0]['embedding']])
        else:
            query_with_prefix = f"query: {query}"
            query_embedding = self.model.encode([query_with_prefix], convert_to_numpy=True, normalize_embeddings=True)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for idx, (dist, i) in enumerate(zip(distances[0], indices[0])):
            if i < len(self.chunks):
                if self.provider == "openai":
                    similarity = dist  # Cosine similarity for normalized vectors
                else:
                    similarity = 1 / (1 + dist)  # Convert L2 distance to similarity
                
                if similarity >= settings.SIMILARITY_THRESHOLD:
                    results.append({
                        'chunk': self.chunks[i],
                        'metadata': self.metadata[i] if self.metadata else {},
                        'similarity': similarity
                    })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    def save_index(self, path):
        """Save index and data"""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, f"index_{self.provider}.faiss"))
        
        with open(os.path.join(path, f"data_{self.provider}.pkl"), 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata,
                'provider': self.provider
            }, f)
    
    def load_index(self, path):
        """Load saved index and data"""
        index_path = os.path.join(path, f"index_{self.provider}.faiss")
        data_path = os.path.join(path, f"data_{self.provider}.pkl")
        
        if os.path.exists(index_path) and os.path.exists(data_path):
            self.index = faiss.read_index(index_path)
            
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']
            return True
        return False