import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEFAULT_LLM = os.getenv("DEFAULT_LLM", "gemini")
    
    # Model Names
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    # Model Settings
    EMBEDDING_MODEL_OPENAI = "text-embedding-ada-002"
    EMBEDDING_MODEL_OS = "intfloat/e5-base-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Search Settings
    TOP_K_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Bajaj Specific Settings
    CONFIDENCE_THRESHOLD = 80
    SUPPORTED_LANGUAGES = ["English", "Hindi"]
    
    # Paths
    DOCUMENTS_PATH = "data/documents"
    EMBEDDINGS_PATH = "data/embeddings"
    
    # Create directories if they don't exist
    os.makedirs(DOCUMENTS_PATH, exist_ok=True)
    os.makedirs(EMBEDDINGS_PATH, exist_ok=True)

settings = Settings()