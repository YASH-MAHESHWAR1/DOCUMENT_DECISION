import re
from typing import Dict, Tuple

class Validators:
    @staticmethod
    def validate_query(query: str) -> Tuple[bool, str]:
        """Validate user query"""
        if not query or len(query.strip()) < 10:
            return False, "Query is too short. Please provide more details."
        
        if len(query) > 500:
            return False, "Query is too long. Please limit to 500 characters."
        
        # Check if query contains at least some meaningful words
        words = query.split()
        if len(words) < 3:
            return False, "Please provide more context in your query."
        
        return True, "Valid query"
    
    @staticmethod
    def validate_file(file) -> Tuple[bool, str]:
        """Validate uploaded file"""
        if not file:
            return False, "No file uploaded"
        
        # Check file extension
        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
        file_ext = file.name.lower().split('.')[-1]
        if f".{file_ext}" not in allowed_extensions:
            return False, f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
        
        # Check file size (max 10MB)
        if file.size > 10 * 1024 * 1024:
            return False, "File size exceeds 10MB limit"
        
        return True, "Valid file"
    
    @staticmethod
    def validate_api_keys(provider: str) -> Tuple[bool, str]:
        """Validate API keys based on provider"""
        from config.settings import settings
        
        if provider == "gemini" and not settings.GEMINI_API_KEY:
            return False, "Gemini API key not configured"
        
        if provider == "openai" and not settings.OPENAI_API_KEY:
            return False, "OpenAI API key not configured"
        
        return True, "API keys valid"