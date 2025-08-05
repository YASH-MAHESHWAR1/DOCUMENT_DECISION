from models.llm_handler import LLMHandler
from models.prompts import QUERY_PARSER_PROMPT
import json
import re
from typing import Optional, Dict

class QueryParser:
    def __init__(self, llm_handler: LLMHandler):
        self.llm = llm_handler
        
    def parse_query(self, query: str) -> Optional[Dict]:
        """Extract structured information from natural language query"""
        prompt = QUERY_PARSER_PROMPT.format(query=query)
        response = self.llm.generate_response(prompt, temperature=0.1)
        
        # Parse the JSON response
        parsed_data = self.llm.parse_json_response(response)
        
        if not parsed_data:
            # Fallback parsing
            parsed_data = self._fallback_parser(query)
            
        return self._validate_parsed_data(parsed_data)
    
    def _fallback_parser(self, query: str) -> Dict:
        """Simple regex-based fallback parser"""
        data = {
            "age": None,
            "gender": None,
            "procedure": None,
            "location": None,
            "policy_duration": None,
            "other_details": []
        }
        
        # Age extraction
        age_match = re.search(r'(\d{1,3})[\s-]*(year|yr|y)[\s-]*(old)?', query, re.I)
        if not age_match:
            age_match = re.search(r'(\d{1,3})[\s-]*(M|F)', query, re.I)
        if age_match:
            data["age"] = age_match.group(1)
        
        # Gender extraction
        gender_match = re.search(r'\b(male|female|M|F)\b', query, re.I)
        if gender_match:
            gender = gender_match.group(1).upper()
            data["gender"] = "M" if gender in ["MALE", "M"] else "F"
        
        # Location extraction
        locations = ["pune", "mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad"]
        for loc in locations:
            if loc in query.lower():
                data["location"] = loc.capitalize()
                break
        
        # Policy duration
        policy_match = re.search(r'(\d+)[\s-]*(month|months)', query, re.I)
        if policy_match:
            data["policy_duration"] = policy_match.group(1)
        
        # Procedure - take the middle part of the query
        words = query.split()
        if len(words) > 3:
            data["procedure"] = " ".join(words[1:-1])
        
        return data
    
    def _validate_parsed_data(self, data: Optional[Dict]) -> Optional[Dict]:
        """Validate and clean parsed data"""
        if not data:
            return None
            
        # Ensure all required fields exist
        required_fields = ["age", "gender", "procedure", "location", "policy_duration", "other_details"]
        for field in required_fields:
            if field not in data:
                data[field] = [] if field == "other_details" else None
        
        # Clean age - Fixed to handle None properly
        if data["age"]:
            try:
                # First check if age is already a string with just numbers
                if isinstance(data["age"], str) and data["age"].isdigit():
                    age = int(data["age"])
                else:
                    # Try to extract numbers from the age string
                    age_match = re.search(r'\d+', str(data["age"]))
                    if age_match:
                        age = int(age_match.group())
                    else:
                        age = None
                
                # Validate age range
                if age and 0 < age < 150:
                    data["age"] = str(age)
                else:
                    data["age"] = None
            except:
                data["age"] = None
        
        return data