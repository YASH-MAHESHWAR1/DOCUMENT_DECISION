from models.llm_handler import LLMHandler
from models.prompts import DECISION_PROMPT, WHAT_IF_PROMPT, MULTILINGUAL_PROMPT
from typing import Dict, List
import json

class DecisionEngine:
    def __init__(self, llm_handler: LLMHandler):
        self.llm = llm_handler
        
    def make_decision(self, query_details: Dict, relevant_clauses: List[Dict]) -> Dict:
        """Make a decision based on query details and relevant clauses"""
        
        # Prepare clause text
        clause_text = self._prepare_clause_text(relevant_clauses)
        
        # Generate decision
        prompt = DECISION_PROMPT.format(
            query_details=json.dumps(query_details, indent=2),
            policy_clauses=clause_text
        )
        
        response = self.llm.generate_response(prompt, temperature=0.2)
        
        # Check if response is valid
        if response:
            decision = self.llm.parse_json_response(response)
        else:
            decision = None
        
        if not decision:
            decision = self._create_default_decision()
        
        # Calculate Bajaj Trust Score with better logic
        decision['bajaj_trust_score'] = self._calculate_trust_score_v2(decision, relevant_clauses, query_details)
        
        return decision
    
    def _calculate_trust_score_v2(self, decision: Dict, clauses: List[Dict], query_details: Dict) -> int:
        """Enhanced trust score calculation"""
        score = 0
        
        # 1. Base score from decision type (0-40 points)
        decision_type = decision.get('decision', '').lower()
        if decision_type == 'approved':
            score += 40
        elif decision_type == 'rejected':
            score += 35
        else:  # requires_review
            score += 20
        
        # 2. Clause relevance score (0-30 points)
        if clauses:
            avg_relevance = sum(c.get('relevance_score', 0) for c in clauses[:3]) / min(len(clauses), 3)
            score += int(avg_relevance * 30)
        
        # 3. Information completeness (0-20 points)
        required_fields = ['age', 'gender', 'procedure', 'location', 'policy_duration']
        provided_fields = sum(1 for field in required_fields if query_details and query_details.get(field))
        score += int((provided_fields / len(required_fields)) * 20)
        
        # 4. Decision clarity bonus (0-10 points)
        if decision.get('referenced_clauses'):
            score += 5
        if decision.get('amount') is not None and decision_type == 'approved':
            score += 5
        
        # 5. Special cases that boost confidence
        if query_details and query_details.get('procedure'):
            procedure = query_details['procedure'].lower()
            
            # Clear exclusions found
            exclusion_keywords = ['cosmetic', 'dental', 'joint replacement', 'bariatric', 'fertility']
            if any(keyword in procedure for keyword in exclusion_keywords) and decision_type == 'rejected':
                score = min(score + 20, 95)
            
            # Emergency procedures
            emergency_keywords = ['emergency', 'accident', 'acute', 'appendectomy', 'fracture']
            if any(keyword in procedure for keyword in emergency_keywords) and decision_type == 'approved':
                score = min(score + 15, 95)
        
        # 6. Policy duration factor
        if query_details and query_details.get('policy_duration'):
            try:
                duration = int(query_details['policy_duration'])
                if duration >= 12:  # Long-standing policy
                    score += 5
            except:
                pass
        
        # Ensure score is within bounds
        return max(10, min(100, score))
    
    def generate_what_if_scenarios(self, query: str, decision: Dict) -> List[str]:
        """Generate what-if scenarios for better understanding"""
        prompt = WHAT_IF_PROMPT.format(
            query=query,
            decision=json.dumps(decision, indent=2)
        )
        
        response = self.llm.generate_response(prompt, temperature=0.5)
        
        # Check if response is valid
        if not response:
            return ["Unable to generate scenarios at this time."]
        
        # Parse scenarios from response
        scenarios = []
        lines = response.split('\n')
        current_scenario = ""
        
        for line in lines:
            if line.strip() and (line[0].isdigit() or line.startswith('-')):
                if current_scenario:
                    scenarios.append(current_scenario.strip())
                current_scenario = line
            else:
                current_scenario += " " + line
        
        if current_scenario:
            scenarios.append(current_scenario.strip())
        
        return scenarios[:3] if scenarios else ["No alternative scenarios could be generated."]
    
    def translate_response(self, response_dict: Dict, language: str) -> Dict:
        """Translate response to specified language"""
        if language == "English":
            return response_dict
            
        prompt = MULTILINGUAL_PROMPT.format(
            language=language,
            response=json.dumps(response_dict, indent=2)
        )
        
        translated = self.llm.generate_response(prompt, temperature=0.3)
        
        # Check if translation was successful
        if translated:
            translated_dict = self.llm.parse_json_response(translated)
            return translated_dict if translated_dict else response_dict
        
        return response_dict
    
    def _prepare_clause_text(self, clauses: List[Dict]) -> str:
        """Prepare clause text for decision making"""
        if not clauses:
            return "No relevant policy clauses found."
            
        clause_texts = []
        for i, clause in enumerate(clauses):
            if clause and 'chunk' in clause:
                relevance = clause.get('relevance_score', 0)
                chunk_text = clause.get('chunk', 'No text available')
                clause_texts.append(f"Clause {i+1} (Relevance: {relevance:.2f}):\n{chunk_text}\n")
        
        return "\n".join(clause_texts) if clause_texts else "No valid clauses to display."
    
    def _calculate_trust_score(self, decision: Dict, clauses: List[Dict]) -> int:
        """DEPRECATED - Use _calculate_trust_score_v2 instead"""
        return self._calculate_trust_score_v2(decision, clauses, {})
    
    def _create_default_decision(self) -> Dict:
        """Create a default decision when parsing fails"""
        return {
            "decision": "requires_review",
            "amount": None,
            "currency": "INR",
            "justification": "Unable to make a clear decision. Manual review required.",
            "referenced_clauses": [],
            "confidence_score": 30,
            "additional_notes": "System was unable to parse the decision properly.",
            "bajaj_trust_score": 30
        }