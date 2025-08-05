from models.embeddings import EmbeddingHandler
from typing import List, Dict, Optional

class SemanticSearch:
    def __init__(self, embedding_handler: EmbeddingHandler):
        self.embeddings = embedding_handler
        
    def search_relevant_clauses(self, query: str, query_details: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant policy clauses based on the query"""
        
        # Enhance query with extracted details
        enhanced_query = self._enhance_query(query, query_details)
        
        # Perform semantic search
        results = self.embeddings.search(enhanced_query)
        
        # Post-process results
        return self._post_process_results(results, query_details)
    
    def _enhance_query(self, query: str, details: Optional[Dict]) -> str:
        """Enhance query with structured information for better search"""
        enhanced_parts = [query]
        
        if details:
            if details.get("procedure"):
                enhanced_parts.append(f"medical procedure: {details['procedure']}")
            if details.get("age"):
                enhanced_parts.append(f"age group: {details['age']} years")
            if details.get("location"):
                enhanced_parts.append(f"location: {details['location']}")
            if details.get("policy_duration"):
                enhanced_parts.append(f"policy duration: {details['policy_duration']} months")
        
        # Add insurance-specific keywords
        enhanced_parts.extend([
            "insurance coverage",
            "claim eligibility",
            "policy terms",
            "exclusions inclusions"
        ])
        
        return " ".join(enhanced_parts)
    
    def _post_process_results(self, results: List[Dict], query_details: Optional[Dict]) -> List[Dict]:
        """Filter and rank results based on relevance"""
        processed_results = []
        
        for result in results:
            # Calculate relevance score
            relevance_score = result.get('similarity', 0)
            
            # Boost score if chunk contains specific keywords
            chunk_lower = result.get('chunk', '').lower()
            
            if query_details and query_details.get("procedure"):
                procedure = query_details['procedure'].lower()
                if procedure in chunk_lower:
                    relevance_score *= 1.2
            
            # Check for policy-specific terms
            policy_terms = ["coverage", "covered", "eligible", "claim", "benefit", "exclusion"]
            term_count = sum(1 for term in policy_terms if term in chunk_lower)
            relevance_score *= (1 + term_count * 0.05)
            
            result['relevance_score'] = min(relevance_score, 1.0)
            processed_results.append(result)
        
        # Sort by relevance score
        processed_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return processed_results[:5]  # Return top 5 results