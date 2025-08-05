import json
from typing import Dict, List
import pandas as pd
import os
from datetime import datetime

class Formatters:
    @staticmethod
    def format_decision_output(decision: Dict, what_if_scenarios: List[str] = None) -> str:
        """Format decision output for display"""
        output = []
        
        # Main decision
        status_emoji = "✅" if decision.get('decision') == 'approved' else "❌" if decision.get('decision') == 'rejected' else "⚠️"
        output.append(f"## {status_emoji} Decision: {decision.get('decision', 'Unknown').upper()}")
        
        # Bajaj Trust Score
        trust_score = decision.get('bajaj_trust_score', 0)
        trust_emoji = "🟢" if trust_score >= 80 else "🟡" if trust_score >= 60 else "🔴"
        output.append(f"\n### {trust_emoji} Bajaj Trust Score™: {trust_score}%")
        
        # Amount
        if decision.get('amount'):
            try:
                amount = float(decision.get('amount'))
                output.append(f"\n### 💰 Amount: ₹{amount:,.2f}")
            except:
                output.append(f"\n### 💰 Amount: ₹{decision.get('amount')}")
        
        # Justification
        output.append(f"\n### 📋 Justification:\n{decision.get('justification', 'No justification provided')}")
        
        # Referenced Clauses
        if decision.get('referenced_clauses'):
            output.append("\n### 📑 Referenced Clauses:")
            for clause in decision['referenced_clauses']:
                output.append(f"- {clause}")
        
        # Additional Notes
        if decision.get('additional_notes'):
            output.append(f"\n### 📝 Additional Notes:\n{decision.get('additional_notes')}")
        
        # What-If Scenarios
        if what_if_scenarios:
            output.append("\n### 🔮 What-If Analysis:")
            for i, scenario in enumerate(what_if_scenarios, 1):
                output.append(f"\n**Scenario {i}:** {scenario}")
        
        return "\n".join(output)
    
    @staticmethod
    def format_query_details(details: Dict) -> str:
        """Format parsed query details for display"""
        if not details:
            return "Unable to parse query details"
        
        df_data = []
        for key, value in details.items():
            if key != "other_details" and value:
                df_data.append({
                    "Field": key.replace("_", " ").title(),
                    "Value": value
                })
        
        if details.get("other_details"):
            for detail in details["other_details"]:
                df_data.append({
                    "Field": "Additional Detail",
                    "Value": detail
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            
            # Try to use to_markdown if tabulate is installed
            try:
                return df.to_markdown(index=False)
            except ImportError:
                # Fallback to simple string formatting
                output = []
                output.append("| Field | Value |")
                output.append("|-------|-------|")
                for _, row in df.iterrows():
                    output.append(f"| {row['Field']} | {row['Value']} |")
                return "\n".join(output)
            except Exception:
                # Ultimate fallback
                output = []
                for _, row in df.iterrows():
                    output.append(f"**{row['Field']}**: {row['Value']}")
                return "\n".join(output)
        
        return "No details extracted"
    
    @staticmethod
    def format_search_results(results: List[Dict]) -> str:
        """Format search results for display"""
        if not results:
            return "No relevant clauses found"
        
        output = []
        for i, result in enumerate(results, 1):
            relevance_score = result.get('relevance_score', 0)
            output.append(f"### 📄 Clause {i} (Relevance: {relevance_score:.1%})")
            
            metadata = result.get('metadata', {})
            doc_name = metadata.get('document', 'Unknown')
            output.append(f"**Document:** {doc_name}")
            
            chunk_text = result.get('chunk', '')
            if len(chunk_text) > 500:
                chunk_text = chunk_text[:500] + '...'
            output.append(f"```\n{chunk_text}\n```")
            output.append("")
        
        return "\n".join(output)
    
    @staticmethod
    def export_to_json(query: str, details: Dict, decision: Dict, scenarios: List[str] = None) -> str:
        """Export complete analysis to JSON format"""
        export_data = {
            "query": query,
            "parsed_details": details,
            "decision": decision,
            "what_if_scenarios": scenarios or [],
            "metadata": {
                "bajaj_trust_score": decision.get('bajaj_trust_score', 0),
                "confidence_level": "High" if decision.get('confidence_score', 0) >= 80 else "Medium" if decision.get('confidence_score', 0) >= 60 else "Low",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def create_summary_report(query: str, decision: Dict) -> str:
        """Create a brief summary report for quick reference"""
        trust_score = decision.get('bajaj_trust_score', 0)
        amount = decision.get('amount', 'N/A')
        
        # Format amount safely
        try:
            if amount != 'N/A':
                amount = f"{float(amount):,}"
        except:
            pass
        
        summary = f"""
### 📊 Claim Pre-Check Summary Report

**Query:** {query}

**Status:** {decision.get('decision', 'Unknown').upper()}
**Trust Score:** {trust_score}%
**Amount:** ₹{amount} if applicable

**Quick Summary:** {decision.get('justification', 'No summary available')[:100]}...

---
*This is a preliminary assessment. Please consult with Bajaj Allianz representatives for final claim processing.*
        """
        return summary
    
    @staticmethod
    def list_documents_summary(documents_path: str) -> Dict:
        """Create a summary of documents in the folder"""
        import glob
        
        doc_summary = {
            'pdf': [],
            'docx': [],
            'txt': [],
            'total_size': 0
        }
        
        try:
            for ext, file_list in [('pdf', '*.pdf'), ('docx', '*.docx'), ('txt', '*.txt')]:
                files = glob.glob(os.path.join(documents_path, file_list))
                for file in files:
                    file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
                    doc_summary[ext].append({
                        'name': os.path.basename(file),
                        'size': f"{file_size:.2f} MB",
                        'modified': datetime.fromtimestamp(os.path.getmtime(file)).strftime("%Y-%m-%d %H:%M")
                    })
                    doc_summary['total_size'] += file_size
        except Exception as e:
            print(f"Error creating document summary: {e}")
        
        return doc_summary