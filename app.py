import streamlit as st
import os
import json
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import glob

# Import custom modules
from config.settings import settings
from models.llm_handler import LLMHandler
from models.embeddings import EmbeddingHandler
from core.document_processor import DocumentProcessor
from core.query_parser import QueryParser
from core.semantic_search import SemanticSearch
from core.decision_engine import DecisionEngine
from utils.validators import Validators
from utils.formatters import Formatters

# Page configuration
st.set_page_config(
    page_title="Bajaj Insurance Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #004B8D 0%, #0066CC 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .trust-score-high { color: #00A652; }
    .trust-score-medium { color: #FF9500; }
    .trust-score-low { color: #FF3B30; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def load_existing_documents():
    """Load and process documents already in the data/documents folder"""
    existing_files = []
    
    # Check for different file types
    for ext in ['*.pdf', '*.docx', '*.doc', '*.txt']:
        files = glob.glob(os.path.join(settings.DOCUMENTS_PATH, ext))
        existing_files.extend(files)
    
    return existing_files

def process_existing_documents(llm_provider):
    """Process existing documents in the documents folder"""
    existing_files = load_existing_documents()
    
    if not existing_files:
        return None, 0
    
    # Initialize handlers
    embeddings_handler = EmbeddingHandler(llm_provider)
    doc_processor = DocumentProcessor()
    
    all_chunks = []
    all_metadata = []
    
    for file_path in existing_files:
        try:
            # Process document
            text = doc_processor.process_document(file_path)
            doc_name = os.path.basename(file_path)
            chunks, metadata = doc_processor.create_chunks(text, doc_name)
            
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)
        except Exception as e:
            st.warning(f"Error processing {os.path.basename(file_path)}: {str(e)}")
    
    if all_chunks:
        # Create embeddings
        embeddings = embeddings_handler.create_embeddings(all_chunks, all_metadata)
        embeddings_handler.chunks = all_chunks
        embeddings_handler.metadata = all_metadata
        embeddings_handler.build_index(embeddings)
        
        # Save embeddings
        embeddings_handler.save_index(settings.EMBEDDINGS_PATH)
        
        return embeddings_handler, len(existing_files)
    
    return None, 0

# Initialize session state
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = False
if 'embeddings_handler' not in st.session_state:
    st.session_state.embeddings_handler = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 Bajaj Insurance Assistant</h1>
    <p>AI-powered insurance claim analyzer with semantic understanding</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # LLM Provider Selection
    llm_provider = st.selectbox(
        "Select AI Provider",
        ["gemini", "openai"],
        index=0 if settings.DEFAULT_LLM == "gemini" else 1
    )
    
    # Validate API keys
    valid, message = Validators.validate_api_keys(llm_provider)
    if not valid:
        st.error(message)
        st.stop()
    
    # Language Selection
    language = st.selectbox(
        "Response Language",
        settings.SUPPORTED_LANGUAGES,
        index=0
    )
    
    # Document Management Section
    st.header("📄 Document Management")
    
    # Check for existing documents
    existing_docs = load_existing_documents()
    if existing_docs:
        st.info(f"Found {len(existing_docs)} documents in the documents folder")
        
        # Show list of existing documents
        with st.expander("View existing documents"):
            for doc in existing_docs:
                st.text(f"📄 {os.path.basename(doc)}")
        
        # Option to process existing documents
        if st.button("Process Existing Documents", type="primary", key="process_existing"):
            with st.spinner("Processing existing documents..."):
                embeddings_handler, num_processed = process_existing_documents(llm_provider)
                
                if embeddings_handler:
                    st.session_state.embeddings_handler = embeddings_handler
                    st.session_state.processed_docs = True
                    st.success(f"✅ Processed {num_processed} existing documents")
                else:
                    st.error("Failed to process existing documents")
    
    # Document Upload
    st.subheader("📤 Upload New Documents")
    uploaded_files = st.file_uploader(
        "Upload Policy Documents",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload insurance policy documents for analysis"
    )
    
    if uploaded_files:
        if st.button("Process New Documents", type="primary", key="process_new"):
            with st.spinner("Processing documents..."):
                # Initialize handlers
                llm_handler = LLMHandler(llm_provider)
                embeddings_handler = EmbeddingHandler(llm_provider)
                doc_processor = DocumentProcessor()
                
                all_chunks = []
                all_metadata = []
                
                # First, load any existing embeddings
                existing_embeddings = EmbeddingHandler(llm_provider)
                if existing_embeddings.load_index(settings.EMBEDDINGS_PATH):
                    all_chunks = existing_embeddings.chunks
                    all_metadata = existing_embeddings.metadata
                
                progress_bar = st.progress(0)
                for idx, file in enumerate(uploaded_files):
                    # Save uploaded file
                    file_path = os.path.join(settings.DOCUMENTS_PATH, file.name)
                    
                    # Check if file already exists
                    if os.path.exists(file_path):
                        if st.checkbox(f"Overwrite existing {file.name}?", key=f"overwrite_{idx}"):
                            with open(file_path, 'wb') as f:
                                f.write(file.getbuffer())
                        else:
                            st.warning(f"Skipped {file.name} - already exists")
                            continue
                    else:
                        with open(file_path, 'wb') as f:
                            f.write(file.getbuffer())
                    
                    # Process document
                    text = doc_processor.process_document(file_path)
                    chunks, metadata = doc_processor.create_chunks(text, file.name)
                    
                    all_chunks.extend(chunks)
                    all_metadata.extend(metadata)
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                # Create embeddings for all chunks
                st.info("Creating embeddings...")
                embeddings = embeddings_handler.create_embeddings(all_chunks, all_metadata)
                embeddings_handler.chunks = all_chunks
                embeddings_handler.metadata = all_metadata
                embeddings_handler.build_index(embeddings)
                
                # Save embeddings
                embeddings_handler.save_index(settings.EMBEDDINGS_PATH)
                
                st.session_state.embeddings_handler = embeddings_handler
                st.session_state.processed_docs = True
                st.success(f"✅ Processed {len(uploaded_files)} new documents. Total chunks: {len(all_chunks)}")
    
    # Load existing embeddings
    st.subheader("💾 Load Previous Session")
    if st.button("Load Previous Embeddings", key="load_embeddings"):
        embeddings_handler = EmbeddingHandler(llm_provider)
        if embeddings_handler.load_index(settings.EMBEDDINGS_PATH):
            st.session_state.embeddings_handler = embeddings_handler
            st.session_state.processed_docs = True
            st.success(f"✅ Loaded embeddings with {len(embeddings_handler.chunks)} chunks")
        else:
            st.error("No previous embeddings found")
    
    # Document management options
    if st.session_state.processed_docs and st.session_state.embeddings_handler:
        st.subheader("🗑️ Manage Documents")
        
        if st.button("Clear All Documents", key="clear_docs"):
            if st.checkbox("Are you sure? This will delete all documents and embeddings"):
                # Clear documents
                for file in load_existing_documents():
                    os.remove(file)
                
                # Clear embeddings
                for file in glob.glob(os.path.join(settings.EMBEDDINGS_PATH, "*")):
                    os.remove(file)
                
                # Reset session state
                st.session_state.processed_docs = False
                st.session_state.embeddings_handler = None
                st.session_state.query_history = []
                
                st.success("✅ All documents and embeddings cleared")
                st.rerun()
    
    # Debug mode
    if st.checkbox("🐛 Debug Mode"):
        st.subheader("Debug Information")
        
        # Basic Status
        debug_info = {
            "System Status": {
                "LLM Provider": llm_provider,
                "Language": language,
                "Documents Processed": st.session_state.processed_docs,
                "Embeddings Ready": st.session_state.embeddings_handler is not None,
            }
        }
        
        # API Keys Status (masked)
        debug_info["API Keys"] = {
            "Gemini": "✅ Configured" if settings.GEMINI_API_KEY else "❌ Missing",
            "OpenAI": "✅ Configured" if settings.OPENAI_API_KEY else "❌ Missing",
        }
        
        # Document Statistics
        if st.session_state.embeddings_handler:
            debug_info["Document Stats"] = {
                "Total Chunks": len(st.session_state.embeddings_handler.chunks),
                "Unique Documents": len(set(m.get('document', 'Unknown') for m in st.session_state.embeddings_handler.metadata)),
                "Index Size": f"{len(st.session_state.embeddings_handler.chunks) * 768 * 4 / (1024*1024):.2f} MB" # Approximate
            }
        
        # Storage Info
        existing_docs = load_existing_documents()
        debug_info["Storage"] = {
            "Documents in Folder": len(existing_docs),
            "Embeddings Saved": os.path.exists(os.path.join(settings.EMBEDDINGS_PATH, f"index_{llm_provider}.faiss")),
            "Total Document Size": f"{sum(os.path.getsize(f) for f in existing_docs) / (1024*1024):.2f} MB" if existing_docs else "0 MB"
        }
        
        # Query History
        if st.session_state.query_history:
            # Safe calculation of average trust score
            trust_scores = [q.get('trust_score', 0) for q in st.session_state.query_history]
            avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0
            
            debug_info["Recent Queries"] = {
                "Total Queries": len(st.session_state.query_history),
                "Last Query Time": st.session_state.query_history[-1]['timestamp'] if st.session_state.query_history else "N/A",
                "Avg Trust Score": f"{avg_trust:.1f}%"
            }
        
        # System Performance
        try:
            import psutil
            debug_info["System Resources"] = {
                "CPU Usage": f"{psutil.cpu_percent()}%",
                "Memory Usage": f"{psutil.virtual_memory().percent}%",
                "Available Memory": f"{psutil.virtual_memory().available / (1024**3):.2f} GB"
            }
        except ImportError:
            debug_info["System Resources"] = "psutil not installed"
        
        # Display debug info
        st.json(debug_info)
        
        # Additional Debug Actions
        st.subheader("Debug Actions")
        
        # Test embeddings
        if st.button("Test Embeddings", key="test_embeddings"):
            try:
                test_text = "Test insurance claim for knee surgery"
                if llm_provider == "openai":
                    import openai
                    response = openai.Embedding.create(
                        input=[test_text],
                        model=settings.EMBEDDING_MODEL_OPENAI
                    )
                    st.success(f"✅ OpenAI Embeddings working. Dimension: {len(response['data'][0]['embedding'])}")
                else:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(settings.EMBEDDING_MODEL_OS)
                    embedding = model.encode([test_text])
                    st.success(f"✅ E5 Embeddings working. Dimension: {embedding.shape[1]}")
            except Exception as e:
                st.error(f"❌ Embedding test failed: {str(e)}")
        
        # Test LLM
        if st.button("Test LLM", key="test_llm"):
            try:
                llm = LLMHandler(llm_provider)
                response = llm.generate_response("Say 'Hello, Bajaj Insurance!'")
                st.success(f"✅ LLM Response: {response}")
            except Exception as e:
                st.error(f"❌ LLM test failed: {str(e)}")
        
        # Export session data
        if st.button("Export Debug Log", key="export_debug"):
            debug_log = {
                "timestamp": datetime.now().isoformat(),
                "debug_info": debug_info,
                "session_state": {
                    "processed_docs": st.session_state.processed_docs,
                    "query_history": st.session_state.query_history
                }
            }
            
            st.download_button(
                label="Download Debug Log",
                data=json.dumps(debug_log, indent=2),
                file_name=f"debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Clear cache
        if st.button("Clear Streamlit Cache", key="clear_cache"):
            st.cache_data.clear()
            st.success("✅ Cache cleared")

# Main content area
if st.session_state.processed_docs:
    # Initialize components
    llm_handler = LLMHandler(llm_provider)
    query_parser = QueryParser(llm_handler)
    semantic_search = SemanticSearch(st.session_state.embeddings_handler)
    decision_engine = DecisionEngine(llm_handler)
    
    # Query input
    st.header("🔍 Insurance Claim Query")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_area(
            "Enter your insurance query",
            placeholder="Example: 46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
            height=100
        )
    
    with col2:
        st.markdown("### 💡 Quick Examples")
        example_queries = [
            "46M, knee surgery, Pune, 3-month policy",
            "25F, maternity expenses, Mumbai, 1-year policy",
            "55M, heart surgery, Delhi, 6-month policy"
        ]
        for ex in example_queries:
            if st.button(ex, key=f"ex_{ex}"):
                query = ex
    
    if st.button("🚀 Analyze Query", type="primary", disabled=not query):
        # Validate query
        valid, message = Validators.validate_query(query)
        if not valid:
            st.error(message)
        else:
            with st.spinner("Analyzing your query..."):
                # Parse query
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Extracted Information")
                    query_details = query_parser.parse_query(query)
                    st.markdown(Formatters.format_query_details(query_details))
                
                with col2:
                    st.subheader("🔍 Semantic Search Results")
                    search_results = semantic_search.search_relevant_clauses(query, query_details)
                    with st.expander("View Relevant Clauses", expanded=True):
                        st.markdown(Formatters.format_search_results(search_results[:3]))
                
                # Make decision
                st.subheader("🎯 Decision Analysis")
                decision = decision_engine.make_decision(query_details, search_results)
                
                # Translate if needed
                if language != "English":
                    decision = decision_engine.translate_response(decision, language)
                
                # Display decision
                st.markdown(Formatters.format_decision_output(decision))
                
                # Trust Score Visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=decision.get('bajaj_trust_score', 0),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Bajaj Trust Score™"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 60], 'color': "lightgray"},
                            {'range': [60, 80], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # What-If Analysis
                if decision.get('bajaj_trust_score', 0) < settings.CONFIDENCE_THRESHOLD:
                    st.subheader("🔮 What-If Analysis")
                    st.info("Since the confidence is below 80%, here are scenarios that might change the outcome:")
                    
                    scenarios = decision_engine.generate_what_if_scenarios(query, decision)
                    for i, scenario in enumerate(scenarios, 1):
                        st.markdown(f"**Scenario {i}:** {scenario}")
                
                # Save to history
                st.session_state.query_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'query': query,
                    'decision': decision.get('decision'),
                    'trust_score': decision.get('bajaj_trust_score', 0)
                })
                
                # Export options
                st.subheader("📥 Export Options")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # JSON Export
                    scenarios = decision_engine.generate_what_if_scenarios(query, decision) if decision.get('bajaj_trust_score', 0) < settings.CONFIDENCE_THRESHOLD else None
                    json_export = Formatters.export_to_json(
                        query, query_details, decision, scenarios
                    )
                    st.download_button(
                        label="Download JSON Report",
                        data=json_export,
                        file_name=f"claim_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Summary Report
                    summary = Formatters.create_summary_report(query, decision)
                    st.download_button(
                        label="Download Summary",
                        data=summary,
                        file_name=f"claim_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col3:
                    # Pre-Check Certificate (Bajaj Special)
                    if st.button("Generate Pre-Check Certificate"):
                        cert = f"""
BAJAJ ALLIANZ INSURANCE
CLAIM PRE-CHECK CERTIFICATE

Date: {datetime.now().strftime("%B %d, %Y")}
Reference: BAJ-{datetime.now().strftime("%Y%m%d%H%M%S")}

Query: {query}
Decision: {decision.get('decision', 'Unknown').upper()}
Trust Score: {decision.get('bajaj_trust_score', 0)}%

This certificate confirms that a preliminary assessment has been 
conducted using AI-powered analysis. Final claim approval is 
subject to standard verification procedures.

---
Digital Verification Code: {hash(query + str(decision)) % 1000000:06d}
                        """
                        st.download_button(
                            label="Download Certificate",
                            data=cert,
                            file_name=f"precheck_certificate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
    
    # Query History
    if st.session_state.query_history:
        with st.expander("📜 Query History"):
            history_df = pd.DataFrame(st.session_state.query_history)
            st.dataframe(
                history_df,
                column_config={
                    "timestamp": "Time",
                    "query": "Query",
                    "decision": st.column_config.TextColumn("Decision", help="Claim decision"),
                    "trust_score": st.column_config.ProgressColumn(
                        "Trust Score",
                        help="Bajaj Trust Score",
                        format="%d%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
                hide_index=True,
                use_container_width=True
            )

else:
    # Welcome screen when no documents are processed
    st.info("👈 Please upload policy documents from the sidebar to begin analysis")
    
    # Feature showcase
    st.header("✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 AI-Powered Analysis
        - Natural language understanding
        - Semantic search, not just keywords
        - Multi-model support (Gemini/OpenAI)
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Bajaj Trust Score™
        - Confidence scoring for decisions
        - What-if scenario analysis
        - Pre-check certificates
        """)
    
    with col3:
        st.markdown("""
        ### 🌐 Multi-Language Support
        - English and Hindi responses
        - Clear explanations
        - Export in multiple formats
        """)
    
    # Sample workflow
    st.header("📋 How It Works")
    st.markdown("""
    1. **Upload Documents**: Add your insurance policy PDFs, Word docs, or text files
    2. **Enter Query**: Type your claim query in natural language
    3. **Get Analysis**: Receive instant decision with justification and trust score
    4. **Export Results**: Download reports and pre-check certificates
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ for Bajaj Allianz | Powered by Advanced AI</p>
        <p>This is a demonstration system. Always consult official channels for final claim processing.</p>
    </div>
    """, unsafe_allow_html=True)