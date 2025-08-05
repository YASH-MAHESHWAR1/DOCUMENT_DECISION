import os
import PyPDF2
from docx import Document
import re
from typing import List, Dict
from config.settings import settings

class DocumentProcessor:
    def __init__(self, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def read_pdf(self, file_path):
        """Extract text from PDF"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text += f"\n[Page {page_num + 1}]\n"
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text
    
    def read_docx(self, file_path):
        """Extract text from Word document"""
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX: {e}")
        return text
    
    def read_txt(self, file_path):
        """Extract text from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT: {e}")
            return ""
    
    def process_document(self, file_path):
        """Process document based on file type"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            text = self.read_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            text = self.read_docx(file_path)
        elif ext in ['.txt', '.text']:
            text = self.read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
        return self.clean_text(text)
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        # Keep important punctuation
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)$$$$\{\}\"\'\/\@\#\$\%\&\*\+\=]', '', text)
        return text.strip()
    
    def create_chunks(self, text, doc_name=""):
        """Create overlapping chunks from text"""
        chunks = []
        metadata = []
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
                current_sentences.append(sentence)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    metadata.append({
                        'document': doc_name,
                        'chunk_index': len(chunks),
                        'sentences': len(current_sentences)
                    })
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_size = 0
                for sent in reversed(current_sentences):
                    if overlap_size + len(sent) <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break
                
                current_chunk = " ".join(overlap_sentences) + " " + sentence + " "
                current_sentences = overlap_sentences + [sentence]
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
            metadata.append({
                'document': doc_name,
                'chunk_index': len(chunks),
                'sentences': len(current_sentences)
            })
        
        return chunks, metadata