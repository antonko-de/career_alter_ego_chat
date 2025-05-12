from typing import List, Dict, Optional
import numpy as np
import faiss
import json
import os
import re
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class Document:
    text: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None

class SimpleRAG:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the RAG system with OpenAI embeddings."""
        self.client = OpenAI()
        self.index = None
        self.documents: List[Document] = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
                
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API."""
        # OpenAI's API has a limit of 8191 tokens per request
        # We'll process in batches of 100 texts to be safe
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            
        return np.array(all_embeddings)
    
    def add_documents(self, documents: List[str], metadata_list: Optional[List[Dict]] = None):
        """Add documents to the vector store with metadata."""
        if metadata_list is None:
            metadata_list = [{"source": f"document_{i}"} for i in range(len(documents))]
            
        # Process and chunk documents
        processed_docs = []
        for doc, metadata in zip(documents, metadata_list):
            cleaned_text = self.preprocess_text(doc)
            chunks = self.chunk_text(cleaned_text)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = i
                processed_docs.append(Document(text=chunk, metadata=chunk_metadata))
        
        # Generate embeddings
        texts = [doc.text for doc in processed_docs]
        embeddings = self.get_embeddings(texts)
        
        # Initialize FAISS index if not exists
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents with their embeddings
        for doc, embedding in zip(processed_docs, embeddings):
            doc.embedding = embedding
            self.documents.append(doc)
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant documents using semantic search."""
        if not self.index or len(self.documents) == 0:
            return []
            
        # Generate query embedding
        query_embedding = self.get_embeddings([query])[0]
        
        # Search in FAISS index
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            k
        )
        
        # Convert distances to similarity scores (1 / (1 + distance))
        similarities = 1 / (1 + distances[0])
        
        # Return results with metadata and scores
        return [{
            "text": self.documents[idx].text,
            "metadata": self.documents[idx].metadata,
            "score": float(similarity)  # Convert to float for JSON serialization
        } for idx, similarity in zip(indices[0], similarities)]
    
    def format_context(self, results: List[Dict]) -> str:
        """Format search results into a coherent context string."""
        context_parts = []
        for result in results:
            source = result["metadata"].get("source", "Unknown")
            context_parts.append(f"From {source}:\n{result['text']}\n")
        return "\n".join(context_parts)
    
    def save(self, directory: str):
        """Save the RAG system to disk."""
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # Save documents and metadata
        documents_data = [{
            "text": doc.text,
            "metadata": doc.metadata,
            "embedding": doc.embedding.tolist() if doc.embedding is not None else None
        } for doc in self.documents]
        
        with open(os.path.join(directory, "documents.json"), "w") as f:
            json.dump(documents_data, f)
    
    def load(self, directory: str):
        """Load the RAG system from disk."""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        
        # Load documents and metadata
        with open(os.path.join(directory, "documents.json"), "r") as f:
            documents_data = json.load(f)
        
        self.documents = [
            Document(
                text=doc["text"],
                metadata=doc["metadata"],
                embedding=np.array(doc["embedding"]) if doc["embedding"] else None
            )
            for doc in documents_data
        ] 