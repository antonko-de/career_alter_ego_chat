from openai import OpenAI
import json
import logging
from pypdf import PdfReader
from typing import List, Dict, Optional
import numpy as np
import faiss
import os
import re
from dataclasses import dataclass
from src.main.python.utils_and_tools import tools, record_user_details, record_unknown_question, send_push_when_engaged

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RAG system
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

# Main class
class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "Anton Kostov"
        
        # Initialize RAG system with custom chunk size
        logger.info("Initializing RAG system...")
        self.rag = SimpleRAG(chunk_size=300, chunk_overlap=50)
        
        # Load and process LinkedIn profile
        reader = PdfReader("resources/Profile.pdf")
        linkedin_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                linkedin_text += text
        
        # Load summary
        with open("resources/summary.txt", "r", encoding="utf-8") as f:
            summary = f.read()
        
        # Add documents to RAG system with metadata
        logger.info("Adding documents to RAG system...")
        self.rag.add_documents(
            documents=[summary, linkedin_text],
            metadata_list=[
                {"source": "summary", "type": "background"},
                {"source": "linkedin", "type": "professional_profile"}
            ]
        )
        
        # Save the processed documents
        logger.info("Saving RAG data...")
        self.rag.save("rag_data")
        logger.info("Initialization complete!")

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool_map = {
                "record_user_details": record_user_details,
                "record_unknown_question": record_unknown_question,
                "send_push_when_engaged": send_push_when_engaged
            }
            tool = tool_map.get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
When the user greets you, use your send_push_when_engaged tool to send a push notification. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        logger.info(f"Received message: {message[:100]}...")
        # Get relevant context from RAG
        results = self.rag.search(message, k=3)
        context = self.rag.format_context(results)
        logger.info(f"Retrieved {len(results)} relevant documents")
        
        # Add context to the message with clear separation
        enhanced_message = f"""Here is the relevant context from my background:

{context}

Based on this context, please answer the following question:
{message}"""
        
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": enhanced_message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        logger.info("Generated response successfully")
        return response.choices[0].message.content 