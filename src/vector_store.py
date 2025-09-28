"""
Vector Database Integration for PyAgenticRAG
Supports multiple vector database providers
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import uuid
import numpy as np

# Vector DB specific imports (will be imported conditionally)
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import qdrant_client
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document representation"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class SearchResult:
    """Search result representation"""
    document: Document
    score: float

class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents by IDs"""
        pass
    
    @abstractmethod
    def update_document(self, document: Document) -> None:
        """Update a document"""
        pass

class EmbeddingModel:
    """Wrapper for embedding models"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.model.encode(text).tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return self.model.encode(texts).tolist()

class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, collection_name: str = "pyagentic_rag", persist_directory: str = "./chroma_db"):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = EmbeddingModel()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "PyAgenticRAG document collection"}
        )
        
        logger.info(f"Initialized ChromaDB collection: {collection_name}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to ChromaDB"""
        if not documents:
            return
        
        # Generate embeddings if not provided
        texts_to_embed = []
        embeddings = []
        
        for doc in documents:
            if doc.embedding is None:
                texts_to_embed.append(doc.content)
            else:
                embeddings.append(doc.embedding)
        
        if texts_to_embed:
            new_embeddings = self.embedding_model.embed_texts(texts_to_embed)
            embeddings.extend(new_embeddings)
        
        # Prepare data for ChromaDB
        ids = [doc.id for doc in documents]
        documents_text = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Add to collection
        self.collection.add(
            documents=documents_text,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for similar documents in ChromaDB"""
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Convert to SearchResult objects
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                document = Document(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] or {}
                )
                
                score = 1.0 - results['distances'][0][i]  # Convert distance to similarity
                search_results.append(SearchResult(document=document, score=score))
        
        logger.info(f"Found {len(search_results)} results for query: {query[:50]}...")
        return search_results
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents by IDs"""
        self.collection.delete(ids=document_ids)
        logger.info(f"Deleted {len(document_ids)} documents from ChromaDB")
    
    def update_document(self, document: Document) -> None:
        """Update a document"""
        # Delete existing and add new (ChromaDB doesn't have native update)
        self.delete_documents([document.id])
        self.add_documents([document])

class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation"""
    
    def __init__(self, collection_name: str = "pyagentic_rag", url: str = "http://localhost:6333"):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client is not installed. Install with: pip install qdrant-client")
        
        self.collection_name = collection_name
        self.embedding_model = EmbeddingModel()
        
        # Initialize Qdrant client
        self.client = qdrant_client.QdrantClient(url=url)
        
        # Create collection if it doesn't exist
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # all-MiniLM-L6-v2 dimension
            )
        except Exception as e:
            logger.info(f"Collection may already exist: {e}")
        
        logger.info(f"Initialized Qdrant collection: {collection_name}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Qdrant"""
        if not documents:
            return
        
        points = []
        for doc in documents:
            if doc.embedding is None:
                embedding = self.embedding_model.embed_text(doc.content)
            else:
                embedding = doc.embedding
            
            point = PointStruct(
                id=doc.id,
                vector=embedding,
                payload={
                    "content": doc.content,
                    **doc.metadata
                }
            )
            points.append(point)
        
        # Add to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Added {len(documents)} documents to Qdrant")
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for similar documents in Qdrant"""
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search in Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Convert to SearchResult objects
        search_results = []
        for result in results:
            document = Document(
                id=str(result.id),
                content=result.payload.get("content", ""),
                metadata={k: v for k, v in result.payload.items() if k != "content"}
            )
            search_results.append(SearchResult(document=document, score=result.score))
        
        logger.info(f"Found {len(search_results)} results for query: {query[:50]}...")
        return search_results
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents by IDs"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=document_ids
        )
        logger.info(f"Deleted {len(document_ids)} documents from Qdrant")
    
    def update_document(self, document: Document) -> None:
        """Update a document"""
        self.add_documents([document])  # Qdrant upsert handles updates

class VectorStoreFactory:
    """Factory for creating vector store instances"""
    
    @staticmethod
    def create_vector_store(provider: str, **kwargs) -> VectorStore:
        """Create a vector store instance based on provider"""
        provider = provider.lower()
        
        if provider == "chroma":
            return ChromaVectorStore(**kwargs)
        elif provider == "qdrant":
            return QdrantVectorStore(**kwargs)
        else:
            raise ValueError(f"Unsupported vector store provider: {provider}")

# Utility functions for document processing
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at word boundary
        if end < len(text):
            last_space = chunk.rfind(' ')
            if last_space > chunk_size // 2:  # Only if space is not too early
                chunk = chunk[:last_space]
                end = start + last_space
        
        chunks.append(chunk)
        start = end - chunk_overlap
        
        if start >= len(text):
            break
    
    return chunks

def create_documents_from_text(
    text: str,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """Create Document objects from text by chunking"""
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    documents = []
    
    base_metadata = metadata or {}
    
    for i, chunk in enumerate(chunks):
        doc_id = f"{source}_{i}_{uuid.uuid4().hex[:8]}"
        doc_metadata = {
            **base_metadata,
            "source": source,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        
        documents.append(Document(
            id=doc_id,
            content=chunk,
            metadata=doc_metadata
        ))
    
    return documents