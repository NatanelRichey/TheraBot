"""
RAG System for TheraBot with DBT Manual
Combines trained adapter + DBT manual retrieval for enhanced responses.
"""

import os
import pickle
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import chromadb
from chromadb.config import Settings

# Import DBT prompts and inference system
from load_adapter_for_inference import TheraBotInference
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DBT_RAGSystem:
    """
    Complete RAG system for TheraBot.
    
    Combines:
    1. Trained LoRA adapter (therapist conversation skills)
    2. DBT manual vector database (knowledge base)
    3. Retrieval augmentation (context injection)
    """
    
    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        adapter_path: str = None,
        dbt_dataset: str = "vjain/dbt",
        use_4bit: bool = True,
        chunk_size: int = 1000,  # Optimal for DBT manual (avg ~1000 chars)
        chunk_overlap: int = 200,
        top_k_retrieval: int = 3,
        vector_db_path: str = "./dbt_vector_db"
    ):
        """
        Initialize RAG system.
        
        Args:
            base_model_name: Base model name
            adapter_path: Path to trained LoRA adapter
            dbt_dataset: HuggingFace dataset name for DBT manual
            use_4bit: Use 4-bit quantization
            chunk_size: Text chunk size for embedding
            chunk_overlap: Overlap between chunks
            top_k_retrieval: Number of chunks to retrieve
            vector_db_path: Path to ChromaDB vector database
        """
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.dbt_dataset_name = dbt_dataset
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k_retrieval
        self.vector_db_path = vector_db_path
        
        logger.info("🚀 Initializing DBT RAG System...")
        
        # Initialize components
        self._init_embedding_model()
        self._init_vector_database()
        self._init_generation_model(use_4bit)
        
        # Load DBT manual if not already indexed
        if not self._check_vector_db_populated():
            logger.info("📥 Vector database empty, loading DBT manual...")
            self._load_and_index_dbt_manual()
        else:
            logger.info("✅ Vector database already populated")
        
        logger.info("✅ DBT RAG System initialized!")
    
    def _init_embedding_model(self):
        """Initialize sentence transformer for embeddings."""
        logger.info("📥 Loading embedding model...")
        # Using a general-purpose model good for retrieval
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Embedding model loaded")
    
    def _init_vector_database(self):
        """Initialize ChromaDB vector database."""
        logger.info(f"📦 Initializing vector database at: {self.vector_db_path}")
        self.chroma_client = chromadb.PersistentClient(
            path=self.vector_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection("dbt_manual")
            logger.info("✅ Found existing DBT manual collection")
        except:
            self.collection = self.chroma_client.create_collection(
                name="dbt_manual",
                metadata={"description": "DBT Manual Knowledge Base"}
            )
            logger.info("✅ Created new DBT manual collection")
    
    def _init_generation_model(self, use_4bit: bool):
        """Initialize TheraBot inference engine."""
        logger.info("📥 Initializing TheraBot generation model...")
        self.therabot = TheraBotInference(
            base_model_name=self.base_model_name,
            adapter_path=self.adapter_path,
            use_4bit=use_4bit
        )
        logger.info("✅ Generation model loaded")
    
    def _check_vector_db_populated(self) -> bool:
        """Check if vector database has data."""
        try:
            count = self.collection.count()
            return count > 0
        except:
            return False
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            
            # Move forward by chunk_size - overlap
            start += chunk_size - overlap
            
            # Don't create tiny final chunk
            if start + overlap >= len(text):
                break
        
        return chunks
    
    def _load_and_index_dbt_manual(self):
        """Load DBT manual dataset and create vector embeddings."""
        logger.info(f"📥 Loading DBT dataset: {self.dbt_dataset_name}")
        
        try:
            ds = load_dataset(self.dbt_dataset_name)
            
            # Process dataset
            all_texts = []
            all_embeddings = []
            all_metadata = []
            
            logger.info("🔄 Processing DBT manual...")
            
            # Iterate through splits
            for split_name, split_data in ds.items():
                logger.info(f"Processing split: {split_name} ({len(split_data)} examples)")
                
                for i, example in enumerate(split_data):
                    # Extract text from DBT manual (using 'page_content' field)
                    text = example.get('page_content', example.get('text', example.get('content', str(example))))
                    
                    if isinstance(text, str) and len(text) > 50:
                        # Chunk long texts
                        chunks = self._chunk_text(text, self.chunk_size, self.chunk_overlap)
                        
                        # Create embeddings for each chunk
                        for j, chunk in enumerate(chunks):
                            all_texts.append(chunk)
                            
                            # Store metadata
                            metadata = {
                                "split": split_name,
                                "example_idx": i,
                                "chunk_idx": j,
                                "text_length": len(chunk),
                                "source": example.get('source', 'dbt_manual')
                            }
                            all_metadata.append(metadata)
            
            logger.info(f"📊 Created {len(all_texts)} chunks")
            
            # Create embeddings in batches
            logger.info("🔢 Creating embeddings...")
            batch_size = 32
            for i in range(0, len(all_texts), batch_size):
                batch = all_texts[i:i+batch_size]
                embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(embeddings.tolist())
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"   Processed {i + len(batch)}/{len(all_texts)} chunks")
            
            # Add to vector database
            logger.info("💾 Adding to vector database...")
            ids = [f"chunk_{i}" for i in range(len(all_texts))]
            
            self.collection.add(
                embeddings=all_embeddings,
                documents=all_texts,
                metadatas=all_metadata,
                ids=ids
            )
            
            logger.info(f"✅ Indexed {len(all_texts)} chunks in vector database")
            
        except Exception as e:
            logger.error(f"❌ Error loading DBT dataset: {e}")
            logger.info("💡 Alternative: Manually populate with text files (see below)")
            raise
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant chunks from DBT manual.
        
        Args:
            query: User's question or statement
            top_k: Number of chunks to retrieve (uses self.top_k if None)
        
        Returns:
            List of relevant chunks with metadata
        """
        top_k = top_k or self.top_k
        
        # Create query embedding
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False).tolist()
        
        # Search vector database
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        retrieved_chunks = []
        for i in range(min(top_k, len(results['documents'][0]))):
            chunk = {
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            }
            retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def chat(
        self,
        user_message: str,
        conversation_history: List[Tuple[str, str]] = None,
        use_rag: bool = True,
        top_k: int = None,
        **kwargs
    ) -> Tuple[str, List[Dict]]:
        """
        Chat with RAG-enhanced responses.
        
        Args:
            user_message: User's message
            conversation_history: Previous conversation
            use_rag: Whether to use RAG retrieval
            top_k: Number of chunks to retrieve
            **kwargs: Additional args for generation
        
        Returns:
            Tuple of (response, retrieved_chunks)
        """
        retrieved_chunks = []
        
        if use_rag:
            # Retrieve relevant chunks
            logger.info(f"🔍 Retrieving relevant DBT information...")
            retrieved_chunks = self.retrieve(user_message, top_k=top_k)
            
            # Log what was retrieved
            logger.info(f"✅ Retrieved {len(retrieved_chunks)} relevant chunks")
            for i, chunk in enumerate(retrieved_chunks[:2]):  # Log first 2
                logger.info(f"   Chunk {i+1}: {chunk['text'][:100]}...")
            
            # Create enriched prompt with retrieved context
            context = "\n\n".join([
                f"[DBT Manual - Section {i+1}]\n{chunk['text']}"
                for i, chunk in enumerate(retrieved_chunks)
            ])
            
            # Inject RAG context into system prompt
            rag_enhanced_message = f"""Relevant DBT Knowledge:
{context}

---

Based on the above DBT manual excerpts, respond to the user's message: {user_message}"""
            
            # Generate with RAG context
            response = self.therabot.chat(
                user_message=rag_enhanced_message,
                conversation_history=conversation_history,
                use_dbt_prompt=False,  # Don't use basic DBT prompt since we have RAG
                **kwargs
            )
        else:
            # Standard generation without RAG
            response = self.therabot.chat(
                user_message=user_message,
                conversation_history=conversation_history,
                use_dbt_prompt=True,
                **kwargs
            )
        
        return response, retrieved_chunks


# Alternative: Manually add text files if dataset load fails
def load_dbt_manual_from_files(file_paths: List[str], rag_system: DBT_RAGSystem):
    """
    Alternative method to load DBT manual from text files.
    Useful if dataset loading fails.
    
    Args:
        file_paths: List of paths to text files
        rag_system: Initialized RAG system
    """
    logger.info("📄 Loading DBT manual from text files...")
    
    all_texts = []
    all_embeddings = []
    all_metadata = []
    
    for file_idx, file_path in enumerate(file_paths):
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Chunk text
            chunks = rag_system._chunk_text(text, rag_system.chunk_size, rag_system.chunk_overlap)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_metadata.append({
                    "file": os.path.basename(file_path),
                    "file_idx": file_idx,
                    "chunk_idx": chunk_idx
                })
    
    # Create embeddings
    logger.info("🔢 Creating embeddings...")
    embeddings = rag_system.embedding_model.encode(all_texts, show_progress_bar=True)
    
    # Add to database
    ids = [f"file_{i}" for i in range(len(all_texts))]
    rag_system.collection.add(
        embeddings=embeddings.tolist(),
        documents=all_texts,
        metadatas=all_metadata,
        ids=ids
    )
    
    logger.info(f"✅ Loaded {len(all_texts)} chunks from {len(file_paths)} files")


# Example usage
def main():
    """Demo RAG system."""
    print("=" * 80)
    print("DBT RAG SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize RAG system
    print("\n🚀 Initializing RAG system...")
    rag_system = DBT_RAGSystem(
        base_model_name="meta-llama/Llama-3.1-8B-Instruct",
        adapter_path="./therapy-model-checkpoints",  # Your adapter path
        use_4bit=True
    )
    
    # Test queries
    test_queries = [
        "Can you teach me the TIPP skills?",
        "What are the PLEASE skills for emotion regulation?",
        "How do I practice mindfulness?",
    ]
    
    print("\n" + "=" * 80)
    print("TESTING RAG RETRIEVAL")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n🤔 Query: {query}")
        
        # Retrieve
        chunks = rag_system.retrieve(query, top_k=2)
        print(f"\n📚 Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\n   Chunk {i+1} (distance: {chunk['distance']:.3f}):")
            print(f"   {chunk['text'][:200]}...")
        
        print("-" * 80)
    
    # Test chat with RAG
    print("\n" + "=" * 80)
    print("TESTING CHAT WITH RAG")
    print("=" * 80)
    
    response, chunks = rag_system.chat(
        user_message="Can you teach me the TIPP skills?",
        use_rag=True
    )
    
    print(f"\n💬 Response: {response}")
    print(f"\n📚 Used {len(chunks)} chunks from DBT manual")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

