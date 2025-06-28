import os
import logging
from typing import List, Optional
from pathlib import Path

# Core LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext,
    Settings,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.vllm import Vllm
from llama_index.vector_stores.milvus import MilvusVectorStore

# Milvus imports
from pymilvus import connections, utility

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    A complete RAG system using LlamaIndex and Milvus
    """
    
    def __init__(
        self,
        collection_name: str = "rag_collection",
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        llm_model: str = "microsoft/DialoGPT-medium",
        vllm_server_url: str = "http://localhost:8000",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_new_tokens: int = 512,
        temperature: float = 0.1
    ):
        """
        Initialize the RAG system
        
        Args:
            collection_name: Name for the Milvus collection
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            embedding_model: HuggingFace embedding model name
            llm_model: HuggingFace model name for vLLM
            vllm_server_url: vLLM server URL
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.collection_name = collection_name
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vllm_server_url = vllm_server_url
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Initialize components
        self._setup_embeddings(embedding_model)
        self._setup_llm(llm_model)
        self._setup_milvus()
        self._setup_node_parser()
        
        self.index = None
        self.query_engine = None
        
    def _setup_embeddings(self, model_name: str):
        """Setup embedding model"""
        logger.info(f"Loading embedding model: {model_name}")
        self.embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            cache_folder="./embeddings_cache"
        )
        Settings.embed_model = self.embed_model
        
    def _setup_llm(self, model_name: str):
        """Setup vLLM"""
        logger.info(f"Setting up vLLM with model: {model_name}")
        self.llm = Vllm(
            model=model_name,
            api_url=self.vllm_server_url,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            dtype="auto",
            trust_remote_code=True
        )
        Settings.llm = self.llm
        
    def _setup_milvus(self):
        """Setup Milvus connection and vector store"""
        logger.info(f"Connecting to Milvus at {self.milvus_host}:{self.milvus_port}")
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=self.milvus_host,
            port=self.milvus_port
        )
        
        # Initialize vector store
        self.vector_store = MilvusVectorStore(
            collection_name=self.collection_name,
            dim=384,  # Dimension for BAAI/bge-small-en-v1.5
            overwrite=False
        )
        
    def _setup_node_parser(self):
        """Setup document node parser"""
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        Settings.node_parser = self.node_parser
        
    def load_documents(self, data_path: str) -> List[Document]:
        """
        Load documents from a directory
        
        Args:
            data_path: Path to directory containing documents
            
        Returns:
            List of loaded documents
        """
        logger.info(f"Loading documents from: {data_path}")
        
        if not os.path.exists(data_path):
            raise ValueError(f"Path does not exist: {data_path}")
            
        reader = SimpleDirectoryReader(
            input_dir=data_path,
            recursive=True,
            exclude_hidden=True
        )
        
        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} documents")
        return documents
        
    def create_index(self, documents: List[Document]):
        """
        Create vector index from documents
        
        Args:
            documents: List of documents to index
        """
        logger.info("Creating vector index...")
        
        # Create storage context with Milvus
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Create index
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        logger.info("Vector index created successfully")
        
    def load_existing_index(self):
        """Load existing index from Milvus"""
        logger.info("Loading existing index from Milvus...")
        
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=storage_context
        )
        
        logger.info("Existing index loaded successfully")
        
    def setup_query_engine(self, similarity_top_k: int = 3):
        """
        Setup query engine for RAG
        
        Args:
            similarity_top_k: Number of similar documents to retrieve
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
            
        logger.info("Setting up query engine...")
        
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode="compact"
        )
        
        logger.info("Query engine setup complete")
        
    def query(self, question: str) -> str:
        """
        Query the RAG system
        
        Args:
            question: Question to ask
            
        Returns:
            Generated response
        """
        if self.query_engine is None:
            raise ValueError("Query engine not setup. Call setup_query_engine() first.")
            
        logger.info(f"Processing query: {question}")
        
        response = self.query_engine.query(question)
        return str(response)
        
    def add_documents(self, documents: List[Document]):
        """
        Add new documents to existing index
        
        Args:
            documents: New documents to add
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
            
        logger.info(f"Adding {len(documents)} new documents to index...")
        
        for doc in documents:
            self.index.insert(doc)
            
        logger.info("Documents added successfully")
        
    def get_collection_stats(self):
        """Get statistics about the Milvus collection"""
        if utility.has_collection(self.collection_name):
            from pymilvus import Collection
            collection = Collection(self.collection_name)
            collection.load()
            return {
                "num_entities": collection.num_entities,
                "collection_name": self.collection_name
            }
        return {"message": "Collection does not exist"}


def main():
    """
    Example usage of the RAG system
    """
    # Initialize RAG system
    rag = RAGSystem(
        collection_name="my_rag_collection",
        milvus_host="localhost",
        milvus_port=19530,
        embedding_model="BAAI/bge-small-en-v1.5",
        llm_model="meta-llama/Llama-2-7b-chat-hf",  # or any HuggingFace model
        vllm_server_url="http://localhost:8000",
        max_new_tokens=512,
        temperature=0.1
    )
    
    # Path to your documents
    data_path = "./data"  # Replace with your document directory
    
    try:
        # Check if collection exists
        if utility.has_collection(rag.collection_name):
            print("Loading existing index...")
            rag.load_existing_index()
        else:
            print("Creating new index...")
            # Load documents
            documents = rag.load_documents(data_path)
            
            # Create index
            rag.create_index(documents)
        
        # Setup query engine
        rag.setup_query_engine(similarity_top_k=3)
        
        # Get collection stats
        stats = rag.get_collection_stats()
        print(f"Collection stats: {stats}")
        
        # Example queries
        questions = [
            "What is the main topic discussed in the documents?",
            "Can you summarize the key points?",
            "What are the important concepts mentioned?"
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            response = rag.query(question)
            print(f"Answer: {response}")
            print("-" * 80)
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()


# Additional utility functions

def setup_vllm_server():
    """
    Instructions for setting up vLLM server
    """
    setup_commands = """
    # Setup vLLM Server
    
    1. Install vLLM:
    pip install vllm
    
    2. Start vLLM server with your model:
    # For Llama 2 7B Chat
    python -m vllm.entrypoints.openai.api_server \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --host 0.0.0.0 \\
        --port 8000 \\
        --served-model-name llama2-chat
    
    # For Mistral 7B Instruct
    python -m vllm.entrypoints.openai.api_server \\
        --model mistralai/Mistral-7B-Instruct-v0.1 \\
        --host 0.0.0.0 \\
        --port 8000
    
    # For CodeLlama
    python -m vllm.entrypoints.openai.api_server \\
        --model codellama/CodeLlama-7b-Instruct-hf \\
        --host 0.0.0.0 \\
        --port 8000
        
    # With GPU memory optimization
    python -m vllm.entrypoints.openai.api_server \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --host 0.0.0.0 \\
        --port 8000 \\
        --gpu-memory-utilization 0.8 \\
        --max-model-len 4096
    
    3. Verify server is running:
    curl http://localhost:8000/v1/models
    
    4. Alternative: Use vLLM with custom configuration
    from vllm import LLM, SamplingParams
    
    llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    """
    print(setup_commands)
    """
    Instructions for setting up Milvus standalone using Docker
    """
def setup_milvus_standalone():
    """
    Instructions for setting up Milvus standalone using Docker
    """
    setup_commands = """
    # Setup Milvus using Docker Compose
    
    1. Create docker-compose.yml:
    
    version: '3.5'
    services:
      etcd:
        container_name: milvus-etcd
        image: quay.io/coreos/etcd:v3.5.5
        environment:
          - ETCD_AUTO_COMPACTION_MODE=revision
          - ETCD_AUTO_COMPACTION_RETENTION=1000
          - ETCD_QUOTA_BACKEND_BYTES=4294967296
          - ETCD_SNAPSHOT_COUNT=50000
        volumes:
          - etcd:/etcd
        command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
        healthcheck:
          test: ["CMD", "etcdctl", "endpoint", "health"]
          interval: 30s
          timeout: 20s
          retries: 3

      minio:
        container_name: milvus-minio
        image: minio/minio:RELEASE.2023-03-20T20-16-18Z
        environment:
          MINIO_ACCESS_KEY: minioadmin
          MINIO_SECRET_KEY: minioadmin
        ports:
          - "9001:9001"
          - "9000:9000"
        volumes:
          - minio:/minio_data
        command: minio server /minio_data --console-address ":9001"
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
          interval: 30s
          timeout: 20s
          retries: 3

      standalone:
        container_name: milvus-standalone
        image: milvusdb/milvus:v2.3.0
        command: ["milvus", "run", "standalone"]
        environment:
          ETCD_ENDPOINTS: etcd:2379
          MINIO_ADDRESS: minio:9000
        volumes:
          - milvus:/var/lib/milvus
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
          interval: 30s
          start_period: 90s
          timeout: 20s
          retries: 3
        ports:
          - "19530:19530"
          - "9091:9091"
        depends_on:
          - "etcd"
          - "minio"

    volumes:
      etcd:
        driver: local
      minio:
        driver: local
      milvus:
        driver: local
    
    2. Run: docker-compose up -d
    3. Install Python dependencies: pip install pymilvus llama-index-vector-stores-milvus
    """
    print(setup_commands)


def requirements_txt():
    """
    Generate requirements.txt for the project
    """
    requirements = """
llama-index
llama-index-vector-stores-milvus
llama-index-embeddings-huggingface
llama-index-llms-vllm
pymilvus
vllm
torch
transformers
sentence-transformers
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())
    print("requirements.txt created")


if __name__ == "__main__":
    # Uncomment to generate requirements.txt
    # requirements_txt()
    
    # Uncomment to see vLLM setup instructions
    # setup_vllm_server()
    
    # Uncomment to see Milvus setup instructions
    # setup_milvus_standalone()
    
    # Run main RAG system
    main()