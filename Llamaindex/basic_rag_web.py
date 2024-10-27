from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

# Configure the embedding model and LLM
Settings.embed_model = JinaEmbedding(
    api_key=os.getenv("JINA_API_KEY"),
    model="jina-embeddings-v3",
    # choose `retrieval.passage` to get passage embeddings
    task="retrieval.passage",
)

Settings.llm = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="mixtral-8x7b-32768",
    temperature=0.7
)

def create_rag_system(data_dir="./data"):
    # Initialize Qdrant client (runs locally by default)
    client = QdrantClient(path="./qdrant_data")
    
    # Create Qdrant vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="my_documents",
        dimension=1024  # Must match embedding dimension
    )
    
    # Load documents from the specified directory
    documents = SimpleDirectoryReader(data_dir).load_data()
    
    # Create vector store index with Qdrant
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store
    )
    
    # Create query engine
    query_engine = index.as_query_engine()
    
    return query_engine

def query_rag(query_engine, question: str):
    # Query the system
    response = query_engine.query(question)
    return response

def main():
    # Initialize the RAG system
    query_engine = create_rag_system()
    
    # Example usage
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        response = query_rag(query_engine, question)
        print(f"\nAnswer: {response}")

if __name__ == "__main__":
    main()