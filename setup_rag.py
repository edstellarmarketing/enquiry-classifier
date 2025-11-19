from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_knowledge_base():
    """Load classification rules and create vector database"""
    
    print("Loading classification rules...")
    # Load the rules document
    loader = TextLoader("classification_rules.txt")
    documents = loader.load()
    
    print("Splitting text into chunks...")
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    
    print("Creating embeddings and vector store...")
    # Create embeddings and store in ChromaDB
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print(f"✓ Knowledge base created successfully!")
    print(f"✓ Stored {len(splits)} document chunks")
    
    return vectorstore

if __name__ == "__main__":
    setup_knowledge_base()
