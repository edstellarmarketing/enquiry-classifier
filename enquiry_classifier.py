from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def format_docs(docs):
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)

def load_classifier():
    """Load the RAG system with classification rules"""
    
    # Load the existing vector database
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Create a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create the LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # Create a custom prompt template
    template = """You are an expert sales enquiry classifier. Use the following classification rules to categorize the enquiry.

Classification Rules:
{context}

Enquiry to classify:
{question}

Based on the rules above, analyze this enquiry and provide:
1. Classification: NO-DETAILED ENQUIRY, SEMI-DETAILED ENQUIRY, or DETAILED ENQUIRY
2. Reason: Brief explanation of why it falls into this category
3. Missing Information: What information is missing (if any)

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def classify_enquiry(enquiry_text):
    """Classify a single enquiry"""
    
    rag_chain = load_classifier()
    result = rag_chain.invoke(enquiry_text)
    return result

def main():
    """Main application loop"""
    
    print("=" * 60)
    print("ENQUIRY CLASSIFICATION SYSTEM")
    print("=" * 60)
    print("\nThis system classifies enquiries into three categories:")
    print("  1. NO-DETAILED ENQUIRY")
    print("  2. SEMI-DETAILED ENQUIRY")
    print("  3. DETAILED ENQUIRY")
    print("\n" + "=" * 60)
    
    while True:
        print("\n\nPaste your enquiry below (or type 'quit' to exit):")
        print("-" * 60)
        
        # Get multiline input
        lines = []
        while True:
            line = input()
            if line.lower() == 'quit':
                print("\nThank you for using the Enquiry Classification System!")
                return
            if line == "":
                break
            lines.append(line)
        
        enquiry = "\n".join(lines)
        
        if not enquiry.strip():
            continue
            
        print("\n" + "=" * 60)
        print("ANALYZING ENQUIRY...")
        print("=" * 60)
        
        # Classify the enquiry
        classification = classify_enquiry(enquiry)
        
        print("\n" + classification)
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
