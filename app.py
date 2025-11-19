import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import os

# Page configuration
st.set_page_config(
    page_title="Enquiry Classification System",
    page_icon="📊",
    layout="centered"
)

# Title and description
st.title("📊 Enquiry Classification System")
st.markdown("---")
st.markdown("""
This system classifies sales enquiries into three categories:
- **NO-DETAILED ENQUIRY**: Missing contact details and training information
- **SEMI-DETAILED ENQUIRY**: Has contact OR training information, but not both
- **DETAILED ENQUIRY**: Complete contact details and detailed training requirements
""")
st.markdown("---")

def format_docs(docs):
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def load_classifier():
    """Load the RAG system with classification rules"""
    
    # Get OpenAI API key from Streamlit secrets
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        st.error("OpenAI API key not found. Please add it to Streamlit secrets.")
        st.stop()
    
    # Load the existing vector database
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Create a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create the LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        temperature=0,
        openai_api_key=openai_api_key
    )
    
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
    try:
        rag_chain = load_classifier()
        result = rag_chain.invoke(enquiry_text)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Main app interface
st.subheader("Enter Enquiry Details")

# Text area for enquiry input
enquiry = st.text_area(
    "Paste your enquiry below:",
    height=200,
    placeholder="Example: Hi, I'm John Smith from ABC Company. Email: john@abc.com. Phone: 555-1234. I need Python training for 10 employees in January..."
)

# Classify button
if st.button("🔍 Classify Enquiry", type="primary"):
    if enquiry.strip():
        with st.spinner("Analyzing enquiry..."):
            classification = classify_enquiry(enquiry)
        
        st.markdown("---")
        st.subheader("Classification Result")
        st.write(classification)
        st.markdown("---")
    else:
        st.warning("Please enter an enquiry to classify.")

# Example enquiries in expander
with st.expander("📝 View Example Enquiries"):
    st.markdown("""
    **Example 1: NO-DETAILED ENQUIRY**
    ```
    I want training for my team
    ```
    
    **Example 2: SEMI-DETAILED ENQUIRY**
    ```
    Hi, I'm John Smith from ABC Company. 
    Email: john@abc.com. Phone: 555-1234. 
    I need training for my team.
    ```
    
    **Example 3: DETAILED ENQUIRY**
    ```
    Hello, I'm Sarah Johnson from XYZ Corporation. 
    Email: sarah@xyz.com, Phone: +1-555-0123. 
    We need Python programming training for 15 employees 
    scheduled for January 2024. We want to focus on data 
    analysis, automation skills, and working with pandas 
    and numpy libraries. Our team has basic programming knowledge.
    ```
    """)

# Footer
st.markdown("---")
st.markdown("*Powered by LangChain and OpenAI*")
