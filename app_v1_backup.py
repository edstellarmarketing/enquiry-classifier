import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import os
import json

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

def load_discovery_rules():
    """Load discovery question rules from JSON file"""
    try:
        with open('discovery_rules.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Discovery rules file not found!")
        return None

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

def extract_classification_type(result_text):
    """Extract the classification type from the result"""
    result_lower = result_text.lower()
    if "no-detailed enquiry" in result_lower:
        return "no_detailed"
    elif "semi-detailed enquiry" in result_lower:
        return "semi_detailed"
    elif "detailed enquiry" in result_lower:
        return "detailed"
    return "unknown"

def get_discovery_questions(classification_type, enquiry_text, discovery_rules):
    """Get relevant discovery questions based on classification and actual missing info"""
    if not discovery_rules:
        return []
    
    # Analyze what's actually present in the enquiry
    enquiry_lower = enquiry_text.lower()
    
    # Check for contact information presence
    has_name = any(word in enquiry_lower for word in ['my name is', 'i am', "i'm", 'this is'])
    has_email = '@' in enquiry_text
    has_phone = any(char.isdigit() for char in enquiry_text) and any(word in enquiry_lower for word in ['phone', 'tel', 'mobile', 'call'])
    has_company = any(word in enquiry_lower for word in ['company', 'organization', 'corp', 'from'])
    
    # Check for training information presence
    has_training_topic = any(word in enquiry_lower for word in ['training', 'course', 'python', 'java', 'management', 'leadership', 'sales'])
    has_participant_count = any(word in enquiry_lower for word in ['employees', 'participants', 'people', 'team members', 'staff']) and any(char.isdigit() for char in enquiry_text)
    has_dates = any(word in enquiry_lower for word in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'date', 'schedule', '2024', '2025'])
    has_objectives = any(word in enquiry_lower for word in ['goal', 'objective', 'focus', 'learn', 'improve', 'skills'])
    
    classification_config = discovery_rules.get("classification_rules", {}).get(classification_type, {})
    categories_to_show = classification_config.get("show_questions_from", [])
    max_questions = classification_config.get("max_questions", 5)
    
    all_questions = []
    
    for category in categories_to_show:
        category_data = discovery_rules.get("discovery_questions", {}).get(category, {})
        questions = category_data.get("questions", [])
        
        for q in questions:
            question_id = q.get("id")
            
            # Filter out questions for information that's already present
            if category == "contact_information":
                if question_id == "contact_name" and has_name:
                    continue
                if question_id == "contact_email" and has_email:
                    continue
                if question_id == "contact_phone" and has_phone:
                    continue
                if question_id == "contact_company" and has_company:
                    continue
            
            if category == "training_details":
                if question_id == "training_topic" and has_training_topic:
                    continue
                if question_id == "participant_count" and has_participant_count:
                    continue
                if question_id == "training_dates" and has_dates:
                    continue
                if question_id == "training_objectives" and has_objectives:
                    continue
            
            all_questions.append({
                "category": category_data.get("category", category),
                "question": q.get("question"),
                "priority": q.get("priority", 999)
            })
    
    # Sort by priority and limit
    all_questions.sort(key=lambda x: x["priority"])
    return all_questions[:max_questions]

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
            classification_type = extract_classification_type(classification)
        
        st.markdown("---")
        st.subheader("Classification Result")
        st.write(classification)
        st.markdown("---")
        
        # Load discovery rules and show relevant questions
        discovery_rules = load_discovery_rules()
        if discovery_rules:
            questions = get_discovery_questions(classification_type, enquiry, discovery_rules)
            
            if questions:
                # Get sales guidance
                classification_config = discovery_rules.get("classification_rules", {}).get(classification_type, {})
                sales_guidance = classification_config.get("sales_guidance", "")
                
                st.subheader("💡 Next Steps for Sales Team")
                
                if sales_guidance:
                    st.success(f"**Guidance:** {sales_guidance}")
                
                st.info("**Use these questions in your follow-up conversation with the client:**")
                
                # Group questions by category
                current_category = None
                for q in questions:
                    if q["category"] != current_category:
                        current_category = q["category"]
                        st.markdown(f"**{current_category}:**")
                    st.markdown(f"• {q['question']}")
                
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
