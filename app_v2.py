import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import os
import json
import re
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="SalesIQ - Enquiry Classification System",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .category-header {
        font-weight: bold;
        color: #1f77b4;
        margin-top: 15px;
    }
    .score-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 SalesIQ - Intelligent Enquiry Classification")
st.markdown("---")

@st.cache_data
def load_salesiq_config():
    """Load the comprehensive SalesIQ configuration"""
    import os
    
    # Try multiple possible locations
    possible_paths = [
        'salesiq_config.json',
        './salesiq_config.json',
        os.path.join(os.getcwd(), 'salesiq_config.json'),
        os.path.join(os.path.dirname(__file__), 'salesiq_config.json')
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            continue
    
    st.error(f"SalesIQ configuration file not found! Current directory: {os.getcwd()}")
    st.error(f"Files in current directory: {os.listdir('.')}")
    return None

def format_docs(docs):
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def load_classifier():
    """Load the RAG system with classification rules"""
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        st.error("OpenAI API key not found. Please add it to Streamlit secrets.")
        st.stop()
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    
    template = """You are an expert sales enquiry classifier. Use the following classification rules to categorize the enquiry.

Classification Rules:
{context}

Enquiry to classify:
{question}

Analyze and provide:
1. Classification: NO-DETAILED ENQUIRY, SEMI-DETAILED ENQUIRY, or DETAILED ENQUIRY
2. Reason: Brief explanation

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def extract_information(enquiry_text, config):
    """Extract information from enquiry using AI hints"""
    extracted = {}
    enquiry_lower = enquiry_text.lower()
    
    # Extract based on patterns
    patterns = {
        'has_email': r'\b[\w.-]+@[\w.-]+\.\w{2,}\b',
        'has_phone': r'\b(?:\+91[\s-]?)?[6-9]\d{9}\b',
        'has_name': ['my name is', 'i am', "i'm", 'this is'],
        'has_company': ['company', 'organization', 'corp', 'from', 'ltd', 'pvt'],
        'has_participants': r'\b\d+\s*(?:people|participants|employees|staff|members)\b',
        'has_budget': ['budget', 'cost', 'price', 'lakh', 'crore', '₹'],
        'has_dates': ['january', 'february', 'march', 'april', 'may', 'june', 
                     'july', 'august', 'september', 'october', 'november', 'december',
                     'q1', 'q2', 'q3', 'q4', '2024', '2025']
    }
    
    # Check email
    email_match = re.search(patterns['has_email'], enquiry_text)
    extracted['contact_email'] = email_match.group(0) if email_match else None
    
    # Check phone
    phone_match = re.search(patterns['has_phone'], enquiry_text)
    extracted['contact_phone'] = phone_match.group(0) if phone_match else None
    
    # Check name
    extracted['contact_name'] = any(word in enquiry_lower for word in patterns['has_name'])
    
    # Check company
    extracted['company_name'] = any(word in enquiry_lower for word in patterns['has_company'])
    
    # Check participants
    participant_match = re.search(patterns['has_participants'], enquiry_lower)
    extracted['num_participants'] = participant_match.group(0) if participant_match else None
    
    # Check budget mention
    extracted['budget_range'] = any(word in enquiry_lower for word in patterns['has_budget'])
    
    # Check dates
    extracted['preferred_start_date'] = any(word in enquiry_lower for word in patterns['has_dates'])
    
    # Check training domain
    domains = config.get('trainingDomains', [])
    for domain in domains:
        keywords = domain.get('keywords', [])
        if any(keyword in enquiry_lower for keyword in keywords):
            extracted['training_domain'] = domain.get('name')
            break
    
    return extracted

def calculate_completeness_score(extracted_info, config):
    """Calculate completeness score based on extracted information"""
    questions = config.get('questions', [])
    total_possible_points = 0
    earned_points = 0
    
    # Map extracted info to questions
    for question in questions:
        data_field = question.get('dataField')
        points = question.get('points', 0)
        is_required = question.get('isRequired', False)
        
        total_possible_points += points
        
        # Check if this field is filled
        if data_field in extracted_info and extracted_info[data_field]:
            earned_points += points
    
    score_percentage = (earned_points / total_possible_points * 100) if total_possible_points > 0 else 0
    
    # Determine threshold category
    thresholds = config.get('scoringConfiguration', {}).get('thresholds', {})
    if score_percentage <= 30:
        category = thresholds.get('incomplete', {})
    elif score_percentage <= 60:
        category = thresholds.get('semiComplete', {})
    else:
        category = thresholds.get('complete', {})
    
    return {
        'score': round(score_percentage, 1),
        'earned_points': earned_points,
        'total_points': total_possible_points,
        'category': category
    }

def get_missing_questions(extracted_info, config):
    """Get questions for missing information, sorted by priority"""
    questions = config.get('questions', [])
    missing_questions = []
    
    priority_order = {'CRITICAL': 1, 'HIGH': 2, 'MEDIUM': 3, 'LOW': 4}
    
    for question in questions:
        data_field = question.get('dataField')
        
        # Check if this information is missing
        if data_field not in extracted_info or not extracted_info[data_field]:
            category_id = question.get('category')
            category_name = get_category_name(category_id, config)
            
            missing_questions.append({
                'question': question.get('questionText'),
                'category': category_name,
                'priority': question.get('priority', 'MEDIUM'),
                'priority_value': priority_order.get(question.get('priority', 'MEDIUM'), 5),
                'points': question.get('points', 0),
                'help_text': question.get('helpText', '')
            })
    
    # Sort by priority
    missing_questions.sort(key=lambda x: (x['priority_value'], -x['points']))
    
    return missing_questions

def get_category_name(category_id, config):
    """Get category name from category ID"""
    categories = config.get('scoringConfiguration', {}).get('categories', [])
    for cat in categories:
        if cat.get('id') == category_id:
            return cat.get('name')
    return "Other"

def classify_enquiry(enquiry_text):
    """Classify a single enquiry"""
    try:
        rag_chain = load_classifier()
        result = rag_chain.invoke(enquiry_text)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Main app interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Enquiry Details")
    enquiry = st.text_area(
        "Paste your enquiry below:",
        height=300,
        placeholder="Example: Hi, I'm John Smith from ABC Company. Email: john@abc.com. Phone: 555-1234. I need Python training for 10 employees in January..."
    )

with col2:
    st.subheader("ℹ️ Quick Guide")
    st.info("""
    **Classification Levels:**
    - 🔴 **Incomplete** (0-30%): Needs major follow-up
    - 🟡 **In Progress** (31-60%): Partial information
    - 🟢 **Complete** (61-100%): Ready for proposal
    
    The system will suggest specific questions to ask based on what's missing.
    """)

# Classify button
if st.button("🔍 Analyze Enquiry", type="primary", use_container_width=True):
    if enquiry.strip():
        config = load_salesiq_config()
        
        if config:
            with st.spinner("Analyzing enquiry..."):
                # Classification
                classification = classify_enquiry(enquiry)
                
                # Information extraction
                extracted_info = extract_information(enquiry, config)
                
                # Calculate score
                score_data = calculate_completeness_score(extracted_info, config)
                
                # Get missing questions
                missing_questions = get_missing_questions(extracted_info, config)
            
            st.markdown("---")
            
            # Display score
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="score-box" style="background-color: {score_data['category'].get('color', '#gray')}20; border: 2px solid {score_data['category'].get('color', '#gray')};">
                    {score_data['category'].get('icon', '')} {score_data['score']}%
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<center><b>{score_data['category'].get('label', 'Unknown')}</b></center>", unsafe_allow_html=True)
            
            with col2:
                st.metric("Points Earned", f"{score_data['earned_points']} / {score_data['total_points']}")
            
            with col3:
                proposal_threshold = config.get('scoringConfiguration', {}).get('proposalUnlockThreshold', 60)
                can_propose = score_data['score'] >= proposal_threshold
                if can_propose:
                    st.success(f"✅ Ready for Proposal\n({proposal_threshold}% threshold met)")
                else:
                    st.warning(f"⏳ Need {proposal_threshold - score_data['score']:.1f}% more")
            
            st.markdown("---")
            
            # Classification result
            with st.expander("🎯 Classification Details", expanded=False):
                st.write(classification)
            
            # Show missing questions grouped by category
            if missing_questions:
                st.subheader("💡 Discovery Questions for Sales Team")
                st.info(f"**Follow these questions in order to complete the enquiry:**")
                
                # Define category order
                category_order = [
                    "Contact Information",
                    "Training Requirements", 
                    "Delivery Preferences",
                    "Commercial Details"
                ]
                
                # Get category metadata from config
                categories_meta = {cat.get('name'): cat for cat in config.get('scoringConfiguration', {}).get('categories', [])}
                
                # Group questions by category
                questions_by_category = {}
                for q in missing_questions:
                    cat = q['category']
                    if cat not in questions_by_category:
                        questions_by_category[cat] = []
                    questions_by_category[cat].append(q)
                
                # Display in order
                question_count = 0
                for category_name in category_order:
                    if category_name in questions_by_category:
                        category_questions = questions_by_category[category_name]
                        
                        # Get category metadata
                        cat_meta = categories_meta.get(category_name, {})
                        cat_color = cat_meta.get('color', '#3B82F6')
                        cat_icon = cat_meta.get('icon', 'info')
                        cat_description = cat_meta.get('description', '')
                        cat_total_points = cat_meta.get('points', 0)
                        
                        # Calculate points missing in this category (only from questions shown)
                        missing_points = sum(q['points'] for q in category_questions)
                        
                        # Calculate what's already collected (estimated)
                        collected_points = max(0, cat_total_points - missing_points)
                        
                        # Create expandable section for each category
                        with st.expander(f"**{category_name}** - {missing_points} points needed ({len(category_questions)} questions)", expanded=True):
                            st.markdown(f"*{cat_description}*")
                            
                            # Show category progress
                            completion = (collected_points / cat_total_points * 100) if cat_total_points > 0 else 0
                            completion = max(0, min(100, completion))  # Clamp between 0-100
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.progress(completion / 100)
                            with col2:
                                st.metric("Complete", f"{completion:.0f}%")
                            
                            st.markdown("---")
                            
                            for q in category_questions:
                                question_count += 1
                                priority_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '⚪'}
                                emoji = priority_emoji.get(q['priority'], '⚪')
                                
                                # Create a nice box for each question
                                st.markdown(f"""
                                <div style="background-color: {cat_color}10; padding: 15px; border-radius: 8px; border-left: 4px solid {cat_color}; margin-bottom: 10px;">
                                    <p style="margin: 0; font-size: 14px; color: #666;">Question {question_count} • {emoji} {q['priority']} Priority • Worth {q['points']} points</p>
                                    <h4 style="margin: 10px 0; color: #333;">Ask the client: {q['question']}</h4>
                                    <p style="margin: 0; font-size: 13px; color: #666; font-style: italic;">💡 {q['help_text']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.info(f"📋 **Total: {question_count} questions to gather complete information**")
                
            else:
                st.success("🎉 All essential information collected! You can proceed with creating a proposal.")
    else:
        st.warning("Please enter an enquiry to analyze.")

# Example enquiries
with st.expander("📝 View Example Enquiries"):
    st.markdown("""
    **Example 1: Incomplete Enquiry**
    ```
    I want training for my team
    ```
    
    **Example 2: Partial Enquiry**
    ```
    Hi, I'm John Smith from ABC Company. 
    Email: john@abc.com. Phone: 555-1234. 
    I need training for my team.
    ```
    
    **Example 3: Complete Enquiry**
    ```
    Hello, I'm Sarah Johnson from XYZ Corporation. 
    Email: sarah@xyz.com, Phone: +1-555-0123. 
    We need Python programming training for 15 employees 
    scheduled for January 2024. We want to focus on data 
    analysis and automation skills with pandas and numpy. 
    Our team has basic programming knowledge.
    Our budget is around ₹3-5 lakhs. The decision maker 
    is our CTO, Venkat Raman.
    ```
    """)

# Footer
st.markdown("---")
st.markdown("*Powered by SalesIQ Configuration Engine • LangChain • OpenAI*")
