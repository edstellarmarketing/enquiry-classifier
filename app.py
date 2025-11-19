import streamlit as st
import json
import os
import re
from training_domain_agent import TrainingDomainAgent
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import openai

# Page configuration
st.set_page_config(
    page_title="SalesIQ v3 - Rules-Based Enquiry System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .score-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .question-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    .question-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .category-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #2c3e50;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .celebration {
        animation: bounce 1s ease infinite;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_json_config(filename: str) -> Optional[Dict]:
    """Load JSON configuration file"""
    possible_paths = [
        filename,
        f'./{filename}',
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(__file__), filename)
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            st.sidebar.error(f"Error loading {filename}: {str(e)}")
            continue
    
    st.error(f"⚠️ Configuration file '{filename}' not found!")
    return None

@st.cache_data
def load_all_configs() -> Dict[str, Any]:
    """Load all configuration files"""
    configs = {
        'discovery_questions': load_json_config('discovery_questions.json'),
        'scoring_rules': load_json_config('scoring_rules.json'),
        'conditional_logic': load_json_config('conditional_logic_rules.json'),
        'proposal_unlock': load_json_config('proposal_unlock_requirements.json'),
        'ai_extraction': load_json_config('ai_extraction_prompts.json'),
        'validation_rules': load_json_config('validation_rules.json'),
        'notification_rules': load_json_config('notification_rules.json'),
        'training_domains': load_json_config('training_domain_logic.json')
    }
    
    return configs

# ============================================================================
# AI-POWERED EXTRACTION FUNCTIONS
# ============================================================================

def get_openai_client():
    """Get OpenAI client with API key"""
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OpenAI API key not found. Please add it to Streamlit secrets or environment variables.")
        st.stop()
    return api_key

def ai_extract_information(enquiry_text: str, configs: Dict) -> Dict:
    """Use AI to extract information from enquiry using configured prompts"""
    
    try:
        api_key = get_openai_client()
        
        # Get extraction configuration
        ai_config = configs.get('ai_extraction', {})
        system_prompt = ai_config.get('system_prompts', {}).get('primary_system_prompt', {}).get('content', '')
        extraction_schema = ai_config.get('extraction_schemas', {}).get('main_extraction_schema', {})
        
        # Build user prompt
        user_prompt = f"""Analyze this training enquiry and extract all available information.

Enquiry Text:
\"\"\"
{enquiry_text}
\"\"\"

Extract information into this JSON structure:
{json.dumps(extraction_schema, indent=2)}

Remember:
- Use null for missing information
- Extract exact names, emails, phone numbers
- Identify training domain from keywords
- Capture all pain points and challenges mentioned
- Note urgency indicators (ASAP, urgent, immediate)
- Return ONLY valid JSON, no other text"""

        # Call OpenAI API
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        # Parse response
        result = response.choices[0].message.content
        
        # Clean up response (remove markdown code blocks if present)
        result = result.strip()
        if result.startswith('```json'):
            result = result[7:]
        if result.startswith('```'):
            result = result[3:]
        if result.endswith('```'):
            result = result[:-3]
        result = result.strip()
        
        extracted_data = json.loads(result)
        
        return extracted_data
        
    except Exception as e:
        st.error(f"AI Extraction Error: {str(e)}")
        return {}

def ai_identify_training_domain(enquiry_text: str, extracted_data: Dict, configs: Dict) -> Dict:
    """Use LangChain Agent with web search to identify training domain"""
    
    try:
        api_key = get_openai_client()
        
        # Create agent
        agent = TrainingDomainAgent(api_key)
        
        # Detect domain using web search
        domain_result = agent.detect_domain(enquiry_text)
        
        # Format to match expected structure
        return {
            "primary_domain": {
                "domain_id": "AI_DETECTED",
                "domain_name": domain_result.get('domain', 'General Training'),
                "confidence_score": 0.9,  # Web-validated confidence
                "reasoning": f"Identified as {domain_result.get('category', 'Other')} category training",
                "matching_keywords": []
            },
            "secondary_domains": [],
            "category": domain_result.get('category', 'Other').lower().replace(' ', '_'),
            "requires_human_review": False,
            "review_reason": None,
            "sub_topics": domain_result.get('sub_topics', [])
        }
        
    except Exception as e:
        st.error(f"Domain Identification Error: {str(e)}")
        return {
            "primary_domain": {
                "domain_id": "ERROR",
                "domain_name": "General Training",
                "confidence_score": 0.0,
                "reasoning": f"Error in identification: {str(e)}",
                "matching_keywords": []
            },
            "secondary_domains": [],
            "category": "other",
            "requires_human_review": True,
            "review_reason": "Agent failed to identify domain"
        }

# ============================================================================
# SCORING AND COMPLETENESS CALCULATION
# ============================================================================

def calculate_completeness_score(extracted_data: Dict, configs: Dict) -> Dict:
    """Calculate completeness score based on extracted data"""
    
    scoring_config = configs.get('scoring_rules', {})
    categories = scoring_config.get('categories', [])
    
    total_points = 0
    earned_points = 0
    category_scores = {}
    filled_fields = []
    missing_critical_fields = []
    
    # Flatten extracted data for easier access
    flat_data = {}
    
    # Extract contact info
    contact = extracted_data.get('contact', {})
    if contact.get('name'):
        flat_data['contact_name'] = contact['name']
    if contact.get('designation'):
        flat_data['contact_designation'] = contact['designation']
    if contact.get('email'):
        flat_data['contact_email'] = contact['email']
    if contact.get('phone'):
        flat_data['contact_phone'] = contact['phone']
    
    # Extract company info
    company = extracted_data.get('company', {})
    if company.get('name'):
        flat_data['company_name'] = company['name']
    if company.get('industry'):
        flat_data['company_industry'] = company['industry']
    
    # Extract training info
    training = extracted_data.get('training', {})
    if training.get('domain'):
        flat_data['training_domain'] = training['domain']
    if training.get('num_participants'):
        flat_data['num_participants'] = training['num_participants']
    if training.get('skill_level'):
        flat_data['participant_skill_level'] = training['skill_level']
    if training.get('pain_points'):
        flat_data['business_pain_points'] = training['pain_points']
    
    # Extract logistics info
    logistics = extracted_data.get('logistics', {})
    if logistics.get('delivery_mode'):
        flat_data['delivery_mode'] = logistics['delivery_mode']
    if logistics.get('timeline'):
        flat_data['preferred_timeline'] = logistics['timeline']
    if logistics.get('location'):
        flat_data['training_location'] = logistics['location']
    
    # Extract commercial info
    commercial = extracted_data.get('commercial', {})
    if commercial.get('budget_indication'):
        flat_data['budget_range'] = commercial['budget_indication']
    if commercial.get('decision_maker'):
        flat_data['decision_maker_name'] = commercial['decision_maker']
    
    # Calculate score for each category
    for category in categories:
        cat_id = category.get('id')
        cat_name = category.get('name')
        cat_points = category.get('points_total', 0)
        cat_earned = 0
        
        fields = category.get('fields', [])
        for field in fields:
            field_name = field.get('field_name')
            field_points = field.get('points', 0)
            is_required = field.get('required', False)
            
            total_points += field_points
            
            # Check if field is filled
            if field_name in flat_data and flat_data[field_name]:
                earned_points += field_points
                cat_earned += field_points
                filled_fields.append(field_name)
            elif is_required:
                missing_critical_fields.append({
                    'field_name': field_name,
                    'display_name': field.get('field_name', '').replace('_', ' ').title(),
                    'points': field_points,
                    'category': cat_name
                })
        
        category_scores[cat_id] = {
            'name': cat_name,
            'earned': cat_earned,
            'total': cat_points,
            'percentage': (cat_earned / cat_points * 100) if cat_points > 0 else 0,
            'color': category.get('color', '#3B82F6'),
            'icon': category.get('icon', '📊')
        }
    
    # Calculate overall percentage
    overall_percentage = (earned_points / total_points * 100) if total_points > 0 else 0
    
    # Determine completeness level
    levels = scoring_config.get('completeness_levels', [])
    current_level = None
    for level in levels:
        min_pct = level.get('min_percentage', 0)
        max_pct = level.get('max_percentage', 100)
        if min_pct <= overall_percentage <= max_pct:
            current_level = level
            break
    
    return {
        'overall_percentage': round(overall_percentage, 1),
        'earned_points': earned_points,
        'total_points': total_points,
        'current_level': current_level,
        'category_scores': category_scores,
        'filled_fields': filled_fields,
        'missing_critical_fields': missing_critical_fields,
        'flat_data': flat_data
    }

# ============================================================================
# PROPOSAL UNLOCK VALIDATION
# ============================================================================

def check_proposal_unlock(score_data: Dict, configs: Dict) -> Dict:
    """Check if proposal can be unlocked based on rules"""
    
    unlock_config = configs.get('proposal_unlock', {})
    global_settings = unlock_config.get('global_settings', {})
    mandatory_fields_config = unlock_config.get('mandatory_fields', {})
    
    # Get threshold
    threshold = global_settings.get('minimum_completeness_percentage', 60)
    current_score = score_data.get('overall_percentage', 0)
    
    # Check completeness threshold
    threshold_met = current_score >= threshold
    
    # Check mandatory fields
    universal_requirements = mandatory_fields_config.get('universal_requirements', [])
    filled_fields = score_data.get('filled_fields', [])
    
    missing_mandatory = []
    for field in universal_requirements:
        field_name = field.get('field_name')
        if field_name not in filled_fields:
            missing_mandatory.append({
                'field_name': field_name,
                'display_name': field.get('display_name'),
                'reason': field.get('reason')
            })
    
    all_mandatory_filled = len(missing_mandatory) == 0
    
    # Determine unlock status
    can_unlock = threshold_met and all_mandatory_filled
    
    return {
        'can_unlock': can_unlock,
        'threshold_met': threshold_met,
        'threshold_required': threshold,
        'current_score': current_score,
        'all_mandatory_filled': all_mandatory_filled,
        'missing_mandatory': missing_mandatory,
        'unlock_message': global_settings.get('celebration_message', '🎉 Proposal Ready!') if can_unlock else f"Need {threshold - current_score:.1f}% more to unlock"
    }

# ============================================================================
# DISCOVERY QUESTIONS GENERATION
# ============================================================================

def get_discovery_questions(score_data: Dict, configs: Dict) -> List[Dict]:
    """Get prioritized discovery questions for missing information"""
    
    questions_config = configs.get('discovery_questions', {})
    all_questions = questions_config.get('questions', [])
    filled_fields = score_data.get('filled_fields', [])
    
    missing_questions = []
    
    for question in all_questions:
        field_mapping = question.get('data_field_mapping')
        
        # Check if this field is already filled
        if field_mapping and field_mapping not in filled_fields:
            # Check show condition
            show_condition = question.get('show_condition')
            if show_condition == 'always' or not show_condition:
                missing_questions.append({
                    'id': question.get('id'),
                    'question_text': question.get('question_text'),
                    'category': question.get('category'),
                    'priority': question.get('priority'),
                    'points': question.get('points_value', 0),
                    'help_text': question.get('help_text', ''),
                    'field_name': field_mapping,
                    'display_order': question.get('display_order', 999)
                })
    
    # Sort by priority and display order
    priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    missing_questions.sort(key=lambda q: (priority_order.get(q['priority'].lower(), 4), q['display_order']))
    
    return missing_questions

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Load all configurations
    with st.spinner("Loading configuration files..."):
        configs = load_all_configs()
    
    # Check if configs loaded successfully
    required_configs = ['discovery_questions', 'scoring_rules', 'ai_extraction']
    missing_configs = [cfg for cfg in required_configs if not configs.get(cfg)]
    
    if missing_configs:
        st.error(f"❌ Missing required configuration files: {', '.join(missing_configs)}")
        st.info("Please ensure all JSON configuration files are in the same directory as this app.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 SalesIQ v3.0")
        st.markdown("**AI-Powered Enquiry System**")
        st.markdown("---")
        
        st.markdown("#### System Status")
        st.success("✅ All configurations loaded")
        st.success("✅ LangChain Agent Active")
        st.metric("Discovery Questions", len(configs.get('discovery_questions', {}).get('questions', [])))
        
        st.markdown("---")
        st.markdown("#### Quick Stats")
        scoring_config = configs.get('scoring_rules', {})
        st.info(f"📊 Total Points: {scoring_config.get('total_points', 100)}")
        st.info(f"🎯 Unlock Threshold: {configs.get('proposal_unlock', {}).get('global_settings', {}).get('minimum_completeness_percentage', 60)}%")
        
        st.markdown("---")
        st.markdown("*Powered by AI + Web Search*")
    
    # Main header
    st.markdown('<h1 class="main-header">🎯 SalesIQ v3 - Intelligent Enquiry Analysis</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered | Web Search | Proposal-Ready**")
    st.markdown("---")
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Training Enquiry Input")
        enquiry_text = st.text_area(
            "Paste the training enquiry below:",
            height=300,
            placeholder="Example:\n\nHi, I'm Rajesh Kumar from Infosys Technologies.\nEmail: rajesh.kumar@infosys.com\nPhone: +91 98765 43210\n\nWe need SEO training for 30 people in our digital marketing team. They're struggling with organic traffic and Google rankings.\n\nOur budget is around ₹4-5 Lakhs."
        )
        
        # Source selection
        col_a, col_b = st.columns(2)
        with col_a:
            enquiry_source = st.selectbox("Enquiry Source", ["Email", "Website Form", "Phone Call", "Chat", "Other"])
        with col_b:
            enquiry_date = st.date_input("Enquiry Date", datetime.now())
    
    with col2:
        st.subheader("ℹ️ How It Works")
        st.info("""
        **1. AI Extraction**
        - Intelligent data extraction
        - Web-powered domain detection
        - Pain point analysis
        
        **2. Rules-Based Scoring**
        - 100-point scoring system
        - Category-wise completeness
        - Automatic validation
        
        **3. Smart Discovery**
        - Prioritized questions
        - Proposal unlock at 60%
        """)
        
        st.markdown("---")
        st.success("""
        **Classification Levels:**
        - 🔴 Incomplete (0-30%)
        - 🟡 Semi-Complete (31-60%)
        - 🟢 Complete (61-100%)
        """)
    
    # Analyze button
    st.markdown("---")
    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    with analyze_col2:
        analyze_button = st.button("🔍 Analyze Enquiry with AI", type="primary", use_container_width=True)
    
    if analyze_button:
        if not enquiry_text.strip():
            st.warning("⚠️ Please enter an enquiry to analyze.")
            return
        
        # Processing steps
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: AI Extraction
        status_text.text("🤖 Step 1/4: AI extracting information...")
        progress_bar.progress(25)
        extracted_data = ai_extract_information(enquiry_text, configs)
        
        # Step 2: Domain Identification (with web search)
        status_text.text("🌐 Step 2/4: Identifying domain (using web search)...")
        progress_bar.progress(50)
        domain_data = ai_identify_training_domain(enquiry_text, extracted_data, configs)
        
        # Step 3: Calculate Completeness
        status_text.text("📊 Step 3/4: Calculating completeness score...")
        progress_bar.progress(75)
        score_data = calculate_completeness_score(extracted_data, configs)
        
        # Step 4: Check Proposal Unlock
        status_text.text("🔓 Step 4/4: Checking proposal unlock status...")
        progress_bar.progress(100)
        unlock_data = check_proposal_unlock(score_data, configs)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Store in session state
        st.session_state.extracted_data = extracted_data
        st.session_state.domain_data = domain_data
        st.session_state.score_data = score_data
        st.session_state.unlock_data = unlock_data
        
        st.success("✅ Analysis complete!")
        st.markdown("---")
    
    # Display results if available
    if 'score_data' in st.session_state:
        score_data = st.session_state.score_data
        unlock_data = st.session_state.unlock_data
        domain_data = st.session_state.domain_data
        extracted_data = st.session_state.extracted_data
        
        # ========================================
        # SCORE DISPLAY
        # ========================================
        st.markdown("## 📊 Completeness Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        current_level = score_data.get('current_level', {})
        
        with col1:
            st.markdown(f"""
            <div class="score-box" style="background-color: {current_level.get('color', '#gray')}20; border: 3px solid {current_level.get('color', '#gray')};">
                {current_level.get('icon', '📊')}<br>{score_data['overall_percentage']}%
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<center><b>{current_level.get('label', 'Unknown')}</b></center>", unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Points Earned", f"{score_data['earned_points']} / {score_data['total_points']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Fields Filled", len(score_data['filled_fields']))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            if unlock_data['can_unlock']:
                st.markdown('<div class="metric-card celebration">', unsafe_allow_html=True)
                st.success("🎉 Proposal\nReady!")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.warning(f"⏳ Need\n{unlock_data['threshold_required'] - unlock_data['current_score']:.1f}% more")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ========================================
        # TRAINING DOMAIN DISPLAY (CLEAN - NO CONFIDENCE)
        # ========================================
        st.markdown("## 🎯 Training Domain Identified")
        
        primary_domain = domain_data.get('primary_domain', {})
        sub_topics = domain_data.get('sub_topics', [])
        category = domain_data.get('category', 'other').replace('_', ' ').title()
        
        domain_col1, domain_col2 = st.columns([2, 1])
        
        with domain_col1:
            st.markdown(f"### {primary_domain.get('domain_name', 'Unknown')}")
            st.write(f"**Category:** {category}")
            
            if sub_topics:
                with st.expander("📚 Sub-topics Covered"):
                    for topic in sub_topics[:5]:
                        st.markdown(f"• {topic}")
        
        with domain_col2:
            # Simple category badge
            st.info(f"**Training Type:**\n{category}")
        
        st.markdown("---")
        
        # ========================================
        # CATEGORY-WISE BREAKDOWN
        # ========================================
        st.markdown("## 📈 Category-Wise Breakdown")
        
        category_scores = score_data.get('category_scores', {})
        
        cols = st.columns(len(category_scores))
        for idx, (cat_id, cat_data) in enumerate(category_scores.items()):
            with cols[idx]:
                st.markdown(f"#### {cat_data['icon']} {cat_data['name']}")
                st.progress(cat_data['percentage'] / 100)
                st.caption(f"{cat_data['earned']}/{cat_data['total']} points • {cat_data['percentage']:.0f}%")
        
        st.markdown("---")
        
        # ========================================
        # EXTRACTED INFORMATION
        # ========================================
        with st.expander("📋 Extracted Information", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Contact Information")
                contact = extracted_data.get('contact', {})
                st.write(f"**Name:** {contact.get('name', 'Not found')}")
                st.write(f"**Designation:** {contact.get('designation', 'Not found')}")
                st.write(f"**Email:** {contact.get('email', 'Not found')}")
                st.write(f"**Phone:** {contact.get('phone', 'Not found')}")
                
                st.markdown("#### Company Information")
                company = extracted_data.get('company', {})
                st.write(f"**Name:** {company.get('name', 'Not found')}")
                st.write(f"**Industry:** {company.get('industry', 'Not found')}")
                st.write(f"**Size:** {company.get('size', 'Not found')}")
            
            with col2:
                st.markdown("#### Training Requirements")
                training = extracted_data.get('training', {})
                st.write(f"**Domain:** {training.get('domain', 'Not found')}")
                st.write(f"**Participants:** {training.get('num_participants', 'Not found')}")
                st.write(f"**Skill Level:** {training.get('skill_level', 'Not found')}")
                
                pain_points = training.get('pain_points', [])
                if pain_points:
                    st.markdown("**Pain Points:**")
                    for pp in pain_points:
                        st.caption(f"• {pp}")
                
                st.markdown("#### Logistics & Commercial")
                logistics = extracted_data.get('logistics', {})
                st.write(f"**Delivery Mode:** {logistics.get('delivery_mode', 'Not found')}")
                st.write(f"**Timeline:** {logistics.get('timeline', 'Not found')}")
                
                commercial = extracted_data.get('commercial', {})
                st.write(f"**Budget:** {commercial.get('budget_indication', 'Not found')}")
                st.write(f"**Decision Maker:** {commercial.get('decision_maker', 'Not found')}")
        
        st.markdown("---")
        
        # ========================================
        # DISCOVERY QUESTIONS
        # ========================================
        if not unlock_data['can_unlock']:
            st.markdown("## 💡 Discovery Questions for Sales Team")
            
            missing_questions = get_discovery_questions(score_data, configs)
            
            if missing_questions:
                st.info(f"📋 **{len(missing_questions)} questions** to complete the enquiry and unlock proposal builder")
                
                # Group by category
                questions_by_cat = {}
                for q in missing_questions:
                    cat = q['category']
                    if cat not in questions_by_cat:
                        questions_by_cat[cat] = []
                    questions_by_cat[cat].append(q)
                
                # Display by category
                for cat_name, questions in questions_by_cat.items():
                    with st.expander(f"**{cat_name}** ({len(questions)} questions)", expanded=True):
                        for idx, q in enumerate(questions, 1):
                            priority_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '⚪'}
                            emoji = priority_emoji.get(q['priority'].lower(), '⚪')
                            
                            st.markdown(f"""
                            <div class="question-card">
                                <p style="margin: 0; font-size: 13px; color: #666;">Question {idx} • {emoji} {q['priority'].upper()} • {q['points']} points</p>
                                <h4 style="margin: 10px 0; color: #1f77b4;">Ask: {q['question_text']}</h4>
                                <p style="margin: 0; font-size: 13px; color: #555; font-style: italic;">💡 {q['help_text']}</p>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.success("🎉 Enquiry is complete! Ready to generate proposal.")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("📄 Generate Proposal", type="primary", use_container_width=True):
                    st.balloons()
                    st.success("✅ Proposal generation feature coming soon!")
    
    # ========================================
    # EXAMPLES
    # ========================================
    st.markdown("---")
    with st.expander("📝 View Example Enquiries"):
        tab1, tab2, tab3 = st.tabs(["Incomplete", "Partial", "Complete"])
        
        with tab1:
            st.code("""I want SEO training for my team""", language="text")
        
        with tab2:
            st.code("""Hi, I'm Rajesh Kumar from Infosys.
Email: rajesh@infosys.com
Phone: +91 9876543210

We need SEO training for our digital marketing team.""", language="text")
        
        with tab3:
            st.code("""Hello, I'm Rajesh Kumar, Manager at Infosys Technologies.
Email: rajesh.kumar@infosys.com
Phone: +91 98765 43210

We need SEO training for 30 people in our digital marketing team. They're struggling with organic traffic and website ranking in Google search.

Participants are at beginner to intermediate level. We prefer hybrid delivery before Q2 2024.

Our budget is around ₹4-5 Lakhs. The decision maker is our VP Marketing, Venkat Raman.

Please send us a proposal.""", language="text")

if __name__ == "__main__":
    main()
