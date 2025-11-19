from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict
import json
import hashlib


class TrainingDomainAgent:
    """Uses LLM with web search to identify training domains"""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=openai_api_key
        )
        self.search_tool = DuckDuckGoSearchRun()
        self.cache = {}  # Simple in-memory cache
    
    def detect_domain(self, enquiry_text: str) -> Dict:
        """
        Detect training domain from enquiry text using web search
        
        Args:
            enquiry_text: The user's training enquiry
            
        Returns:
            Dict with domain, category, and sub_topics
        """
        # Create cache key
        cache_key = hashlib.md5(enquiry_text.lower().strip().encode()).hexdigest()
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # First, try to extract training topic keywords
            topic_keywords = self._extract_keywords(enquiry_text)
            
            # Search the web for information about the training topic
            search_query = f"{topic_keywords} training curriculum topics"
            search_results = self.search_tool.run(search_query)
            
            # Use LLM to analyze and structure the results
            domain_data = self._analyze_with_llm(enquiry_text, search_results)
            
            # Cache the result
            self.cache[cache_key] = domain_data
            
            return domain_data
            
        except Exception as e:
            print(f"Error: {e}")
            return self._get_fallback()
    
    def _extract_keywords(self, enquiry_text: str) -> str:
        """Extract training-related keywords from enquiry"""
        # Simple keyword extraction
        enquiry_lower = enquiry_text.lower()
        
        # Common patterns
        if 'training' in enquiry_lower:
            words = enquiry_lower.split('training')[0].split()
            if words:
                return words[-1]  # Word before "training"
        
        # Look for other indicators
        for indicator in ['course', 'program', 'learn', 'skill']:
            if indicator in enquiry_lower:
                words = enquiry_lower.split(indicator)[0].split()
                if words:
                    return words[-1]
        
        return enquiry_text[:50]  # First 50 chars as fallback
    
    def _analyze_with_llm(self, enquiry_text: str, search_results: str) -> Dict:
        """Use LLM to analyze enquiry and search results"""
        
        prompt = ChatPromptTemplate.from_template("""
You are a training domain expert. Analyze the enquiry and web search results to identify the training domain.

Enquiry: {enquiry}

Web Search Results:
{search_results}

Task:
1. Identify the primary training domain/topic
2. Categorize into one of these: Technical, Business, Digital Marketing, Data & Analytics, Cloud & Infrastructure, Soft Skills, Compliance & Security, or Other
3. List 3-5 specific sub-topics that would be covered

Return ONLY a valid JSON object (no extra text before or after):
{{
  "domain": "Specific training domain name",
  "category": "Category name",
  "sub_topics": ["topic1", "topic2", "topic3", "topic4", "topic5"]
}}
""")
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({
            "enquiry": enquiry_text,
            "search_results": search_results[:1000]  # Limit to first 1000 chars
        })
        
        return self._parse_output(result)
    
    def _parse_output(self, output_text: str) -> Dict:
        """Parse LLM output to extract structured data"""
        try:
            # Find JSON in the output
            start_idx = output_text.find('{')
            end_idx = output_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = output_text[start_idx:end_idx]
                data = json.loads(json_str)
                
                # Validate required fields
                if 'domain' in data and 'category' in data:
                    return {
                        'domain': data.get('domain', 'General Training'),
                        'category': data.get('category', 'Other'),
                        'sub_topics': data.get('sub_topics', [])[:5]  # Max 5
                    }
            
            return self._get_fallback()
                
        except Exception as e:
            print(f"Parsing error: {e}")
            return self._get_fallback()
    
    def _get_fallback(self) -> Dict:
        """Fallback response when detection fails"""
        return {
            "domain": "General Training",
            "category": "Other",
            "sub_topics": []
        }