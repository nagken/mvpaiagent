"""
Classifier Agent - Uses LLM to predict product classification codes
"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import Dict, Any
import json


class ClassifierAgent:
    """
    Uses LLM to predict product classification codes based on 
    product description and similar product context.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.2)
        self.prompt_template = PromptTemplate.from_template("""
You are an AI product classification assistant for a catalog system.

Your task is to analyze a new product description and predict the correct classification codes 
based on similar products in the catalog.

SIMILAR PRODUCTS FOR REFERENCE:
{context}

NEW PRODUCT TO CLASSIFY:
Description: {description}

Based on the similar products above, predict the most appropriate classification codes.

IMPORTANT RULES:
1. code_level_1 must be a high-level category (LAPTOPS, MONITORS, ACCESSORIES, etc.)
2. code_level_2 must be a subcategory that makes sense with code_level_1
3. vendor should be extracted from the product description if mentioned
4. price_range should be estimated: LOW, MEDIUM, HIGH, VERY_HIGH
5. Provide a clear rationale for your classification

Return your response as valid JSON with this exact format:
{{
    "code_level_1": "CATEGORY",
    "code_level_2": "SUBCATEGORY", 
    "vendor": "VENDOR_NAME",
    "price_range": "ESTIMATED_RANGE",
    "confidence": 0.95,
    "rationale": "Explanation of classification reasoning"
}}
""")
    
    def run(self, description: str, context: str) -> str:
        """
        Classify a product description using LLM with similar product context.
        
        Args:
            description: New product description to classify
            context: Context from similar products
            
        Returns:
            JSON string with classification prediction
        """
        try:
            # Format the prompt
            prompt = self.prompt_template.format(
                description=description,
                context=context
            )
            
            print(f"Classifying product: '{description[:50]}...'")
            
            # Get LLM response
            response = self.llm.predict(prompt)
            
            print(f"Classification complete")
            
            return response.strip()
            
        except Exception as e:
            print(f"Error during classification: {e}")
            # Return error response in expected format
            error_response = {
                "code_level_1": "UNKNOWN",
                "code_level_2": "UNKNOWN",
                "vendor": "UNKNOWN",
                "price_range": "UNKNOWN",
                "confidence": 0.0,
                "rationale": f"Classification failed: {str(e)}"
            }
            return json.dumps(error_response)
    
    def validate_response_format(self, response: str) -> Dict[str, Any]:
        """
        Validate and parse the LLM response.
        
        Args:
            response: JSON string response from LLM
            
        Returns:
            Parsed and validated response dictionary
        """
        try:
            data = json.loads(response)
            
            # Ensure required fields exist
            required_fields = ["code_level_1", "code_level_2", "vendor", "price_range", "rationale"]
            for field in required_fields:
                if field not in data:
                    data[field] = "UNKNOWN"
            
            # Ensure confidence exists and is valid
            if "confidence" not in data or not isinstance(data["confidence"], (int, float)):
                data["confidence"] = 0.5
            
            return data
            
        except json.JSONDecodeError as e:
            print(f"Invalid JSON response from LLM: {e}")
            return {
                "code_level_1": "UNKNOWN",
                "code_level_2": "UNKNOWN", 
                "vendor": "UNKNOWN",
                "price_range": "UNKNOWN",
                "confidence": 0.0,
                "rationale": f"JSON parsing failed: {str(e)}"
            }