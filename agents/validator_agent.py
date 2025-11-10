"""
Validator Agent - Validates classification predictions against catalog constraints
"""

import json
import pandas as pd
from typing import Dict, Any, List


class ValidatorAgent:
    """
    Validates classification predictions against known catalog constraints
    and business rules.
    """
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self._load_valid_values()
    
    def _load_valid_values(self):
        """Load valid values for each classification field."""
        self.valid_level_1 = set(self.df["code_level_1"].unique())
        self.valid_level_2 = set(self.df["code_level_2"].unique())
        self.valid_vendors = set(self.df["vendor"].unique())
        self.valid_price_ranges = {"LOW", "MEDIUM", "HIGH", "VERY_HIGH"}
        
        # Create level_1 -> level_2 mapping for validation
        self.level_mappings = {}
        for _, row in self.df.iterrows():
            l1 = row["code_level_1"]
            l2 = row["code_level_2"]
            if l1 not in self.level_mappings:
                self.level_mappings[l1] = set()
            self.level_mappings[l1].add(l2)
    
    def run(self, prediction_json: str) -> Dict[str, Any]:
        """
        Validate a classification prediction.
        
        Args:
            prediction_json: JSON string containing classification prediction
            
        Returns:
            Validation result with corrected values and validation status
        """
        try:
            data = json.loads(prediction_json)
            
            validation_result = {
                "original_prediction": data.copy(),
                "validated_prediction": {},
                "validation_errors": [],
                "is_valid": True,
                "confidence_adjusted": False
            }
            
            # Validate and correct each field
            validated_data = self._validate_fields(data, validation_result)
            validation_result["validated_prediction"] = validated_data
            
            # Check if any errors occurred
            validation_result["is_valid"] = len(validation_result["validation_errors"]) == 0
            
            print(f"Validation complete - Valid: {validation_result['is_valid']}")
            if validation_result["validation_errors"]:
                for error in validation_result["validation_errors"]:
                    print(f"Warning: {error}")
            
            return validation_result
            
        except json.JSONDecodeError as e:
            print(f"JSON validation error: {e}")
            return {
                "original_prediction": prediction_json,
                "validated_prediction": {
                    "code_level_1": "UNKNOWN",
                    "code_level_2": "UNKNOWN",
                    "vendor": "UNKNOWN", 
                    "price_range": "UNKNOWN",
                    "confidence": 0.0,
                    "rationale": f"JSON parsing failed: {str(e)}"
                },
                "validation_errors": [f"Invalid JSON format: {str(e)}"],
                "is_valid": False,
                "confidence_adjusted": True
            }
        except Exception as e:
            print(f"Validation error: {e}")
            return {
                "error": str(e),
                "is_valid": False
            }
    
    def _validate_fields(self, data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and correct individual fields."""
        validated = data.copy()
        
        # Validate code_level_1
        if data.get("code_level_1") not in self.valid_level_1:
            result["validation_errors"].append(f"Invalid code_level_1: {data.get('code_level_1')}")
            validated["code_level_1"] = self._suggest_level_1(data.get("code_level_1", ""))
        
        # Validate code_level_2 against level_1
        level_1 = validated["code_level_1"]
        level_2 = data.get("code_level_2")
        
        if level_1 in self.level_mappings:
            if level_2 not in self.level_mappings[level_1]:
                result["validation_errors"].append(f"Invalid code_level_2: {level_2} for category {level_1}")
                validated["code_level_2"] = list(self.level_mappings[level_1])[0]  # Use first valid option
        else:
            validated["code_level_2"] = "UNKNOWN"
        
        # Validate vendor (more lenient - allow new vendors)
        if data.get("vendor", "").strip() == "":
            validated["vendor"] = "UNKNOWN"
        
        # Validate price_range
        if data.get("price_range") not in self.valid_price_ranges:
            result["validation_errors"].append(f"Invalid price_range: {data.get('price_range')}")
            validated["price_range"] = "MEDIUM"  # Default fallback
        
        # Adjust confidence if validation failed
        if result["validation_errors"]:
            original_confidence = validated.get("confidence", 0.5)
            validated["confidence"] = max(0.1, original_confidence * 0.7)  # Reduce confidence
            result["confidence_adjusted"] = True
        
        return validated
    
    def _suggest_level_1(self, invalid_level: str) -> str:
        """Suggest a valid level_1 code based on similarity."""
        # Simple fallback logic - could be enhanced with similarity matching
        invalid_upper = invalid_level.upper()
        
        # Check for partial matches
        for valid_code in self.valid_level_1:
            if invalid_upper in valid_code or valid_code in invalid_upper:
                return valid_code
        
        # Default fallback
        return "ACCESSORIES"
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation constraints and statistics."""
        return {
            "valid_level_1_codes": sorted(list(self.valid_level_1)),
            "valid_level_2_codes": sorted(list(self.valid_level_2)), 
            "valid_vendors": sorted(list(self.valid_vendors)),
            "valid_price_ranges": sorted(list(self.valid_price_ranges)),
            "level_mappings": {k: sorted(list(v)) for k, v in self.level_mappings.items()}
        }