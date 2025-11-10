"""
Ingestion Agent - Handles product catalog data loading
"""

import pandas as pd
from typing import Dict, Any


class IngestionAgent:
    """
    Loads and preprocesses product catalog data for the classification pipeline.
    """
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
    
    def run(self) -> pd.DataFrame:
        """
        Load catalog data and return as DataFrame.
        
        Returns:
            pd.DataFrame: Product catalog data
        """
        try:
            df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(df)} products from catalog")
            return df
        except Exception as e:
            print(f"Error loading catalog: {e}")
            raise
    
    def get_schema(self) -> Dict[str, Any]:
        """Get catalog schema information."""
        df = pd.read_csv(self.csv_path)
        return {
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict()
        }