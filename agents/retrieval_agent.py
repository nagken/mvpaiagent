"""
Retrieval Agent - Handles semantic search for similar products
"""

from utils.vector_store import VectorStore
from typing import List, Dict, Any


class RetrievalAgent:
    """
    Performs semantic retrieval of similar products using vector similarity search.
    """
    
    def __init__(self, store: VectorStore, top_k: int = 3):
        self.store = store
        self.top_k = top_k
    
    def run(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve semantically similar products for a given query.
        
        Args:
            query: Product description to search for
            
        Returns:
            List of similar product records with similarity scores
        """
        try:
            results = self.store.search(query, self.top_k)
            print(f"Retrieved {len(results)} similar products for: '{query[:50]}...'")
            
            # Log similarity scores
            for result in results:
                print(f"  {result['description'][:40]}... (similarity: {result['similarity_score']:.3f})")
            
            return results
        except Exception as e:
            print(f"Error during retrieval: {e}")
            raise
    
    def get_context_summary(self, results: List[Dict[str, Any]]) -> str:
        """
        Create a formatted context summary for the classifier.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Formatted context string
        """
        context_lines = []
        for result in results:
            context_lines.append(
                f"Product: {result['description']}\n"
                f"Category: {result['code_level_1']} -> {result['code_level_2']}\n"
                f"Vendor: {result['vendor']}, Price: {result['price_range']}\n"
                f"Similarity: {result['similarity_score']:.3f}\n"
            )
        
        return "Similar Products:\n" + "\n".join(context_lines)