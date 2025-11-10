"""
Vector Store Utility for AI Product Classification MVP
-----------------------------------------------------
Manages FAISS-based semantic search for product similarity matching.
"""

import faiss
import numpy as np
import os
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from typing import List, Dict, Any
import json


class VectorStore:
    """
    FAISS-based vector store for product similarity search.
    Handles embedding generation, indexing, and semantic retrieval.
    """
    
    def __init__(self, csv_path: str, index_path: str):
        self.csv_path = csv_path
        self.index_path = index_path
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        self.df = pd.read_csv(csv_path)
        self.index = None
        
        # Load existing index or build new one
        if os.path.exists(index_path):
            print(f"Loading existing FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
        else:
            print(f"Building new FAISS index...")
            self._build_index()

    def _build_index(self):
        """Build FAISS index from product descriptions."""
        texts = self.df["description"].tolist()
        print(f"Generating embeddings for {len(texts)} products...")
        
        # Generate embeddings
        vectors = np.array(self.embedder.embed_documents(texts)).astype("float32")
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        
        # Save index
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index saved to {self.index_path}")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar products using semantic similarity.
        
        Args:
            query: Product description to search for
            k: Number of top matches to return
            
        Returns:
            List of similar product records with metadata
        """
        # Generate query embedding
        q_vec = np.array(self.embedder.embed_query(query)).astype("float32").reshape(1, -1)
        
        # Search FAISS index
        distances, indices = self.index.search(q_vec, k)
        
        # Get matching records with similarity scores
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            record = self.df.iloc[idx].to_dict()
            record['similarity_score'] = float(1.0 / (1.0 + distance))  # Convert L2 to similarity
            record['rank'] = i + 1
            results.append(record)
        
        return results

    def get_valid_codes(self) -> Dict[str, List[str]]:
        """Get all valid classification codes from the catalog."""
        return {
            "code_level_1": sorted(self.df["code_level_1"].unique().tolist()),
            "code_level_2": sorted(self.df["code_level_2"].unique().tolist()),
            "vendors": sorted(self.df["vendor"].unique().tolist()),
            "price_ranges": sorted(self.df["price_range"].unique().tolist())
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_products": len(self.df),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.index.d if self.index else 0,
            "categories": len(self.df["code_level_1"].unique()),
            "subcategories": len(self.df["code_level_2"].unique())
        }