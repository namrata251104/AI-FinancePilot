import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import streamlit as st
from datetime import datetime
import uuid
import os

class VectorStore:
    """Manages vector storage and semantic search for financial transactions."""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self._initialize_chroma()
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Configure ChromaDB to use in-memory storage for simplicity
            self.client = chromadb.Client(Settings(
                is_persistent=False,  # Use in-memory storage
                anonymized_telemetry=False
            ))
            
            # Create or get collection
            collection_name = "financial_transactions"
            try:
                self.collection = self.client.get_collection(collection_name)
            except Exception:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Financial transaction embeddings"}
                )
            
            st.success("✅ Vector database initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ Error initializing vector database: {str(e)}")
            self.client = None
            self.collection = None
    
    def add_transactions(self, df: pd.DataFrame) -> bool:
        """Add transactions to the vector store."""
        if self.collection is None:
            st.error("Vector database not initialized")
            return False
        
        try:
            # Clear existing data
            self.collection.delete()
            
            # Prepare documents and metadata
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in df.iterrows():
                # Create a searchable text representation
                doc_text = self._create_document_text(row)
                documents.append(doc_text)
                
                # Prepare metadata
                metadata = {
                    "date": row['date'].strftime('%Y-%m-%d'),
                    "amount": float(row['amount']),
                    "category": str(row.get('category', 'Other')),
                    "description": str(row['description']),
                    "transaction_type": str(row.get('transaction_type', 'unknown')),
                    "month": int(row.get('month', 1)),
                    "year": int(row.get('year', 2023))
                }
                metadatas.append(metadata)
                
                # Generate unique ID
                ids.append(str(uuid.uuid4()))
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            
            st.success(f"✅ Added {len(documents)} transactions to vector database")
            return True
            
        except Exception as e:
            st.error(f"❌ Error adding transactions to vector store: {str(e)}")
            return False
    
    def _create_document_text(self, row: pd.Series) -> str:
        """Create a searchable text representation of a transaction."""
        date_str = row['date'].strftime('%Y-%m-%d %B')
        amount_str = f"${abs(row['amount']):.2f}"
        category = row.get('category', 'Other')
        description = str(row['description'])
        
        # Create comprehensive text for better semantic search
        doc_text = f"""
        Transaction on {date_str}
        Amount: {amount_str}
        Category: {category}
        Description: {description}
        Type: {'expense' if row['amount'] < 0 else 'income'}
        """
        
        return doc_text.strip()
    
    def search_transactions(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search for transactions using semantic similarity."""
        if self.collection is None:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count())
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents']) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'document': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else 0
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            st.error(f"❌ Error searching transactions: {str(e)}")
            return []
    
    def search_by_category(self, category: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search transactions by category."""
        if self.collection is None:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[f"category {category} expenses"],
                where={"category": category},
                n_results=n_results
            )
            
            return self._format_results(results)
            
        except Exception as e:
            st.error(f"❌ Error searching by category: {str(e)}")
            return []
    
    def search_by_date_range(self, start_date: str, end_date: str, n_results: int = 50) -> List[Dict[str, Any]]:
        """Search transactions within a date range."""
        if self.collection is None:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[f"transactions between {start_date} and {end_date}"],
                where={
                    "$and": [
                        {"date": {"$gte": start_date}},
                        {"date": {"$lte": end_date}}
                    ]
                },
                n_results=n_results
            )
            
            return self._format_results(results)
            
        except Exception as e:
            st.error(f"❌ Error searching by date range: {str(e)}")
            return []
    
    def search_by_amount_range(self, min_amount: float, max_amount: float, n_results: int = 20) -> List[Dict[str, Any]]:
        """Search transactions by amount range."""
        if self.collection is None:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[f"transactions between ${min_amount} and ${max_amount}"],
                where={
                    "$and": [
                        {"amount": {"$gte": min_amount}},
                        {"amount": {"$lte": max_amount}}
                    ]
                },
                n_results=n_results
            )
            
            return self._format_results(results)
            
        except Exception as e:
            st.error(f"❌ Error searching by amount range: {str(e)}")
            return []
    
    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format search results consistently."""
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                result = {
                    'document': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else 0
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.collection is None:
            return {}
        
        try:
            count = self.collection.count()
            return {
                'total_transactions': count,
                'collection_name': self.collection.name,
                'is_active': True
            }
        except Exception as e:
            return {
                'total_transactions': 0,
                'collection_name': 'N/A',
                'is_active': False,
                'error': str(e)
            }
