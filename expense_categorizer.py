import pandas as pd
import streamlit as st
import numpy as np
from typing import List, Dict, Any
import re
import os

class ExpenseCategorizer:
    """Automatically categorizes expenses using transformer models."""
    
    def __init__(self):
        self.categories = [
            "Food & Dining",
            "Transportation", 
            "Shopping",
            "Bills & Utilities",
            "Entertainment",
            "Health & Medical",
            "Travel",
            "Education",
            "Insurance",
            "Investment",
            "Income",
            "Transfer",
            "Other"
        ]
        
        # Initialize the classification pipeline
        self.classifier = None
        self._initialize_classifier()
        
        # Keyword mappings for fallback categorization
        self.keyword_mappings = {
            "Food & Dining": [
                "restaurant", "cafe", "food", "dining", "pizza", "burger", "starbucks",
                "mcdonald", "subway", "grocery", "market", "supermarket", "uber eats",
                "doordash", "grubhub", "delivery", "takeout"
            ],
            "Transportation": [
                "gas", "fuel", "uber", "lyft", "taxi", "bus", "train", "metro",
                "parking", "toll", "car", "vehicle", "auto", "mechanic", "tire"
            ],
            "Shopping": [
                "amazon", "walmart", "target", "mall", "store", "retail", "clothing",
                "shoes", "electronics", "books", "home depot", "lowes"
            ],
            "Bills & Utilities": [
                "electric", "electricity", "gas bill", "water", "internet", "phone",
                "cable", "subscription", "netflix", "spotify", "utility", "bill"
            ],
            "Entertainment": [
                "movie", "cinema", "theater", "concert", "game", "sport", "gym",
                "fitness", "netflix", "spotify", "entertainment", "club", "bar"
            ],
            "Health & Medical": [
                "doctor", "hospital", "pharmacy", "medical", "health", "dental",
                "vision", "clinic", "prescription", "medicine", "cvs", "walgreens"
            ],
            "Travel": [
                "airline", "flight", "hotel", "airbnb", "rental car", "vacation",
                "travel", "booking", "expedia", "trip", "resort"
            ],
            "Education": [
                "school", "university", "college", "tuition", "education", "book",
                "course", "training", "certification"
            ],
            "Insurance": [
                "insurance", "premium", "policy", "coverage", "deductible"
            ],
            "Investment": [
                "investment", "stock", "bond", "mutual fund", "401k", "ira",
                "retirement", "savings", "dividend", "brokerage"
            ],
            "Income": [
                "salary", "payroll", "wage", "bonus", "refund", "cashback",
                "deposit", "payment received", "income"
            ],
            "Transfer": [
                "transfer", "atm", "withdrawal", "deposit", "check", "wire",
                "payment to", "send money"
            ]
        }
    
    def _initialize_classifier(self):
        """Initialize the transformer-based classifier."""
        # For now, use keyword-based categorization only
        # AI model will be added when transformers dependency is resolved
        self.classifier = None
        st.info("🔍 Using keyword-based categorization (AI categorization coming soon!)")
    
    def categorize_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize all transactions in the dataframe."""
        df_copy = df.copy()
        
        # Add category column if it doesn't exist
        if 'category' not in df_copy.columns:
            df_copy['category'] = 'Other'
        
        total_transactions = len(df_copy)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (idx, row) in enumerate(df_copy.iterrows()):
            # Update progress
            progress = min(1.0, (i + 1) / total_transactions)
            progress_bar.progress(progress)
            status_text.text(f"Categorizing transaction {i + 1} of {total_transactions}")
            
            # Categorize the transaction
            category = self._categorize_single_transaction(row['description'], row.get('amount', 0))
            df_copy.at[idx, 'category'] = category
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return df_copy
    
    def _categorize_single_transaction(self, description: str, amount: float = 0) -> str:
        """Categorize a single transaction."""
        description = str(description).lower().strip()
        
        # First, try keyword-based categorization for speed
        keyword_category = self._categorize_by_keywords(description, amount)
        
        if keyword_category != "Other":
            return keyword_category
        
        # If keyword categorization fails and transformer is available, use AI
        if self.classifier is not None:
            try:
                result = self.classifier(description, self.categories)
                # Return the category with highest score if confidence is reasonable
                if result['scores'][0] > 0.3:  # Confidence threshold
                    return result['labels'][0]
            except Exception as e:
                # Fallback to keyword categorization if AI fails
                pass
        
        return "Other"
    
    def _categorize_by_keywords(self, description: str, amount: float = 0) -> str:
        """Categorize transaction based on keyword matching."""
        description_lower = description.lower()
        
        # Special handling for income (positive amounts with specific keywords)
        if amount > 0:
            income_keywords = ["salary", "payroll", "deposit", "refund", "cashback", "dividend"]
            if any(keyword in description_lower for keyword in income_keywords):
                return "Income"
        
        # Check each category's keywords
        for category, keywords in self.keyword_mappings.items():
            if any(keyword in description_lower for keyword in keywords):
                return category
        
        return "Other"
    
    def get_category_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get distribution of categories in the dataset."""
        if 'category' not in df.columns:
            return {}
        
        category_counts = df['category'].value_counts()
        category_amounts = df.groupby('category')['amount'].sum().abs()
        
        return {
            'counts': category_counts.to_dict(),
            'amounts': category_amounts.to_dict(),
            'percentages': (category_counts / len(df) * 100).to_dict()
        }
    
    def suggest_custom_categories(self, df: pd.DataFrame) -> List[str]:
        """Suggest custom categories based on frequent transaction patterns."""
        if 'description' not in df.columns:
            return []
        
        # Extract common words from descriptions
        all_descriptions = ' '.join(df['description'].astype(str).str.lower())
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_descriptions)
        
        # Count word frequency
        from collections import Counter
        word_counts = Counter(words)
        
        # Get most common words that aren't already in our categories
        common_words = [word for word, count in word_counts.most_common(20) 
                       if count > 5 and len(word) > 3]
        
        # Filter out common stop words and already categorized terms
        stop_words = {'the', 'and', 'for', 'with', 'from', 'payment', 'transaction'}
        suggestions = [word.title() for word in common_words 
                      if word not in stop_words]
        
        return suggestions[:10]  # Return top 10 suggestions
