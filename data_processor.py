import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np
from typing import Optional, Dict, Any
import io

class DataProcessor:
    """Handles loading and preprocessing of financial data files."""
    
    def __init__(self):
        self.required_columns = ['date', 'description', 'amount']
        self.column_mappings = {
            # Common variations of column names
            'date': ['date', 'transaction_date', 'posting_date', 'Date', 'Transaction Date'],
            'description': ['description', 'desc', 'memo', 'Description', 'Memo', 'Transaction Description'],
            'amount': ['amount', 'Amount', 'transaction_amount', 'debit', 'credit', 'Transaction Amount']
        }
    
    def load_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load and preprocess uploaded financial data file."""
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV or Excel files.")
                return None
            
            # Clean and standardize the dataframe
            df_cleaned = self._clean_dataframe(df)
            
            if df_cleaned is not None:
                st.success(f"Successfully loaded {len(df_cleaned)} transactions")
                return df_cleaned
            else:
                return None
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    
    def _clean_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Clean and standardize the dataframe structure."""
        try:
            # Map columns to standard names
            df_mapped = self._map_columns(df)
            
            if df_mapped is None:
                return None
            
            # Clean data types and values
            df_cleaned = self._clean_data_types(df_mapped)
            
            # Remove invalid transactions
            df_cleaned = self._remove_invalid_transactions(df_cleaned)
            
            # Add derived columns
            df_cleaned = self._add_derived_columns(df_cleaned)
            
            return df_cleaned
            
        except Exception as e:
            st.error(f"Error cleaning data: {str(e)}")
            return None
    
    def _map_columns(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Map column names to standard format."""
        df_copy = df.copy()
        column_mapping = {}
        
        # Find matching columns for each required field
        for required_col in self.required_columns:
            found_column = None
            
            for col in df_copy.columns:
                if col.lower().strip() in [mapping.lower() for mapping in self.column_mappings[required_col]]:
                    found_column = col
                    break
            
            if found_column:
                column_mapping[found_column] = required_col
            else:
                st.error(f"Required column '{required_col}' not found. Available columns: {list(df_copy.columns)}")
                return None
        
        # Rename columns
        df_copy = df_copy.rename(columns=column_mapping)
        
        # Keep only required columns and any additional useful columns
        columns_to_keep = self.required_columns.copy()
        
        # Add any remaining columns that might be useful
        for col in df_copy.columns:
            if col not in columns_to_keep and col.lower() in ['category', 'type', 'account', 'balance']:
                columns_to_keep.append(col)
        
        return df_copy[columns_to_keep]
    
    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert data types."""
        df_copy = df.copy()
        
        # Clean and convert date column
        try:
            df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
        except Exception as e:
            st.warning(f"Date conversion issues: {str(e)}")
            df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce', infer_datetime_format=True)
        
        # Clean and convert amount column
        if df_copy['amount'].dtype == 'object':
            # Remove currency symbols and convert to numeric
            df_copy['amount'] = df_copy['amount'].astype(str).str.replace(r'[^\d.-]', '', regex=True)
            df_copy['amount'] = pd.to_numeric(df_copy['amount'], errors='coerce')
        
        # Clean description column
        df_copy['description'] = df_copy['description'].astype(str).str.strip()
        
        return df_copy
    
    def _remove_invalid_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove transactions with invalid or missing data."""
        initial_count = len(df)
        
        # Remove rows with missing essential data
        df_clean = df.dropna(subset=['date', 'description', 'amount'])
        
        # Remove transactions with zero amounts (optional)
        df_clean = df_clean[df_clean['amount'] != 0]
        
        # Remove future dates
        df_clean = df_clean[df_clean['date'] <= datetime.now()]
        
        removed_count = initial_count - len(df_clean)
        if removed_count > 0:
            st.info(f"Removed {removed_count} invalid transactions")
        
        return df_clean
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful derived columns."""
        df_copy = df.copy()
        
        # Add transaction type (debit/credit)
        df_copy['transaction_type'] = df_copy['amount'].apply(
            lambda x: 'credit' if x > 0 else 'debit'
        )
        
        # Add absolute amount for easier analysis
        df_copy['abs_amount'] = df_copy['amount'].abs()
        
        # Add month and year columns
        df_copy['month'] = df_copy['date'].dt.month
        df_copy['year'] = df_copy['date'].dt.year
        df_copy['month_year'] = df_copy['date'].dt.to_period('M')
        
        # Add day of week
        df_copy['day_of_week'] = df_copy['date'].dt.day_name()
        
        return df_copy
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the dataset."""
        return {
            'total_transactions': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'total_debits': df[df['amount'] < 0]['amount'].sum(),
            'total_credits': df[df['amount'] > 0]['amount'].sum(),
            'net_amount': df['amount'].sum(),
            'avg_transaction': df['amount'].mean(),
            'unique_descriptions': df['description'].nunique()
        }
