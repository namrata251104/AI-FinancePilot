import streamlit as st
import pandas as pd
import numpy as np
from typing import Any, Optional, List, Dict
import re
from datetime import datetime
import io

def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    if pd.isna(amount):
        return "$0.00"
    return f"${amount:,.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage string."""
    if pd.isna(value):
        return "0.0%"
    return f"{value:.1f}%"

def validate_file(uploaded_file) -> bool:
    """Validate uploaded file format and basic structure."""
    if uploaded_file is None:
        return False
    
    # Check file extension
    valid_extensions = ['.csv', '.xlsx', '.xls']
    if not any(uploaded_file.name.lower().endswith(ext) for ext in valid_extensions):
        st.error("Please upload a CSV or Excel file.")
        return False
    
    # Check file size (limit to 50MB)
    if uploaded_file.size > 50 * 1024 * 1024:
        st.error("File size too large. Please upload a file smaller than 50MB.")
        return False
    
    try:
        # Try to read the file to check if it's valid
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, nrows=5)  # Read only first 5 rows for validation
        else:
            df = pd.read_excel(uploaded_file, nrows=5)
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Check if file has minimum required columns
        if len(df.columns) < 3:
            st.error("File must have at least 3 columns (date, description, amount).")
            return False
        
        return True
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return False

def clean_amount_string(amount_str: str) -> float:
    """Clean and convert amount string to float."""
    if pd.isna(amount_str):
        return 0.0
    
    # Convert to string and remove common currency symbols and formatting
    cleaned = str(amount_str).strip()
    cleaned = re.sub(r'[^\d.-]', '', cleaned)
    
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def parse_date_string(date_str: str) -> Optional[datetime]:
    """Parse various date string formats."""
    if pd.isna(date_str):
        return None
    
    # Common date formats to try
    formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y/%m/%d',
        '%m-%d-%Y',
        '%d-%m-%Y',
        '%B %d, %Y',
        '%b %d, %Y',
        '%d %B %Y',
        '%d %b %Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    
    # If none of the formats work, try pandas to_datetime
    try:
        return pd.to_datetime(date_str)
    except:
        return None

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect likely column types in the dataframe."""
    column_types = {}
    
    for col in df.columns:
        col_lower = col.lower().strip()
        sample_values = df[col].dropna().head(10)
        
        # Date detection
        if any(keyword in col_lower for keyword in ['date', 'time', 'when']):
            column_types[col] = 'date'
        elif len(sample_values) > 0:
            # Try to parse a few values as dates
            date_count = 0
            for val in sample_values:
                if parse_date_string(str(val)) is not None:
                    date_count += 1
            if date_count >= len(sample_values) * 0.7:  # 70% are valid dates
                column_types[col] = 'date'
        
        # Amount detection
        if any(keyword in col_lower for keyword in ['amount', 'price', 'cost', 'value', 'debit', 'credit']):
            column_types[col] = 'amount'
        elif len(sample_values) > 0:
            # Check if values look like amounts
            numeric_count = 0
            for val in sample_values:
                try:
                    clean_amount_string(str(val))
                    numeric_count += 1
                except:
                    pass
            if numeric_count >= len(sample_values) * 0.8:  # 80% are numeric
                column_types[col] = 'amount'
        
        # Description detection
        if any(keyword in col_lower for keyword in ['description', 'desc', 'memo', 'detail', 'merchant']):
            column_types[col] = 'description'
        elif col not in column_types:
            # Check if it's mostly text
            if len(sample_values) > 0 and all(isinstance(val, str) or len(str(val)) > 5 for val in sample_values):
                column_types[col] = 'description'
    
    return column_types

def get_date_range_options() -> List[Dict[str, Any]]:
    """Get predefined date range options."""
    now = datetime.now()
    
    return [
        {
            'label': 'Last 7 days',
            'start': now - pd.Timedelta(days=7),
            'end': now
        },
        {
            'label': 'Last 30 days',
            'start': now - pd.Timedelta(days=30),
            'end': now
        },
        {
            'label': 'Last 3 months',
            'start': now - pd.Timedelta(days=90),
            'end': now
        },
        {
            'label': 'Last 6 months',
            'start': now - pd.Timedelta(days=180),
            'end': now
        },
        {
            'label': 'Last year',
            'start': now - pd.Timedelta(days=365),
            'end': now
        },
        {
            'label': 'Year to date',
            'start': datetime(now.year, 1, 1),
            'end': now
        }
    ]

def calculate_financial_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate various financial metrics from the transaction data."""
    metrics = {}
    
    if len(df) == 0:
        return metrics
    
    # Basic metrics
    metrics['total_transactions'] = len(df)
    metrics['total_income'] = df[df['amount'] > 0]['amount'].sum()
    metrics['total_expenses'] = abs(df[df['amount'] < 0]['amount'].sum())
    metrics['net_flow'] = metrics['total_income'] - metrics['total_expenses']
    
    # Average metrics
    metrics['avg_income_per_transaction'] = df[df['amount'] > 0]['amount'].mean() if len(df[df['amount'] > 0]) > 0 else 0
    metrics['avg_expense_per_transaction'] = abs(df[df['amount'] < 0]['amount'].mean()) if len(df[df['amount'] < 0]) > 0 else 0
    
    # Date range
    if 'date' in df.columns:
        metrics['date_range'] = {
            'start': df['date'].min(),
            'end': df['date'].max(),
            'days': (df['date'].max() - df['date'].min()).days
        }
        
        # Monthly averages
        if metrics['date_range']['days'] > 30:
            months = metrics['date_range']['days'] / 30.44  # Average days per month
            metrics['avg_monthly_income'] = metrics['total_income'] / months
            metrics['avg_monthly_expenses'] = metrics['total_expenses'] / months
    
    # Category metrics (if available)
    if 'category' in df.columns:
        expense_by_category = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        metrics['top_expense_category'] = expense_by_category.idxmax() if len(expense_by_category) > 0 else None
        metrics['top_expense_amount'] = expense_by_category.max() if len(expense_by_category) > 0 else 0
        metrics['category_count'] = df['category'].nunique()
    
    return metrics

def export_to_csv(df: pd.DataFrame, filename: str = None) -> str:
    """Export dataframe to CSV string."""
    if filename is None:
        filename = f"financial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return df.to_csv(index=False)

def create_sample_data_template() -> pd.DataFrame:
    """Create a sample data template for users to understand the expected format."""
    sample_data = {
        'date': [
            '2024-01-15',
            '2024-01-16',
            '2024-01-17',
            '2024-01-18',
            '2024-01-19'
        ],
        'description': [
            'Grocery Store Purchase',
            'Coffee Shop',
            'Gas Station',
            'Salary Deposit',
            'Restaurant Dinner'
        ],
        'amount': [
            -85.50,
            -4.75,
            -45.00,
            2500.00,
            -67.25
        ]
    }
    
    return pd.DataFrame(sample_data)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations."""
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    
    return sanitized

def format_large_number(number: float) -> str:
    """Format large numbers with appropriate suffixes."""
    if abs(number) >= 1_000_000:
        return f"${number/1_000_000:.1f}M"
    elif abs(number) >= 1_000:
        return f"${number/1_000:.1f}K"
    else:
        return f"${number:.2f}"
