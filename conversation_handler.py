import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime, timedelta
import re
from openai import OpenAI

class ConversationHandler:
    """Handles conversational AI interactions for financial queries."""
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "sk-default-key")
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-5"
        
        self.conversation_memory = []
        
    def get_response(self, user_query: str, vector_store, df: pd.DataFrame) -> str:
        """Generate a response to user's financial query."""
        try:
            # Search for relevant transactions
            relevant_transactions = vector_store.search_transactions(user_query, n_results=10)
            
            # Analyze the query to determine response type
            query_analysis = self._analyze_query(user_query, df)
            
            # Generate contextual response
            response = self._generate_response(user_query, relevant_transactions, query_analysis, df)
            
            # Store in conversation memory
            self.conversation_memory.append({
                "query": user_query,
                "response": response,
                "timestamp": datetime.now()
            })
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error while analyzing your financial data: {str(e)}"
    
    def _analyze_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the user query to understand intent and extract parameters."""
        query_lower = query.lower()
        
        analysis = {
            "intent": "general",
            "time_period": None,
            "category": None,
            "amount_range": None,
            "needs_calculation": False,
            "needs_visualization": False
        }
        
        # Detect intent
        if any(word in query_lower for word in ["how much", "total", "sum", "spent", "spend"]):
            analysis["intent"] = "spending_analysis"
            analysis["needs_calculation"] = True
            
        elif any(word in query_lower for word in ["compare", "comparison", "vs", "versus"]):
            analysis["intent"] = "comparison"
            analysis["needs_calculation"] = True
            
        elif any(word in query_lower for word in ["trend", "pattern", "over time", "monthly"]):
            analysis["intent"] = "trend_analysis"
            analysis["needs_visualization"] = True
            
        elif any(word in query_lower for word in ["category", "categories", "type", "types"]):
            analysis["intent"] = "category_analysis"
            
        elif any(word in query_lower for word in ["budget", "recommend", "advice", "suggest"]):
            analysis["intent"] = "recommendation"
        
        # Extract time periods
        time_patterns = {
            "last month": "last_month",
            "this month": "current_month",
            "last year": "last_year",
            "this year": "current_year",
            "last 6 months": "last_6_months",
            "last week": "last_week"
        }
        
        for pattern, period in time_patterns.items():
            if pattern in query_lower:
                analysis["time_period"] = period
                break
        
        # Extract categories
        categories = df['category'].unique() if 'category' in df.columns else []
        for category in categories:
            if category.lower() in query_lower:
                analysis["category"] = category
                break
        
        return analysis
    
    def _generate_response(self, query: str, relevant_transactions: List[Dict], 
                          analysis: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate AI response based on query analysis and relevant data."""
        
        # Prepare context for the AI
        context = self._prepare_context(relevant_transactions, analysis, df)
        
        # Create the prompt
        system_prompt = """You are a knowledgeable personal finance assistant. 
        Analyze the provided financial data and answer the user's question accurately.
        Provide specific numbers, dates, and actionable insights.
        Format your response in a clear, conversational manner.
        If calculations are needed, show the math.
        Always be helpful and encouraging about financial management."""
        
        user_prompt = f"""
        User Question: {query}
        
        Financial Data Context:
        {context}
        
        Please provide a comprehensive answer based on the data provided.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Fallback to rule-based response
            return self._generate_fallback_response(query, analysis, df)
    
    def _prepare_context(self, relevant_transactions: List[Dict], 
                        analysis: Dict[str, Any], df: pd.DataFrame) -> str:
        """Prepare context string for AI prompt."""
        context_parts = []
        
        # Add relevant transactions
        if relevant_transactions:
            context_parts.append("Relevant Transactions:")
            for i, tx in enumerate(relevant_transactions[:5]):  # Limit to top 5
                metadata = tx['metadata']
                context_parts.append(
                    f"- {metadata['date']}: {metadata['description']} "
                    f"${abs(metadata['amount']):.2f} ({metadata['category']})"
                )
        
        # Add summary statistics based on analysis
        if analysis["intent"] == "spending_analysis":
            context_parts.append(self._get_spending_summary(df, analysis))
        elif analysis["intent"] == "category_analysis":
            context_parts.append(self._get_category_summary(df))
        elif analysis["intent"] == "trend_analysis":
            context_parts.append(self._get_trend_summary(df))
        
        return "\n".join(context_parts)
    
    def _get_spending_summary(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """Generate spending summary based on time period and category."""
        summary_parts = ["Spending Summary:"]
        
        # Filter by time period if specified
        filtered_df = self._filter_by_time_period(df, analysis["time_period"])
        
        # Filter by category if specified
        if analysis["category"]:
            filtered_df = filtered_df[filtered_df['category'] == analysis["category"]]
        
        # Calculate totals
        total_spent = filtered_df[filtered_df['amount'] < 0]['amount'].sum()
        total_income = filtered_df[filtered_df['amount'] > 0]['amount'].sum()
        
        summary_parts.append(f"Total Expenses: ${abs(total_spent):.2f}")
        summary_parts.append(f"Total Income: ${total_income:.2f}")
        summary_parts.append(f"Net: ${total_income + total_spent:.2f}")
        summary_parts.append(f"Number of transactions: {len(filtered_df)}")
        
        return "\n".join(summary_parts)
    
    def _get_category_summary(self, df: pd.DataFrame) -> str:
        """Generate category breakdown summary."""
        if 'category' not in df.columns:
            return "Category information not available."
        
        category_spending = df[df['amount'] < 0].groupby('category')['amount'].sum().abs().sort_values(ascending=False)
        
        summary_parts = ["Category Breakdown:"]
        for category, amount in category_spending.head(5).items():
            summary_parts.append(f"- {category}: ${amount:.2f}")
        
        return "\n".join(summary_parts)
    
    def _get_trend_summary(self, df: pd.DataFrame) -> str:
        """Generate trend analysis summary."""
        monthly_spending = df[df['amount'] < 0].groupby('month_year')['amount'].sum().abs()
        
        summary_parts = ["Monthly Spending Trend:"]
        for month, amount in monthly_spending.tail(6).items():
            summary_parts.append(f"- {month}: ${amount:.2f}")
        
        # Add trend direction
        if len(monthly_spending) >= 2:
            recent_change = monthly_spending.iloc[-1] - monthly_spending.iloc[-2]
            trend = "increasing" if recent_change > 0 else "decreasing"
            summary_parts.append(f"Recent trend: {trend} by ${abs(recent_change):.2f}")
        
        return "\n".join(summary_parts)
    
    def _filter_by_time_period(self, df: pd.DataFrame, time_period: str) -> pd.DataFrame:
        """Filter dataframe by specified time period."""
        if not time_period:
            return df
        
        now = datetime.now()
        
        if time_period == "last_month":
            start_date = (now - timedelta(days=30)).replace(day=1)
            end_date = now.replace(day=1) - timedelta(days=1)
        elif time_period == "current_month":
            start_date = now.replace(day=1)
            end_date = now
        elif time_period == "last_year":
            start_date = datetime(now.year - 1, 1, 1)
            end_date = datetime(now.year - 1, 12, 31)
        elif time_period == "current_year":
            start_date = datetime(now.year, 1, 1)
            end_date = now
        elif time_period == "last_6_months":
            start_date = now - timedelta(days=180)
            end_date = now
        elif time_period == "last_week":
            start_date = now - timedelta(days=7)
            end_date = now
        else:
            return df
        
        return df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    def _generate_fallback_response(self, query: str, analysis: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate a basic response when AI is not available."""
        
        if analysis["intent"] == "spending_analysis":
            filtered_df = self._filter_by_time_period(df, analysis["time_period"])
            if analysis["category"]:
                filtered_df = filtered_df[filtered_df['category'] == analysis["category"]]
            
            total_spent = abs(filtered_df[filtered_df['amount'] < 0]['amount'].sum())
            period_text = analysis["time_period"].replace("_", " ") if analysis["time_period"] else "all time"
            category_text = f" on {analysis['category']}" if analysis["category"] else ""
            
            return f"You spent ${total_spent:.2f}{category_text} during {period_text}."
        
        elif analysis["intent"] == "category_analysis":
            if 'category' in df.columns:
                top_category = df[df['amount'] < 0].groupby('category')['amount'].sum().abs().idxmax()
                return f"Your highest spending category is {top_category}."
        
        return "I can help you analyze your financial data. Please try rephrasing your question or upload your transaction data first."
    
    def get_budget_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate budget recommendations based on spending patterns."""
        recommendations = []
        
        if 'category' not in df.columns:
            return ["Upload transaction data to get personalized budget recommendations."]
        
        # Analyze spending patterns
        monthly_spending = df[df['amount'] < 0].groupby('month_year')['amount'].sum().abs()
        category_spending = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        
        # High spending categories
        if len(category_spending) > 0:
            top_category = category_spending.idxmax()
            top_amount = category_spending.max()
            recommendations.append(
                f"💡 Your highest spending is on {top_category} (${top_amount:.2f}). "
                f"Consider setting a monthly budget for this category."
            )
        
        # Trend analysis
        if len(monthly_spending) >= 2:
            recent_change = monthly_spending.iloc[-1] - monthly_spending.iloc[-2]
            if recent_change > 0:
                recommendations.append(
                    f"⚠️ Your spending increased by ${recent_change:.2f} last month. "
                    f"Consider reviewing your recent purchases."
                )
            else:
                recommendations.append(
                    f"✅ Good job! You reduced spending by ${abs(recent_change):.2f} last month."
                )
        
        # Average recommendations
        avg_monthly = monthly_spending.mean() if len(monthly_spending) > 0 else 0
        recommendations.append(
            f"📊 Your average monthly spending is ${avg_monthly:.2f}. "
            f"Consider setting aside 20% of your income for savings."
        )
        
        return recommendations
