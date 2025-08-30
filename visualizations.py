import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import streamlit as st

class FinanceVisualizer:
    """Creates interactive visualizations for financial data."""
    
    def __init__(self):
        self.color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
            '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43'
        ]
    
    def create_monthly_trend(self, df: pd.DataFrame) -> go.Figure:
        """Create a monthly spending trend chart."""
        # Group by month and calculate spending
        monthly_data = df.groupby('month_year').agg({
            'amount': lambda x: x[x < 0].sum() * -1  # Convert negative to positive for expenses
        }).reset_index()
        
        monthly_data['month_year_str'] = monthly_data['month_year'].astype(str)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_data['month_year_str'],
            y=monthly_data['amount'],
            mode='lines+markers',
            name='Monthly Spending',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Monthly Spending Trend',
            xaxis_title='Month',
            yaxis_title='Amount ($)',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_category_pie_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create a pie chart for spending by category."""
        if 'category' not in df.columns:
            return go.Figure()
        
        # Calculate spending by category (only expenses)
        category_spending = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        
        fig = go.Figure(data=[go.Pie(
            labels=category_spending.index,
            values=category_spending.values,
            hole=0.3,
            marker_colors=self.color_palette[:len(category_spending)]
        )])
        
        fig.update_layout(
            title='Spending Distribution by Category',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_spending_vs_income_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create a chart comparing monthly spending vs income."""
        monthly_data = df.groupby('month_year').agg({
            'amount': [
                lambda x: x[x < 0].sum() * -1,  # Expenses (positive)
                lambda x: x[x > 0].sum()        # Income
            ]
        })
        
        monthly_data.columns = ['Expenses', 'Income']
        monthly_data = monthly_data.reset_index()
        monthly_data['month_year_str'] = monthly_data['month_year'].astype(str)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=monthly_data['month_year_str'],
            y=monthly_data['Income'],
            name='Income',
            marker_color='#4ECDC4'
        ))
        
        fig.add_trace(go.Bar(
            x=monthly_data['month_year_str'],
            y=monthly_data['Expenses'],
            name='Expenses',
            marker_color='#FF6B6B'
        ))
        
        fig.update_layout(
            title='Monthly Income vs Expenses',
            xaxis_title='Month',
            yaxis_title='Amount ($)',
            barmode='group',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_daily_spending_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create a heatmap showing spending patterns by day of week and hour."""
        # Add day of week and hour columns
        df_copy = df.copy()
        df_copy['day_of_week'] = df_copy['date'].dt.day_name()
        df_copy['day_num'] = df_copy['date'].dt.dayofweek
        
        # Calculate daily spending
        daily_spending = df_copy[df_copy['amount'] < 0].groupby(['day_num', 'day_of_week'])['amount'].sum().abs().reset_index()
        
        # Create pivot table for heatmap
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig = go.Figure(data=go.Bar(
            x=daily_spending['day_of_week'],
            y=daily_spending['amount'],
            marker_color=self.color_palette[0]
        ))
        
        fig.update_layout(
            title='Spending by Day of Week',
            xaxis_title='Day',
            yaxis_title='Total Spending ($)',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_transaction_timeline(self, df: pd.DataFrame, category: str = None) -> go.Figure:
        """Create a timeline of transactions."""
        df_filtered = df.copy()
        
        if category:
            df_filtered = df_filtered[df_filtered['category'] == category]
        
        # Sort by date
        df_filtered = df_filtered.sort_values('date')
        
        # Create different colors for income vs expenses
        colors = ['green' if x > 0 else 'red' for x in df_filtered['amount']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_filtered['date'],
            y=df_filtered['amount'],
            mode='markers',
            marker=dict(
                color=colors,
                size=abs(df_filtered['amount']) / df_filtered['amount'].abs().max() * 20 + 5,
                opacity=0.7
            ),
            text=df_filtered['description'],
            hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Amount: $%{y:.2f}<extra></extra>',
            name='Transactions'
        ))
        
        fig.update_layout(
            title=f'Transaction Timeline{" - " + category if category else ""}',
            xaxis_title='Date',
            yaxis_title='Amount ($)',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_spending_breakdown_sunburst(self, df: pd.DataFrame) -> go.Figure:
        """Create a sunburst chart for hierarchical spending breakdown."""
        if 'category' not in df.columns:
            return go.Figure()
        
        # Prepare data for sunburst
        spending_data = df[df['amount'] < 0].copy()
        spending_data['amount'] = spending_data['amount'].abs()
        
        # Create hierarchical structure
        sunburst_data = []
        
        # Add root
        total_spending = spending_data['amount'].sum()
        
        # Add categories
        for category in spending_data['category'].unique():
            category_amount = spending_data[spending_data['category'] == category]['amount'].sum()
            sunburst_data.append({
                'ids': category,
                'labels': category,
                'parents': '',
                'values': category_amount
            })
        
        # Convert to DataFrame for plotly
        sunburst_df = pd.DataFrame(sunburst_data)
        
        fig = go.Figure(go.Sunburst(
            ids=sunburst_df['ids'],
            labels=sunburst_df['labels'],
            parents=sunburst_df['parents'],
            values=sunburst_df['values'],
            branchvalues="total",
            marker=dict(colors=self.color_palette)
        ))
        
        fig.update_layout(
            title='Spending Breakdown',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_budget_vs_actual_chart(self, df: pd.DataFrame, budget_dict: Dict[str, float] = None) -> go.Figure:
        """Create a chart comparing budgeted vs actual spending by category."""
        if 'category' not in df.columns or not budget_dict:
            return go.Figure()
        
        # Calculate actual spending by category
        actual_spending = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        
        # Prepare data for comparison
        categories = list(set(actual_spending.index) | set(budget_dict.keys()))
        actual_values = [actual_spending.get(cat, 0) for cat in categories]
        budget_values = [budget_dict.get(cat, 0) for cat in categories]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=categories,
            y=budget_values,
            name='Budget',
            marker_color='#4ECDC4',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            x=categories,
            y=actual_values,
            name='Actual',
            marker_color='#FF6B6B'
        ))
        
        fig.update_layout(
            title='Budget vs Actual Spending',
            xaxis_title='Category',
            yaxis_title='Amount ($)',
            barmode='group',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def generate_insights(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Generate textual insights from the financial data."""
        insights = {
            'spending': [],
            'trends': [],
            'recommendations': []
        }
        
        if len(df) == 0:
            return insights
        
        # Spending insights
        if 'category' in df.columns:
            top_category = df[df['amount'] < 0].groupby('category')['amount'].sum().abs().idxmax()
            top_amount = df[df['amount'] < 0].groupby('category')['amount'].sum().abs().max()
            insights['spending'].append(f"Your highest spending category is {top_category} with ${top_amount:.2f}")
            
            # Average transaction size
            avg_expense = df[df['amount'] < 0]['amount'].abs().mean()
            insights['spending'].append(f"Your average expense transaction is ${avg_expense:.2f}")
        
        # Trend insights
        if 'month_year' in df.columns:
            monthly_spending = df[df['amount'] < 0].groupby('month_year')['amount'].sum().abs()
            if len(monthly_spending) >= 2:
                trend_change = monthly_spending.iloc[-1] - monthly_spending.iloc[-2]
                if trend_change > 0:
                    insights['trends'].append(f"Your spending increased by ${trend_change:.2f} last month")
                else:
                    insights['trends'].append(f"Your spending decreased by ${abs(trend_change):.2f} last month")
        
        # Day of week patterns
        if 'day_of_week' in df.columns:
            daily_spending = df[df['amount'] < 0].groupby('day_of_week')['amount'].sum().abs()
            highest_day = daily_spending.idxmax()
            insights['trends'].append(f"You tend to spend the most on {highest_day}s")
        
        return insights
