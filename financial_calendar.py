import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import calendar

class FinancialCalendar:
    """Interactive financial calendar with spending patterns and insights."""
    
    def __init__(self):
        self.color_scheme = {
            'income': '#4CAF50',
            'expense': '#F44336',
            'recurring': '#FF9800',
            'unusual': '#9C27B0'
        }
    
    def create_calendar_view(self, df: pd.DataFrame, year: int = None, month: int = None) -> Dict[str, Any]:
        """Create calendar view of financial transactions."""
        
        if year is None:
            year = datetime.now().year
        if month is None:
            month = datetime.now().month
        
        # Filter data for the specified month/year
        calendar_data = df[
            (df['date'].dt.year == year) & 
            (df['date'].dt.month == month)
        ].copy()
        
        # Group transactions by day
        daily_summary = self._create_daily_summary(calendar_data)
        
        # Identify patterns
        patterns = self._identify_spending_patterns(calendar_data)
        
        # Create calendar heatmap data
        calendar_matrix = self._create_calendar_matrix(daily_summary, year, month)
        
        return {
            'calendar_matrix': calendar_matrix,
            'daily_summary': daily_summary,
            'patterns': patterns,
            'month_stats': self._calculate_month_stats(calendar_data),
            'year': year,
            'month': month
        }
    
    def _create_daily_summary(self, df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Create summary for each day of the month."""
        daily_summary = {}
        
        for day in range(1, 32):  # Maximum days in a month
            day_data = df[df['date'].dt.day == day]
            
            if len(day_data) > 0:
                total_income = day_data[day_data['amount'] > 0]['amount'].sum()
                total_expense = abs(day_data[day_data['amount'] < 0]['amount'].sum())
                net_flow = total_income - total_expense
                
                # Categorize transactions
                categories = {}
                if 'category' in day_data.columns:
                    categories = day_data[day_data['amount'] < 0].groupby('category')['amount'].sum().abs().to_dict()
                
                daily_summary[day] = {
                    'total_income': total_income,
                    'total_expense': total_expense,
                    'net_flow': net_flow,
                    'transaction_count': len(day_data),
                    'categories': categories,
                    'transactions': day_data[['description', 'amount', 'category']].to_dict('records') if 'category' in day_data.columns else day_data[['description', 'amount']].to_dict('records'),
                    'day_type': self._classify_day_type(day_data, total_expense)
                }
        
        return daily_summary
    
    def _classify_day_type(self, day_data: pd.DataFrame, total_expense: float) -> str:
        """Classify the type of spending day."""
        
        if len(day_data) == 0:
            return 'inactive'
        
        # Check for income
        if any(day_data['amount'] > 0):
            return 'income_day'
        
        # Check for high spending
        if total_expense > 200:  # Threshold for high spending
            return 'high_spending'
        
        # Check for recurring bills
        recurring_keywords = ['bill', 'payment', 'subscription', 'rent', 'mortgage', 'insurance']
        if any(keyword in desc.lower() for desc in day_data['description'].astype(str) for keyword in recurring_keywords):
            return 'bills_day'
        
        # Regular spending
        if total_expense > 0:
            return 'regular_spending'
        
        return 'inactive'
    
    def _identify_spending_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify spending patterns within the month."""
        patterns = {
            'busiest_days': [],
            'highest_spending_days': [],
            'recurring_patterns': [],
            'weekend_vs_weekday': {},
            'category_patterns': {}
        }
        
        if len(df) == 0:
            return patterns
        
        # Busiest days (most transactions)
        daily_counts = df.groupby(df['date'].dt.day).size()
        patterns['busiest_days'] = daily_counts.nlargest(3).to_dict()
        
        # Highest spending days
        daily_spending = df[df['amount'] < 0].groupby(df['date'].dt.day)['amount'].sum().abs()
        patterns['highest_spending_days'] = daily_spending.nlargest(3).to_dict()
        
        # Weekend vs Weekday analysis
        df['is_weekend'] = df['date'].dt.weekday >= 5
        weekend_spending = abs(df[(df['amount'] < 0) & (df['is_weekend'])]['amount'].sum())
        weekday_spending = abs(df[(df['amount'] < 0) & (~df['is_weekend'])]['amount'].sum())
        
        patterns['weekend_vs_weekday'] = {
            'weekend_total': weekend_spending,
            'weekday_total': weekday_spending,
            'weekend_preference': weekend_spending > weekday_spending
        }
        
        # Category patterns by day of week
        if 'category' in df.columns:
            category_by_day = df[df['amount'] < 0].groupby([df['date'].dt.day_name(), 'category'])['amount'].sum().abs()
            patterns['category_patterns'] = category_by_day.to_dict()
        
        return patterns
    
    def _create_calendar_matrix(self, daily_summary: Dict[int, Dict[str, Any]], year: int, month: int) -> List[List[Dict[str, Any]]]:
        """Create matrix representation of calendar for visualization."""
        
        # Get calendar layout
        cal = calendar.monthcalendar(year, month)
        calendar_matrix = []
        
        for week in cal:
            week_data = []
            for day in week:
                if day == 0:
                    # Empty cell for days not in this month
                    week_data.append({
                        'day': 0,
                        'empty': True,
                        'spending': 0,
                        'income': 0,
                        'net': 0,
                        'type': 'empty'
                    })
                else:
                    day_info = daily_summary.get(day, {})
                    week_data.append({
                        'day': day,
                        'empty': False,
                        'spending': day_info.get('total_expense', 0),
                        'income': day_info.get('total_income', 0),
                        'net': day_info.get('net_flow', 0),
                        'transactions': day_info.get('transaction_count', 0),
                        'type': day_info.get('day_type', 'inactive'),
                        'categories': day_info.get('categories', {})
                    })
            calendar_matrix.append(week_data)
        
        return calendar_matrix
    
    def _calculate_month_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate monthly statistics."""
        
        total_income = df[df['amount'] > 0]['amount'].sum()
        total_expense = abs(df[df['amount'] < 0]['amount'].sum())
        transaction_count = len(df)
        
        # Day-wise breakdown
        daily_avg_expense = total_expense / max(1, len(df['date'].dt.day.unique())) if len(df) > 0 else 0
        
        # Most active day
        most_active_day = df['date'].dt.day_name().value_counts().index[0] if len(df) > 0 else 'N/A'
        
        return {
            'total_income': total_income,
            'total_expense': total_expense,
            'net_flow': total_income - total_expense,
            'transaction_count': transaction_count,
            'daily_avg_expense': daily_avg_expense,
            'most_active_day': most_active_day,
            'spending_days': len(df[df['amount'] < 0]['date'].dt.day.unique()),
            'income_days': len(df[df['amount'] > 0]['date'].dt.day.unique())
        }
    
    def create_calendar_heatmap(self, calendar_data: Dict[str, Any]) -> go.Figure:
        """Create calendar heatmap visualization."""
        
        calendar_matrix = calendar_data['calendar_matrix']
        year = calendar_data['year']
        month = calendar_data['month']
        
        # Prepare data for heatmap
        spending_values = []
        day_labels = []
        hover_text = []
        
        for week_idx, week in enumerate(calendar_matrix):
            week_spending = []
            week_labels = []
            week_hover = []
            
            for day_data in week:
                if day_data['empty']:
                    week_spending.append(0)
                    week_labels.append('')
                    week_hover.append('')
                else:
                    week_spending.append(day_data['spending'])
                    week_labels.append(str(day_data['day']))
                    
                    hover_info = f"Day {day_data['day']}<br>"
                    hover_info += f"Spending: ${day_data['spending']:.2f}<br>"
                    hover_info += f"Income: ${day_data['income']:.2f}<br>"
                    hover_info += f"Net: ${day_data['net']:.2f}<br>"
                    hover_info += f"Transactions: {day_data['transactions']}"
                    week_hover.append(hover_info)
            
            spending_values.append(week_spending)
            day_labels.append(week_labels)
            hover_text.append(week_hover)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=spending_values,
            text=day_labels,
            hovertext=hover_text,
            hovertemplate='%{hovertext}<extra></extra>',
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Daily Spending ($)")
        ))
        
        # Add day of week labels
        weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig.update_layout(
            title=f'Financial Calendar - {calendar.month_name[month]} {year}',
            xaxis=dict(
                tickvals=list(range(7)),
                ticktext=weekdays,
                title='Day of Week'
            ),
            yaxis=dict(
                tickvals=list(range(len(calendar_matrix))),
                ticktext=[f'Week {i+1}' for i in range(len(calendar_matrix))],
                title='Week'
            ),
            height=400,
            width=600
        )
        
        return fig
    
    def create_spending_pattern_chart(self, patterns: Dict[str, Any]) -> go.Figure:
        """Create visualization of spending patterns."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Busiest Days', 'Highest Spending Days', 'Weekend vs Weekday', 'Daily Average'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}], 
                   [{'type': 'pie'}, {'type': 'bar'}]]
        )
        
        # Busiest days
        if patterns['busiest_days']:
            days = list(patterns['busiest_days'].keys())
            counts = list(patterns['busiest_days'].values())
            
            fig.add_trace(go.Bar(
                x=[f"Day {d}" for d in days],
                y=counts,
                name='Transactions',
                marker_color='lightblue'
            ), row=1, col=1)
        
        # Highest spending days
        if patterns['highest_spending_days']:
            spend_days = list(patterns['highest_spending_days'].keys())
            spend_amounts = list(patterns['highest_spending_days'].values())
            
            fig.add_trace(go.Bar(
                x=[f"Day {d}" for d in spend_days],
                y=spend_amounts,
                name='Spending ($)',
                marker_color='lightcoral'
            ), row=1, col=2)
        
        # Weekend vs Weekday pie chart
        weekend_weekday = patterns['weekend_vs_weekday']
        if weekend_weekday:
            fig.add_trace(go.Pie(
                labels=['Weekend', 'Weekday'],
                values=[weekend_weekday['weekend_total'], weekend_weekday['weekday_total']],
                name='Spending Distribution'
            ), row=2, col=1)
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Spending Patterns Analysis"
        )
        
        return fig
    
    def get_calendar_insights(self, calendar_data: Dict[str, Any]) -> List[str]:
        """Generate insights from calendar analysis."""
        insights = []
        
        patterns = calendar_data['patterns']
        month_stats = calendar_data['month_stats']
        
        # Most active day insight
        if month_stats['most_active_day'] != 'N/A':
            insights.append(f"📅 You're most active on {month_stats['most_active_day']}s with financial transactions")
        
        # Weekend vs weekday insight
        weekend_data = patterns['weekend_vs_weekday']
        if weekend_data and weekend_data['weekend_preference']:
            weekend_pct = (weekend_data['weekend_total'] / (weekend_data['weekend_total'] + weekend_data['weekday_total'])) * 100
            insights.append(f"🏖️ You spend {weekend_pct:.1f}% of your money on weekends")
        else:
            insights.append(f"💼 You spend more during weekdays than weekends")
        
        # High spending days
        if patterns['highest_spending_days']:
            max_day = max(patterns['highest_spending_days'], key=patterns['highest_spending_days'].get)
            max_amount = patterns['highest_spending_days'][max_day]
            insights.append(f"💸 Your highest spending day was day {max_day} with ${max_amount:.2f}")
        
        # Spending frequency
        spending_days = month_stats['spending_days']
        total_days_in_month = 31  # Approximate
        spending_frequency = (spending_days / total_days_in_month) * 100
        
        if spending_frequency > 80:
            insights.append(f"🛍️ You make purchases almost daily ({spending_frequency:.0f}% of days)")
        elif spending_frequency > 50:
            insights.append(f"🛒 You shop regularly ({spending_frequency:.0f}% of days have transactions)")
        else:
            insights.append(f"💰 You're selective with spending ({spending_frequency:.0f}% of days have purchases)")
        
        # Income pattern
        income_days = month_stats['income_days']
        if income_days <= 2:
            insights.append(f"💼 You receive income on {income_days} day(s) per month - consider diversifying income sources")
        
        return insights
    
    def create_monthly_flow_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create monthly cash flow visualization."""
        
        daily_flow = df.groupby(df['date'].dt.day).agg({
            'amount': lambda x: [x[x > 0].sum(), x[x < 0].sum()]
        })
        
        days = daily_flow.index
        income_values = [flow[0] for flow in daily_flow['amount']]
        expense_values = [abs(flow[1]) for flow in daily_flow['amount']]
        net_values = [income - expense for income, expense in zip(income_values, expense_values)]
        
        fig = go.Figure()
        
        # Income bars
        fig.add_trace(go.Bar(
            x=days,
            y=income_values,
            name='Income',
            marker_color='green',
            opacity=0.7
        ))
        
        # Expense bars (negative)
        fig.add_trace(go.Bar(
            x=days,
            y=[-exp for exp in expense_values],
            name='Expenses',
            marker_color='red',
            opacity=0.7
        ))
        
        # Net flow line
        fig.add_trace(go.Scatter(
            x=days,
            y=net_values,
            mode='lines+markers',
            name='Net Flow',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title='Daily Cash Flow',
            xaxis_title='Day of Month',
            yaxis_title='Amount ($)',
            template='plotly_white',
            height=400,
            barmode='relative'
        )
        
        return fig