import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import streamlit as st

class FinancialHealthAnalyzer:
    """Calculates personalized financial health scores and provides insights."""
    
    def __init__(self):
        self.score_weights = {
            'savings_rate': 0.25,
            'spending_consistency': 0.20,
            'category_balance': 0.15,
            'debt_management': 0.15,
            'income_stability': 0.15,
            'emergency_fund': 0.10
        }
    
    def calculate_health_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive financial health score (0-100)."""
        scores = {}
        
        # 1. Savings Rate Score (25%)
        scores['savings_rate'] = self._calculate_savings_rate_score(df)
        
        # 2. Spending Consistency Score (20%)
        scores['spending_consistency'] = self._calculate_consistency_score(df)
        
        # 3. Category Balance Score (15%)
        scores['category_balance'] = self._calculate_category_balance_score(df)
        
        # 4. Debt Management Score (15%)
        scores['debt_management'] = self._calculate_debt_management_score(df)
        
        # 5. Income Stability Score (15%)
        scores['income_stability'] = self._calculate_income_stability_score(df)
        
        # 6. Emergency Fund Score (10%)
        scores['emergency_fund'] = self._calculate_emergency_fund_score(df)
        
        # Calculate weighted total
        total_score = sum(scores[key] * self.score_weights[key] for key in scores.keys())
        
        return {
            'total_score': round(total_score, 1),
            'component_scores': scores,
            'grade': self._get_grade(total_score),
            'insights': self._generate_health_insights(scores, df),
            'improvement_tips': self._generate_improvement_tips(scores)
        }
    
    def _calculate_savings_rate_score(self, df: pd.DataFrame) -> float:
        """Calculate savings rate score based on income vs expenses."""
        monthly_data = df.groupby('month_year').agg({
            'amount': lambda x: [x[x > 0].sum(), x[x < 0].sum()]
        })
        
        savings_rates = []
        for month_data in monthly_data['amount']:
            income = month_data[0]
            expenses = abs(month_data[1])
            if income > 0:
                savings_rate = (income - expenses) / income * 100
                savings_rates.append(max(0, savings_rate))
        
        if not savings_rates:
            return 50.0
        
        avg_savings_rate = np.mean(savings_rates)
        
        # Score based on savings rate
        if avg_savings_rate >= 20:
            return 100.0
        elif avg_savings_rate >= 15:
            return 85.0
        elif avg_savings_rate >= 10:
            return 70.0
        elif avg_savings_rate >= 5:
            return 55.0
        elif avg_savings_rate >= 0:
            return 40.0
        else:
            return 20.0
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate spending consistency score."""
        monthly_expenses = df[df['amount'] < 0].groupby('month_year')['amount'].sum().abs()
        
        if len(monthly_expenses) < 2:
            return 75.0
        
        # Calculate coefficient of variation
        cv = monthly_expenses.std() / monthly_expenses.mean()
        
        # Score based on consistency (lower CV = higher score)
        if cv <= 0.1:
            return 100.0
        elif cv <= 0.2:
            return 85.0
        elif cv <= 0.3:
            return 70.0
        elif cv <= 0.5:
            return 55.0
        else:
            return 40.0
    
    def _calculate_category_balance_score(self, df: pd.DataFrame) -> float:
        """Calculate category balance score based on recommended spending ratios."""
        if 'category' not in df.columns:
            return 70.0
        
        category_spending = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        total_expenses = category_spending.sum()
        
        if total_expenses == 0:
            return 70.0
        
        # Recommended spending ratios
        recommended_ratios = {
            'Food & Dining': 0.15,
            'Transportation': 0.15,
            'Bills & Utilities': 0.25,
            'Entertainment': 0.05,
            'Shopping': 0.10,
            'Health & Medical': 0.05
        }
        
        balance_score = 100.0
        for category, recommended in recommended_ratios.items():
            if category in category_spending:
                actual_ratio = category_spending[category] / total_expenses
                deviation = abs(actual_ratio - recommended)
                # Penalize deviations
                balance_score -= min(30, deviation * 200)
        
        return max(40.0, balance_score)
    
    def _calculate_debt_management_score(self, df: pd.DataFrame) -> float:
        """Calculate debt management score."""
        # Look for debt-related transactions
        debt_keywords = ['loan', 'credit card', 'mortgage', 'debt', 'interest']
        debt_transactions = df[df['description'].str.lower().str.contains('|'.join(debt_keywords), na=False)]
        
        total_debt_payments = abs(debt_transactions[debt_transactions['amount'] < 0]['amount'].sum())
        total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
        
        if total_expenses == 0:
            return 80.0
        
        debt_ratio = total_debt_payments / total_expenses
        
        # Score based on debt-to-expense ratio
        if debt_ratio <= 0.1:
            return 100.0
        elif debt_ratio <= 0.2:
            return 85.0
        elif debt_ratio <= 0.3:
            return 70.0
        elif debt_ratio <= 0.4:
            return 55.0
        else:
            return 40.0
    
    def _calculate_income_stability_score(self, df: pd.DataFrame) -> float:
        """Calculate income stability score."""
        income_data = df[df['amount'] > 0].groupby('month_year')['amount'].sum()
        
        if len(income_data) < 2:
            return 75.0
        
        # Calculate coefficient of variation for income
        cv = income_data.std() / income_data.mean()
        
        # Score based on income stability
        if cv <= 0.05:
            return 100.0
        elif cv <= 0.1:
            return 85.0
        elif cv <= 0.2:
            return 70.0
        elif cv <= 0.3:
            return 55.0
        else:
            return 40.0
    
    def _calculate_emergency_fund_score(self, df: pd.DataFrame) -> float:
        """Calculate emergency fund score based on savings patterns."""
        # Look for savings-related transactions
        savings_keywords = ['savings', 'emergency', 'fund', 'investment']
        savings_transactions = df[df['description'].str.lower().str.contains('|'.join(savings_keywords), na=False)]
        
        monthly_expenses = abs(df[df['amount'] < 0].groupby('month_year')['amount'].sum().mean())
        total_savings = savings_transactions[savings_transactions['amount'] > 0]['amount'].sum()
        
        if monthly_expenses == 0:
            return 70.0
        
        # Calculate months of expenses covered
        months_covered = total_savings / monthly_expenses
        
        # Score based on emergency fund coverage
        if months_covered >= 6:
            return 100.0
        elif months_covered >= 3:
            return 80.0
        elif months_covered >= 1:
            return 60.0
        elif months_covered >= 0.5:
            return 40.0
        else:
            return 20.0
    
    def _get_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "C+"
        elif score >= 65:
            return "C"
        elif score >= 60:
            return "D+"
        elif score >= 55:
            return "D"
        else:
            return "F"
    
    def _generate_health_insights(self, scores: Dict[str, float], df: pd.DataFrame) -> List[str]:
        """Generate personalized financial health insights."""
        insights = []
        
        # Analyze strongest and weakest areas
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_scores[0]
        weakest = sorted_scores[-1]
        
        insights.append(f"💪 Your strongest area is {strongest[0].replace('_', ' ').title()} ({strongest[1]:.1f}/100)")
        insights.append(f"⚠️ Your weakest area is {weakest[0].replace('_', ' ').title()} ({weakest[1]:.1f}/100)")
        
        # Specific insights based on data
        if 'category' in df.columns:
            top_expense = df[df['amount'] < 0].groupby('category')['amount'].sum().abs().idxmax()
            insights.append(f"📊 Your largest expense category is {top_expense}")
        
        # Trend insights
        monthly_spending = df[df['amount'] < 0].groupby('month_year')['amount'].sum().abs()
        if len(monthly_spending) >= 2:
            trend = "increasing" if monthly_spending.iloc[-1] > monthly_spending.iloc[0] else "decreasing"
            insights.append(f"📈 Your spending trend is {trend} over time")
        
        return insights
    
    def _generate_improvement_tips(self, scores: Dict[str, float]) -> List[str]:
        """Generate personalized improvement tips."""
        tips = []
        
        for component, score in scores.items():
            if score < 70:  # Areas needing improvement
                if component == 'savings_rate':
                    tips.append("💡 Try to save at least 20% of your income. Start with the 50/30/20 rule.")
                elif component == 'spending_consistency':
                    tips.append("📊 Create a monthly budget to make your spending more predictable.")
                elif component == 'category_balance':
                    tips.append("⚖️ Review your spending categories. Consider reducing non-essential expenses.")
                elif component == 'debt_management':
                    tips.append("💳 Focus on paying down high-interest debt first to improve your score.")
                elif component == 'income_stability':
                    tips.append("💼 Consider diversifying income sources or building a more stable income stream.")
                elif component == 'emergency_fund':
                    tips.append("🚨 Build an emergency fund covering 3-6 months of expenses.")
        
        if not tips:
            tips.append("🎉 Great job! Your financial health is strong across all areas!")
        
        return tips

    def create_health_score_visualization(self, health_data: Dict[str, Any]):
        """Create visual representation of financial health score."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create gauge chart for overall score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = health_data['total_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Financial Health Score<br><span style='font-size:0.8em;color:gray'>{health_data['grade']}</span>"},
            delta = {'reference': 75},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def create_component_scores_chart(self, component_scores: Dict[str, float]):
        """Create radar chart for component scores."""
        import plotly.graph_objects as go
        
        categories = [name.replace('_', ' ').title() for name in component_scores.keys()]
        values = list(component_scores.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Scores',
            line_color='rgb(255, 107, 107)'
        ))
        
        # Add ideal scores for comparison
        ideal_scores = [90] * len(categories)
        fig.add_trace(go.Scatterpolar(
            r=ideal_scores,
            theta=categories,
            fill='toself',
            name='Ideal Scores',
            opacity=0.3,
            line_color='rgb(76, 175, 80)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Financial Health Components",
            height=400
        )
        
        return fig