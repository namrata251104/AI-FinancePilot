import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Any, Optional
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

class GoalTracker:
    """Advanced goal tracking and financial recommendations system."""
    
    def __init__(self):
        self.goal_types = [
            'Emergency Fund',
            'Vacation/Travel',
            'Home Down Payment',
            'Car Purchase',
            'Debt Payoff',
            'Investment/Retirement',
            'Education Fund',
            'Wedding',
            'Custom Goal'
        ]
    
    def create_goal(self, name: str, target_amount: float, target_date: date, 
                   goal_type: str = 'Custom Goal', current_amount: float = 0) -> Dict[str, Any]:
        """Create a new financial goal."""
        days_remaining = (target_date - date.today()).days
        months_remaining = max(1, days_remaining / 30.44)
        
        monthly_savings_needed = max(0, (target_amount - current_amount) / months_remaining)
        
        return {
            'name': name,
            'type': goal_type,
            'target_amount': target_amount,
            'current_amount': current_amount,
            'target_date': target_date,
            'days_remaining': days_remaining,
            'months_remaining': months_remaining,
            'monthly_savings_needed': monthly_savings_needed,
            'progress_percentage': (current_amount / target_amount * 100) if target_amount > 0 else 0,
            'status': self._get_goal_status(current_amount, target_amount, days_remaining),
            'created_date': date.today()
        }
    
    def analyze_goal_feasibility(self, goal: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze if a goal is achievable based on current financial patterns."""
        
        # Calculate current financial capacity
        monthly_income = df[df['amount'] > 0].groupby('month_year')['amount'].sum().mean()
        monthly_expenses = abs(df[df['amount'] < 0].groupby('month_year')['amount'].sum().mean())
        current_savings_rate = max(0, (monthly_income - monthly_expenses) / monthly_income) if monthly_income > 0 else 0
        current_monthly_savings = monthly_income * current_savings_rate
        
        # Goal requirements
        required_monthly_savings = goal['monthly_savings_needed']
        
        # Feasibility analysis
        if current_monthly_savings >= required_monthly_savings:
            feasibility = 'Easy'
            recommendation = "You're already saving enough to reach this goal!"
        elif current_monthly_savings >= required_monthly_savings * 0.8:
            feasibility = 'Achievable'
            shortage = required_monthly_savings - current_monthly_savings
            recommendation = f"You need to save an additional ${shortage:.2f} per month. Consider reducing discretionary spending."
        elif current_monthly_savings >= required_monthly_savings * 0.5:
            feasibility = 'Challenging'
            shortage = required_monthly_savings - current_monthly_savings
            recommendation = f"This goal is challenging. You need ${shortage:.2f} more per month. Consider extending the timeline or reducing the target amount."
        else:
            feasibility = 'Difficult'
            shortage = required_monthly_savings - current_monthly_savings
            recommendation = f"This goal may be unrealistic with current spending. You need ${shortage:.2f} more per month. Consider major budget changes or extending the timeline significantly."
        
        return {
            'feasibility': feasibility,
            'current_monthly_savings': current_monthly_savings,
            'required_monthly_savings': required_monthly_savings,
            'monthly_shortage': max(0, required_monthly_savings - current_monthly_savings),
            'recommendation': recommendation,
            'success_probability': min(100, max(10, (current_monthly_savings / required_monthly_savings) * 100))
        }
    
    def generate_savings_strategies(self, goal: Dict[str, Any], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate personalized savings strategies to reach the goal."""
        strategies = []
        
        if 'category' not in df.columns:
            return self._get_default_strategies(goal)
        
        # Analyze spending categories for optimization opportunities
        category_spending = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        total_expenses = category_spending.sum()
        
        monthly_shortage = goal['monthly_savings_needed'] - self._calculate_current_savings(df)
        
        if monthly_shortage <= 0:
            strategies.append({
                'strategy': 'Stay on Track',
                'description': "You're already saving enough! Keep up the good work.",
                'potential_savings': 0,
                'difficulty': 'Easy',
                'category': 'Maintenance'
            })
            return strategies
        
        # Strategy 1: Reduce largest expense categories
        for category, amount in category_spending.head(3).items():
            monthly_amount = amount / max(1, len(df['month_year'].unique()))
            potential_reduction = monthly_amount * 0.15  # 15% reduction
            
            if potential_reduction >= monthly_shortage * 0.3:  # Could cover 30% of shortage
                strategies.append({
                    'strategy': f'Reduce {category} Spending',
                    'description': f"Cut {category} spending by 15% to save ${potential_reduction:.2f}/month",
                    'potential_savings': potential_reduction,
                    'difficulty': 'Medium',
                    'category': category
                })
        
        # Strategy 2: Target discretionary spending
        discretionary_categories = ['Entertainment', 'Shopping', 'Dining']
        for category in discretionary_categories:
            if category in category_spending:
                monthly_amount = category_spending[category] / max(1, len(df['month_year'].unique()))
                potential_reduction = monthly_amount * 0.25  # 25% reduction
                
                strategies.append({
                    'strategy': f'Cut {category}',
                    'description': f"Reduce {category} by 25% to save ${potential_reduction:.2f}/month",
                    'potential_savings': potential_reduction,
                    'difficulty': 'Easy' if potential_reduction < 100 else 'Medium',
                    'category': category
                })
        
        # Strategy 3: Increase income
        current_monthly_income = df[df['amount'] > 0].groupby('month_year')['amount'].sum().mean()
        income_increase_needed = monthly_shortage / 0.7  # Assume 30% goes to taxes/expenses
        
        strategies.append({
            'strategy': 'Increase Income',
            'description': f"Increase monthly income by ${income_increase_needed:.2f} through side work or raises",
            'potential_savings': monthly_shortage,
            'difficulty': 'Hard',
            'category': 'Income'
        })
        
        # Strategy 4: Combination approach
        if len(strategies) > 1:
            total_potential = sum(s['potential_savings'] for s in strategies[:2])
            if total_potential >= monthly_shortage:
                strategies.append({
                    'strategy': 'Combination Approach',
                    'description': f"Combine multiple small changes to save ${total_potential:.2f}/month",
                    'potential_savings': total_potential,
                    'difficulty': 'Medium',
                    'category': 'Mixed'
                })
        
        # Sort by potential impact and difficulty
        strategies.sort(key=lambda x: (x['potential_savings'], -self._get_difficulty_score(x['difficulty'])), reverse=True)
        
        return strategies[:5]  # Return top 5 strategies
    
    def track_goal_progress(self, goal: Dict[str, Any], transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Track progress towards a specific goal."""
        
        # Simple progress tracking (in real app, this would link to specific savings transactions)
        current_savings = self._calculate_current_savings(transactions_df)
        
        # Calculate if on track
        elapsed_time = (date.today() - goal['created_date']).days
        total_time = (goal['target_date'] - goal['created_date']).days
        expected_progress = (elapsed_time / max(1, total_time)) * 100
        
        actual_progress = goal['progress_percentage']
        
        return {
            'current_amount': goal['current_amount'],
            'target_amount': goal['target_amount'],
            'progress_percentage': actual_progress,
            'expected_progress': expected_progress,
            'ahead_behind': actual_progress - expected_progress,
            'monthly_savings_needed': goal['monthly_savings_needed'],
            'days_remaining': goal['days_remaining'],
            'projected_completion': self._project_completion_date(goal, current_savings),
            'on_track': abs(actual_progress - expected_progress) <= 10  # Within 10% tolerance
        }
    
    def get_goal_recommendations(self, goals: List[Dict[str, Any]], df: pd.DataFrame) -> List[str]:
        """Get personalized recommendations across all goals."""
        recommendations = []
        
        if not goals:
            recommendations.append("💡 Start by setting your first financial goal! Emergency funds are usually a great place to begin.")
            return recommendations
        
        # Analyze goal priorities
        urgent_goals = [g for g in goals if g['days_remaining'] < 90]
        large_goals = [g for g in goals if g['target_amount'] > 10000]
        
        if urgent_goals:
            recommendations.append(f"⏰ Focus on urgent goals: {', '.join([g['name'] for g in urgent_goals[:2]])}")
        
        # Check for conflicting goals
        total_monthly_needed = sum(g['monthly_savings_needed'] for g in goals)
        current_capacity = self._calculate_current_savings(df)
        
        if total_monthly_needed > current_capacity * 1.2:
            recommendations.append("⚖️ You have conflicting goals. Consider prioritizing or extending timelines for some goals.")
        
        # Emergency fund recommendation
        emergency_goals = [g for g in goals if 'emergency' in g['name'].lower()]
        if not emergency_goals:
            recommendations.append("🚨 Consider adding an emergency fund goal (3-6 months of expenses).")
        
        # Diversification recommendation
        goal_types = set(g['type'] for g in goals)
        if len(goal_types) == 1 and len(goals) > 1:
            recommendations.append("🎯 Consider diversifying your goals across different categories (savings, investment, debt payoff).")
        
        return recommendations
    
    def _get_goal_status(self, current: float, target: float, days_remaining: int) -> str:
        """Determine goal status."""
        progress = (current / target * 100) if target > 0 else 0
        
        if progress >= 100:
            return 'Completed'
        elif days_remaining < 0:
            return 'Overdue'
        elif days_remaining < 30:
            return 'Critical'
        elif progress >= 75:
            return 'On Track'
        elif progress >= 50:
            return 'Behind'
        else:
            return 'Far Behind'
    
    def _calculate_current_savings(self, df: pd.DataFrame) -> float:
        """Calculate current monthly savings capacity."""
        monthly_income = df[df['amount'] > 0].groupby('month_year')['amount'].sum().mean()
        monthly_expenses = abs(df[df['amount'] < 0].groupby('month_year')['amount'].sum().mean())
        return max(0, monthly_income - monthly_expenses)
    
    def _get_difficulty_score(self, difficulty: str) -> int:
        """Convert difficulty to numeric score."""
        scores = {'Easy': 3, 'Medium': 2, 'Hard': 1}
        return scores.get(difficulty, 1)
    
    def _project_completion_date(self, goal: Dict[str, Any], current_monthly_savings: float) -> date:
        """Project when goal will be completed at current savings rate."""
        if current_monthly_savings <= 0:
            return goal['target_date']  # Use original target if no savings
        
        remaining_amount = goal['target_amount'] - goal['current_amount']
        months_needed = remaining_amount / current_monthly_savings
        
        projected_date = date.today() + timedelta(days=int(months_needed * 30.44))
        return projected_date
    
    def _get_default_strategies(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get default strategies when category data is not available."""
        return [
            {
                'strategy': 'Create a Budget',
                'description': 'Track your spending by category to identify saving opportunities',
                'potential_savings': goal['monthly_savings_needed'] * 0.3,
                'difficulty': 'Easy',
                'category': 'Planning'
            },
            {
                'strategy': 'Automate Savings',
                'description': 'Set up automatic transfers to a dedicated savings account',
                'potential_savings': goal['monthly_savings_needed'],
                'difficulty': 'Easy',
                'category': 'Automation'
            },
            {
                'strategy': 'Reduce Discretionary Spending',
                'description': 'Cut back on dining out, entertainment, and non-essential purchases',
                'potential_savings': goal['monthly_savings_needed'] * 0.5,
                'difficulty': 'Medium',
                'category': 'Spending'
            }
        ]
    
    def create_goal_progress_chart(self, goal: Dict[str, Any]) -> go.Figure:
        """Create progress visualization for a goal."""
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = goal['progress_percentage'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{goal['name']}<br><span style='font-size:0.8em;color:gray'>${goal['current_amount']:,.0f} / ${goal['target_amount']:,.0f}</span>"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
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
    
    def create_savings_strategy_chart(self, strategies: List[Dict[str, Any]]) -> go.Figure:
        """Create visualization of savings strategies."""
        
        if not strategies:
            return go.Figure()
        
        strategy_names = [s['strategy'] for s in strategies[:5]]
        potential_savings = [s['potential_savings'] for s in strategies[:5]]
        difficulties = [s['difficulty'] for s in strategies[:5]]
        
        # Color mapping for difficulty
        color_map = {'Easy': 'green', 'Medium': 'orange', 'Hard': 'red'}
        colors = [color_map.get(d, 'blue') for d in difficulties]
        
        fig = go.Figure(data=[
            go.Bar(
                x=strategy_names,
                y=potential_savings,
                marker_color=colors,
                text=[f"${x:.0f}" for x in potential_savings],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Potential Savings Strategies',
            xaxis_title='Strategy',
            yaxis_title='Potential Monthly Savings ($)',
            template='plotly_white',
            height=400
        )
        
        return fig