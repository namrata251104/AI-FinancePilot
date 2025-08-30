import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import streamlit as st

class SmartAlertsSystem:
    """AI-powered spending alerts and notifications."""
    
    def __init__(self):
        self.alert_thresholds = {
            'spending_spike': 1.5,    # 50% above normal
            'budget_warning': 0.8,    # 80% of budget used
            'unusual_transaction': 2.0, # 2x std deviation
            'category_overspend': 1.3  # 30% above category average
        }
    
    def generate_alerts(self, df: pd.DataFrame, budgets: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Generate comprehensive smart alerts."""
        alerts = []
        
        # Current month data
        current_month = datetime.now().strftime('%Y-%m')
        current_month_data = df[df['date'].dt.strftime('%Y-%m') == current_month]
        
        # 1. Budget alerts
        if budgets:
            alerts.extend(self._check_budget_alerts(current_month_data, budgets))
        
        # 2. Spending spike alerts
        alerts.extend(self._check_spending_spikes(df, current_month_data))
        
        # 3. Unusual transaction alerts
        alerts.extend(self._check_unusual_transactions(df, current_month_data))
        
        # 4. Category overspending alerts
        alerts.extend(self._check_category_overspending(df, current_month_data))
        
        # 5. Trend alerts
        alerts.extend(self._check_trend_alerts(df))
        
        # 6. Recurring expense alerts
        alerts.extend(self._check_recurring_expense_alerts(df))
        
        # Sort by priority
        alerts.sort(key=lambda x: self._get_alert_priority(x['type']), reverse=True)
        
        return alerts[:10]  # Return top 10 most important alerts
    
    def _check_budget_alerts(self, current_data: pd.DataFrame, budgets: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for budget-related alerts."""
        alerts = []
        
        if 'category' not in current_data.columns:
            return alerts
        
        current_spending = current_data[current_data['amount'] < 0].groupby('category')['amount'].sum().abs()
        
        for category, budget in budgets.items():
            if category in current_spending:
                spent = current_spending[category]
                percentage_used = (spent / budget) * 100
                
                if percentage_used >= 100:
                    alerts.append({
                        'type': 'budget_exceeded',
                        'category': category,
                        'message': f"🚨 Budget exceeded! You've spent ${spent:.2f} of your ${budget:.2f} budget for {category}",
                        'severity': 'high',
                        'amount': spent,
                        'budget': budget,
                        'percentage': percentage_used
                    })
                elif percentage_used >= 80:
                    alerts.append({
                        'type': 'budget_warning',
                        'category': category,
                        'message': f"⚠️ Budget warning! You've used {percentage_used:.1f}% of your {category} budget",
                        'severity': 'medium',
                        'amount': spent,
                        'budget': budget,
                        'percentage': percentage_used
                    })
        
        return alerts
    
    def _check_spending_spikes(self, df: pd.DataFrame, current_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for unusual spending spikes."""
        alerts = []
        
        # Calculate historical monthly average
        monthly_spending = df[df['amount'] < 0].groupby('month_year')['amount'].sum().abs()
        
        if len(monthly_spending) < 2:
            return alerts
        
        historical_avg = monthly_spending[:-1].mean()  # Exclude current month
        current_spending = abs(current_data[current_data['amount'] < 0]['amount'].sum())
        
        if current_spending > historical_avg * self.alert_thresholds['spending_spike']:
            increase_percentage = ((current_spending - historical_avg) / historical_avg) * 100
            alerts.append({
                'type': 'spending_spike',
                'message': f"📈 Spending spike detected! This month's spending is {increase_percentage:.1f}% above your average",
                'severity': 'high' if increase_percentage > 100 else 'medium',
                'current_amount': current_spending,
                'average_amount': historical_avg,
                'increase_percentage': increase_percentage
            })
        
        return alerts
    
    def _check_unusual_transactions(self, df: pd.DataFrame, current_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for unusually large transactions."""
        alerts = []
        
        # Calculate transaction size statistics
        transaction_sizes = abs(df[df['amount'] < 0]['amount'])
        mean_transaction = transaction_sizes.mean()
        std_transaction = transaction_sizes.std()
        
        if std_transaction == 0:
            return alerts
        
        # Check current month's large transactions
        current_transactions = current_data[current_data['amount'] < 0]
        
        for _, transaction in current_transactions.iterrows():
            amount = abs(transaction['amount'])
            z_score = (amount - mean_transaction) / std_transaction
            
            if z_score > self.alert_thresholds['unusual_transaction']:
                alerts.append({
                    'type': 'unusual_transaction',
                    'message': f"💳 Unusual transaction: ${amount:.2f} at {transaction['description']} ({z_score:.1f}x your typical spending)",
                    'severity': 'medium',
                    'amount': amount,
                    'description': transaction['description'],
                    'date': transaction['date'].strftime('%Y-%m-%d'),
                    'z_score': z_score
                })
        
        return alerts
    
    def _check_category_overspending(self, df: pd.DataFrame, current_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for category-specific overspending."""
        alerts = []
        
        if 'category' not in df.columns:
            return alerts
        
        # Calculate historical category averages
        historical_category_avg = df[df['amount'] < 0].groupby(['category', 'month_year'])['amount'].sum().abs().groupby('category').mean()
        
        # Current month category spending
        current_category_spending = current_data[current_data['amount'] < 0].groupby('category')['amount'].sum().abs()
        
        for category in current_category_spending.index:
            if category in historical_category_avg:
                current_amount = current_category_spending[category]
                historical_avg = historical_category_avg[category]
                
                if current_amount > historical_avg * self.alert_thresholds['category_overspend']:
                    increase_percentage = ((current_amount - historical_avg) / historical_avg) * 100
                    alerts.append({
                        'type': 'category_overspend',
                        'category': category,
                        'message': f"📊 {category} spending is {increase_percentage:.1f}% above average this month",
                        'severity': 'medium',
                        'current_amount': current_amount,
                        'average_amount': historical_avg,
                        'increase_percentage': increase_percentage
                    })
        
        return alerts
    
    def _check_trend_alerts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for concerning spending trends."""
        alerts = []
        
        monthly_spending = df[df['amount'] < 0].groupby('month_year')['amount'].sum().abs()
        
        if len(monthly_spending) < 3:
            return alerts
        
        # Calculate trend
        months = list(range(len(monthly_spending)))
        spending_values = monthly_spending.values
        trend_slope = np.polyfit(months, spending_values, 1)[0]
        
        # Check for concerning upward trend
        if trend_slope > monthly_spending.mean() * 0.1:  # 10% increase per month
            alerts.append({
                'type': 'upward_trend',
                'message': f"📈 Your spending has been increasing by ${trend_slope:.2f} per month over the last {len(monthly_spending)} months",
                'severity': 'medium',
                'trend_slope': trend_slope,
                'trend_direction': 'increasing'
            })
        
        return alerts
    
    def _check_recurring_expense_alerts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for missing or changed recurring expenses."""
        alerts = []
        
        # Find potential recurring expenses (same description, similar amounts)
        recurring_candidates = df.groupby('description').agg({
            'amount': ['count', 'std'],
            'date': ['min', 'max']
        }).reset_index()
        
        recurring_candidates.columns = ['description', 'count', 'std', 'first_date', 'last_date']
        
        # Filter for likely recurring expenses
        recurring_expenses = recurring_candidates[
            (recurring_candidates['count'] >= 3) &
            (recurring_candidates['std'] < 10)  # Low variation in amount
        ]
        
        current_month = datetime.now().strftime('%Y-%m')
        
        for _, expense in recurring_expenses.iterrows():
            last_occurrence = expense['last_date']
            days_since = (datetime.now() - last_occurrence).days
            
            # Alert if recurring expense is overdue
            if days_since > 35:  # More than 35 days (allowing for monthly variation)
                alerts.append({
                    'type': 'missing_recurring',
                    'message': f"🔄 Missing recurring expense: '{expense['description']}' last seen {days_since} days ago",
                    'severity': 'low',
                    'description': expense['description'],
                    'days_since': days_since,
                    'last_occurrence': last_occurrence.strftime('%Y-%m-%d')
                })
        
        return alerts
    
    def _get_alert_priority(self, alert_type: str) -> int:
        """Get priority score for alert sorting."""
        priority_scores = {
            'budget_exceeded': 10,
            'spending_spike': 9,
            'budget_warning': 8,
            'unusual_transaction': 7,
            'category_overspend': 6,
            'upward_trend': 5,
            'missing_recurring': 3
        }
        return priority_scores.get(alert_type, 1)
    
    def create_alerts_summary(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of alerts by severity."""
        summary = {
            'total_alerts': len(alerts),
            'high_severity': len([a for a in alerts if a.get('severity') == 'high']),
            'medium_severity': len([a for a in alerts if a.get('severity') == 'medium']),
            'low_severity': len([a for a in alerts if a.get('severity') == 'low']),
            'alert_types': {}
        }
        
        # Count by type
        for alert in alerts:
            alert_type = alert['type']
            if alert_type not in summary['alert_types']:
                summary['alert_types'][alert_type] = 0
            summary['alert_types'][alert_type] += 1
        
        return summary
    
    def display_alerts(self, alerts: List[Dict[str, Any]]):
        """Display alerts in Streamlit interface."""
        if not alerts:
            st.success("🎉 No alerts! Your spending looks healthy.")
            return
        
        st.subheader(f"🚨 Smart Alerts ({len(alerts)})")
        
        # Group by severity
        high_alerts = [a for a in alerts if a.get('severity') == 'high']
        medium_alerts = [a for a in alerts if a.get('severity') == 'medium']
        low_alerts = [a for a in alerts if a.get('severity') == 'low']
        
        # Display high severity alerts
        if high_alerts:
            st.error("🔴 High Priority Alerts")
            for alert in high_alerts:
                st.error(alert['message'])
        
        # Display medium severity alerts
        if medium_alerts:
            st.warning("🟡 Medium Priority Alerts")
            for alert in medium_alerts:
                st.warning(alert['message'])
        
        # Display low severity alerts
        if low_alerts:
            st.info("🔵 Low Priority Alerts")
            for alert in low_alerts:
                st.info(alert['message'])