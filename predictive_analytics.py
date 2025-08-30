import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class PredictiveAnalytics:
    """Advanced predictive analytics for financial forecasting."""
    
    def __init__(self):
        pass
    
    def predict_future_spending(self, df: pd.DataFrame, months_ahead: int = 3) -> Dict[str, Any]:
        """Predict future spending patterns using trend analysis."""
        
        # Prepare monthly spending data
        monthly_data = df[df['amount'] < 0].groupby('month_year')['amount'].sum().abs()
        
        if len(monthly_data) < 2:
            return self._default_prediction(months_ahead)
        
        # Simple linear trend analysis
        months = list(range(len(monthly_data)))
        spending_values = monthly_data.values
        
        # Calculate trend
        trend_slope = np.polyfit(months, spending_values, 1)[0]
        trend_intercept = np.polyfit(months, spending_values, 1)[1]
        
        # Generate predictions
        future_months = []
        predictions = []
        last_month_idx = len(monthly_data)
        
        for i in range(1, months_ahead + 1):
            future_month_idx = last_month_idx + i
            predicted_value = trend_slope * future_month_idx + trend_intercept
            
            # Add seasonal adjustment (simple)
            seasonal_factor = self._get_seasonal_factor(monthly_data, i)
            adjusted_prediction = max(0, predicted_value * seasonal_factor)
            
            predictions.append(adjusted_prediction)
            
            # Generate future month names
            last_date = monthly_data.index[-1]
            if hasattr(last_date, 'to_timestamp'):
                last_timestamp = last_date.to_timestamp()
            else:
                last_timestamp = pd.to_datetime(str(last_date))
            
            future_date = last_timestamp + pd.DateOffset(months=i)
            future_months.append(future_date.strftime('%Y-%m'))
        
        return {
            'predictions': predictions,
            'future_months': future_months,
            'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
            'monthly_change': abs(trend_slope),
            'confidence_level': self._calculate_confidence(monthly_data, predictions)
        }
    
    def predict_category_spending(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict future spending by category."""
        if 'category' not in df.columns:
            return {}
        
        category_predictions = {}
        
        for category in df['category'].unique():
            category_data = df[(df['category'] == category) & (df['amount'] < 0)]
            monthly_category = category_data.groupby('month_year')['amount'].sum().abs()
            
            if len(monthly_category) >= 2:
                # Simple trend prediction
                trend = np.polyfit(range(len(monthly_category)), monthly_category.values, 1)[0]
                last_value = monthly_category.iloc[-1]
                next_month_prediction = max(0, last_value + trend)
                
                category_predictions[category] = {
                    'predicted_amount': next_month_prediction,
                    'trend': 'increasing' if trend > 0 else 'decreasing',
                    'confidence': min(100, max(50, 100 - abs(trend) / last_value * 100))
                }
        
        return category_predictions
    
    def detect_spending_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect unusual spending patterns."""
        anomalies = []
        
        # Monthly spending anomalies
        monthly_spending = df[df['amount'] < 0].groupby('month_year')['amount'].sum().abs()
        
        if len(monthly_spending) >= 3:
            mean_spending = monthly_spending.mean()
            std_spending = monthly_spending.std()
            
            for month, amount in monthly_spending.items():
                z_score = abs(amount - mean_spending) / std_spending if std_spending > 0 else 0
                
                if z_score > 2:  # More than 2 standard deviations
                    anomalies.append({
                        'type': 'monthly_spending',
                        'month': str(month),
                        'amount': amount,
                        'deviation': f"{z_score:.1f} std deviations",
                        'severity': 'high' if z_score > 3 else 'medium'
                    })
        
        # Category spending anomalies
        if 'category' in df.columns:
            for category in df['category'].unique():
                category_data = df[(df['category'] == category) & (df['amount'] < 0)]
                monthly_category = category_data.groupby('month_year')['amount'].sum().abs()
                
                if len(monthly_category) >= 3:
                    mean_cat = monthly_category.mean()
                    std_cat = monthly_category.std()
                    
                    for month, amount in monthly_category.items():
                        z_score = abs(amount - mean_cat) / std_cat if std_cat > 0 else 0
                        
                        if z_score > 2.5:
                            anomalies.append({
                                'type': 'category_spending',
                                'category': category,
                                'month': str(month),
                                'amount': amount,
                                'deviation': f"{z_score:.1f} std deviations",
                                'severity': 'high' if z_score > 3 else 'medium'
                            })
        
        return anomalies
    
    def forecast_budget_risks(self, df: pd.DataFrame, budgets: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Forecast potential budget overruns."""
        risks = []
        
        if not budgets:
            # Create default budgets based on historical data
            if 'category' in df.columns:
                monthly_avg = df[df['amount'] < 0].groupby('category')['amount'].sum().abs() / max(1, len(df['month_year'].unique()))
                budgets = {cat: amount * 1.1 for cat, amount in monthly_avg.items()}  # 10% buffer
        
        if budgets and 'category' in df.columns:
            category_predictions = self.predict_category_spending(df)
            
            for category, budget in budgets.items():
                if category in category_predictions:
                    predicted = category_predictions[category]['predicted_amount']
                    if predicted > budget:
                        overage = predicted - budget
                        percentage_over = (overage / budget) * 100
                        
                        risks.append({
                            'category': category,
                            'budget': budget,
                            'predicted': predicted,
                            'overage': overage,
                            'percentage_over': percentage_over,
                            'risk_level': 'high' if percentage_over > 20 else 'medium'
                        })
        
        return risks
    
    def _get_seasonal_factor(self, monthly_data: pd.Series, months_ahead: int) -> float:
        """Calculate simple seasonal adjustment factor."""
        if len(monthly_data) < 12:
            return 1.0  # No seasonal adjustment for insufficient data
        
        # Simple seasonal pattern (could be enhanced)
        current_month = datetime.now().month
        future_month = ((current_month + months_ahead - 1) % 12) + 1
        
        # Basic seasonal factors (could be data-driven)
        seasonal_factors = {
            1: 0.9,   # January (post-holiday)
            2: 0.95,  # February
            3: 1.0,   # March
            4: 1.05,  # April
            5: 1.0,   # May
            6: 1.1,   # June (summer activities)
            7: 1.15,  # July (vacation)
            8: 1.1,   # August
            9: 1.0,   # September
            10: 1.05, # October
            11: 1.2,  # November (holiday prep)
            12: 1.3   # December (holidays)
        }
        
        return seasonal_factors.get(future_month, 1.0)
    
    def _calculate_confidence(self, historical_data: pd.Series, predictions: List[float]) -> float:
        """Calculate confidence level for predictions."""
        if len(historical_data) < 3:
            return 60.0
        
        # Calculate historical variance
        variance = historical_data.var()
        mean_value = historical_data.mean()
        
        if mean_value == 0:
            return 50.0
        
        # Confidence based on data consistency
        cv = (variance ** 0.5) / mean_value  # Coefficient of variation
        
        if cv < 0.1:
            return 90.0
        elif cv < 0.2:
            return 80.0
        elif cv < 0.3:
            return 70.0
        else:
            return 60.0
    
    def _default_prediction(self, months_ahead: int) -> Dict[str, Any]:
        """Return default prediction when insufficient data."""
        return {
            'predictions': [0] * months_ahead,
            'future_months': [],
            'trend_direction': 'stable',
            'monthly_change': 0,
            'confidence_level': 50.0
        }
    
    def create_prediction_chart(self, df: pd.DataFrame, prediction_data: Dict[str, Any]) -> go.Figure:
        """Create visualization for spending predictions."""
        # Historical data
        monthly_spending = df[df['amount'] < 0].groupby('month_year')['amount'].sum().abs()
        
        fig = go.Figure()
        
        # Historical spending
        fig.add_trace(go.Scatter(
            x=[str(month) for month in monthly_spending.index],
            y=monthly_spending.values,
            mode='lines+markers',
            name='Historical Spending',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Predictions
        if prediction_data['predictions']:
            fig.add_trace(go.Scatter(
                x=prediction_data['future_months'],
                y=prediction_data['predictions'],
                mode='lines+markers',
                name='Predicted Spending',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6, symbol='diamond')
            ))
        
        fig.update_layout(
            title='Spending Forecast',
            xaxis_title='Month',
            yaxis_title='Amount ($)',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_risk_alert_chart(self, risks: List[Dict[str, Any]]) -> go.Figure:
        """Create visualization for budget risks."""
        if not risks:
            return go.Figure()
        
        categories = [risk['category'] for risk in risks]
        budgets = [risk['budget'] for risk in risks]
        predicted = [risk['predicted'] for risk in risks]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=categories,
            y=budgets,
            name='Budget',
            marker_color='green',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            x=categories,
            y=predicted,
            name='Predicted Spending',
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Budget Risk Analysis',
            xaxis_title='Category',
            yaxis_title='Amount ($)',
            barmode='group',
            template='plotly_white',
            height=400
        )
        
        return fig