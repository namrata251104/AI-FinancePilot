import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta, date
import os
from typing import List, Dict, Any
import json

from data_processor import DataProcessor
from expense_categorizer import ExpenseCategorizer
from vector_store import VectorStore
from conversation_handler import ConversationHandler
from visualizations import FinanceVisualizer
from utils import format_currency, validate_file
from financial_health import FinancialHealthAnalyzer
from predictive_analytics import PredictiveAnalytics
from smart_alerts import SmartAlertsSystem
from goal_tracker import GoalTracker
from financial_calendar import FinancialCalendar

# Configure Streamlit page
st.set_page_config(
    page_title="AI Finance Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'categorizer' not in st.session_state:
    st.session_state.categorizer = ExpenseCategorizer()
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
if 'conversation_handler' not in st.session_state:
    st.session_state.conversation_handler = ConversationHandler()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = FinanceVisualizer()
if 'health_analyzer' not in st.session_state:
    st.session_state.health_analyzer = FinancialHealthAnalyzer()
if 'predictive_analytics' not in st.session_state:
    st.session_state.predictive_analytics = PredictiveAnalytics()
if 'smart_alerts' not in st.session_state:
    st.session_state.smart_alerts = SmartAlertsSystem()
if 'goal_tracker' not in st.session_state:
    st.session_state.goal_tracker = GoalTracker()
if 'financial_calendar' not in st.session_state:
    st.session_state.financial_calendar = FinancialCalendar()
if 'transactions_df' not in st.session_state:
    st.session_state.transactions_df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_goals' not in st.session_state:
    st.session_state.user_goals = []
if 'budgets' not in st.session_state:
    st.session_state.budgets = {}

def main():
    # Enhanced header with branding
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0; font-size: 3em;'>🚀 FinancePilot</h1>
        <h3 style='color: white; margin: 0; font-weight: 300;'>Your AI-Powered Financial Co-Pilot</h3>
        <p style='color: white; margin: 10px 0 0 0; opacity: 0.9;'>Transform your financial data into actionable insights with next-level AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    if st.session_state.transactions_df is None:
        st.markdown("""
        <div style='text-align: center; margin: 30px 0;'>
            <h2>🌟 Revolutionary Features</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style='text-align: center; padding: 20px; border: 2px solid #667eea; border-radius: 10px; margin: 10px;'>
                <h3>🏥</h3>
                <h4>Health Score</h4>
                <p>AI-powered financial wellness scoring</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 20px; border: 2px solid #667eea; border-radius: 10px; margin: 10px;'>
                <h3>🔮</h3>
                <h4>Predictions</h4>
                <p>Future spending forecasts & trends</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='text-align: center; padding: 20px; border: 2px solid #667eea; border-radius: 10px; margin: 10px;'>
                <h3>🚨</h3>
                <h4>Smart Alerts</h4>
                <p>Proactive spending notifications</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style='text-align: center; padding: 20px; border: 2px solid #667eea; border-radius: 10px; margin: 10px;'>
                <h3>🎯</h3>
                <h4>Goal Tracking</h4>
                <p>Personalized savings strategies</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced sidebar with branding
    with st.sidebar:
        # Sidebar Branding Header
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(45deg, #667eea, #764ba2); border-radius: 10px; margin-bottom: 25px;'>
            <h2 style='color: white; margin: 0; font-size: 1.8em;'>🚀 FinancePilot</h2>
            <p style='color: white; margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;'>Control Center</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data Upload Section
        st.markdown("""
        <div style='text-align: center; padding: 15px; background: linear-gradient(45deg, #4facfe, #00f2fe); border-radius: 10px; margin-bottom: 20px;'>
            <h4 style='color: white; margin: 0;'>📁 Data Upload</h4>
            <p style='color: white; margin: 0; font-size: 0.9em;'>CSV & Excel supported</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload your bank statement",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel files with transaction data"
        )
        
        if uploaded_file is not None:
            if validate_file(uploaded_file):
                with st.spinner("Processing your financial data..."):
                    # Process the uploaded file
                    df = st.session_state.data_processor.load_file(uploaded_file)
                    
                    if df is not None:
                        # Categorize expenses
                        df_categorized = st.session_state.categorizer.categorize_transactions(df)
                        
                        # Store in vector database
                        st.session_state.vector_store.add_transactions(df_categorized)
                        
                        # Update session state
                        st.session_state.transactions_df = df_categorized
                        
                        st.success(f"✅ Processed {len(df_categorized)} transactions!")
                        
                        # Display basic stats
                        st.subheader("📊 Quick Stats")
                        total_transactions = len(df_categorized)
                        total_amount = df_categorized['amount'].sum()
                        avg_transaction = df_categorized['amount'].mean()
                        
                        st.metric("Total Transactions", total_transactions)
                        st.metric("Total Amount", format_currency(total_amount))
                        st.metric("Average Transaction", format_currency(avg_transaction))
                        
                        # Gamification elements
                        st.markdown("---")
                        st.markdown("### 🏆 Your Achievements")
                        
                        # Calculate achievements
                        categories = len(df_categorized['category'].unique()) if 'category' in df_categorized.columns else 0
                        months_tracked = len(df_categorized['month_year'].unique()) if 'month_year' in df_categorized.columns else 0
                        
                        # Achievement badges
                        achievements = []
                        if total_transactions >= 100:
                            achievements.append("🥇 Transaction Master")
                        elif total_transactions >= 50:
                            achievements.append("🥈 Data Enthusiast")
                        elif total_transactions >= 10:
                            achievements.append("🥉 Getting Started")
                        
                        if categories >= 8:
                            achievements.append("🎨 Category Expert")
                        elif categories >= 5:
                            achievements.append("📊 Category Tracker")
                        
                        if months_tracked >= 6:
                            achievements.append("📅 Long-term Planner")
                        elif months_tracked >= 3:
                            achievements.append("⏰ Consistent Tracker")
                        
                        for achievement in achievements:
                            st.success(achievement)
                        
                        # Progress to next level
                        st.markdown("### 📈 Level Progress")
                        if total_transactions < 100:
                            progress = total_transactions / 100
                            st.progress(progress)
                            st.caption(f"{100 - total_transactions} more transactions to Master level!")
                        else:
                            st.balloons()
                            st.success("🎉 Master level achieved!")
            else:
                st.error("❌ Invalid file format or structure")
    
    # Main content area
    if st.session_state.transactions_df is not None:
        df = st.session_state.transactions_df
        
        # Create tabs with better organization and flow
        tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "🏠 Dashboard",
            "💬 AI Chat", 
            "📈 Analytics", 
            "🔍 Explorer", 
            "🏥 Health",
            "🔮 Predictions",
            "🚨 Alerts", 
            "🎯 Goals",
            "📅 Calendar"
        ])
        
        with tab0:
            # MAIN DASHBOARD - Overview of everything
            st.markdown("""
            <div style='text-align: center; margin-bottom: 30px;'>
                <h2 style='color: #667eea;'>📊 Financial Dashboard</h2>
                <p style='color: #666; font-size: 1.1em;'>Your complete financial overview at a glance</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick Stats Cards
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate key metrics
            total_transactions = len(df)
            total_spending = abs(df[df['amount'] < 0]['amount'].sum())
            total_income = df[df['amount'] > 0]['amount'].sum()
            net_worth = total_income - total_spending
            
            with col1:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; text-align: center; margin: 5px;'>
                    <h3 style='color: white; margin: 0;'>💳</h3>
                    <h2 style='color: white; margin: 10px 0;'>{}</h2>
                    <p style='color: white; margin: 0; opacity: 0.9;'>Total Transactions</p>
                </div>
                """.format(total_transactions), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; text-align: center; margin: 5px;'>
                    <h3 style='color: white; margin: 0;'>💸</h3>
                    <h2 style='color: white; margin: 10px 0;'>{}</h2>
                    <p style='color: white; margin: 0; opacity: 0.9;'>Total Spending</p>
                </div>
                """.format(format_currency(total_spending)), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 10px; text-align: center; margin: 5px;'>
                    <h3 style='color: white; margin: 0;'>💰</h3>
                    <h2 style='color: white; margin: 10px 0;'>{}</h2>
                    <p style='color: white; margin: 0; opacity: 0.9;'>Total Income</p>
                </div>
                """.format(format_currency(total_income)), unsafe_allow_html=True)
            
            with col4:
                net_color = "#43a047" if net_worth >= 0 else "#e53e3e"
                net_icon = "📈" if net_worth >= 0 else "📉"
                st.markdown("""
                <div style='background: {}; padding: 20px; border-radius: 10px; text-align: center; margin: 5px;'>
                    <h3 style='color: white; margin: 0;'>{}</h3>
                    <h2 style='color: white; margin: 10px 0;'>{}</h2>
                    <p style='color: white; margin: 0; opacity: 0.9;'>Net Flow</p>
                </div>
                """.format(net_color, net_icon, format_currency(net_worth)), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Three-column layout for dashboard sections
            dash_col1, dash_col2, dash_col3 = st.columns([1, 1, 1])
            
            with dash_col1:
                # Health Score Card
                st.markdown("""
                <div style='background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h4 style='color: #667eea; margin-top: 0;'>🏥 Health Score</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Mini health score
                health_data = st.session_state.health_analyzer.calculate_health_score(df)
                score = health_data['total_score']
                grade = health_data['grade']
                
                st.markdown(f"""
                <div style='text-align: center; padding: 10px;'>
                    <h1 style='color: #667eea; margin: 0; font-size: 3em;'>{score}</h1>
                    <h3 style='color: #666; margin: 0;'>Grade: {grade}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 View Full Health Analysis", key="health_dash"):
                    st.info("💡 Switch to the Health tab for detailed analysis!")
            
            with dash_col2:
                # Recent Alerts Card
                st.markdown("""
                <div style='background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h4 style='color: #f5576c; margin-top: 0;'>🚨 Smart Alerts</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Quick alerts preview
                alerts = st.session_state.smart_alerts.generate_alerts(df, st.session_state.budgets)
                
                if alerts:
                    alert_count = len(alerts)
                    high_priority = len([a for a in alerts if a.get('severity') == 'high'])
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px;'>
                        <h2 style='color: #f5576c; margin: 0;'>{alert_count}</h2>
                        <p style='color: #666; margin: 5px 0;'>Active Alerts</p>
                        <p style='color: #e53e3e; margin: 0; font-weight: bold;'>{high_priority} High Priority</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show top alert
                    if alerts:
                        top_alert = alerts[0]
                        st.warning(f"⚠️ {top_alert['message'][:50]}...")
                else:
                    st.success("🎉 No alerts - you're doing great!")
                
                if st.button("🔍 View All Alerts", key="alerts_dash"):
                    st.info("💡 Switch to the Alerts tab for detailed information!")
            
            with dash_col3:
                # Goals Progress Card
                st.markdown("""
                <div style='background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h4 style='color: #43a047; margin-top: 0;'>🎯 Goals Progress</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if st.session_state.user_goals:
                    total_goals = len(st.session_state.user_goals)
                    completed_goals = len([g for g in st.session_state.user_goals if g['progress_percentage'] >= 100])
                    avg_progress = np.mean([g['progress_percentage'] for g in st.session_state.user_goals])
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px;'>
                        <h2 style='color: #43a047; margin: 0;'>{total_goals}</h2>
                        <p style='color: #666; margin: 5px 0;'>Active Goals</p>
                        <p style='color: #43a047; margin: 0; font-weight: bold;'>{avg_progress:.1f}% Avg Progress</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show next goal
                    next_goal = min(st.session_state.user_goals, key=lambda x: x['days_remaining'])
                    st.info(f"⏰ Next: {next_goal['name']} ({next_goal['days_remaining']} days)")
                else:
                    st.markdown("""
                    <div style='text-align: center; padding: 20px;'>
                        <p style='color: #666;'>No goals set yet</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.button("🎯 Manage Goals", key="goals_dash"):
                    st.info("💡 Switch to the Goals tab to create and manage your financial goals!")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Quick Actions Section
            st.markdown("""
            <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;'>
                <h3 style='color: white; text-align: center; margin: 0;'>⚡ Quick Actions</h3>
            </div>
            """, unsafe_allow_html=True)
            
            action_col1, action_col2, action_col3, action_col4 = st.columns(4)
            
            with action_col1:
                if st.button("💬 Ask AI Assistant", key="quick_chat"):
                    st.info("💡 Go to AI Chat tab to ask questions about your finances!")
            
            with action_col2:
                if st.button("📈 View Analytics", key="quick_analytics"):
                    st.info("💡 Go to Analytics tab for detailed spending analysis!")
            
            with action_col3:
                if st.button("🔮 See Predictions", key="quick_predictions"):
                    st.info("💡 Go to Predictions tab to forecast future spending!")
            
            with action_col4:
                if st.button("📅 Calendar View", key="quick_calendar"):
                    st.info("💡 Go to Calendar tab to see spending patterns by date!")
            
            # Recent Activity Section
            st.markdown("""
            <div style='margin-top: 30px;'>
                <h3 style='color: #667eea;'>📋 Recent Activity</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Show recent transactions in a nice format
            recent_transactions = df.head(5)
            
            for _, transaction in recent_transactions.iterrows():
                amount_color = "#e53e3e" if transaction['amount'] < 0 else "#43a047"
                amount_symbol = "💸" if transaction['amount'] < 0 else "💰"
                
                st.markdown(f"""
                <div style='background: white; border-left: 4px solid {amount_color}; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <strong>{transaction['description']}</strong><br>
                            <small style='color: #666;'>{transaction['date'].strftime('%Y-%m-%d') if hasattr(transaction['date'], 'strftime') else str(transaction['date'])}</small>
                        </div>
                        <div style='text-align: right;'>
                            <span style='font-size: 1.2em;'>{amount_symbol}</span>
                            <strong style='color: {amount_color}; font-size: 1.1em;'>{format_currency(abs(transaction['amount']))}</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab1:
            st.header("💬 Ask Your Finance Assistant")
            st.markdown("Ask questions about your spending habits in natural language!")
            
            # Chat interface
            user_question = st.text_input(
                "Ask a question about your finances:",
                placeholder="e.g., How much did I spend on food last month?",
                key="user_input"
            )
            
            if st.button("Ask") and user_question:
                with st.spinner("Analyzing your financial data..."):
                    # Get response from conversation handler
                    response = st.session_state.conversation_handler.get_response(
                        user_question, 
                        st.session_state.vector_store,
                        df
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": response,
                        "timestamp": datetime.now()
                    })
            
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("💭 Conversation History")
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                    with st.expander(f"Q: {chat['question'][:50]}...", expanded=(i==0)):
                        st.write(f"**Question:** {chat['question']}")
                        st.write(f"**Answer:** {chat['answer']}")
                        st.write(f"*{chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}*")
        
        with tab2:
            st.header("📈 Financial Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Spending by category pie chart
                st.subheader("Spending by Category")
                category_spending = df.groupby('category')['amount'].sum().abs()
                fig_pie = px.pie(
                    values=category_spending.values,
                    names=category_spending.index,
                    title="Expense Distribution by Category"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Monthly spending trend
                st.subheader("Monthly Spending Trend")
                monthly_spending = st.session_state.visualizer.create_monthly_trend(df)
                st.plotly_chart(monthly_spending, use_container_width=True)
            
            # Top spending categories
            st.subheader("📊 Top Spending Categories")
            top_categories = df.groupby('category')['amount'].sum().abs().sort_values(ascending=False).head(10)
            fig_bar = px.bar(
                x=top_categories.index,
                y=top_categories.values,
                title="Top 10 Spending Categories",
                labels={'x': 'Category', 'y': 'Amount'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab3:
            st.header("🔍 Transaction Explorer")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                categories = ['All'] + list(df['category'].unique())
                selected_category = st.selectbox("Filter by Category", categories)
            
            with col2:
                date_range = st.date_input(
                    "Date Range",
                    value=(df['date'].min(), df['date'].max()),
                    min_value=df['date'].min(),
                    max_value=df['date'].max()
                )
            
            with col3:
                amount_range = st.slider(
                    "Amount Range",
                    float(df['amount'].min()),
                    float(df['amount'].max()),
                    (float(df['amount'].min()), float(df['amount'].max()))
                )
            
            # Filter data
            filtered_df = df.copy()
            
            if selected_category != 'All':
                filtered_df = filtered_df[filtered_df['category'] == selected_category]
            
            if len(date_range) == 2:
                filtered_df = filtered_df[
                    (filtered_df['date'] >= pd.to_datetime(date_range[0])) &
                    (filtered_df['date'] <= pd.to_datetime(date_range[1]))
                ]
            
            filtered_df = filtered_df[
                (filtered_df['amount'] >= amount_range[0]) &
                (filtered_df['amount'] <= amount_range[1])
            ]
            
            # Display filtered transactions
            st.subheader(f"📋 Transactions ({len(filtered_df)} found)")
            
            if len(filtered_df) > 0:
                # Format for display
                display_df = filtered_df.copy()
                display_df['amount'] = display_df['amount'].apply(format_currency)
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                
                st.dataframe(
                    display_df[['date', 'description', 'amount', 'category']],
                    use_container_width=True
                )
                
                # Download filtered data
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=csv,
                    file_name=f"filtered_transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No transactions found matching the selected filters.")
        
        with tab4:
            # Enhanced Health Score Layout
            st.markdown("""
            <div style='text-align: center; margin-bottom: 30px;'>
                <h2 style='color: #667eea;'>🏥 Financial Health Analysis</h2>
                <p style='color: #666; font-size: 1.1em;'>Comprehensive assessment of your financial wellness</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate health score
            health_data = st.session_state.health_analyzer.calculate_health_score(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Health Score Card
                st.markdown("""
                <div style='background: white; border: 1px solid #e0e0e0; border-radius: 15px; padding: 25px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h4 style='color: #667eea; margin-top: 0; text-align: center;'>📊 Overall Health Score</h4>
                </div>
                """, unsafe_allow_html=True)
                
                health_fig = st.session_state.health_analyzer.create_health_score_visualization(health_data)
                st.plotly_chart(health_fig, use_container_width=True)
                
                # Health insights in cards
                st.markdown("""
                <div style='background: white; border: 1px solid #e0e0e0; border-radius: 15px; padding: 25px; margin: 15px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h4 style='color: #4facfe; margin-top: 0;'>💡 Key Insights</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for insight in health_data['insights']:
                    st.markdown(f"""
                    <div style='background: #f8f9fa; border-left: 4px solid #4facfe; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                        <p style='margin: 0; color: #333;'>{insight}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Component scores card
                st.markdown("""
                <div style='background: white; border: 1px solid #e0e0e0; border-radius: 15px; padding: 25px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h4 style='color: #667eea; margin-top: 0; text-align: center;'>🎯 Component Analysis</h4>
                </div>
                """, unsafe_allow_html=True)
                
                component_fig = st.session_state.health_analyzer.create_component_scores_chart(health_data['component_scores'])
                st.plotly_chart(component_fig, use_container_width=True)
                
                # Improvement tips in cards
                st.markdown("""
                <div style='background: white; border: 1px solid #e0e0e0; border-radius: 15px; padding: 25px; margin: 15px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h4 style='color: #43a047; margin-top: 0;'>🚀 Improvement Tips</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for tip in health_data['improvement_tips']:
                    st.markdown(f"""
                    <div style='background: #f1f8e9; border-left: 4px solid #43a047; padding: 15px; margin: 10px 0; border-radius: 5px;'>
                        <p style='margin: 0; color: #333;'>{tip}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab5:
            st.header("🔮 Financial Predictions")
            
            # Spending predictions
            predictions = st.session_state.predictive_analytics.predict_future_spending(df, months_ahead=3)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Prediction chart
                pred_fig = st.session_state.predictive_analytics.create_prediction_chart(df, predictions)
                st.plotly_chart(pred_fig, use_container_width=True)
                
                # Prediction summary
                st.subheader("📊 Forecast Summary")
                if predictions['predictions']:
                    avg_prediction = np.mean(predictions['predictions'])
                    st.metric("Avg Monthly Spending (Next 3 months)", format_currency(avg_prediction))
                    st.metric("Trend Direction", predictions['trend_direction'].title())
                    st.metric("Confidence Level", f"{predictions['confidence_level']:.1f}%")
            
            with col2:
                # Category predictions
                category_predictions = st.session_state.predictive_analytics.predict_category_spending(df)
                
                if category_predictions:
                    st.subheader("📈 Category Forecasts")
                    for category, pred_data in list(category_predictions.items())[:5]:
                        st.write(f"**{category}**")
                        st.write(f"Predicted: {format_currency(pred_data['predicted_amount'])}")
                        st.write(f"Trend: {pred_data['trend']} ({pred_data['confidence']:.0f}% confidence)")
                        st.write("---")
                
                # Anomalies
                anomalies = st.session_state.predictive_analytics.detect_spending_anomalies(df)
                if anomalies:
                    st.subheader("⚠️ Detected Anomalies")
                    for anomaly in anomalies[:3]:
                        severity_color = "🔴" if anomaly['severity'] == 'high' else "🟡"
                        st.write(f"{severity_color} {anomaly['type'].replace('_', ' ').title()}")
                        if 'month' in anomaly:
                            st.write(f"Month: {anomaly['month']}")
                        st.write(f"Amount: {format_currency(anomaly['amount'])}")
                        st.write("---")
        
        with tab6:
            # Enhanced Alerts Layout
            st.markdown("""
            <div style='text-align: center; margin-bottom: 30px;'>
                <h2 style='color: #f5576c;'>🚨 Smart Financial Alerts</h2>
                <p style='color: #666; font-size: 1.1em;'>Proactive notifications to keep your finances on track</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate and display alerts
            alerts = st.session_state.smart_alerts.generate_alerts(df, st.session_state.budgets)
            
            if alerts:
                # Alert summary cards at top
                alert_summary = st.session_state.smart_alerts.create_alerts_summary(alerts)
                
                sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                
                with sum_col1:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%); padding: 20px; border-radius: 10px; text-align: center; margin: 5px;'>
                        <h3 style='color: white; margin: 0;'>🚨</h3>
                        <h2 style='color: white; margin: 10px 0;'>{alert_summary['total_alerts']}</h2>
                        <p style='color: white; margin: 0; opacity: 0.9;'>Total Alerts</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with sum_col2:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #e53e3e 0%, #ff6b35 100%); padding: 20px; border-radius: 10px; text-align: center; margin: 5px;'>
                        <h3 style='color: white; margin: 0;'>🔴</h3>
                        <h2 style='color: white; margin: 10px 0;'>{alert_summary['high_severity']}</h2>
                        <p style='color: white; margin: 0; opacity: 0.9;'>High Priority</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with sum_col3:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #ff9800 0%, #ffb74d 100%); padding: 20px; border-radius: 10px; text-align: center; margin: 5px;'>
                        <h3 style='color: white; margin: 0;'>🟡</h3>
                        <h2 style='color: white; margin: 10px 0;'>{alert_summary['medium_severity']}</h2>
                        <p style='color: white; margin: 0; opacity: 0.9;'>Medium Priority</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with sum_col4:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #2196f3 0%, #64b5f6 100%); padding: 20px; border-radius: 10px; text-align: center; margin: 5px;'>
                        <h3 style='color: white; margin: 0;'>🔵</h3>
                        <h2 style='color: white; margin: 10px 0;'>{alert_summary['low_severity']}</h2>
                        <p style='color: white; margin: 0; opacity: 0.9;'>Low Priority</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Alert details in cards
                st.markdown("""
                <div style='background: white; border: 1px solid #e0e0e0; border-radius: 15px; padding: 25px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h4 style='color: #f5576c; margin-top: 0;'>📋 Alert Details</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Display alerts with better styling
                for alert in alerts:
                    severity_colors = {
                        'high': '#ffebee',
                        'medium': '#fff3e0', 
                        'low': '#e3f2fd'
                    }
                    border_colors = {
                        'high': '#f44336',
                        'medium': '#ff9800',
                        'low': '#2196f3'
                    }
                    
                    severity = alert.get('severity', 'low')
                    bg_color = severity_colors.get(severity, '#f5f5f5')
                    border_color = border_colors.get(severity, '#ddd')
                    
                    st.markdown(f"""
                    <div style='background: {bg_color}; border-left: 4px solid {border_color}; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                        <p style='margin: 0; color: #333; font-weight: 500;'>{alert['message']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align: center; padding: 50px; background: linear-gradient(135deg, #43a047 0%, #66bb6a 100%); border-radius: 15px; margin: 20px 0;'>
                    <h1 style='color: white; margin: 0; font-size: 4em;'>🎉</h1>
                    <h2 style='color: white; margin: 20px 0;'>All Clear!</h2>
                    <p style='color: white; opacity: 0.9; font-size: 1.2em;'>No alerts detected - your finances look healthy!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Budget Management Card
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='background: white; border: 1px solid #e0e0e0; border-radius: 15px; padding: 25px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h4 style='color: #43a047; margin-top: 0;'>💰 Budget Management</h4>
            </div>
            """, unsafe_allow_html=True)
            
            budget_col1, budget_col2 = st.columns(2)
            
            with budget_col1:
                st.markdown("##### Set New Budget")
                if 'category' in df.columns:
                    categories = df['category'].unique()
                    selected_category = st.selectbox("Category", categories, key="budget_category")
                    budget_amount = st.number_input(f"Monthly Budget ($)", min_value=0.0, step=50.0, key="budget_amount")
                    
                    if st.button("💾 Set Budget", key="set_budget_btn"):
                        st.session_state.budgets[selected_category] = budget_amount
                        st.success(f"✅ Budget set for {selected_category}: {format_currency(budget_amount)}")
                        st.rerun()
            
            with budget_col2:
                st.markdown("##### Current Budgets")
                if st.session_state.budgets:
                    for category, amount in st.session_state.budgets.items():
                        st.markdown(f"""
                        <div style='background: #f8f9fa; border: 1px solid #e9ecef; padding: 10px; margin: 5px 0; border-radius: 5px; display: flex; justify-content: space-between;'>
                            <span style='font-weight: 500;'>{category}</span>
                            <span style='color: #43a047; font-weight: bold;'>{format_currency(amount)}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No budgets set yet. Set your first budget to get started!")
        
        with tab7:
            st.header("🎯 Financial Goals")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("➕ Create New Goal")
                
                goal_name = st.text_input("Goal Name", placeholder="e.g., Emergency Fund")
                goal_type = st.selectbox("Goal Type", st.session_state.goal_tracker.goal_types)
                target_amount = st.number_input("Target Amount ($)", min_value=0.0, step=100.0)
                target_date = st.date_input("Target Date", min_value=datetime.now().date())
                current_amount = st.number_input("Current Amount ($)", min_value=0.0, step=50.0)
                
                if st.button("Create Goal"):
                    new_goal = st.session_state.goal_tracker.create_goal(
                        goal_name, target_amount, target_date, goal_type, current_amount
                    )
                    st.session_state.user_goals.append(new_goal)
                    st.success(f"Goal '{goal_name}' created successfully!")
                    st.rerun()
            
            with col2:
                st.subheader("📋 Your Goals")
                
                if st.session_state.user_goals:
                    for i, goal in enumerate(st.session_state.user_goals):
                        with st.expander(f"{goal['name']} ({goal['progress_percentage']:.1f}% complete)"):
                            
                            # Goal progress chart
                            progress_fig = st.session_state.goal_tracker.create_goal_progress_chart(goal)
                            st.plotly_chart(progress_fig, use_container_width=True)
                            
                            # Goal details
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"**Target:** {format_currency(goal['target_amount'])}")
                                st.write(f"**Current:** {format_currency(goal['current_amount'])}")
                                st.write(f"**Target Date:** {goal['target_date']}")
                            with col_b:
                                st.write(f"**Status:** {goal['status']}")
                                st.write(f"**Days Remaining:** {goal['days_remaining']}")
                                st.write(f"**Monthly Savings Needed:** {format_currency(goal['monthly_savings_needed'])}")
                            
                            # Feasibility analysis
                            feasibility = st.session_state.goal_tracker.analyze_goal_feasibility(goal, df)
                            st.write(f"**Feasibility:** {feasibility['feasibility']}")
                            st.write(f"**Success Probability:** {feasibility['success_probability']:.1f}%")
                            
                            # Remove goal button
                            if st.button(f"Remove Goal", key=f"remove_{i}"):
                                st.session_state.user_goals.pop(i)
                                st.rerun()
                else:
                    st.info("No goals set yet. Create your first goal to get started!")
            
            # Goal strategies
            if st.session_state.user_goals:
                st.subheader("💡 Savings Strategies")
                for goal in st.session_state.user_goals[:2]:  # Show strategies for first 2 goals
                    strategies = st.session_state.goal_tracker.generate_savings_strategies(goal, df)
                    if strategies:
                        st.write(f"**Strategies for {goal['name']}:**")
                        strategy_fig = st.session_state.goal_tracker.create_savings_strategy_chart(strategies)
                        st.plotly_chart(strategy_fig, use_container_width=True)
                        
                        for strategy in strategies[:3]:
                            st.write(f"• **{strategy['strategy']}**: {strategy['description']}")
        
        with tab8:
            st.header("📅 Financial Calendar")
            
            # Month/Year selector
            col1, col2 = st.columns(2)
            with col1:
                selected_year = st.selectbox("Year", range(2020, 2030), index=4)  # Default to 2024
            with col2:
                selected_month = st.selectbox("Month", range(1, 13), index=datetime.now().month-1)
            
            # Generate calendar data
            calendar_data = st.session_state.financial_calendar.create_calendar_view(df, selected_year, selected_month)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Calendar heatmap
                calendar_fig = st.session_state.financial_calendar.create_calendar_heatmap(calendar_data)
                st.plotly_chart(calendar_fig, use_container_width=True)
                
                # Month statistics
                st.subheader("📊 Month Statistics")
                month_stats = calendar_data['month_stats']
                st.metric("Total Spending", format_currency(month_stats['total_spending']))
                st.metric("Total Income", format_currency(month_stats['total_income']))
                st.metric("Net Flow", format_currency(month_stats['net_flow']))
                st.metric("Transaction Count", month_stats['transaction_count'])
                st.metric("Most Active Day", month_stats['most_active_day'])
            
            with col2:
                # Spending patterns
                if calendar_data['patterns']:
                    pattern_fig = st.session_state.financial_calendar.create_spending_pattern_chart(calendar_data['patterns'])
                    st.plotly_chart(pattern_fig, use_container_width=True)
                
                # Calendar insights
                insights = st.session_state.financial_calendar.get_calendar_insights(calendar_data)
                st.subheader("💡 Calendar Insights")
                for insight in insights:
                    st.info(insight)
            
            # Daily cash flow
            st.subheader("💰 Daily Cash Flow")
            flow_fig = st.session_state.financial_calendar.create_monthly_flow_chart(df)
            st.plotly_chart(flow_fig, use_container_width=True)
    
    else:
        # Enhanced welcome screen with proper branding
        st.markdown("""
        <div style='text-align: center; padding: 50px 20px; margin: 50px 0;'>
            <h1 style='color: #667eea; font-size: 4em; margin: 0; font-weight: 300;'>🚀 AI FinancePilot</h1>
            <h2 style='color: #764ba2; margin: 30px 0; font-weight: 300; font-size: 2em;'>Your Intelligent Financial Co-Pilot</h2>
            <p style='color: #666; font-size: 1.4em; margin: 40px 0; max-width: 800px; margin-left: auto; margin-right: auto; line-height: 1.6;'>
                Transform your financial data into actionable insights with revolutionary AI-powered analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Getting Started Card
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin: 40px 0; text-align: center;'>
            <h3 style='margin: 0 0 20px 0; font-size: 1.8em;'>👆 Get Started</h3>
            <p style='margin: 0; opacity: 0.9; font-size: 1.2em;'>Upload your bank statement using the sidebar to unlock powerful financial insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Requirements and Features
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background: white; border: 1px solid #e0e0e0; border-radius: 15px; padding: 25px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h4 style='color: #667eea; margin-top: 0;'>📋 File Requirements</h4>
                <p style='margin: 0; color: #333;'>Your CSV/Excel file should contain:</p>
                <ul style='color: #666; margin: 15px 0;'>
                    <li><strong>Date</strong> column (transaction dates)</li>
                    <li><strong>Description</strong> column (transaction details)</li>
                    <li><strong>Amount</strong> column (transaction amounts)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: white; border: 1px solid #e0e0e0; border-radius: 15px; padding: 25px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h4 style='color: #43a047; margin-top: 0;'>🎯 What You Can Do</h4>
                <ul style='color: #666; margin: 15px 0;'>
                    <li>🤖 <strong>AI Chat</strong>: Natural language queries</li>
                    <li>📊 <strong>Analytics</strong>: Beautiful charts & insights</li>
                    <li>🏥 <strong>Health Score</strong>: Financial wellness rating</li>
                    <li>🔮 <strong>Predictions</strong>: Future spending forecasts</li>
                    <li>🚨 <strong>Smart Alerts</strong>: Proactive notifications</li>
                    <li>🎯 <strong>Goals</strong>: Savings strategy recommendations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Example Questions Card
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 25px; border-radius: 15px; margin: 30px 0;'>
            <h4 style='margin: 0 0 20px 0; text-align: center;'>💬 Example Questions You Can Ask</h4>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 0;'>
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
                    <p style='margin: 0; opacity: 0.9;'>"How much did I spend on food last month?"</p>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
                    <p style='margin: 0; opacity: 0.9;'>"What's my biggest expense category?"</p>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
                    <p style='margin: 0; opacity: 0.9;'>"Show me my travel expenses for the last 6 months"</p>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
                    <p style='margin: 0; opacity: 0.9;'>"How does my spending compare to last year?"</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
