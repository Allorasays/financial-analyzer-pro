"""
Test script for database functionality
Run this to verify database operations work correctly
"""

import streamlit as st
from database import DatabaseManager

def test_database():
    """Test database functionality"""
    st.title("🗄️ Database Test - Financial Analyzer Pro")
    
    # Initialize database
    db = DatabaseManager()
    
    # Test database stats
    st.subheader("📊 Database Statistics")
    stats = db.get_database_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Users", stats.get('users_count', 0))
    with col2:
        st.metric("Portfolios", stats.get('portfolios_count', 0))
    with col3:
        st.metric("Watchlists", stats.get('watchlists_count', 0))
    
    # Test user creation
    st.subheader("👤 Test User Creation")
    
    with st.form("test_user_form"):
        username = st.text_input("Username", value="testuser")
        email = st.text_input("Email", value="test@example.com")
        password = st.text_input("Password", type="password", value="testpass123")
        full_name = st.text_input("Full Name", value="Test User")
        
        if st.form_submit_button("Create Test User"):
            success = db.create_user(username, email, password, full_name)
            if success:
                st.success("✅ User created successfully!")
            else:
                st.error("❌ Failed to create user (may already exist)")
    
    # Test user authentication
    st.subheader("🔐 Test User Authentication")
    
    with st.form("test_auth_form"):
        auth_username = st.text_input("Username", value="testuser", key="auth_username")
        auth_password = st.text_input("Password", type="password", value="testpass123", key="auth_password")
        
        if st.form_submit_button("Test Authentication"):
            user = db.authenticate_user(auth_username, auth_password)
            if user:
                st.success(f"✅ Authentication successful! Welcome {user['full_name']}")
                st.json(user)
            else:
                st.error("❌ Authentication failed")
    
    # Test preferences
    st.subheader("⚙️ Test User Preferences")
    
    if st.button("Test Preferences Operations"):
        # Get test user ID
        user = db.authenticate_user("testuser", "testpass123")
        if user:
            user_id = user['id']
            
            # Get preferences
            prefs = db.get_user_preferences(user_id)
            st.write("Current preferences:", prefs)
            
            # Update preferences
            new_prefs = {
                'default_timeframe': '6mo',
                'favorite_indicators': ['RSI', 'MACD', 'Bollinger'],
                'risk_tolerance': 'High',
                'theme': 'Dark'
            }
            
            success = db.update_user_preferences(user_id, new_prefs)
            if success:
                st.success("✅ Preferences updated successfully!")
                
                # Get updated preferences
                updated_prefs = db.get_user_preferences(user_id)
                st.write("Updated preferences:", updated_prefs)
            else:
                st.error("❌ Failed to update preferences")
        else:
            st.error("❌ Please create a test user first")
    
    # Test portfolio operations
    st.subheader("💼 Test Portfolio Operations")
    
    if st.button("Test Portfolio Operations"):
        user = db.authenticate_user("testuser", "testpass123")
        if user:
            user_id = user['id']
            
            # Get portfolios
            portfolios = db.get_user_portfolios(user_id)
            st.write("User portfolios:", portfolios)
            
            # Create test portfolio
            success = db.create_portfolio(user_id, "Test Portfolio", "A test portfolio")
            if success:
                st.success("✅ Portfolio created successfully!")
                
                # Get updated portfolios
                portfolios = db.get_user_portfolios(user_id)
                st.write("Updated portfolios:", portfolios)
                
                # Add test position
                if portfolios:
                    portfolio_id = portfolios[0]['id']
                    success = db.add_position_to_portfolio(
                        portfolio_id, "AAPL", 10, 150.00, "Test position"
                    )
                    if success:
                        st.success("✅ Position added successfully!")
                        
                        # Get portfolio positions
                        positions = db.get_portfolio_positions(portfolio_id)
                        st.write("Portfolio positions:", positions)
                    else:
                        st.error("❌ Failed to add position")
            else:
                st.error("❌ Failed to create portfolio")
        else:
            st.error("❌ Please create a test user first")
    
    # Test watchlist operations
    st.subheader("👀 Test Watchlist Operations")
    
    if st.button("Test Watchlist Operations"):
        user = db.authenticate_user("testuser", "testpass123")
        if user:
            user_id = user['id']
            
            # Get watchlists
            watchlists = db.get_user_watchlists(user_id)
            st.write("User watchlists:", watchlists)
            
            # Update watchlist
            if watchlists:
                watchlist_id = watchlists[0]['id']
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
                success = db.update_watchlist(watchlist_id, symbols)
                if success:
                    st.success("✅ Watchlist updated successfully!")
                    
                    # Get updated watchlist
                    watchlists = db.get_user_watchlists(user_id)
                    st.write("Updated watchlists:", watchlists)
                else:
                    st.error("❌ Failed to update watchlist")
        else:
            st.error("❌ Please create a test user first")
    
    # Final stats
    st.subheader("📈 Final Database Statistics")
    final_stats = db.get_database_stats()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Users", final_stats.get('users_count', 0))
    with col2:
        st.metric("Portfolios", final_stats.get('portfolios_count', 0))
    with col3:
        st.metric("Positions", final_stats.get('portfolio_positions_count', 0))
    with col4:
        st.metric("Watchlists", final_stats.get('watchlists_count', 0))
    with col5:
        st.metric("Templates", final_stats.get('analysis_templates_count', 0))

if __name__ == "__main__":
    test_database()
