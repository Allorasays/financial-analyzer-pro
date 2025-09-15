"""
Simple database test without Streamlit
"""

from database_simple import DatabaseManager

def test_database_simple():
    """Test database functionality without Streamlit"""
    print("🗄️ Testing Financial Analyzer Pro Database...")
    
    # Initialize database
    db = DatabaseManager()
    print("✅ Database initialized successfully")
    
    # Test database stats
    stats = db.get_database_stats()
    print(f"📊 Database stats: {stats}")
    
    # Test user creation
    print("\n👤 Testing user creation...")
    success = db.create_user("testuser", "test@example.com", "testpass123", "Test User")
    if success:
        print("✅ User created successfully")
    else:
        print("❌ User creation failed (may already exist)")
    
    # Test user authentication
    print("\n🔐 Testing user authentication...")
    user = db.authenticate_user("testuser", "testpass123")
    if user:
        print(f"✅ Authentication successful! User: {user['full_name']}")
        user_id = user['id']
    else:
        print("❌ Authentication failed")
        return
    
    # Test preferences
    print("\n⚙️ Testing user preferences...")
    prefs = db.get_user_preferences(user_id)
    print(f"Current preferences: {prefs}")
    
    # Update preferences
    new_prefs = {
        'default_timeframe': '6mo',
        'favorite_indicators': ['RSI', 'MACD', 'Bollinger'],
        'risk_tolerance': 'High',
        'theme': 'Dark'
    }
    
    success = db.update_user_preferences(user_id, new_prefs)
    if success:
        print("✅ Preferences updated successfully")
        updated_prefs = db.get_user_preferences(user_id)
        print(f"Updated preferences: {updated_prefs}")
    else:
        print("❌ Failed to update preferences")
    
    # Test portfolio operations
    print("\n💼 Testing portfolio operations...")
    portfolios = db.get_user_portfolios(user_id)
    print(f"User portfolios: {portfolios}")
    
    # Create test portfolio
    success = db.create_portfolio(user_id, "Test Portfolio", "A test portfolio")
    if success:
        print("✅ Portfolio created successfully")
        portfolios = db.get_user_portfolios(user_id)
        print(f"Updated portfolios: {portfolios}")
        
        # Add test position
        if portfolios:
            portfolio_id = portfolios[0]['id']
            success = db.add_position_to_portfolio(
                portfolio_id, "AAPL", 10, 150.00, "Test position"
            )
            if success:
                print("✅ Position added successfully")
                positions = db.get_portfolio_positions(portfolio_id)
                print(f"Portfolio positions: {positions}")
            else:
                print("❌ Failed to add position")
    else:
        print("❌ Failed to create portfolio")
    
    # Test watchlist operations
    print("\n👀 Testing watchlist operations...")
    watchlists = db.get_user_watchlists(user_id)
    print(f"User watchlists: {watchlists}")
    
    # Update watchlist
    if watchlists:
        watchlist_id = watchlists[0]['id']
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        success = db.update_watchlist(watchlist_id, symbols)
        if success:
            print("✅ Watchlist updated successfully")
            watchlists = db.get_user_watchlists(user_id)
            print(f"Updated watchlists: {watchlists}")
        else:
            print("❌ Failed to update watchlist")
    
    # Final stats
    print("\n📈 Final database statistics:")
    final_stats = db.get_database_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 Database test completed successfully!")

if __name__ == "__main__":
    test_database_simple()
