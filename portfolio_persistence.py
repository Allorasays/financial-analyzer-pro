"""
Portfolio Persistence System for Financial Analyzer Pro - Phase 3
Handles saving, loading, and managing user portfolios
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from database_manager import DatabaseManager
from auth_system import AuthSystem

class PortfolioPersistence:
    def __init__(self):
        self.db = DatabaseManager()
        self.auth = AuthSystem()
    
    def get_user_portfolios(self) -> List[Dict]:
        """Get all portfolios for current user"""
        if not self.auth.is_authenticated():
            return []
        
        user_id = self.auth.get_current_user()['id']
        return self.db.get_user_portfolios(user_id)
    
    def create_portfolio(self, name: str, description: str = None, is_public: bool = False) -> Optional[int]:
        """Create a new portfolio for current user"""
        if not self.auth.is_authenticated():
            st.error("Please sign in to create portfolios")
            return None
        
        user_id = self.auth.get_current_user()['id']
        portfolio_id = self.db.create_portfolio(user_id, name, description, is_public)
        
        if portfolio_id:
            st.success(f"Portfolio '{name}' created successfully!")
            return portfolio_id
        else:
            st.error("Failed to create portfolio. Please try again.")
            return None
    
    def save_portfolio_data(self, portfolio_id: int, portfolio_data: pd.DataFrame) -> bool:
        """Save portfolio data to database"""
        try:
            # Clear existing positions
            conn = self.db.db_path
            # This would be implemented in database_manager.py
            
            # Add new positions
            for _, row in portfolio_data.iterrows():
                self.db.add_portfolio_position(
                    portfolio_id=portfolio_id,
                    symbol=row['Symbol'],
                    shares=row['Shares'],
                    purchase_price=row['Purchase Price'],
                    purchase_date=row.get('Purchase Date', datetime.now().strftime('%Y-%m-%d')),
                    notes=row.get('Notes', '')
                )
            
            return True
            
        except Exception as e:
            st.error(f"Failed to save portfolio: {str(e)}")
            return False
    
    def load_portfolio_data(self, portfolio_id: int) -> Optional[pd.DataFrame]:
        """Load portfolio data from database"""
        try:
            positions = self.db.get_portfolio_positions(portfolio_id)
            
            if not positions:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for pos in positions:
                data.append({
                    'Symbol': pos['symbol'],
                    'Shares': pos['shares'],
                    'Purchase Price': pos['purchase_price'],
                    'Purchase Date': pos['purchase_date'],
                    'Notes': pos['notes']
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            st.error(f"Failed to load portfolio: {str(e)}")
            return None
    
    def show_portfolio_manager(self):
        """Display portfolio management interface"""
        st.header("💼 Portfolio Manager")
        
        if not self.auth.is_authenticated():
            st.warning("Please sign in to manage your portfolios")
            return
        
        # Portfolio management tabs
        tab1, tab2, tab3 = st.tabs(["My Portfolios", "Create Portfolio", "Import/Export"])
        
        with tab1:
            self.show_my_portfolios()
        
        with tab2:
            self.show_create_portfolio()
        
        with tab3:
            self.show_import_export()
    
    def show_my_portfolios(self):
        """Display user's portfolios"""
        portfolios = self.get_user_portfolios()
        
        if not portfolios:
            st.info("You don't have any portfolios yet. Create one in the 'Create Portfolio' tab!")
            return
        
        # Portfolio selection
        portfolio_names = [f"{p['name']} ({p['created_at'][:10]})" for p in portfolios]
        selected_idx = st.selectbox("Select Portfolio", range(len(portfolio_names)), 
                                  format_func=lambda x: portfolio_names[x])
        
        if selected_idx is not None:
            selected_portfolio = portfolios[selected_idx]
            st.subheader(f"📊 {selected_portfolio['name']}")
            
            if selected_portfolio['description']:
                st.write(f"**Description:** {selected_portfolio['description']}")
            
            # Load and display portfolio data
            portfolio_data = self.load_portfolio_data(selected_portfolio['id'])
            
            if not portfolio_data.empty:
                # Calculate current values (simplified)
                portfolio_data['Current Price'] = 0.0  # Would be fetched from yfinance
                portfolio_data['Value'] = portfolio_data['Shares'] * portfolio_data['Current Price']
                portfolio_data['P&L'] = portfolio_data['Value'] - (portfolio_data['Shares'] * portfolio_data['Purchase Price'])
                portfolio_data['P&L %'] = (portfolio_data['P&L'] / (portfolio_data['Shares'] * portfolio_data['Purchase Price'])) * 100
                
                # Display portfolio
                st.dataframe(portfolio_data, use_container_width=True)
                
                # Portfolio summary
                total_value = portfolio_data['Value'].sum()
                total_cost = (portfolio_data['Shares'] * portfolio_data['Purchase Price']).sum()
                total_pnl = total_value - total_cost
                total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Value", f"${total_value:,.2f}")
                with col2:
                    st.metric("Total Cost", f"${total_cost:,.2f}")
                with col3:
                    st.metric("Total P&L", f"${total_pnl:,.2f}")
                with col4:
                    st.metric("P&L %", f"{total_pnl_pct:.2f}%")
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Add Position", key=f"add_{selected_portfolio['id']}"):
                        st.session_state.add_position_portfolio = selected_portfolio['id']
                        st.rerun()
                
                with col2:
                    if st.button("Export Portfolio", key=f"export_{selected_portfolio['id']}"):
                        self.export_portfolio(selected_portfolio['id'], portfolio_data)
                
                with col3:
                    if st.button("Delete Portfolio", key=f"delete_{selected_portfolio['id']}"):
                        st.session_state.delete_portfolio = selected_portfolio['id']
                        st.rerun()
            else:
                st.info("This portfolio is empty. Add some positions to get started!")
    
    def show_create_portfolio(self):
        """Display portfolio creation form"""
        st.subheader("📝 Create New Portfolio")
        
        with st.form("create_portfolio_form"):
            name = st.text_input("Portfolio Name", placeholder="e.g., My Growth Portfolio")
            description = st.text_area("Description (Optional)", placeholder="Describe your investment strategy...")
            is_public = st.checkbox("Make this portfolio public", help="Other users can view this portfolio")
            
            if st.form_submit_button("Create Portfolio", type="primary"):
                if name:
                    portfolio_id = self.create_portfolio(name, description, is_public)
                    if portfolio_id:
                        st.session_state.current_portfolio = portfolio_id
                        st.rerun()
                else:
                    st.error("Please enter a portfolio name")
    
    def show_import_export(self):
        """Display import/export functionality"""
        st.subheader("📥 Import/Export Portfolios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Export Portfolio**")
            portfolios = self.get_user_portfolios()
            if portfolios:
                portfolio_names = [p['name'] for p in portfolios]
                selected_portfolio = st.selectbox("Select Portfolio to Export", portfolio_names)
                
                if st.button("Export as CSV"):
                    portfolio_data = self.load_portfolio_data(
                        next(p['id'] for p in portfolios if p['name'] == selected_portfolio)
                    )
                    if not portfolio_data.empty:
                        csv = portfolio_data.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{selected_portfolio}_portfolio.csv",
                            mime="text/csv"
                        )
            else:
                st.info("No portfolios to export")
        
        with col2:
            st.write("**Import Portfolio**")
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Validate required columns
                    required_cols = ['Symbol', 'Shares', 'Purchase Price']
                    if all(col in df.columns for col in required_cols):
                        st.success("File uploaded successfully!")
                        st.dataframe(df.head())
                        
                        portfolio_name = st.text_input("Portfolio Name", value=f"Imported_{datetime.now().strftime('%Y%m%d')}")
                        
                        if st.button("Import Portfolio"):
                            portfolio_id = self.create_portfolio(portfolio_name, "Imported portfolio")
                            if portfolio_id:
                                # Add positions
                                for _, row in df.iterrows():
                                    self.db.add_portfolio_position(
                                        portfolio_id=portfolio_id,
                                        symbol=row['Symbol'],
                                        shares=row['Shares'],
                                        purchase_price=row['Purchase Price'],
                                        purchase_date=row.get('Purchase Date', datetime.now().strftime('%Y-%m-%d')),
                                        notes=row.get('Notes', '')
                                    )
                                st.success("Portfolio imported successfully!")
                                st.rerun()
                    else:
                        st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                        
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
    
    def export_portfolio(self, portfolio_id: int, portfolio_data: pd.DataFrame):
        """Export portfolio data"""
        if not portfolio_data.empty:
            csv = portfolio_data.to_csv(index=False)
            st.download_button(
                label="Download Portfolio CSV",
                data=csv,
                file_name=f"portfolio_{portfolio_id}.csv",
                mime="text/csv"
            )
    
    def show_watchlist_manager(self):
        """Display watchlist management interface"""
        st.header("👀 Watchlist Manager")
        
        if not self.auth.is_authenticated():
            st.warning("Please sign in to manage your watchlists")
            return
        
        # Watchlist management tabs
        tab1, tab2 = st.tabs(["My Watchlists", "Create Watchlist"])
        
        with tab1:
            self.show_my_watchlists()
        
        with tab2:
            self.show_create_watchlist()
    
    def show_my_watchlists(self):
        """Display user's watchlists"""
        if not self.auth.is_authenticated():
            return
        
        user_id = self.auth.get_current_user()['id']
        watchlists = self.db.get_user_watchlists(user_id)
        
        if not watchlists:
            st.info("You don't have any watchlists yet. Create one in the 'Create Watchlist' tab!")
            return
        
        # Watchlist selection
        watchlist_names = [f"{w['name']} ({len(w['symbols'])} symbols)" for w in watchlists]
        selected_idx = st.selectbox("Select Watchlist", range(len(watchlist_names)), 
                                  format_func=lambda x: watchlist_names[x])
        
        if selected_idx is not None:
            selected_watchlist = watchlists[selected_idx]
            st.subheader(f"👀 {selected_watchlist['name']}")
            
            # Display symbols
            if selected_watchlist['symbols']:
                st.write("**Symbols:**")
                for symbol in selected_watchlist['symbols']:
                    st.write(f"• {symbol}")
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Edit Watchlist", key=f"edit_{selected_watchlist['id']}"):
                        st.session_state.edit_watchlist = selected_watchlist['id']
                        st.rerun()
                
                with col2:
                    if st.button("Delete Watchlist", key=f"delete_{selected_watchlist['id']}"):
                        st.session_state.delete_watchlist = selected_watchlist['id']
                        st.rerun()
            else:
                st.info("This watchlist is empty. Add some symbols to get started!")
    
    def show_create_watchlist(self):
        """Display watchlist creation form"""
        st.subheader("📝 Create New Watchlist")
        
        with st.form("create_watchlist_form"):
            name = st.text_input("Watchlist Name", placeholder="e.g., Tech Stocks")
            symbols_text = st.text_area("Symbols (one per line)", placeholder="AAPL\nMSFT\nGOOGL\nTSLA")
            
            if st.form_submit_button("Create Watchlist", type="primary"):
                if name and symbols_text:
                    symbols = [s.strip().upper() for s in symbols_text.split('\n') if s.strip()]
                    
                    if symbols:
                        user_id = self.auth.get_current_user()['id']
                        watchlist_id = self.db.create_watchlist(user_id, name, symbols)
                        
                        if watchlist_id:
                            st.success(f"Watchlist '{name}' created with {len(symbols)} symbols!")
                            st.rerun()
                        else:
                            st.error("Failed to create watchlist. Please try again.")
                    else:
                        st.error("Please enter at least one symbol")
                else:
                    st.error("Please enter both name and symbols")
