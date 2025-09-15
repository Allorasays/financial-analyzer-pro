import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedFinancialAnalytics:
    def __init__(self):
        self.risk_free_rate = 0.04  # 4% risk-free rate
        self.market_return = 0.10   # 10% expected market return
    
    def get_enhanced_financial_data(self, symbol: str) -> Dict:
        """Get comprehensive financial data for analysis"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            # Get historical data
            hist = ticker.history(period="5y")
            
            # Get financial statements
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            
            # Get quarterly data
            quarterly_financials = ticker.quarterly_financials
            quarterly_balance_sheet = ticker.quarterly_balance_sheet
            quarterly_cashflow = ticker.quarterly_cashflow
            
            return {
                'info': info,
                'historical': hist,
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cashflow': cashflow,
                'quarterly_financials': quarterly_financials,
                'quarterly_balance_sheet': quarterly_balance_sheet,
                'quarterly_cashflow': quarterly_cashflow
            }
        except Exception as e:
            st.error(f"Error fetching financial data: {str(e)}")
            return {}
    
    def calculate_financial_ratios(self, data: Dict) -> Dict:
        """Calculate comprehensive financial ratios"""
        try:
            ratios = {}
            
            if not data or not data.get('financials').empty:
                return ratios
            
            financials = data['financials']
            balance_sheet = data['balance_sheet']
            cashflow = data['cashflow']
            
            # Get most recent year data
            if len(financials.columns) > 0:
                latest_year = financials.columns[0]
                
                # Revenue and profitability ratios
                revenue = financials.loc['Total Revenue', latest_year] if 'Total Revenue' in financials.index else 0
                net_income = financials.loc['Net Income', latest_year] if 'Net Income' in financials.index else 0
                gross_profit = financials.loc['Gross Profit', latest_year] if 'Gross Profit' in financials.index else 0
                operating_income = financials.loc['Operating Income', latest_year] if 'Operating Income' in financials.index else 0
                
                # Balance sheet data
                total_assets = balance_sheet.loc['Total Assets', latest_year] if 'Total Assets' in balance_sheet.index else 0
                total_liabilities = balance_sheet.loc['Total Liabilities', latest_year] if 'Total Liabilities' in balance_sheet.index else 0
                total_equity = balance_sheet.loc['Total Stockholder Equity', latest_year] if 'Total Stockholder Equity' in balance_sheet.index else 0
                current_assets = balance_sheet.loc['Current Assets', latest_year] if 'Current Assets' in balance_sheet.index else 0
                current_liabilities = balance_sheet.loc['Current Liabilities', latest_year] if 'Current Liabilities' in balance_sheet.index else 0
                cash = balance_sheet.loc['Cash And Cash Equivalents', latest_year] if 'Cash And Cash Equivalents' in balance_sheet.index else 0
                
                # Cash flow data
                operating_cashflow = cashflow.loc['Total Cash From Operating Activities', latest_year] if 'Total Cash From Operating Activities' in cashflow.index else 0
                capex = cashflow.loc['Capital Expenditures', latest_year] if 'Capital Expenditures' in cashflow.index else 0
                free_cashflow = operating_cashflow - abs(capex) if operating_cashflow and capex else 0
                
                # Profitability Ratios
                ratios['gross_margin'] = (gross_profit / revenue * 100) if revenue > 0 else 0
                ratios['operating_margin'] = (operating_income / revenue * 100) if revenue > 0 else 0
                ratios['net_margin'] = (net_income / revenue * 100) if revenue > 0 else 0
                ratios['roe'] = (net_income / total_equity * 100) if total_equity > 0 else 0
                ratios['roa'] = (net_income / total_assets * 100) if total_assets > 0 else 0
                
                # Liquidity Ratios
                ratios['current_ratio'] = current_assets / current_liabilities if current_liabilities > 0 else 0
                ratios['quick_ratio'] = (current_assets - cash) / current_liabilities if current_liabilities > 0 else 0
                ratios['cash_ratio'] = cash / current_liabilities if current_liabilities > 0 else 0
                
                # Leverage Ratios
                ratios['debt_to_equity'] = total_liabilities / total_equity if total_equity > 0 else 0
                ratios['debt_to_assets'] = total_liabilities / total_assets if total_assets > 0 else 0
                ratios['equity_ratio'] = total_equity / total_assets if total_assets > 0 else 0
                
                # Efficiency Ratios
                ratios['asset_turnover'] = revenue / total_assets if total_assets > 0 else 0
                ratios['inventory_turnover'] = 0  # Would need inventory data
                ratios['receivables_turnover'] = 0  # Would need receivables data
                
                # Cash Flow Ratios
                ratios['operating_cashflow_margin'] = (operating_cashflow / revenue * 100) if revenue > 0 else 0
                ratios['free_cashflow_margin'] = (free_cashflow / revenue * 100) if revenue > 0 else 0
                ratios['cash_conversion_ratio'] = operating_cashflow / net_income if net_income > 0 else 0
                
                # Growth Ratios (compare with previous year)
                if len(financials.columns) > 1:
                    prev_year = financials.columns[1]
                    prev_revenue = financials.loc['Total Revenue', prev_year] if 'Total Revenue' in financials.index else 0
                    prev_net_income = financials.loc['Net Income', prev_year] if 'Net Income' in financials.index else 0
                    
                    ratios['revenue_growth'] = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
                    ratios['net_income_growth'] = ((net_income - prev_net_income) / prev_net_income * 100) if prev_net_income > 0 else 0
                else:
                    ratios['revenue_growth'] = 0
                    ratios['net_income_growth'] = 0
                
                # Valuation Ratios (if market data available)
                market_cap = data['info'].get('marketCap', 0)
                shares_outstanding = data['info'].get('sharesOutstanding', 0)
                
                if market_cap and shares_outstanding:
                    ratios['pe_ratio'] = market_cap / net_income if net_income > 0 else 0
                    ratios['pb_ratio'] = market_cap / total_equity if total_equity > 0 else 0
                    ratios['ps_ratio'] = market_cap / revenue if revenue > 0 else 0
                    ratios['ev_ebitda'] = 0  # Would need enterprise value calculation
                else:
                    ratios['pe_ratio'] = 0
                    ratios['pb_ratio'] = 0
                    ratios['ps_ratio'] = 0
                    ratios['ev_ebitda'] = 0
            
            return ratios
            
        except Exception as e:
            st.error(f"Error calculating financial ratios: {str(e)}")
            return {}
    
    def calculate_dcf_valuation(self, data: Dict, symbol: str) -> Dict:
        """Calculate Discounted Cash Flow valuation"""
        try:
            if not data or data['quarterly_cashflow'].empty:
                return {}
            
            quarterly_cashflow = data['quarterly_cashflow']
            info = data['info']
            
            # Get free cash flow data
            if 'Free Cash Flow' in quarterly_cashflow.index:
                fcf_data = quarterly_cashflow.loc['Free Cash Flow'].dropna()
            elif 'Total Cash From Operating Activities' in quarterly_cashflow.index and 'Capital Expenditures' in quarterly_cashflow.index:
                operating_cf = quarterly_cashflow.loc['Total Cash From Operating Activities'].dropna()
                capex = quarterly_cashflow.loc['Capital Expenditures'].dropna()
                fcf_data = operating_cf - abs(capex)
            else:
                return {}
            
            if len(fcf_data) < 4:  # Need at least 1 year of data
                return {}
            
            # Calculate growth rates
            fcf_values = fcf_data.values
            growth_rates = []
            
            for i in range(1, len(fcf_values)):
                if fcf_values[i-1] != 0:
                    growth_rate = (fcf_values[i] - fcf_values[i-1]) / abs(fcf_values[i-1])
                    growth_rates.append(growth_rate)
            
            # Average growth rate (last 3 quarters)
            avg_growth_rate = np.mean(growth_rates[-3:]) if len(growth_rates) >= 3 else np.mean(growth_rates) if growth_rates else 0.05
            
            # Cap growth rate between -50% and 50%
            avg_growth_rate = max(-0.5, min(0.5, avg_growth_rate))
            
            # Terminal growth rate (assume 3% long-term)
            terminal_growth_rate = 0.03
            
            # Discount rate (WACC approximation)
            beta = info.get('beta', 1.0)
            risk_free_rate = self.risk_free_rate
            market_risk_premium = self.market_return - risk_free_rate
            cost_of_equity = risk_free_rate + beta * market_risk_premium
            
            # Assume cost of debt is 5%
            cost_of_debt = 0.05
            tax_rate = 0.25  # Assume 25% tax rate
            
            # Simple WACC calculation (assuming 100% equity for simplicity)
            wacc = cost_of_equity
            
            # Project FCF for next 5 years
            current_fcf = fcf_values[0]
            projected_fcf = []
            
            for year in range(1, 6):
                projected_fcf.append(current_fcf * ((1 + avg_growth_rate) ** year))
            
            # Calculate terminal value
            terminal_fcf = projected_fcf[-1] * (1 + terminal_growth_rate)
            terminal_value = terminal_fcf / (wacc - terminal_growth_rate)
            
            # Discount projected FCF and terminal value
            present_value_fcf = []
            for i, fcf in enumerate(projected_fcf):
                pv = fcf / ((1 + wacc) ** (i + 1))
                present_value_fcf.append(pv)
            
            terminal_pv = terminal_value / ((1 + wacc) ** 5)
            
            # Enterprise value
            enterprise_value = sum(present_value_fcf) + terminal_pv
            
            # Get current market cap and shares outstanding
            market_cap = info.get('marketCap', 0)
            shares_outstanding = info.get('sharesOutstanding', 0)
            
            if market_cap and shares_outstanding:
                current_price = market_cap / shares_outstanding
                dcf_price = enterprise_value / shares_outstanding
                upside_downside = ((dcf_price - current_price) / current_price) * 100
            else:
                current_price = 0
                dcf_price = enterprise_value / 1000000000  # Assume 1B shares
                upside_downside = 0
            
            return {
                'dcf_price': dcf_price,
                'current_price': current_price,
                'upside_downside': upside_downside,
                'enterprise_value': enterprise_value,
                'wacc': wacc * 100,
                'growth_rate': avg_growth_rate * 100,
                'terminal_growth_rate': terminal_growth_rate * 100,
                'projected_fcf': projected_fcf,
                'terminal_value': terminal_value,
                'present_value_fcf': present_value_fcf,
                'terminal_pv': terminal_pv
            }
            
        except Exception as e:
            st.error(f"Error calculating DCF valuation: {str(e)}")
            return {}
    
    def calculate_risk_score(self, data: Dict, ratios: Dict) -> Dict:
        """Calculate comprehensive risk score (0-100, lower is better)"""
        try:
            risk_factors = {}
            total_risk = 0
            max_risk = 0
            
            # Financial Stability Risk (30% weight)
            stability_risk = 0
            stability_max = 30
            
            # Debt levels
            debt_to_equity = ratios.get('debt_to_equity', 0)
            if debt_to_equity > 2.0:
                stability_risk += 15
            elif debt_to_equity > 1.0:
                stability_risk += 10
            elif debt_to_equity > 0.5:
                stability_risk += 5
            
            # Current ratio
            current_ratio = ratios.get('current_ratio', 0)
            if current_ratio < 1.0:
                stability_risk += 15
            elif current_ratio < 1.5:
                stability_risk += 10
            elif current_ratio < 2.0:
                stability_risk += 5
            
            risk_factors['financial_stability'] = min(stability_risk, stability_max)
            total_risk += risk_factors['financial_stability']
            max_risk += stability_max
            
            # Profitability Risk (25% weight)
            profitability_risk = 0
            profitability_max = 25
            
            # Profit margins
            net_margin = ratios.get('net_margin', 0)
            if net_margin < 0:
                profitability_risk += 15
            elif net_margin < 5:
                profitability_risk += 10
            elif net_margin < 10:
                profitability_risk += 5
            
            # ROE
            roe = ratios.get('roe', 0)
            if roe < 0:
                profitability_risk += 10
            elif roe < 10:
                profitability_risk += 5
            
            risk_factors['profitability'] = min(profitability_risk, profitability_max)
            total_risk += risk_factors['profitability']
            max_risk += profitability_max
            
            # Growth Risk (20% weight)
            growth_risk = 0
            growth_max = 20
            
            # Revenue growth
            revenue_growth = ratios.get('revenue_growth', 0)
            if revenue_growth < -20:
                growth_risk += 15
            elif revenue_growth < -10:
                growth_risk += 10
            elif revenue_growth < 0:
                growth_risk += 5
            
            # Net income growth
            net_income_growth = ratios.get('net_income_growth', 0)
            if net_income_growth < -30:
                growth_risk += 5
            elif net_income_growth < -20:
                growth_risk += 3
            elif net_income_growth < -10:
                growth_risk += 1
            
            risk_factors['growth'] = min(growth_risk, growth_max)
            total_risk += risk_factors['growth']
            max_risk += growth_max
            
            # Market Risk (15% weight)
            market_risk = 0
            market_max = 15
            
            # P/E ratio
            pe_ratio = ratios.get('pe_ratio', 0)
            if pe_ratio > 50:
                market_risk += 10
            elif pe_ratio > 30:
                market_risk += 5
            elif pe_ratio < 0:
                market_risk += 5
            
            # Beta (if available)
            beta = data.get('info', {}).get('beta', 1.0)
            if beta > 2.0:
                market_risk += 5
            elif beta > 1.5:
                market_risk += 3
            elif beta < 0.5:
                market_risk += 2
            
            risk_factors['market'] = min(market_risk, market_max)
            total_risk += risk_factors['market']
            max_risk += market_max
            
            # Liquidity Risk (10% weight)
            liquidity_risk = 0
            liquidity_max = 10
            
            # Quick ratio
            quick_ratio = ratios.get('quick_ratio', 0)
            if quick_ratio < 0.5:
                liquidity_risk += 10
            elif quick_ratio < 1.0:
                liquidity_risk += 5
            elif quick_ratio < 1.5:
                liquidity_risk += 2
            
            risk_factors['liquidity'] = min(liquidity_risk, liquidity_max)
            total_risk += risk_factors['liquidity']
            max_risk += liquidity_max
            
            # Calculate overall risk score
            risk_score = (total_risk / max_risk * 100) if max_risk > 0 else 0
            
            # Risk level classification
            if risk_score < 20:
                risk_level = "Low"
                risk_color = "green"
            elif risk_score < 40:
                risk_level = "Moderate"
                risk_color = "orange"
            elif risk_score < 60:
                risk_level = "High"
                risk_color = "red"
            else:
                risk_level = "Very High"
                risk_color = "darkred"
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'risk_factors': risk_factors,
                'total_risk': total_risk,
                'max_risk': max_risk
            }
            
        except Exception as e:
            st.error(f"Error calculating risk score: {str(e)}")
            return {'risk_score': 50, 'risk_level': 'Unknown', 'risk_color': 'gray', 'risk_factors': {}}
    
    def generate_investment_recommendation(self, dcf_data: Dict, risk_data: Dict, ratios: Dict) -> Dict:
        """Generate AI-powered investment recommendation"""
        try:
            recommendation = "HOLD"
            confidence = 50
            reasoning = []
            
            # DCF Analysis
            upside_downside = dcf_data.get('upside_downside', 0)
            if upside_downside > 20:
                recommendation = "STRONG BUY"
                confidence += 30
                reasoning.append(f"DCF shows {upside_downside:.1f}% upside potential")
            elif upside_downside > 10:
                recommendation = "BUY"
                confidence += 20
                reasoning.append(f"DCF shows {upside_downside:.1f}% upside potential")
            elif upside_downside < -20:
                recommendation = "SELL"
                confidence += 30
                reasoning.append(f"DCF shows {upside_downside:.1f}% downside risk")
            elif upside_downside < -10:
                recommendation = "WEAK HOLD"
                confidence += 20
                reasoning.append(f"DCF shows {upside_downside:.1f}% downside risk")
            
            # Risk Analysis
            risk_score = risk_data.get('risk_score', 50)
            if risk_score < 20:
                confidence += 20
                reasoning.append("Low risk profile")
            elif risk_score > 60:
                confidence -= 20
                reasoning.append("High risk profile")
            
            # Financial Health
            net_margin = ratios.get('net_margin', 0)
            if net_margin > 15:
                confidence += 15
                reasoning.append("Strong profitability")
            elif net_margin < 0:
                confidence -= 15
                reasoning.append("Negative profitability")
            
            roe = ratios.get('roe', 0)
            if roe > 20:
                confidence += 10
                reasoning.append("High ROE")
            elif roe < 5:
                confidence -= 10
                reasoning.append("Low ROE")
            
            # Growth Analysis
            revenue_growth = ratios.get('revenue_growth', 0)
            if revenue_growth > 20:
                confidence += 15
                reasoning.append("Strong revenue growth")
            elif revenue_growth < -10:
                confidence -= 15
                reasoning.append("Declining revenue")
            
            # Valuation
            pe_ratio = ratios.get('pe_ratio', 0)
            if 0 < pe_ratio < 15:
                confidence += 10
                reasoning.append("Attractive valuation")
            elif pe_ratio > 30:
                confidence -= 10
                reasoning.append("Expensive valuation")
            
            # Cap confidence between 0 and 100
            confidence = max(0, min(100, confidence))
            
            return {
                'recommendation': recommendation,
                'confidence': confidence,
                'reasoning': reasoning,
                'upside_downside': upside_downside,
                'risk_score': risk_score
            }
            
        except Exception as e:
            st.error(f"Error generating recommendation: {str(e)}")
            return {
                'recommendation': 'HOLD',
                'confidence': 50,
                'reasoning': ['Unable to analyze'],
                'upside_downside': 0,
                'risk_score': 50
            }
    
    def create_analytics_dashboard(self, symbol: str) -> None:
        """Create comprehensive analytics dashboard"""
        st.header(f"🔬 Advanced Analytics: {symbol}")
        
        # Get data
        with st.spinner("Loading comprehensive financial data..."):
            data = self.get_enhanced_financial_data(symbol)
        
        if not data:
            st.error("Unable to load financial data for analysis")
            return
        
        # Calculate analytics
        ratios = self.calculate_financial_ratios(data)
        dcf_data = self.calculate_dcf_valuation(data, symbol)
        risk_data = self.calculate_risk_score(data, ratios)
        recommendation = self.generate_investment_recommendation(dcf_data, risk_data, ratios)
        
        # Create tabs for different analytics
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "💰 DCF Valuation", "⚠️ Risk Analysis", "📈 Financial Ratios", "🎯 Recommendation"
        ])
        
        with tab1:
            self.show_analytics_overview(dcf_data, risk_data, recommendation, ratios)
        
        with tab2:
            self.show_dcf_analysis(dcf_data)
        
        with tab3:
            self.show_risk_analysis(risk_data)
        
        with tab4:
            self.show_financial_ratios(ratios)
        
        with tab5:
            self.show_investment_recommendation(recommendation, dcf_data, risk_data)
    
    def show_analytics_overview(self, dcf_data: Dict, risk_data: Dict, recommendation: Dict, ratios: Dict):
        """Show analytics overview"""
        st.subheader("📊 Investment Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            dcf_price = dcf_data.get('dcf_price', 0)
            current_price = dcf_data.get('current_price', 0)
            if dcf_price and current_price:
                st.metric(
                    "DCF Price",
                    f"${dcf_price:.2f}",
                    f"${dcf_price - current_price:+.2f}"
                )
        
        with col2:
            risk_score = risk_data.get('risk_score', 0)
            risk_level = risk_data.get('risk_level', 'Unknown')
            st.metric(
                "Risk Score",
                f"{risk_score:.0f}/100",
                risk_level
            )
        
        with col3:
            rec = recommendation.get('recommendation', 'HOLD')
            confidence = recommendation.get('confidence', 50)
            st.metric(
                "Recommendation",
                rec,
                f"{confidence}% confidence"
            )
        
        with col4:
            upside = recommendation.get('upside_downside', 0)
            st.metric(
                "Upside/Downside",
                f"{upside:+.1f}%",
                "vs DCF"
            )
        
        # Key metrics
        st.subheader("Key Financial Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Profitability**")
            st.write(f"Net Margin: {ratios.get('net_margin', 0):.1f}%")
            st.write(f"ROE: {ratios.get('roe', 0):.1f}%")
            st.write(f"ROA: {ratios.get('roa', 0):.1f}%")
        
        with col2:
            st.write("**Valuation**")
            st.write(f"P/E Ratio: {ratios.get('pe_ratio', 0):.1f}")
            st.write(f"P/B Ratio: {ratios.get('pb_ratio', 0):.1f}")
            st.write(f"P/S Ratio: {ratios.get('ps_ratio', 0):.1f}")
    
    def show_dcf_analysis(self, dcf_data: Dict):
        """Show DCF analysis details"""
        st.subheader("💰 Discounted Cash Flow Analysis")
        
        if not dcf_data:
            st.warning("Insufficient data for DCF analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**DCF Valuation**")
            st.write(f"Fair Value: ${dcf_data.get('dcf_price', 0):.2f}")
            st.write(f"Current Price: ${dcf_data.get('current_price', 0):.2f}")
            st.write(f"Upside/Downside: {dcf_data.get('upside_downside', 0):+.1f}%")
            st.write(f"Enterprise Value: ${dcf_data.get('enterprise_value', 0):,.0f}")
        
        with col2:
            st.write("**Assumptions**")
            st.write(f"WACC: {dcf_data.get('wacc', 0):.1f}%")
            st.write(f"Growth Rate: {dcf_data.get('growth_rate', 0):.1f}%")
            st.write(f"Terminal Growth: {dcf_data.get('terminal_growth_rate', 0):.1f}%")
        
        # DCF breakdown chart
        if dcf_data.get('projected_fcf'):
            fig = go.Figure()
            
            years = list(range(1, 6))
            projected_fcf = dcf_data['projected_fcf']
            
            fig.add_trace(go.Bar(
                x=years,
                y=projected_fcf,
                name='Projected FCF',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Projected Free Cash Flow (5 Years)",
                xaxis_title="Year",
                yaxis_title="Free Cash Flow ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_risk_analysis(self, risk_data: Dict):
        """Show risk analysis details"""
        st.subheader("⚠️ Risk Analysis")
        
        risk_score = risk_data.get('risk_score', 0)
        risk_level = risk_data.get('risk_level', 'Unknown')
        risk_color = risk_data.get('risk_color', 'gray')
        
        # Risk score gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 40], 'color': "yellow"},
                    {'range': [40, 60], 'color': "orange"},
                    {'range': [60, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"**Risk Level: {risk_level}**")
        
        # Risk factors breakdown
        risk_factors = risk_data.get('risk_factors', {})
        if risk_factors:
            st.subheader("Risk Factors Breakdown")
            
            factors_data = []
            for factor, score in risk_factors.items():
                factors_data.append({
                    'Factor': factor.replace('_', ' ').title(),
                    'Risk Score': score,
                    'Max Score': 30 if factor == 'financial_stability' else 
                               25 if factor == 'profitability' else
                               20 if factor == 'growth' else
                               15 if factor == 'market' else 10
                })
            
            df = pd.DataFrame(factors_data)
            df['Percentage'] = (df['Risk Score'] / df['Max Score'] * 100).round(1)
            
            fig = px.bar(df, x='Factor', y='Percentage', 
                        title='Risk Factors by Category',
                        color='Percentage',
                        color_continuous_scale='Reds')
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_financial_ratios(self, ratios: Dict):
        """Show financial ratios analysis"""
        st.subheader("📈 Financial Ratios Analysis")
        
        if not ratios:
            st.warning("No financial ratios data available")
            return
        
        # Categorize ratios
        profitability_ratios = {
            'Gross Margin': ratios.get('gross_margin', 0),
            'Operating Margin': ratios.get('operating_margin', 0),
            'Net Margin': ratios.get('net_margin', 0),
            'ROE': ratios.get('roe', 0),
            'ROA': ratios.get('roa', 0)
        }
        
        liquidity_ratios = {
            'Current Ratio': ratios.get('current_ratio', 0),
            'Quick Ratio': ratios.get('quick_ratio', 0),
            'Cash Ratio': ratios.get('cash_ratio', 0)
        }
        
        leverage_ratios = {
            'Debt to Equity': ratios.get('debt_to_equity', 0),
            'Debt to Assets': ratios.get('debt_to_assets', 0),
            'Equity Ratio': ratios.get('equity_ratio', 0)
        }
        
        valuation_ratios = {
            'P/E Ratio': ratios.get('pe_ratio', 0),
            'P/B Ratio': ratios.get('pb_ratio', 0),
            'P/S Ratio': ratios.get('ps_ratio', 0)
        }
        
        # Display ratios in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Profitability Ratios**")
            for name, value in profitability_ratios.items():
                st.write(f"{name}: {value:.2f}")
            
            st.write("**Liquidity Ratios**")
            for name, value in liquidity_ratios.items():
                st.write(f"{name}: {value:.2f}")
        
        with col2:
            st.write("**Leverage Ratios**")
            for name, value in leverage_ratios.items():
                st.write(f"{name}: {value:.2f}")
            
            st.write("**Valuation Ratios**")
            for name, value in valuation_ratios.items():
                st.write(f"{name}: {value:.2f}")
        
        # Growth metrics
        st.subheader("Growth Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Revenue Growth: {ratios.get('revenue_growth', 0):.1f}%")
        with col2:
            st.write(f"Net Income Growth: {ratios.get('net_income_growth', 0):.1f}%")
    
    def show_investment_recommendation(self, recommendation: Dict, dcf_data: Dict, risk_data: Dict):
        """Show investment recommendation"""
        st.subheader("🎯 Investment Recommendation")
        
        rec = recommendation.get('recommendation', 'HOLD')
        confidence = recommendation.get('confidence', 50)
        reasoning = recommendation.get('reasoning', [])
        
        # Recommendation display
        if rec == "STRONG BUY":
            st.success(f"🟢 {rec} - {confidence}% Confidence")
        elif rec == "BUY":
            st.success(f"🟢 {rec} - {confidence}% Confidence")
        elif rec == "WEAK HOLD":
            st.warning(f"🟡 {rec} - {confidence}% Confidence")
        elif rec == "HOLD":
            st.info(f"🟡 {rec} - {confidence}% Confidence")
        elif rec == "SELL":
            st.error(f"🔴 {rec} - {confidence}% Confidence")
        else:
            st.info(f"🟡 {rec} - {confidence}% Confidence")
        
        # Reasoning
        st.subheader("Analysis Reasoning")
        for reason in reasoning:
            st.write(f"• {reason}")
        
        # Summary metrics
        st.subheader("Key Metrics Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("DCF Upside/Downside", f"{recommendation.get('upside_downside', 0):+.1f}%")
        
        with col2:
            st.metric("Risk Score", f"{recommendation.get('risk_score', 0):.0f}/100")
        
        with col3:
            st.metric("Confidence", f"{confidence}%")
        
        # Investment thesis
        st.subheader("Investment Thesis")
        
        upside = recommendation.get('upside_downside', 0)
        risk_score = recommendation.get('risk_score', 50)
        
        if upside > 20 and risk_score < 30:
            st.success("**Strong Investment Case**: High upside potential with low risk profile. Consider significant position.")
        elif upside > 10 and risk_score < 40:
            st.success("**Good Investment Case**: Positive upside with moderate risk. Consider adding to portfolio.")
        elif upside > 0 and risk_score < 50:
            st.info("**Moderate Investment Case**: Some upside potential with acceptable risk. Consider small position.")
        elif upside < -10 or risk_score > 60:
            st.error("**Poor Investment Case**: High downside risk or high risk profile. Consider avoiding or reducing position.")
        else:
            st.info("**Neutral Investment Case**: Mixed signals. Monitor closely before making investment decisions.")




