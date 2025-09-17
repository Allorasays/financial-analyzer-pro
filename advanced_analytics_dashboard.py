import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_advanced_analytics_dashboard():
    """Create comprehensive advanced analytics dashboard"""
    
    st.markdown("""
    <style>
    .analytics-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .correlation-heatmap {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("📊 Advanced Analytics Dashboard")
    
    # Market overview
    st.subheader("📈 Market Overview")
    
    # Get market data
    symbols = ['^GSPC', '^IXIC', '^DJI', '^VIX', '^RUT']
    market_data = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            if not hist.empty:
                market_data[symbol] = hist
        except Exception as e:
            st.warning(f"Could not fetch {symbol}: {str(e)}")
    
    if market_data:
        # Calculate returns
        returns_data = {}
        for symbol, data in market_data.items():
            returns = data['Close'].pct_change().dropna()
            returns_data[symbol] = returns
        
        # Create returns dataframe
        returns_df = pd.DataFrame(returns_data)
        
        # Market performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sp500_return = returns_df['^GSPC'].mean() * 252 * 100
            st.metric("S&P 500 Annual Return", f"{sp500_return:.2f}%")
        
        with col2:
            nasdaq_return = returns_df['^IXIC'].mean() * 252 * 100
            st.metric("NASDAQ Annual Return", f"{nasdaq_return:.2f}%")
        
        with col3:
            dow_return = returns_df['^DJI'].mean() * 252 * 100
            st.metric("DOW Annual Return", f"{dow_return:.2f}%")
        
        with col4:
            vix_current = market_data['^VIX']['Close'].iloc[-1]
            st.metric("VIX (Fear Index)", f"{vix_current:.2f}")
        
        # Correlation heatmap
        st.subheader("🔗 Market Correlation Matrix")
        
        correlation_matrix = returns_df.corr()
        
        fig_heatmap = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Market Indices Correlation Heatmap",
            color_continuous_scale="RdBu_r"
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Volatility analysis
        st.subheader("📊 Volatility Analysis")
        
        volatility_data = {}
        for symbol, data in market_data.items():
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            volatility_data[symbol] = volatility
        
        vol_df = pd.DataFrame(list(volatility_data.items()), columns=['Index', 'Volatility'])
        
        fig_vol = px.bar(
            vol_df,
            x='Index',
            y='Volatility',
            title="Annualized Volatility by Index",
            labels={'Volatility': 'Volatility (%)', 'Index': 'Market Index'}
        )
        fig_vol.update_layout(height=400)
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Rolling volatility
        st.subheader("📈 Rolling Volatility (30-day)")
        
        rolling_vol_data = {}
        for symbol, data in market_data.items():
            returns = data['Close'].pct_change().dropna()
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100
            rolling_vol_data[symbol] = rolling_vol
        
        rolling_vol_df = pd.DataFrame(rolling_vol_data)
        
        fig_rolling = go.Figure()
        for symbol in rolling_vol_df.columns:
            fig_rolling.add_trace(go.Scatter(
                x=rolling_vol_df.index,
                y=rolling_vol_df[symbol],
                mode='lines',
                name=symbol,
                line=dict(width=2)
            ))
        
        fig_rolling.update_layout(
            title="30-Day Rolling Volatility",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            height=400
        )
        st.plotly_chart(fig_rolling, use_container_width=True)
    
    # Sector analysis
    st.subheader("🏭 Sector Analysis")
    
    sector_symbols = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
        'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
        'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'HD']
    }
    
    sector_returns = {}
    sector_volatility = {}
    
    for sector, symbols in sector_symbols.items():
        sector_data = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1mo")
                if not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    sector_data.extend(returns.tolist())
            except:
                continue
        
        if sector_data:
            sector_returns[sector] = np.mean(sector_data) * 252 * 100
            sector_volatility[sector] = np.std(sector_data) * np.sqrt(252) * 100
    
    if sector_returns:
        sector_df = pd.DataFrame({
            'Sector': list(sector_returns.keys()),
            'Return': list(sector_returns.values()),
            'Volatility': list(sector_volatility.values())
        })
        
        # Sector performance scatter plot
        fig_sector = px.scatter(
            sector_df,
            x='Volatility',
            y='Return',
            size='Volatility',
            color='Sector',
            title="Sector Performance: Return vs Volatility",
            labels={'Volatility': 'Volatility (%)', 'Return': 'Annual Return (%)'}
        )
        fig_sector.update_layout(height=500)
        st.plotly_chart(fig_sector, use_container_width=True)
        
        # Sector table
        st.subheader("📊 Sector Performance Table")
        display_sector_df = sector_df.copy()
        display_sector_df['Return'] = display_sector_df['Return'].apply(lambda x: f"{x:.2f}%")
        display_sector_df['Volatility'] = display_sector_df['Volatility'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display_sector_df, use_container_width=True)
    
    # Market breadth analysis
    st.subheader("📊 Market Breadth Analysis")
    
    # Get individual stock data for breadth analysis
    individual_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC']
    stock_data = {}
    
    for symbol in individual_stocks:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            if not hist.empty:
                stock_data[symbol] = hist
        except:
            continue
    
    if stock_data:
        # Calculate daily advances/declines
        advances = []
        declines = []
        dates = []
        
        for date in stock_data[list(stock_data.keys())[0]].index:
            daily_advances = 0
            daily_declines = 0
            
            for symbol, data in stock_data.items():
                if date in data.index:
                    if len(data.loc[:date]) >= 2:
                        prev_close = data.loc[:date]['Close'].iloc[-2]
                        curr_close = data.loc[:date]['Close'].iloc[-1]
                        if curr_close > prev_close:
                            daily_advances += 1
                        elif curr_close < prev_close:
                            daily_declines += 1
            
            advances.append(daily_advances)
            declines.append(daily_declines)
            dates.append(date)
        
        breadth_df = pd.DataFrame({
            'Date': dates,
            'Advances': advances,
            'Declines': declines
        })
        breadth_df['Net Advances'] = breadth_df['Advances'] - breadth_df['Declines']
        breadth_df['Advance/Decline Ratio'] = breadth_df['Advances'] / breadth_df['Declines']
        
        # Plot market breadth
        fig_breadth = go.Figure()
        
        fig_breadth.add_trace(go.Scatter(
            x=breadth_df['Date'],
            y=breadth_df['Net Advances'],
            mode='lines',
            name='Net Advances',
            line=dict(color='blue', width=2)
        ))
        
        fig_breadth.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Neutral")
        
        fig_breadth.update_layout(
            title="Market Breadth: Net Advances",
            xaxis_title="Date",
            yaxis_title="Net Advances",
            height=400
        )
        
        st.plotly_chart(fig_breadth, use_container_width=True)
        
        # Advance/Decline ratio
        fig_ratio = go.Figure()
        
        fig_ratio.add_trace(go.Scatter(
            x=breadth_df['Date'],
            y=breadth_df['Advance/Decline Ratio'],
            mode='lines',
            name='A/D Ratio',
            line=dict(color='green', width=2)
        ))
        
        fig_ratio.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Neutral (1.0)")
        
        fig_ratio.update_layout(
            title="Advance/Decline Ratio",
            xaxis_title="Date",
            yaxis_title="A/D Ratio",
            height=400
        )
        
        st.plotly_chart(fig_ratio, use_container_width=True)
    
    # Economic indicators
    st.subheader("📊 Economic Indicators")
    
    # Get economic data (simplified)
    economic_indicators = {
        '10-Year Treasury': '^TNX',
        '30-Year Treasury': '^TYX',
        'Dollar Index': 'DX-Y.NYB',
        'Gold': 'GC=F',
        'Oil': 'CL=F'
    }
    
    econ_data = {}
    for name, symbol in economic_indicators.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            if not hist.empty:
                econ_data[name] = hist
        except:
            continue
    
    if econ_data:
        # Economic indicators table
        econ_metrics = []
        for name, data in econ_data.items():
            current_value = data['Close'].iloc[-1]
            prev_value = data['Close'].iloc[-2] if len(data) > 1 else current_value
            change = ((current_value - prev_value) / prev_value) * 100
            
            econ_metrics.append({
                'Indicator': name,
                'Current Value': f"{current_value:.2f}",
                'Change': f"{change:+.2f}%"
            })
        
        econ_df = pd.DataFrame(econ_metrics)
        st.dataframe(econ_df, use_container_width=True)
        
        # Economic indicators chart
        fig_econ = go.Figure()
        
        for name, data in econ_data.items():
            fig_econ.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name=name,
                line=dict(width=2)
            ))
        
        fig_econ.update_layout(
            title="Economic Indicators",
            xaxis_title="Date",
            yaxis_title="Value",
            height=400
        )
        
        st.plotly_chart(fig_econ, use_container_width=True)
    
    # Market sentiment analysis
    st.subheader("😊 Market Sentiment Analysis")
    
    # Calculate sentiment indicators
    sentiment_indicators = {}
    
    if '^VIX' in market_data:
        vix_data = market_data['^VIX']
        current_vix = vix_data['Close'].iloc[-1]
        vix_avg = vix_data['Close'].mean()
        
        if current_vix < vix_avg * 0.8:
            vix_sentiment = "Very Bullish"
            vix_color = "green"
        elif current_vix < vix_avg:
            vix_sentiment = "Bullish"
            vix_color = "lightgreen"
        elif current_vix < vix_avg * 1.2:
            vix_sentiment = "Neutral"
            vix_color = "yellow"
        else:
            vix_sentiment = "Bearish"
            vix_color = "red"
        
        sentiment_indicators['VIX Sentiment'] = {
            'value': f"{current_vix:.2f}",
            'sentiment': vix_sentiment,
            'color': vix_color
        }
    
    # Market breadth sentiment
    if 'breadth_df' in locals():
        recent_net_advances = breadth_df['Net Advances'].tail(5).mean()
        if recent_net_advances > 5:
            breadth_sentiment = "Very Bullish"
            breadth_color = "green"
        elif recent_net_advances > 0:
            breadth_sentiment = "Bullish"
            breadth_color = "lightgreen"
        elif recent_net_advances > -5:
            breadth_sentiment = "Neutral"
            breadth_color = "yellow"
        else:
            breadth_sentiment = "Bearish"
            breadth_color = "red"
        
        sentiment_indicators['Market Breadth'] = {
            'value': f"{recent_net_advances:.1f}",
            'sentiment': breadth_sentiment,
            'color': breadth_color
        }
    
    # Display sentiment indicators
    if sentiment_indicators:
        col1, col2 = st.columns(2)
        
        for i, (indicator, data) in enumerate(sentiment_indicators.items()):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{indicator}</h4>
                    <p><strong>Value:</strong> {data['value']}</p>
                    <p><strong>Sentiment:</strong> <span style="color: {data['color']}">{data['sentiment']}</span></p>
                </div>
                """, unsafe_allow_html=True)
    
    # Summary insights
    st.subheader("💡 Market Insights")
    
    insights = []
    
    if '^VIX' in market_data:
        vix_current = market_data['^VIX']['Close'].iloc[-1]
        if vix_current < 20:
            insights.append("✅ Low VIX suggests market complacency and potential for volatility")
        elif vix_current > 30:
            insights.append("⚠️ High VIX indicates market fear and potential buying opportunity")
    
    if 'breadth_df' in locals():
        recent_breadth = breadth_df['Net Advances'].tail(5).mean()
        if recent_breadth > 5:
            insights.append("📈 Strong market breadth suggests healthy market participation")
        elif recent_breadth < -5:
            insights.append("📉 Weak market breadth indicates limited participation")
    
    if sector_returns:
        best_sector = max(sector_returns.items(), key=lambda x: x[1])
        worst_sector = min(sector_returns.items(), key=lambda x: x[1])
        insights.append(f"🏆 {best_sector[0]} sector leading with {best_sector[1]:.2f}% return")
        insights.append(f"📉 {worst_sector[0]} sector lagging with {worst_sector[1]:.2f}% return")
    
    for insight in insights:
        st.info(insight)

def main():
    st.set_page_config(
        page_title="Advanced Analytics Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    create_advanced_analytics_dashboard()

if __name__ == "__main__":
    main()
