"""
Enhanced ML Dashboard for Financial Analyzer Pro
Advanced machine learning visualization and analysis tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional

from enhanced_ml_service import enhanced_ml_service
from sentiment_analysis_service import sentiment_service

class EnhancedMLDashboard:
    """Enhanced ML dashboard with advanced visualization and analysis"""
    
    def __init__(self):
        self.model_performance_history = {}
        self.prediction_history = {}
        
    def display_ml_analysis_dashboard(self, symbol: str, data: pd.DataFrame, period: str = "1y"):
        """Display comprehensive ML analysis dashboard"""
        st.header(f"🤖 Enhanced ML Analysis - {symbol}")
        
        # Configuration section
        with st.expander("⚙️ ML Configuration", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                lookback_days = st.slider("Lookback Days", 30, 200, 60, help="Number of days to look back for sequences")
                forecast_horizon = st.slider("Forecast Horizon", 1, 30, 5, help="Days ahead to predict")
            
            with col2:
                include_sentiment = st.checkbox("Include Sentiment Analysis", value=True, help="Add news and social media sentiment")
                include_technical = st.checkbox("Include Technical Indicators", value=True, help="Add advanced technical indicators")
                include_volatility = st.checkbox("Include Volatility Features", value=True, help="Add volatility-based features")
            
            with col3:
                model_types = st.multiselect(
                    "Model Types",
                    ["Traditional", "Gradient Boosting", "Neural Networks", "Ensemble"],
                    default=["Traditional", "Gradient Boosting", "Ensemble"],
                    help="Select which types of models to train"
                )
        
        # Prepare data
        with st.spinner("Preparing data and features..."):
            # Get sentiment features if enabled
            sentiment_features = {}
            if include_sentiment:
                try:
                    sentiment_features = sentiment_service.get_sentiment_features(symbol, 7)
                    st.success(f"✅ Added {len(sentiment_features)} sentiment features")
                except Exception as e:
                    st.warning(f"⚠️ Sentiment analysis failed: {str(e)}")
            
            # Prepare ML data
            ml_data = enhanced_ml_service.prepare_ml_data(
                data, 
                lookback=lookback_days, 
                forecast_horizon=forecast_horizon
            )
            
            if ml_data is None:
                st.error("❌ Failed to prepare ML data")
                return
            
            X, y, X_sequences, y_sequences = ml_data
            
            # Add sentiment features to X
            if sentiment_features:
                for feature_name, feature_value in sentiment_features.items():
                    X[feature_name] = feature_value
            
            st.success(f"✅ Prepared {len(X)} samples with {len(X.columns)} features")
        
        # Train models
        if st.button("🚀 Train Enhanced ML Models", type="primary"):
            with st.spinner("Training advanced ML models..."):
                # Train models
                model_results = enhanced_ml_service.train_ensemble_models(
                    X, y, X_sequences, y_sequences
                )
                
                if 'error' in model_results:
                    st.error(f"❌ Training failed: {model_results['error']}")
                    return
                
                # Store results in session state
                st.session_state[f'ml_results_{symbol}'] = model_results
                st.session_state[f'ml_data_{symbol}'] = (X, y, X_sequences, y_sequences)
                
                st.success("✅ Model training completed!")
        
        # Display results if available
        if f'ml_results_{symbol}' in st.session_state:
            model_results = st.session_state[f'ml_results_{symbol}']
            X, y, X_sequences, y_sequences = st.session_state[f'ml_data_{symbol}']
            
            # Model comparison
            self._display_model_comparison(model_results)
            
            # Feature importance
            self._display_feature_importance(model_results, X.columns.tolist())
            
            # Predictions visualization
            self._display_predictions_visualization(model_results, y)
            
            # Model performance over time
            self._display_performance_over_time(model_results, X, y)
            
            # Prediction accuracy analysis
            self._display_accuracy_analysis(model_results, y)
            
            # Sentiment impact analysis
            if sentiment_features:
                self._display_sentiment_impact_analysis(symbol, model_results)
    
    def _display_model_comparison(self, model_results: Dict[str, Any]):
        """Display model comparison dashboard"""
        st.subheader("📊 Model Performance Comparison")
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in model_results.items():
            if name != 'ensemble' and 'r2' in result:
                comparison_data.append({
                    'Model': name,
                    'R² Score': result['r2'],
                    'RMSE': result['rmse'],
                    'MAE': result['mae'],
                    'MAPE': result['mape'],
                    'Training Time': result.get('training_time', 0)
                })
        
        if not comparison_data:
            st.warning("No model results to display")
            return
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('R² Score', ascending=False)
        
        # Display comparison table
        st.dataframe(df_comparison, use_container_width=True)
        
        # Best model highlight
        best_model = df_comparison.iloc[0]
        st.success(f"🏆 Best Model: {best_model['Model']} (R² = {best_model['R² Score']:.4f})")
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            fig_r2 = px.bar(
                df_comparison, 
                x='Model', 
                y='R² Score',
                title="R² Score Comparison",
                color='R² Score',
                color_continuous_scale='Viridis'
            )
            fig_r2.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            # RMSE comparison
            fig_rmse = px.bar(
                df_comparison, 
                x='Model', 
                y='RMSE',
                title="RMSE Comparison (Lower is Better)",
                color='RMSE',
                color_continuous_scale='Reds'
            )
            fig_rmse.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_rmse, use_container_width=True)
    
    def _display_feature_importance(self, model_results: Dict[str, Any], feature_names: List[str]):
        """Display feature importance analysis"""
        st.subheader("🎯 Feature Importance Analysis")
        
        # Get feature importance from all models
        importance_df = enhanced_ml_service.get_feature_importance(model_results, feature_names)
        
        if importance_df.empty:
            st.warning("No feature importance data available")
            return
        
        # Display top features
        top_features = importance_df.head(20)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Feature importance bar chart
            fig = px.bar(
                top_features,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 20 Most Important Features",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature categories
            st.subheader("📊 Feature Categories")
            
            # Categorize features
            categories = {
                'Price': [f for f in top_features['Feature'] if any(x in f.lower() for x in ['price', 'close', 'open', 'high', 'low'])],
                'Volume': [f for f in top_features['Feature'] if 'volume' in f.lower()],
                'Technical': [f for f in top_features['Feature'] if any(x in f.lower() for x in ['rsi', 'macd', 'bb', 'stoch', 'williams', 'cci'])],
                'Moving Averages': [f for f in top_features['Feature'] if any(x in f.lower() for x in ['sma', 'ema'])],
                'Volatility': [f for f in top_features['Feature'] if 'volatility' in f.lower()],
                'Momentum': [f for f in top_features['Feature'] if any(x in f.lower() for x in ['momentum', 'roc', 'proc'])],
                'Sentiment': [f for f in top_features['Feature'] if any(x in f.lower() for x in ['sentiment', 'news', 'social'])],
                'Trend': [f for f in top_features['Feature'] if 'trend' in f.lower()]
            }
            
            for category, features in categories.items():
                if features:
                    st.write(f"**{category}**: {len(features)} features")
                    for feature in features[:5]:  # Show top 5
                        st.write(f"• {feature}")
                    if len(features) > 5:
                        st.write(f"• ... and {len(features) - 5} more")
        
        # Feature correlation heatmap
        if len(top_features) > 5:
            st.subheader("🔗 Feature Correlation Heatmap")
            
            # Get correlation matrix for top features
            try:
                # This would need the actual feature data
                st.info("Feature correlation analysis would require the full feature matrix")
            except Exception as e:
                st.warning(f"Correlation analysis not available: {str(e)}")
    
    def _display_predictions_visualization(self, model_results: Dict[str, Any], y_true: pd.Series):
        """Display predictions visualization"""
        st.subheader("📈 Predictions vs Actual")
        
        # Get predictions from best model
        best_model_name = None
        best_r2 = -np.inf
        
        for name, result in model_results.items():
            if name != 'ensemble' and 'r2' in result and result['r2'] > best_r2:
                best_r2 = result['r2']
                best_model_name = name
        
        if best_model_name is None:
            st.warning("No valid model predictions found")
            return
        
        best_predictions = model_results[best_model_name]['predictions']
        
        # Create predictions vs actual plot
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=y_true.values,
            y=y_true.values,
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='gray', dash='dash')
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=y_true.values,
            y=best_predictions,
            mode='markers',
            name=f'{best_model_name} Predictions',
            marker=dict(color='blue', size=6)
        ))
        
        fig.update_layout(
            title=f"Predictions vs Actual ({best_model_name})",
            xaxis_title="Actual Price",
            yaxis_title="Predicted Price",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residuals plot
        residuals = y_true.values - best_predictions
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residuals histogram
            fig_hist = px.histogram(
                x=residuals,
                title="Residuals Distribution",
                nbins=30
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Residuals over time
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Scatter(
                y=residuals,
                mode='lines',
                name='Residuals',
                line=dict(color='red')
            ))
            fig_resid.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_resid.update_layout(
                title="Residuals Over Time",
                yaxis_title="Residuals",
                height=400
            )
            st.plotly_chart(fig_resid, use_container_width=True)
    
    def _display_performance_over_time(self, model_results: Dict[str, Any], X: pd.DataFrame, y: pd.Series):
        """Display model performance over time"""
        st.subheader("📊 Performance Over Time")
        
        # Calculate rolling performance metrics
        window_size = min(50, len(y) // 4)
        
        if window_size < 10:
            st.warning("Not enough data for rolling performance analysis")
            return
        
        # Get predictions from ensemble model
        if 'ensemble' in model_results:
            predictions = model_results['ensemble']['predictions']
        else:
            # Use best individual model
            best_model_name = max(
                [name for name in model_results.keys() if name != 'ensemble' and 'r2' in model_results[name]],
                key=lambda x: model_results[x]['r2']
            )
            predictions = model_results[best_model_name]['predictions']
        
        # Calculate rolling R²
        rolling_r2 = []
        for i in range(window_size, len(y)):
            y_window = y.iloc[i-window_size:i]
            pred_window = predictions[i-window_size:i]
            r2 = 1 - (np.sum((y_window - pred_window)**2) / np.sum((y_window - y_window.mean())**2))
            rolling_r2.append(r2)
        
        # Create rolling performance chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=rolling_r2,
            mode='lines',
            name='Rolling R² Score',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Random Performance")
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Good Performance")
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Excellent Performance")
        
        fig.update_layout(
            title=f"Rolling R² Score (Window: {window_size})",
            xaxis_title="Time Period",
            yaxis_title="R² Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average R²", f"{np.mean(rolling_r2):.3f}")
        with col2:
            st.metric("Max R²", f"{np.max(rolling_r2):.3f}")
        with col3:
            st.metric("Min R²", f"{np.min(rolling_r2):.3f}")
        with col4:
            st.metric("R² Std", f"{np.std(rolling_r2):.3f}")
    
    def _display_accuracy_analysis(self, model_results: Dict[str, Any], y_true: pd.Series):
        """Display prediction accuracy analysis"""
        st.subheader("🎯 Prediction Accuracy Analysis")
        
        # Calculate accuracy metrics for all models
        accuracy_data = []
        
        for name, result in model_results.items():
            if name != 'ensemble' and 'predictions' in result:
                predictions = result['predictions']
                
                # Calculate accuracy metrics
                mae = np.mean(np.abs(y_true.values - predictions))
                mape = np.mean(np.abs((y_true.values - predictions) / y_true.values)) * 100
                
                # Direction accuracy (up/down prediction)
                actual_direction = np.diff(y_true.values) > 0
                pred_direction = np.diff(predictions) > 0
                direction_accuracy = np.mean(actual_direction == pred_direction) * 100
                
                accuracy_data.append({
                    'Model': name,
                    'MAE': mae,
                    'MAPE': mape,
                    'Direction Accuracy': direction_accuracy
                })
        
        if not accuracy_data:
            st.warning("No accuracy data available")
            return
        
        df_accuracy = pd.DataFrame(accuracy_data)
        
        # Display accuracy table
        st.dataframe(df_accuracy, use_container_width=True)
        
        # Accuracy visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # MAPE comparison
            fig_mape = px.bar(
                df_accuracy,
                x='Model',
                y='MAPE',
                title="Mean Absolute Percentage Error (Lower is Better)",
                color='MAPE',
                color_continuous_scale='Reds'
            )
            fig_mape.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_mape, use_container_width=True)
        
        with col2:
            # Direction accuracy
            fig_dir = px.bar(
                df_accuracy,
                x='Model',
                y='Direction Accuracy',
                title="Direction Accuracy (Higher is Better)",
                color='Direction Accuracy',
                color_continuous_scale='Greens'
            )
            fig_dir.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_dir, use_container_width=True)
    
    def _display_sentiment_impact_analysis(self, symbol: str, model_results: Dict[str, Any]):
        """Display sentiment impact analysis"""
        st.subheader("📰 Sentiment Impact Analysis")
        
        # Get sentiment data
        try:
            news_sentiment = sentiment_service.analyze_news_sentiment(symbol, 7)
            social_sentiment = sentiment_service.analyze_social_sentiment(symbol, "twitter", 7)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📰 News Sentiment")
                if 'aggregate_sentiment' in news_sentiment:
                    agg = news_sentiment['aggregate_sentiment']
                    st.metric("Overall Sentiment", agg.get('overall_sentiment', 'neutral').title())
                    st.metric("Polarity", f"{agg.get('average_polarity', 0):.3f}")
                    st.metric("Confidence", f"{agg.get('confidence_score', 0):.3f}")
                    st.metric("Articles", agg.get('total_items', 0))
            
            with col2:
                st.subheader("🐦 Social Sentiment")
                if 'aggregate_sentiment' in social_sentiment:
                    agg = social_sentiment['aggregate_sentiment']
                    st.metric("Overall Sentiment", agg.get('overall_sentiment', 'neutral').title())
                    st.metric("Polarity", f"{agg.get('average_polarity', 0):.3f}")
                    st.metric("Confidence", f"{agg.get('confidence_score', 0):.3f}")
                    st.metric("Posts", agg.get('total_items', 0))
            
            # Sentiment impact on predictions
            st.subheader("🎯 Sentiment Impact on Predictions")
            
            # Check if sentiment features were used
            sentiment_features = [col for col in st.session_state.get(f'ml_data_{symbol}', (None, None, None, None))[0].columns 
                               if any(x in col.lower() for x in ['sentiment', 'news', 'social'])]
            
            if sentiment_features:
                st.success(f"✅ Sentiment features used in model: {', '.join(sentiment_features)}")
                
                # Show sentiment feature importance
                importance_df = enhanced_ml_service.get_feature_importance(
                    model_results, 
                    st.session_state[f'ml_data_{symbol}'][0].columns.tolist()
                )
                
                sentiment_importance = importance_df[
                    importance_df['Feature'].str.contains('|'.join(['sentiment', 'news', 'social']), case=False)
                ]
                
                if not sentiment_importance.empty:
                    fig = px.bar(
                        sentiment_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Sentiment Feature Importance"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No sentiment features found in top features")
            else:
                st.info("No sentiment features were used in the model")
                
        except Exception as e:
            st.warning(f"Sentiment analysis failed: {str(e)}")

# Global instance
enhanced_ml_dashboard = EnhancedMLDashboard()
