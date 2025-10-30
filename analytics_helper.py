"""
Web Analytics Helper for MONETA Financial Analyzer
Tracks user interactions and feature usage in Streamlit web app
"""
import streamlit as st
import json
from datetime import datetime
from pathlib import Path

# Simple file-based analytics (can be replaced with Google Analytics, Mixpanel, etc.)
ANALYTICS_DIR = Path("analytics_logs")
ANALYTICS_DIR.mkdir(exist_ok=True)

def log_event(event_name: str, **params):
    """Log an analytics event"""
    event_data = {
        "timestamp": datetime.now().isoformat(),
        "event": event_name,
        "params": params,
        "session_id": st.session_state.get("session_id", "unknown")
    }
    
    # Write to log file
    log_file = ANALYTICS_DIR / f"events_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(event_data) + "\n")
    
    # Store in session state for real-time dashboard
    if "analytics_events" not in st.session_state:
        st.session_state.analytics_events = []
    st.session_state.analytics_events.append(event_data)

def log_page_view(page_name: str):
    """Log a page view"""
    log_event("page_view", page=page_name)

def log_feature_used(feature_name: str, **context):
    """Log feature usage"""
    log_event("feature_used", feature=feature_name, **context)

def log_stock_analysis(ticker: str, analysis_type: str):
    """Log stock analysis"""
    log_event("stock_analysis", ticker=ticker, analysis_type=analysis_type)

def log_prediction_requested(ticker: str, prediction_type: str):
    """Log prediction request"""
    log_event("prediction_requested", ticker=ticker, prediction_type=prediction_type)

def get_analytics_summary():
    """Get analytics summary for dashboard"""
    events = st.session_state.get("analytics_events", [])
    if not events:
        return {"total_events": 0}
    
    from collections import Counter
    event_counts = Counter(e["event"] for e in events)
    
    return {
        "total_events": len(events),
        "event_counts": dict(event_counts),
        "unique_sessions": len(set(e.get("session_id", "unknown") for e in events))
    }

