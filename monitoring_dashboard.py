import requests
import streamlit as st
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict

st.set_page_config(page_title="MONETA API Monitoring", page_icon="📡", layout="wide")

st.title("📡 MONETA API Monitoring Dashboard")
st.markdown("---")

# Sidebar configuration
api_base = st.sidebar.text_input("Backend URL", value="http://localhost:8000")
refresh_interval = st.sidebar.selectbox("Auto-refresh (seconds)", [5, 10, 30, 60], index=2)
enable_alerts = st.sidebar.checkbox("Enable Alerts", value=True)
alert_threshold = st.sidebar.slider("Error Rate Threshold (%)", 0, 20, 5)

# Initialize session state for metrics history
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = defaultdict(list)
if 'alert_log' not in st.session_state:
    st.session_state.alert_log = []

@st.cache_data(ttl=5)
def fetch_status(base: str):
    """Fetch API status with error handling"""
    try:
        r = requests.get(f"{base}/api/system/status", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timeout", "type": "timeout"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection failed", "type": "connection"}
    except Exception as e:
        return {"error": str(e), "type": "unknown"}

@st.cache_data(ttl=5)
def fetch_health(base: str):
    """Fetch health endpoint"""
    try:
        r = requests.get(f"{base}/health", timeout=5)
        return r.json() if r.status_code == 200 else {"status": "unhealthy"}
    except:
        return {"status": "error"}

def check_alerts(data, threshold):
    """Check for alert conditions"""
    alerts = []
    if "error" in data:
        alerts.append({
            "severity": "critical",
            "message": f"API Status Error: {data.get('error')}",
            "timestamp": datetime.now()
        })
    else:
        services = data.get("services", {})
        for name, status in services.items():
            if not status.get("healthy", False):
                alerts.append({
                    "severity": "high",
                    "message": f"{name} service is down",
                    "timestamp": datetime.now()
                })
            # Check rate limits
            rate_info = status.get("rate_limit", "")
            if "rate limited" in rate_info.lower():
                alerts.append({
                    "severity": "medium",
                    "message": f"{name} is rate limited",
                    "timestamp": datetime.now()
                })
    return alerts

def send_alert_placeholder(alert):
    """Placeholder for alert notification (Slack/Email/SMS)"""
    # In production, integrate with:
    # - Slack webhook
    # - Email (SMTP)
    # - SMS (Twilio)
    return True

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

# Fetch data
data = fetch_status(api_base)
health_data = fetch_health(api_base)

# Overall status
if "error" not in data:
    overall_status = "✅ Healthy" if health_data.get("status") == "ok" else "⚠️ Degraded"
    overall_color = "green" if health_data.get("status") == "ok" else "orange"
else:
    overall_status = "❌ Down"
    overall_color = "red"

with col1:
    st.metric("Overall Status", overall_status, delta=None)

# Count healthy services
if "error" not in data:
    services = data.get("services", {})
    healthy_count = sum(1 for s in services.values() if s.get("healthy", False))
    total_count = len(services)
    
    with col2:
        st.metric("Services Healthy", f"{healthy_count}/{total_count}")
    
    with col3:
        uptime_pct = (healthy_count / total_count * 100) if total_count > 0 else 0
        st.metric("Uptime", f"{uptime_pct:.1f}%")
    
    with col4:
        last_update = datetime.now().strftime("%H:%M:%S")
        st.metric("Last Update", last_update)

st.markdown("---")

# Service status cards
if "error" not in data:
    services = data.get("services", {})
    
    st.subheader("🔌 Service Status")
    cols = st.columns(4)
    
    for i, (name, status) in enumerate(services.items()):
        with cols[i % 4]:
            healthy = status.get("healthy", False)
            rate_limit = status.get("rate_limit", "N/A")
            
            # Status badge
            status_badge = "✅" if healthy else "❌"
            status_text = "Healthy" if healthy else "Down"
            status_color = "green" if healthy else "red"
            
            st.markdown(f"""
            <div style="border: 2px solid {status_color}; padding: 10px; border-radius: 5px; margin: 5px;">
                <h4>{status_badge} {name}</h4>
                <p><strong>Status:</strong> {status_text}</p>
                <p><strong>Rate Limit:</strong> {rate_limit}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Update metrics history
            st.session_state.metrics_history[name].append({
                "timestamp": datetime.now(),
                "healthy": healthy,
                "rate_limit": rate_limit
            })
            
            # Keep only last 100 entries
            if len(st.session_state.metrics_history[name]) > 100:
                st.session_state.metrics_history[name] = st.session_state.metrics_history[name][-100:]
else:
    st.error(f"❌ **Connection Error**: {data.get('error', 'Unknown error')}")
    st.info("💡 **Tip**: Ensure the backend is running and the URL is correct")

# Alerts section
if enable_alerts:
    st.markdown("---")
    st.subheader("🚨 Alerts")
    
    alerts = check_alerts(data, alert_threshold)
    
    if alerts:
        for alert in alerts:
            severity_color = {
                "critical": "red",
                "high": "orange",
                "medium": "yellow"
            }.get(alert["severity"], "gray")
            
            st.markdown(f"""
            <div style="border-left: 4px solid {severity_color}; padding: 10px; margin: 5px 0; background-color: #f0f0f0;">
                <strong>[{alert['severity'].upper()}]</strong> {alert['message']}<br>
                <small>{alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Log alert
            if alert not in st.session_state.alert_log[-10:]:
                st.session_state.alert_log.append(alert)
                send_alert_placeholder(alert)
    else:
        st.success("✅ No active alerts")

# Metrics visualization
st.markdown("---")
st.subheader("📊 Service Health Over Time")

if "error" not in data:
    service_names = list(data.get("services", {}).keys())
    selected_services = st.multiselect("Select services to visualize", service_names, default=service_names[:4])
    
    if selected_services and st.session_state.metrics_history:
        import pandas as pd
        
        # Prepare data for chart
        chart_data = []
        for service in selected_services:
            history = st.session_state.metrics_history.get(service, [])
            if history:
                for entry in history[-20:]:  # Last 20 data points
                    chart_data.append({
                        "timestamp": entry["timestamp"],
                        "service": service,
                        "healthy": 1 if entry["healthy"] else 0
                    })
        
        if chart_data:
            df = pd.DataFrame(chart_data)
            st.line_chart(df.pivot(index="timestamp", columns="service", values="healthy"))

# Raw data
with st.expander("📋 Raw Status JSON"):
    st.json(data if "error" not in data else {"error": data.get("error")})

# Auto-refresh
if refresh_interval > 0:
    time.sleep(refresh_interval)
    st.rerun()
