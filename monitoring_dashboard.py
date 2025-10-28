import requests
import streamlit as st

st.set_page_config(page_title="API Monitoring", page_icon="📡", layout="wide")

st.title("📡 MONETA API Monitoring Dashboard")
api_base = st.sidebar.text_input("Backend URL", value="http://localhost:8000")

@st.cache_data(ttl=30)
def fetch_status(base: str):
    try:
        r = requests.get(f"{base}/api/system/status", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

data = fetch_status(api_base)

if "error" in data:
    st.error(f"Failed to fetch status: {data['error']}")
else:
    cols = st.columns(3)
    for i, (name, status) in enumerate(data.get("services", {}).items()):
        with cols[i % 3]:
            healthy = status.get("healthy", False)
            rate = status.get("rate_limit", "-")
            st.metric(label=name, value="✅ Healthy" if healthy else "❌ Down", delta=f"rate: {rate}")

    st.subheader("Raw Status JSON")
    st.json(data)


