import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import time
import json
import psutil
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

class PerformanceMonitor:
    """Performance monitoring and analytics system"""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'response_times': [],
            'error_counts': [],
            'user_actions': [],
            'api_calls': [],
            'cache_hits': [],
            'cache_misses': []
        }
        self.start_time = time.time()
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # CPU and memory usage
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                self.metrics['cpu_usage'].append({
                    'timestamp': datetime.now(),
                    'value': cpu_percent
                })
                
                self.metrics['memory_usage'].append({
                    'timestamp': datetime.now(),
                    'value': memory.percent
                })
                
                # Keep only last 1000 data points
                for key in self.metrics:
                    if len(self.metrics[key]) > 1000:
                        self.metrics[key] = self.metrics[key][-1000:]
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(10)
    
    def record_response_time(self, endpoint: str, response_time: float):
        """Record API response time"""
        self.metrics['response_times'].append({
            'timestamp': datetime.now(),
            'endpoint': endpoint,
            'response_time': response_time
        })
    
    def record_error(self, error_type: str, error_message: str):
        """Record error occurrence"""
        self.metrics['error_counts'].append({
            'timestamp': datetime.now(),
            'error_type': error_type,
            'error_message': error_message
        })
    
    def record_user_action(self, action: str, details: dict = None):
        """Record user action"""
        self.metrics['user_actions'].append({
            'timestamp': datetime.now(),
            'action': action,
            'details': details or {}
        })
    
    def record_api_call(self, endpoint: str, method: str, status_code: int):
        """Record API call"""
        self.metrics['api_calls'].append({
            'timestamp': datetime.now(),
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code
        })
    
    def record_cache_event(self, hit: bool, key: str):
        """Record cache hit/miss"""
        if hit:
            self.metrics['cache_hits'].append({
                'timestamp': datetime.now(),
                'key': key
            })
        else:
            self.metrics['cache_misses'].append({
                'timestamp': datetime.now(),
                'key': key
            })
    
    def get_performance_summary(self):
        """Get performance summary"""
        uptime = time.time() - self.start_time
        
        # Calculate averages
        cpu_avg = np.mean([m['value'] for m in self.metrics['cpu_usage'][-100:]]) if self.metrics['cpu_usage'] else 0
        memory_avg = np.mean([m['value'] for m in self.metrics['memory_usage'][-100:]]) if self.metrics['memory_usage'] else 0
        
        # Response times
        response_times = [m['response_time'] for m in self.metrics['response_times']]
        avg_response_time = np.mean(response_times) if response_times else 0
        max_response_time = np.max(response_times) if response_times else 0
        
        # Error rate
        total_errors = len(self.metrics['error_counts'])
        total_actions = len(self.metrics['user_actions'])
        error_rate = (total_errors / total_actions * 100) if total_actions > 0 else 0
        
        # Cache hit rate
        cache_hits = len(self.metrics['cache_hits'])
        cache_misses = len(self.metrics['cache_misses'])
        cache_hit_rate = (cache_hits / (cache_hits + cache_misses) * 100) if (cache_hits + cache_misses) > 0 else 0
        
        return {
            'uptime_hours': uptime / 3600,
            'cpu_usage_avg': cpu_avg,
            'memory_usage_avg': memory_avg,
            'avg_response_time': avg_response_time,
            'max_response_time': max_response_time,
            'total_errors': total_errors,
            'error_rate': error_rate,
            'total_actions': total_actions,
            'cache_hit_rate': cache_hit_rate,
            'total_api_calls': len(self.metrics['api_calls'])
        }

class UserAnalytics:
    """User analytics and behavior tracking"""
    
    def __init__(self):
        self.user_sessions = {}
        self.feature_usage = {}
        self.user_flows = {}
        self.performance_metrics = {}
        
    def track_session_start(self, session_id: str, user_agent: str = None):
        """Track user session start"""
        self.user_sessions[session_id] = {
            'start_time': datetime.now(),
            'user_agent': user_agent,
            'actions': [],
            'features_used': set(),
            'pages_visited': [],
            'session_duration': 0
        }
    
    def track_session_end(self, session_id: str):
        """Track user session end"""
        if session_id in self.user_sessions:
            session = self.user_sessions[session_id]
            session['end_time'] = datetime.now()
            session['session_duration'] = (session['end_time'] - session['start_time']).total_seconds()
    
    def track_feature_usage(self, session_id: str, feature: str, details: dict = None):
        """Track feature usage"""
        if session_id in self.user_sessions:
            self.user_sessions[session_id]['features_used'].add(feature)
            self.user_sessions[session_id]['actions'].append({
                'timestamp': datetime.now(),
                'action': 'feature_usage',
                'feature': feature,
                'details': details or {}
            })
            
            # Track feature usage globally
            if feature not in self.feature_usage:
                self.feature_usage[feature] = 0
            self.feature_usage[feature] += 1
    
    def track_page_visit(self, session_id: str, page: str):
        """Track page visit"""
        if session_id in self.user_sessions:
            self.user_sessions[session_id]['pages_visited'].append({
                'timestamp': datetime.now(),
                'page': page
            })
    
    def get_user_analytics_summary(self):
        """Get user analytics summary"""
        total_sessions = len(self.user_sessions)
        active_sessions = len([s for s in self.user_sessions.values() if 'end_time' not in s])
        
        # Calculate average session duration
        completed_sessions = [s for s in self.user_sessions.values() if 'end_time' in s]
        avg_session_duration = np.mean([s['session_duration'] for s in completed_sessions]) if completed_sessions else 0
        
        # Most used features
        most_used_features = sorted(self.feature_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Page popularity
        page_visits = {}
        for session in self.user_sessions.values():
            for visit in session['pages_visited']:
                page = visit['page']
                page_visits[page] = page_visits.get(page, 0) + 1
        
        most_visited_pages = sorted(page_visits.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'avg_session_duration': avg_session_duration,
            'most_used_features': most_used_features,
            'most_visited_pages': most_visited_pages,
            'total_features_used': len(self.feature_usage)
        }

def create_monitoring_dashboard():
    """Create comprehensive monitoring dashboard"""
    
    st.markdown("""
    <style>
    .monitoring-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-healthy { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-critical { background-color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("📊 System Monitoring & Analytics")
    
    # Initialize monitoring
    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()
        st.session_state.performance_monitor.start_monitoring()
    
    if 'user_analytics' not in st.session_state:
        st.session_state.user_analytics = UserAnalytics()
    
    performance_monitor = st.session_state.performance_monitor
    user_analytics = st.session_state.user_analytics
    
    # System status
    st.subheader("🟢 System Status")
    
    performance_summary = performance_monitor.get_performance_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        uptime_hours = performance_summary['uptime_hours']
        status_class = "status-healthy" if uptime_hours > 0 else "status-critical"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Uptime</h4>
            <p><span class="status-indicator {status_class}"></span>{uptime_hours:.1f} hours</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        cpu_usage = performance_summary['cpu_usage_avg']
        status_class = "status-healthy" if cpu_usage < 70 else "status-warning" if cpu_usage < 90 else "status-critical"
        st.markdown(f"""
        <div class="metric-card">
            <h4>CPU Usage</h4>
            <p><span class="status-indicator {status_class}"></span>{cpu_usage:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        memory_usage = performance_summary['memory_usage_avg']
        status_class = "status-healthy" if memory_usage < 70 else "status-warning" if memory_usage < 90 else "status-critical"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Memory Usage</h4>
            <p><span class="status-indicator {status_class}"></span>{memory_usage:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        error_rate = performance_summary['error_rate']
        status_class = "status-healthy" if error_rate < 5 else "status-warning" if error_rate < 15 else "status-critical"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Error Rate</h4>
            <p><span class="status-indicator {status_class}"></span>{error_rate:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance charts
    st.subheader("📈 Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU Usage Chart
        if performance_monitor.metrics['cpu_usage']:
            cpu_data = performance_monitor.metrics['cpu_usage'][-50:]  # Last 50 points
            cpu_df = pd.DataFrame(cpu_data)
            
            fig_cpu = go.Figure()
            fig_cpu.add_trace(go.Scatter(
                x=cpu_df['timestamp'],
                y=cpu_df['value'],
                mode='lines',
                name='CPU Usage %',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig_cpu.update_layout(
                title="CPU Usage Over Time",
                xaxis_title="Time",
                yaxis_title="CPU Usage (%)",
                height=300
            )
            
            st.plotly_chart(fig_cpu, use_container_width=True)
    
    with col2:
        # Memory Usage Chart
        if performance_monitor.metrics['memory_usage']:
            memory_data = performance_monitor.metrics['memory_usage'][-50:]  # Last 50 points
            memory_df = pd.DataFrame(memory_data)
            
            fig_memory = go.Figure()
            fig_memory.add_trace(go.Scatter(
                x=memory_df['timestamp'],
                y=memory_df['value'],
                mode='lines',
                name='Memory Usage %',
                line=dict(color='#ff7f0e', width=2)
            ))
            
            fig_memory.update_layout(
                title="Memory Usage Over Time",
                xaxis_title="Time",
                yaxis_title="Memory Usage (%)",
                height=300
            )
            
            st.plotly_chart(fig_memory, use_container_width=True)
    
    # Response times
    st.subheader("⚡ Response Times")
    
    if performance_monitor.metrics['response_times']:
        response_data = performance_monitor.metrics['response_times'][-100:]  # Last 100 requests
        response_df = pd.DataFrame(response_data)
        
        # Group by endpoint
        endpoint_times = response_df.groupby('endpoint')['response_time'].agg(['mean', 'max', 'count']).reset_index()
        endpoint_times = endpoint_times.sort_values('mean', ascending=False)
        
        fig_response = px.bar(
            endpoint_times,
            x='endpoint',
            y='mean',
            title="Average Response Time by Endpoint",
            labels={'mean': 'Response Time (ms)', 'endpoint': 'Endpoint'}
        )
        fig_response.update_layout(height=400)
        st.plotly_chart(fig_response, use_container_width=True)
        
        # Response time distribution
        fig_dist = px.histogram(
            response_df,
            x='response_time',
            title="Response Time Distribution",
            labels={'response_time': 'Response Time (ms)', 'count': 'Frequency'}
        )
        fig_dist.update_layout(height=300)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # User analytics
    st.subheader("👥 User Analytics")
    
    user_summary = user_analytics.get_user_analytics_summary()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Sessions", user_summary['total_sessions'])
    with col2:
        st.metric("Active Sessions", user_summary['active_sessions'])
    with col3:
        st.metric("Avg Session Duration", f"{user_summary['avg_session_duration']:.1f}s")
    
    # Feature usage
    if user_summary['most_used_features']:
        st.subheader("🔥 Most Used Features")
        
        features_df = pd.DataFrame(user_summary['most_used_features'], columns=['Feature', 'Usage Count'])
        
        fig_features = px.bar(
            features_df,
            x='Usage Count',
            y='Feature',
            orientation='h',
            title="Feature Usage Statistics"
        )
        fig_features.update_layout(height=400)
        st.plotly_chart(fig_features, use_container_width=True)
    
    # Page popularity
    if user_summary['most_visited_pages']:
        st.subheader("📄 Most Visited Pages")
        
        pages_df = pd.DataFrame(user_summary['most_visited_pages'], columns=['Page', 'Visits'])
        
        fig_pages = px.pie(
            pages_df,
            values='Visits',
            names='Page',
            title="Page Visit Distribution"
        )
        fig_pages.update_layout(height=400)
        st.plotly_chart(fig_pages, use_container_width=True)
    
    # Error analysis
    st.subheader("🚨 Error Analysis")
    
    if performance_monitor.metrics['error_counts']:
        error_data = performance_monitor.metrics['error_counts'][-50:]  # Last 50 errors
        error_df = pd.DataFrame(error_data)
        
        # Error types
        error_types = error_df['error_type'].value_counts()
        
        fig_errors = px.bar(
            x=error_types.index,
            y=error_types.values,
            title="Error Types Distribution",
            labels={'x': 'Error Type', 'y': 'Count'}
        )
        fig_errors.update_layout(height=300)
        st.plotly_chart(fig_errors, use_container_width=True)
        
        # Recent errors table
        st.subheader("Recent Errors")
        recent_errors = error_df.tail(10)
        st.dataframe(recent_errors, use_container_width=True)
    
    # Cache performance
    st.subheader("💾 Cache Performance")
    
    cache_hit_rate = performance_summary['cache_hit_rate']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Cache Hit Rate", f"{cache_hit_rate:.1f}%")
    
    with col2:
        total_cache_events = len(performance_monitor.metrics['cache_hits']) + len(performance_monitor.metrics['cache_misses'])
        st.metric("Total Cache Events", total_cache_events)
    
    # System recommendations
    st.subheader("💡 System Recommendations")
    
    recommendations = []
    
    if performance_summary['cpu_usage_avg'] > 80:
        recommendations.append("⚠️ High CPU usage detected. Consider optimizing algorithms or scaling resources.")
    
    if performance_summary['memory_usage_avg'] > 80:
        recommendations.append("⚠️ High memory usage detected. Consider implementing memory optimization.")
    
    if performance_summary['error_rate'] > 10:
        recommendations.append("🚨 High error rate detected. Review error logs and fix critical issues.")
    
    if performance_summary['avg_response_time'] > 2000:
        recommendations.append("🐌 Slow response times detected. Consider optimizing database queries and caching.")
    
    if performance_summary['cache_hit_rate'] < 50:
        recommendations.append("💾 Low cache hit rate. Consider improving caching strategy.")
    
    if not recommendations:
        recommendations.append("✅ System is performing well. No immediate actions required.")
    
    for recommendation in recommendations:
        st.info(recommendation)
    
    # Export monitoring data
    st.subheader("📤 Export Monitoring Data")
    
    if st.button("Export Performance Data", type="primary"):
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_summary': performance_summary,
            'user_analytics': user_summary,
            'system_info': {
                'python_version': '3.11.0',
                'streamlit_version': '1.28.0',
                'platform': 'Linux'
            }
        }
        
        # Convert to JSON
        report_json = json.dumps(report, indent=2, default=str)
        
        st.download_button(
            label="Download Monitoring Report",
            data=report_json,
            file_name=f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Real-time monitoring controls
    st.subheader("🎛️ Monitoring Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Monitoring", type="primary"):
            performance_monitor.start_monitoring()
            st.success("Monitoring started!")
    
    with col2:
        if st.button("Stop Monitoring", type="secondary"):
            performance_monitor.stop_monitoring()
            st.success("Monitoring stopped!")

def main():
    st.set_page_config(
        page_title="System Monitoring & Analytics",
        page_icon="📊",
        layout="wide"
    )
    
    create_monitoring_dashboard()

if __name__ == "__main__":
    main()
