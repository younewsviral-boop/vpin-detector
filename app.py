import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="VPIN Calculator",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Bitcoin VPIN & Jump Detection")
st.markdown("Live VPIN calculation from trade data")
st.markdown("---")

# Initialize session state
if 'trades' not in st.session_state:
    st.session_state.trades = pd.DataFrame()
if 'vpin_values' not in st.session_state:
    st.session_state.vpin_values = pd.DataFrame()

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Parameters
    bucket_size = st.slider("Bucket Size (BTC)", 10, 200, 50)
    window_size = st.slider("Window Size", 5, 50, 16)
    
    # Actions
    if st.button("🚀 Generate Sample Data", use_container_width=True):
        # Create realistic sample data
        np.random.seed(42)
        n_trades = 1000
        dates = pd.date_range(
            end=datetime.now(),
            periods=n_trades,
            freq='1min'
        )
        
        # Price with realistic volatility
        base_price = 50000
        returns = np.random.normal(0, 0.002, n_trades)
        prices = base_price * np.exp(np.cumsum(returns))
        
        st.session_state.trades = pd.DataFrame({
            'timestamp': dates,
            'price': prices.round(2),
            'volume': np.random.lognormal(-1, 1, n_trades).round(4),
            'direction': np.random.choice([1, -1], n_trades, p=[0.52, 0.48])
        })
        
        st.success(f"✅ Generated {n_trades} sample trades!")
    
    if st.button("📊 Calculate VPIN", use_container_width=True):
        if len(st.session_state.trades) > 0:
            # Simple VPIN calculation
            st.session_state.vpin_values = calculate_simple_vpin(
                st.session_state.trades,
                bucket_size,
                window_size
            )
            st.success(f"✅ Calculated VPIN for {len(st.session_state.vpin_values)} buckets")
        else:
            st.warning("⚠️ Please generate data first")
    
    if st.button("🔄 Reset All", use_container_width=True):
        st.session_state.trades = pd.DataFrame()
        st.session_state.vpin_values = pd.DataFrame()
        st.success("✅ Reset complete!")

# MAIN DASHBOARD
if len(st.session_state.trades) > 0:
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", len(st.session_state.trades))
    
    with col2:
        current_price = st.session_state.trades['price'].iloc[-1]
        st.metric("Current Price", f"${current_price:,.0f}")
    
    with col3:
        if len(st.session_state.vpin_values) > 0:
            current_vpin = st.session_state.vpin_values['vpin'].iloc[-1]
            st.metric("Current VPIN", f"{current_vpin:.4f}")
        else:
            st.metric("Current VPIN", "N/A")
    
    with col4:
        if len(st.session_state.vpin_values) > 0:
            avg_vpin = st.session_state.vpin_values['vpin'].mean()
            if avg_vpin < 0.2:
                level = "🟢 LOW"
            elif avg_vpin < 0.3:
                level = "🟡 MEDIUM"
            else:
                level = "🔴 HIGH"
            st.metric("Toxicity", level)
        else:
            st.metric("Toxicity", "N/A")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Charts", "📊 Data", "📥 Export"])
    
    with tab1:
        # Create chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Bitcoin Price", "VPIN")
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=st.session_state.trades['timestamp'],
                y=st.session_state.trades['price'],
                name="Price",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # VPIN chart
        if len(st.session_state.vpin_values) > 0:
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.vpin_values['timestamp'],
                    y=st.session_state.vpin_values['vpin'],
                    name="VPIN",
                    line=dict(color='orange', width=2),
                    fill='tozeroy'
                ),
                row=2, col=1
            )
            
            # Add threshold lines
            fig.add_hline(y=0.3, line_dash="dash", line_color="red", 
                         opacity=0.5, row=2, col=1)
            fig.add_hline(y=0.2, line_dash="dot", line_color="yellow", 
                         opacity=0.5, row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Show data
        st.subheader("Recent Trades (Last 20)")
        st.dataframe(
            st.session_state.trades.tail(20),
            use_container_width=True,
            height=400
        )
        
        if len(st.session_state.vpin_values) > 0:
            st.subheader("VPIN Values (Last 20)")
            st.dataframe(
                st.session_state.vpin_values.tail(20),
                use_container_width=True,
                height=400
            )
    
    with tab3:
        # Export options
        st.subheader("📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv1 = st.session_state.trades.to_csv(index=False)
            st.download_button(
                label="Download Trades CSV",
                data=csv1,
                file_name="bitcoin_trades.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if len(st.session_state.vpin_values) > 0:
                csv2 = st.session_state.vpin_values.to_csv(index=False)
                st.download_button(
                    label="Download VPIN CSV",
                    data=csv2,
                    file_name="vpin_values.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Instructions
        st.markdown("---")
        st.info("""
        **How to use:**
        1. Click **"Generate Sample Data"** in sidebar
        2. Adjust bucket size and window if needed
        3. Click **"Calculate VPIN"**
        4. View charts and export data
        """)
        
else:
    # Welcome screen
    st.info("👈 **Start by generating sample data from the sidebar**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Features")
        st.markdown("""
        - ✅ **VPIN Calculation**
        - ✅ **Realistic Sample Data**
        - ✅ **Interactive Charts**
        - ✅ **CSV Export**
        - ✅ **Toxicity Levels**
        - ✅ **100% Free Hosting**
        """)
    
    with col2:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. Click **"Generate Sample Data"**
        2. Click **"Calculate VPIN"**
        3. View charts in **"Charts"** tab
        4. Export data in **"Export"** tab
        """)
    
    # Quick action
    if st.button("✨ Quick Start - Generate Data Now", use_container_width=True):
        st.rerun()

# ============ FUNCTIONS ============

def calculate_simple_vpin(trades_df, bucket_volume=50, window_size=16):
    """Calculate VPIN from trade data"""
    if len(trades_df) < 100:
        return pd.DataFrame()
    
    trades_df = trades_df.sort_values('timestamp').copy()
    
    # Create volume buckets
    trades_df['cum_vol'] = trades_df['volume'].cumsum()
    trades_df['bucket_id'] = (trades_df['cum_vol'] // bucket_volume).astype(int)
    
    # Calculate bucket stats
    bucket_stats = []
    for bucket_id, group in trades_df.groupby('bucket_id'):
        buy_vol = group[group['direction'] == 1]['volume'].sum()
        sell_vol = group[group['direction'] == -1]['volume'].sum()
        total_vol = buy_vol + sell_vol
        
        bucket_stats.append({
            'bucket_id': bucket_id,
            'timestamp': group['timestamp'].iloc[-1],
            'buy_volume': buy_vol,
            'sell_volume': sell_vol,
            'total_volume': total_vol,
            'imbalance': abs(buy_vol - sell_vol)
        })
    
    buckets_df = pd.DataFrame(bucket_stats)
    
    # Calculate rolling VPIN
    vpin_values = []
    for i in range(window_size, len(buckets_df)):
        window = buckets_df.iloc[i-window_size:i]
        sum_imbalance = window['imbalance'].sum()
        avg_volume = window['total_volume'].mean()
        
        if avg_volume > 0:
            vpin = sum_imbalance / (window_size * avg_volume)
            vpin_values.append({
                'timestamp': buckets_df.iloc[i]['timestamp'],
                'vpin': vpin
            })
    
    return pd.DataFrame(vpin_values)

# Footer
st.markdown("---")
st.caption("VPIN Calculator | Built with Streamlit | Data updates in real-time")
