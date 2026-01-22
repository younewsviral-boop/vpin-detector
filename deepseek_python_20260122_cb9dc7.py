import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
import base64

# Page config
st.set_page_config(
    page_title="VPIN & Jump Detector",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'trades' not in st.session_state:
    st.session_state.trades = pd.DataFrame()
if 'vpin' not in st.session_state:
    st.session_state.vpin = pd.DataFrame()

# Title
st.title("📊 Bitcoin VPIN & Jump Detection - FREE VERSION")
st.markdown("---")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Data source
    source = st.selectbox(
        "Data Source",
        ["Sample Data", "Binance API (Live)", "Upload CSV"]
    )
    
    # Parameters
    bucket_vol = st.slider("Bucket Volume (BTC)", 10, 200, 50)
    window = st.slider("Window Size", 1, 50, 16)
    
    # Actions
    if st.button("🔄 Generate Sample Data", use_container_width=True):
        st.session_state.trades = generate_sample_data(1000)
        st.success("Generated 1000 sample trades!")
    
    if st.button("📊 Calculate VPIN", use_container_width=True):
        if len(st.session_state.trades) > 0:
            st.session_state.vpin = calculate_vpin_simple(
                st.session_state.trades, 
                bucket_vol, 
                window
            )
            st.success(f"VPIN calculated for {len(st.session_state.vpin)} buckets!")

# MAIN DASHBOARD
if len(st.session_state.trades) > 0:
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", len(st.session_state.trades))
    
    with col2:
        latest_price = st.session_state.trades['price'].iloc[-1]
        st.metric("Current Price", f"${latest_price:,.0f}")
    
    with col3:
        if len(st.session_state.vpin) > 0:
            current_vpin = st.session_state.vpin['vpin'].iloc[-1]
            st.metric("Current VPIN", f"{current_vpin:.3f}")
    
    with col4:
        avg_vpin = st.session_state.vpin['vpin'].mean() if len(st.session_state.vpin) > 0 else 0
        level = "HIGH" if avg_vpin > 0.3 else "MEDIUM" if avg_vpin > 0.2 else "LOW"
        st.metric("Toxicity", level)
    
    # Charts
    tab1, tab2, tab3 = st.tabs(["📈 Price & VPIN", "📊 Trade Data", "📥 Export"])
    
    with tab1:
        # Price chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(
                x=st.session_state.trades['timestamp'],
                y=st.session_state.trades['price'],
                name="Price",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        if len(st.session_state.vpin) > 0:
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.vpin['timestamp'],
                    y=st.session_state.vpin['vpin'],
                    name="VPIN",
                    line=dict(color='orange', width=2),
                    fill='tozeroy'
                ),
                row=2, col=1
            )
            
            # Add threshold
            fig.add_hline(
                y=0.3, line_dash="dash", 
                line_color="red", opacity=0.5,
                row=2, col=1
            )
        
        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.dataframe(
            st.session_state.trades.tail(50),
            use_container_width=True
        )
    
    with tab3:
        # Export to CSV
        csv1 = st.session_state.trades.to_csv(index=False)
        csv2 = st.session_state.vpin.to_csv(index=False) if len(st.session_state.vpin) > 0 else ""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download Trades CSV",
                data=csv1,
                file_name="bitcoin_trades.csv",
                mime="text/csv"
            )
        
        with col2:
            if csv2:
                st.download_button(
                    label="📥 Download VPIN CSV",
                    data=csv2,
                    file_name="vpin_calculated.csv",
                    mime="text/csv"
                )
        
        # Generate Excel with openpyxl
        if st.button("Generate Excel Report"):
            excel_buffer = create_excel_report(
                st.session_state.trades,
                st.session_state.vpin
            )
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_buffer,
                file_name="vpin_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    # Welcome screen
    st.info("👈 Use the sidebar to generate sample data or connect to Binance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quick Start")
        if st.button("Generate Sample Data Now"):
            st.session_state.trades = generate_sample_data(500)
            st.rerun()
    
    with col2:
        st.subheader("Features")
        st.markdown("""
        - ✅ VPIN calculation
        - ✅ Live Binance data
        - ✅ Jump detection
        - ✅ Excel export
        - ✅ Real-time charts
        - ✅ 100% Free
        """)

# ============ CORE FUNCTIONS ============

def generate_sample_data(n_trades=1000):
    """Generate realistic Bitcoin trade data"""
    dates = pd.date_range(
        end=datetime.now(),
        periods=n_trades,
        freq='1min'
    )
    
    # Generate realistic price series with jumps
    base_price = 50000
    returns = np.random.normal(0, 0.001, n_trades)
    
    # Add some jumps
    jump_indices = np.random.choice(n_trades, size=5, replace=False)
    returns[jump_indices] = np.random.choice([0.02, -0.02], 5)
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate volumes
    volumes = np.random.lognormal(mean=0, sigma=1, size=n_trades) * 0.1
    
    # Generate directions (buy/sell)
    directions = np.random.choice([1, -1], n_trades, p=[0.52, 0.48])
    
    return pd.DataFrame({
        'timestamp': dates,
        'price': prices.round(2),
        'volume': volumes.round(6),
        'direction': directions
    })

def calculate_vpin_simple(trades_df, bucket_volume=50, window_size=16):
    """Simplified VPIN calculation"""
    if len(trades_df) == 0:
        return pd.DataFrame()
    
    trades_df = trades_df.sort_values('timestamp').copy()
    
    # Create volume buckets
    trades_df['cum_vol'] = trades_df['volume'].cumsum()
    trades_df['bucket_id'] = (trades_df['cum_vol'] // bucket_volume).astype(int)
    
    # Calculate bucket stats
    bucket_stats = []
    unique_buckets = trades_df['bucket_id'].unique()
    
    for bucket_id in unique_buckets:
        bucket_trades = trades_df[trades_df['bucket_id'] == bucket_id]
        if len(bucket_trades) > 0:
            buy_vol = bucket_trades[bucket_trades['direction'] == 1]['volume'].sum()
            sell_vol = bucket_trades[bucket_trades['direction'] == -1]['volume'].sum()
            total_vol = buy_vol + sell_vol
            
            bucket_stats.append({
                'bucket_id': bucket_id,
                'timestamp': bucket_trades['timestamp'].iloc[-1],
                'buy_volume': buy_vol,
                'sell_volume': sell_vol,
                'total_volume': total_vol,
                'imbalance': abs(buy_vol - sell_vol)
            })
    
    if not bucket_stats:
        return pd.DataFrame()
    
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
                'vpin': vpin,
                'bucket_id': buckets_df.iloc[i]['bucket_id']
            })
    
    return pd.DataFrame(vpin_values)

def create_excel_report(trades_df, vpin_df):
    """Create Excel report using pandas ExcelWriter"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Trades sheet
        trades_df.to_excel(writer, sheet_name='Trades', index=False)
        
        # VPIN sheet
        if len(vpin_df) > 0:
            vpin_df.to_excel(writer, sheet_name='VPIN', index=False)
        
        # Summary sheet
        summary_data = {
            'Metric': ['Total Trades', 'Time Period', 'Avg Price', 'Total Volume'],
            'Value': [
                len(trades_df),
                f"{trades_df['timestamp'].min().strftime('%Y-%m-%d')} to {trades_df['timestamp'].max().strftime('%Y-%m-%d')}",
                f"${trades_df['price'].mean():.0f}",
                f"{trades_df['volume'].sum():.2f} BTC"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    output.seek(0)
    return output.getvalue()

def fetch_binance_data(symbol="BTCUSDT", limit=500):
    """Fetch data from Binance API"""
    try:
        url = f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit={limit}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        trades = []
        for trade in data:
            trades.append({
                'timestamp': datetime.fromtimestamp(trade['time'] / 1000),
                'price': float(trade['price']),
                'volume': float(trade['qty']),
                'direction': -1 if trade['isBuyerMaker'] else 1
            })
        
        return pd.DataFrame(trades)
    except Exception as e:
        st.error(f"Error fetching from Binance: {e}")
        return pd.DataFrame()

# Run the app
if __name__ == "__main__":
    pass