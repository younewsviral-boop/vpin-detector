import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="VPIN Detector",
    page_icon="📊",
    layout="wide"
)

# ====================== VPIN CALCULATION FUNCTIONS ======================

def calculate_vpin(volume, buy_volume, sell_volume, bucket_size=1, time_bars=False):
    """
    Calculate VPIN (Volume-Synchronized Probability of Informed Trading)
    
    Parameters:
    -----------
    volume : array-like
        Total volume per trade or time bar
    buy_volume : array-like
        Buy volume per trade or time bar
    sell_volume : array-like
        Sell volume per trade or time bar
    bucket_size : int
        Volume buckets (e.g., 1 = 1% of total volume, 50 = 50 trades per bucket)
    time_bars : bool
        If True, treat input as time bars. If False, as trade bars.
    
    Returns:
    --------
    vpin_series : pandas Series
        VPIN values
    buy_sell_imbalance : pandas Series
        Buy-sell imbalance
    """
    # Ensure numpy arrays
    volume = np.array(volume)
    buy_volume = np.array(buy_volume)
    sell_volume = np.array(sell_volume)
    
    # Validate inputs
    if len(volume) != len(buy_volume) or len(volume) != len(sell_volume):
        raise ValueError("All input arrays must have the same length")
    
    # Calculate order flow imbalance
    ofi = buy_volume - sell_volume
    
    if time_bars:
        # For time bars: VPIN = |buy_volume - sell_volume| / volume
        vpin_values = np.abs(ofi) / volume
        vpin_values = np.where(volume > 0, vpin_values, 0)
    else:
        # For trade bars: Bucket by number of trades
        n_buckets = len(volume) // bucket_size
        vpin_values = []
        
        for i in range(n_buckets):
            start_idx = i * bucket_size
            end_idx = min(start_idx + bucket_size, len(volume))
            
            bucket_volume = volume[start_idx:end_idx].sum()
            bucket_buys = buy_volume[start_idx:end_idx].sum()
            bucket_sells = sell_volume[start_idx:end_idx].sum()
            
            if bucket_volume > 0:
                vpin = abs(bucket_buys - bucket_sells) / bucket_volume
                vpin_values.append(vpin)
            else:
                vpin_values.append(0)
        
        vpin_values = np.array(vpin_values)
    
    return vpin_values, ofi


def calculate_advanced_vpin(df, volume_bucket_percent=1):
    """
    Advanced VPIN calculation from trade data
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain columns: ['price', 'volume', 'side'] or ['price', 'volume', 'buy', 'sell']
    volume_bucket_percent : float
        Percentage of total volume per bucket (e.g., 1 = 1% buckets)
    
    Returns:
    --------
    result_df : pandas DataFrame
        VPIN results with timestamps
    """
    # Identify buy/sell columns
    if 'side' in df.columns:
        # If we have a 'side' column (e.g., 'buy' or 'sell')
        df['buy_volume'] = np.where(df['side'].str.lower() == 'buy', df['volume'], 0)
        df['sell_volume'] = np.where(df['side'].str.lower() == 'sell', df['volume'], 0)
    elif 'buy' in df.columns and 'sell' in df.columns:
        # If we already have buy/sell columns
        df['buy_volume'] = df['buy']
        df['sell_volume'] = df['sell']
    else:
        # Simulate buy/sell (50/50 split for demo)
        df['buy_volume'] = df['volume'] * np.random.uniform(0.4, 0.6, len(df))
        df['sell_volume'] = df['volume'] - df['buy_volume']
    
    # Sort by time if timestamp exists
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    # Calculate total volume
    total_volume = df['volume'].sum()
    bucket_volume = total_volume * (volume_bucket_percent / 100)
    
    # Create volume buckets
    vpin_values = []
    bucket_start_times = []
    current_bucket_volume = 0
    current_bucket_buys = 0
    current_bucket_sells = 0
    
    for idx, row in df.iterrows():
        current_bucket_volume += row['volume']
        current_bucket_buys += row['buy_volume']
        current_bucket_sells += row['sell_volume']
        
        if current_bucket_volume >= bucket_volume:
            # Calculate VPIN for this bucket
            if current_bucket_volume > 0:
                vpin = abs(current_bucket_buys - current_bucket_sells) / current_bucket_volume
            else:
                vpin = 0
            
            vpin_values.append(vpin)
            
            # Get timestamp for this bucket
            if 'timestamp' in df.columns:
                bucket_start_times.append(row['timestamp'])
            else:
                bucket_start_times.append(idx)
            
            # Reset for next bucket
            current_bucket_volume = 0
            current_bucket_buys = 0
            current_bucket_sells = 0
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'timestamp': bucket_start_times[:len(vpin_values)],
        'VPIN': vpin_values
    })
    
    return result_df


def generate_sample_data(n_trades=1000, start_date='2024-01-01'):
    """
    Generate sample trade data for testing
    
    Returns:
    --------
    df : pandas DataFrame
        Sample trade data with columns: timestamp, price, volume, side
    """
    # Generate timestamps
    start = pd.Timestamp(start_date)
    timestamps = [start + timedelta(seconds=i*10) for i in range(n_trades)]
    
    # Generate price data (random walk with drift)
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.01, n_trades)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate volume data (higher during "informed" periods)
    volumes = np.random.lognormal(mean=5, sigma=1.2, size=n_trades)
    
    # Create informed trading periods (higher VPIN)
    informed_periods = np.zeros(n_trades)
    for i in range(0, n_trades, 200):
        informed_periods[i:i+50] = 1
    
    # Generate buy/sell imbalance during informed periods
    buy_prob = 0.5 + 0.3 * informed_periods * np.random.random(n_trades)
    sides = np.where(np.random.random(n_trades) < buy_prob, 'buy', 'sell')
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes,
        'side': sides
    })
    
    return df

# ====================== STREAMLIT APP ======================

def main():
    st.title("📊 VPIN Detector")
    st.markdown("""
    **Volume-Synchronized Probability of Informed Trading (VPIN)** 
    is a metric that estimates the probability of informed trading in financial markets.
    """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Sample Data", "Upload CSV"]
        )
        
        # Algorithm parameters
        st.subheader("VPIN Parameters")
        bucket_size = st.slider(
            "Bucket Size (trades)",
            min_value=10,
            max_value=200,
            value=50,
            help="Number of trades per bucket"
        )
        
        vpin_threshold = st.slider(
            "High VPIN Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="VPIN values above this are considered high"
        )
        
        window_size = st.slider(
            "Moving Average Window",
            min_value=5,
            max_value=100,
            value=20,
            help="Window for VPIN moving average"
        )
        
        calculate_btn = st.button("Calculate VPIN", type="primary")
    
    # Main content area
    if data_source == "Sample Data":
        st.info("Using sample trade data. Adjust parameters in sidebar.")
        n_trades = st.slider("Number of trades", 100, 5000, 1000)
        df = generate_sample_data(n_trades=n_trades)
    else:
        uploaded_file = st.file_uploader("Upload trade data CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} trades")
        else:
            st.warning("Please upload a CSV file or use sample data")
            return
    
    # Show data preview
    with st.expander("Data Preview", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            st.metric("Total Trades", len(df))
            st.metric("Total Volume", f"{df['volume'].sum():,.0f}")
            if 'side' in df.columns:
                buy_count = (df['side'].str.lower() == 'buy').sum()
                st.metric("Buy/Sell Ratio", f"{buy_count/len(df):.1%}")
    
    if calculate_btn and not df.empty:
        # Calculate VPIN
        with st.spinner("Calculating VPIN..."):
            try:
                # Prepare data
                if 'buy_volume' not in df.columns or 'sell_volume' not in df.columns:
                    df['buy_volume'] = np.where(df['side'].str.lower() == 'buy', df['volume'], 0)
                    df['sell_volume'] = np.where(df['side'].str.lower() == 'sell', df['volume'], 0)
                
                # Calculate basic VPIN
                vpin_values, ofi = calculate_vpin(
                    df['volume'].values,
                    df['buy_volume'].values,
                    df['sell_volume'].values,
                    bucket_size=bucket_size,
                    time_bars=False
                )
                
                # Calculate advanced VPIN (volume buckets)
                vpin_df = calculate_advanced_vpin(df, volume_bucket_percent=1)
                
                # Add moving average
                vpin_df['VPIN_MA'] = vpin_df['VPIN'].rolling(window=window_size).mean()
                vpin_df['High_VPIN'] = vpin_df['VPIN'] > vpin_threshold
                
                # Calculate statistics
                avg_vpin = vpin_df['VPIN'].mean()
                max_vpin = vpin_df['VPIN'].max()
                high_vpin_count = vpin_df['High_VPIN'].sum()
                high_vpin_percent = high_vpin_count / len(vpin_df)
                
                # Display results
                st.subheader("📈 VPIN Analysis Results")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Average VPIN", f"{avg_vpin:.3f}")
                col2.metric("Maximum VPIN", f"{max_vpin:.3f}")
                col3.metric("High VPIN Buckets", high_vpin_count)
                col4.metric("High VPIN %", f"{high_vpin_percent:.1%}")
                
                # Plot VPIN
                fig1 = go.Figure()
                
                # Add VPIN line
                fig1.add_trace(go.Scatter(
                    x=vpin_df['timestamp'] if 'timestamp' in vpin_df.columns else range(len(vpin_df)),
                    y=vpin_df['VPIN'],
                    mode='lines',
                    name='VPIN',
                    line=dict(color='blue', width=2)
                ))
                
                # Add moving average
                fig1.add_trace(go.Scatter(
                    x=vpin_df['timestamp'] if 'timestamp' in vpin_df.columns else range(len(vpin_df)),
                    y=vpin_df['VPIN_MA'],
                    mode='lines',
                    name=f'MA ({window_size})',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                
                # Add threshold line
                fig1.add_hline(
                    y=vpin_threshold,
                    line_dash="dot",
                    line_color="red",
                    annotation_text=f"Threshold ({vpin_threshold})",
                    annotation_position="bottom right"
                )
                
                # Highlight high VPIN areas
                high_vpin_mask = vpin_df['High_VPIN']
                if high_vpin_mask.any():
                    high_vpin_x = vpin_df.loc[high_vpin_mask, 'timestamp'] if 'timestamp' in vpin_df.columns else vpin_df.index[high_vpin_mask]
                    high_vpin_y = vpin_df.loc[high_vpin_mask, 'VPIN']
                    fig1.add_trace(go.Scatter(
                        x=high_vpin_x,
                        y=high_vpin_y,
                        mode='markers',
                        name='High VPIN',
                        marker=dict(color='red', size=8, symbol='triangle-up')
                    ))
                
                # Update layout
                fig1.update_layout(
                    title="VPIN Over Time",
                    xaxis_title="Time / Bucket",
                    yaxis_title="VPIN Value",
                    hovermode='x unified',
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
                # Order Flow Imbalance plot
                st.subheader("💰 Order Flow Imbalance")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=range(len(ofi)),
                    y=ofi,
                    mode='lines',
                    name='OFI',
                    line=dict(color='green', width=1)
                ))
                fig2.update_layout(
                    title="Order Flow Imbalance (Buys - Sells)",
                    xaxis_title="Trade Index",
                    yaxis_title="Imbalance",
                    height=300,
                    template="plotly_white"
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # High VPIN periods table
                if high_vpin_count > 0:
                    st.subheader("⚠️ High VPIN Periods")
                    high_vpin_periods = vpin_df[vpin_df['High_VPIN']].copy()
                    high_vpin_periods['VPIN'] = high_vpin_periods['VPIN'].round(3)
                    
                    # Calculate duration if timestamps available
                    if 'timestamp' in high_vpin_periods.columns:
                        high_vpin_periods = high_vpin_periods.sort_values('timestamp')
                    
                    st.dataframe(high_vpin_periods, use_container_width=True)
                
                # Download results
                st.subheader("📥 Export Results")
                csv = vpin_df.to_csv(index=False)
                st.download_button(
                    label="Download VPIN Data (CSV)",
                    data=csv,
                    file_name="vpin_results.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error calculating VPIN: {str(e)}")
                st.info("Make sure your data has the required columns: 'volume', and either 'side' or 'buy_volume'/'sell_volume'")
    
    # Information section
    with st.expander("ℹ️ About VPIN"):
        st.markdown("""
        ### What is VPIN?
        VPIN (Volume-Synchronized Probability of Informed Trading) measures the probability 
        that a given trade originates from an informed trader rather than a noise trader.
        
        ### Interpretation:
        - **High VPIN (> 0.7)**: High probability of informed trading
        - **Moderate VPIN (0.3-0.7)**: Mixed trading
        - **Low VPIN (< 0.3)**: Mostly noise/uninformed trading
        
        ### Formula (simplified):
        ```
        VPIN = |V_buy - V_sell| / (V_buy + V_sell)
        ```
        Where:
        - V_buy = Total buy volume in bucket
        - V_sell = Total sell volume in bucket
        
        ### Recommended Settings:
        - **Bucket Size**: 30-100 trades for intraday analysis
        - **Threshold**: 0.6-0.8 for high VPIN detection
        - **Moving Average**: 20 periods for smoothing
        """)

if __name__ == "__main__":
    main()
