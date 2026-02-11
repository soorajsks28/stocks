import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime
import pytz

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Market Master Pro UI", layout="wide")

# --- CUSTOM CSS (ATTRACTIVE UI) ---
st.markdown("""
    <style>
    /* General Settings */
    .stApp { background-color: #0e1117; }
    
    /* Card Styling */
    .info-card { background-color: #1E1E1E; padding: 20px; border-radius: 12px; border-left: 5px solid #007bff; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .success-card { background-color: #1b3a2b; padding: 20px; border-radius: 12px; border-left: 5px solid #00e676; margin-bottom: 20px; }
    .risk-card { background-color: #3a1b1b; padding: 20px; border-radius: 12px; border-left: 5px solid #ff5252; margin-bottom: 20px; }
    
    /* Fonts */
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 600; }
    .big-price { font-size: 36px; font-weight: bold; color: #ffffff; }
    .math-text { font-family: 'Courier New', monospace; font-size: 14px; color: #bbb; }
    
    /* Section Dividers */
    .section-divider { border-top: 2px solid #333; margin: 30px 0; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Market Master AI: Visual Edition")

# --- HELPER: TIME STRATEGY ---
def get_time_strategy():
    IST = pytz.timezone('Asia/Kolkata')
    now = datetime.now(IST)
    
    if now.hour < 10: 
        return "☕ **Morning Setup (9:15 - 10:00):** Market ko settle hone dein. Volatility high hoti hai. Limit orders use karein."
    elif now.hour < 13: 
        return "✅ **Prime Trading Hour:** Trend clear hai. Entry lene ka best time."
    elif now.hour >= 13 and now.hour < 14:
        return "⏸️ **Lunch Time (Volume Low):** Fake breakouts se bachein. Wait karein."
    elif now.hour >= 14: 
        return "🚀 **European Move (2:00 PM+):** Market mein tezi aa sakti hai. Stop Loss trail karein."
    else:
        return "🌙 **After Market:** Kal ke liye analysis karein."

# --- HELPER: CALCULATE DATA ---
def process_stock_data(symbol):
    try:
        # Download Data
        df = yf.download(symbol+".NS", period="6mo", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return None

        # Indicators
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        bb = BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Upper'] = bb.bollinger_hband()
        
        return df
    except: return None

# --- HELPER: PLOT CHART ---
def plot_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='orange', width=1), name='50 Day Trend'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='Resistance'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='Support'))
    fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    return fig

# --- HELPER: DISPLAY DETAILS FUNCTION ---
def display_deep_dive(ticker, df, budget, mode):
    curr_price = df['Close'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    sma50 = df['SMA50'].iloc[-1]
    
    # Logic Generation
    signal = "HOLD / WATCH"
    color_code = "orange"
    
    # Simple Logic for Display
    if rsi < 35: 
        signal = "BUY (Oversold)"
        color_code = "#00e676" # Green
    elif curr_price > sma50 and rsi > 50: 
        signal = "BUY (Uptrend)"
        color_code = "#00e676" # Green
    elif rsi > 70: 
        signal = "SELL (Overbought)"
        color_code = "#ff5252" # Red
    
    # 1. TOP DETAILS SECTION
    st.markdown(f"""
    <div class="info-card" style="border-left: 5px solid {color_code};">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h2 style="margin:0;">{ticker}</h2>
                <span class="big-price">₹{curr_price:.2f}</span>
            </div>
            <div style="text-align:right;">
                <h3 style="color:{color_code}; margin:0;">{signal}</h3>
                <p style="margin:0; color:#aaa;">RSI Strength: {rsi:.1f}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. CHART SECTION
    st.subheader("📊 Live Technical Chart")
    st.plotly_chart(plot_chart(df, ticker), use_container_width=True)
    
    # 3. MATHEMATICAL EQUATIONS SECTION
    st.markdown("### 🧮 Mathematical Logic (Behind the Scene)")
    
    # Calculating targets based on ATR
    sl_value = curr_price - (atr * 1.5)
    tgt_value = curr_price + (atr * 2.5)
    
    # Avoid division by zero if price is weird, though unlikely
    if curr_price > 0:
        qty = int(budget // curr_price)
    else:
        qty = 0
    
    col_math1, col_math2 = st.columns(2)
    with col_math1:
        st.markdown(f"""
        <div class="info-card">
            <h4>📉 Why this Signal?</h4>
            <ul style="list-style-type:none; padding:0;">
                <li><b>RSI Equation:</b> Value is {rsi:.1f} <br><span class="math-text">(If < 30 = Cheap, > 70 = Expensive)</span></li>
                <li><b>Trend Check:</b> Price {'>' if curr_price > sma50 else '<'} SMA(50) <br><span class="math-text">(Above 50 SMA = Bullish)</span></li>
                <li><b>Volatility (ATR):</b> ₹{atr:.2f} daily movement</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_math2:
        st.markdown(f"""
        <div class="info-card">
            <h4>🎯 Target & SL Formula</h4>
            <p class="math-text">
            <b>Stop Loss = Price - (1.5 × ATR)</b><br>
            {curr_price:.0f} - (1.5 × {atr:.0f}) = <b style="color:#ff5252;">₹{sl_value:.2f}</b>
            </p>
            <p class="math-text">
            <b>Target = Price + (2.5 × ATR)</b><br>
            {curr_price:.0f} + (2.5 × {atr:.0f}) = <b style="color:#00e676;">₹{tgt_value:.2f}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 4. PRO TIPS SECTION
    st.subheader("💡 Pro Tips & Action Plan")
    
    time_advice = get_time_strategy()
    
    st.markdown(f"""
    <div class="success-card">
        <h3 style="margin-top:0;">🚀 Execution Strategy</h3>
        <p><b>1. Quantity to Buy:</b> {qty} Shares (Total Investment: ₹{qty*curr_price:.0f})</p>
        <p><b>2. Best Time to Act:</b> {time_advice}</p>
        <p><b>3. Exit Strategy:</b></p>
        <ul>
            <li>Agar price <b>₹{tgt_value:.2f}</b> touch kare -> <b>Profit Book karein.</b></li>
            <li>Agar price <b>₹{sl_value:.2f}</b> touch kare -> <b>Loss accept karein (Exit).</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# --- MAIN APP LOGIC ---
tab1, tab2 = st.tabs(["🏠 Home & Search", "⚡ Scanner (Safe vs High Risk)"])

# TAB 1: SEARCH (LEGACY)
with tab1:
    st.header("🔎 Manual Analysis")
    tick = st.text_input("Stock Symbol", "RELIANCE")
    if st.button("Check Stock"):
        with st.spinner('Analyzing...'):
            d = process_stock_data(tick.upper())
            if d is not None:
                display_deep_dive(tick.upper(), d, 25000, "Intraday")
            else:
                st.error("Invalid Stock Symbol or Data Not Available")

# TAB 2: THE SCANNER (REQUESTED FLOW)
with tab2:
    st.header("🤖 AI Market Scanner")
    
    c1, c2 = st.columns(2)
    with c1:
        budget = st.number_input("Capital (₹)", value=50000, step=5000)
    with c2:
        mode = st.selectbox("Mode", ["Intraday", "Delivery"])
        
    if st.button("🔄 Scan Market Now"):
        
        # LIST OF STOCKS
        all_stocks = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'TATAMOTORS', 'ITC', 
            'LT', 'BAJFINANCE', 'MARUTI', 'TITAN', 'ULTRACEMCO', 'SUNPHARMA', 'ADANIENT', 
            'JINDALSTEL', 'DLF', 'IDEA', 'YESBANK', 'ZOMATO', 'PAYTM', 'WIPRO', 'HCLTECH'
        ]
        
        safe_picks = []
        risky_picks = []
        
        bar = st.progress(0, "Scanning NSE Stocks...")
        
        for i, s in enumerate(all_stocks):
            bar.progress((i+1)/len(all_stocks))
            try:
                df = process_stock_data(s)
                if df is None: continue
                
                price = df['Close'].iloc[-1]
                rsi = df['RSI'].iloc[-1]
                atr = df['ATR'].iloc[-1]
                volatility_pct = (atr / price) * 100
                
                # Logic for SAFE (Steady movement, reasonable RSI)
                if volatility_pct < 2.5 and (rsi < 40 or rsi > 60):
                    safe_picks.append({"Stock": s, "Price": price, "Data": df})
                    
                # Logic for HIGH RISK (High Volatility)
                elif volatility_pct >= 2.5: 
                    risky_picks.append({"Stock": s, "Price": price, "Data": df, "Vol": volatility_pct})
            except: continue
            
        bar.empty()
        
        # SAVE TO SESSION STATE
        st.session_state['safe_list'] = safe_picks[:7] # Limit to Top 7
        st.session_state['risk_list'] = risky_picks[:6] # Limit to Top 6
        st.session_state['scanned'] = True

    # --- DISPLAY RESULTS ---
    if st.session_state.get('scanned'):
        
        # SECTION A: SAFE STOCKS
        st.markdown("## 🛡️ Top Safe & Steady Picks")
        safe_list = st.session_state['safe_list']
        
        if safe_list:
            # Dropdown for interaction
            safe_options = [f"{x['Stock']} (₹{x['Price']:.2f})" for x in safe_list]
            s_opt = st.selectbox("👇 Select a Safe Company to Analyze:", safe_options, key="safe_select")
            
            # Find selected data
            s_ticker = s_opt.split(" (")[0]
            s_data_obj = next(x for x in safe_list if x['Stock'] == s_ticker)
            
            # Show Deep Dive
            st.markdown("---")
            display_deep_dive(s_ticker, s_data_obj['Data'], budget, mode)
        else:
            st.info("No safe patterns found today. Market might be choppy.")
            
        # FIX FOR PREVIOUS ERROR: Single Quotes used here
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # SECTION B: HIGH RISK STOCKS
        st.markdown("## 🔥 Top High Risk (Jackpot Zone)")
        st.caption("Warning: In stocks mein movement bohot tez hoti hai. High Risk, High Reward.")
        
        risk_list = st.session_state['risk_list']
        
        if risk_list:
            # Dropdown for interaction
            risk_options = [f"{x['Stock']} (Vol: {x['Vol']:.1f}%)" for x in risk_list]
            r_opt = st.selectbox("👇 Select a High Risk Company:", risk_options, key="risk_select")
            
            # Find selected data
            r_ticker = r_opt.split(" (")[0]
            r_data_obj = next(x for x in risk_list if x['Stock'] == r_ticker)
            
            # Show Deep Dive (Using Risk Card Style for header)
            st.markdown("---")
            st.markdown(f'<div class="risk-card"><h2 style="color:white; margin:0;">⚠️ ANALYZING HIGH RISK: {r_ticker}</h2></div>', unsafe_allow_html=True)
            display_deep_dive(r_ticker, r_data_obj['Data'], budget, mode)
            
        else:
            st.info("No high volatility stocks found currently.")



#key
 #python -m streamlit run stocks.py 
  
    
 # WIPRO

 # 230.72
