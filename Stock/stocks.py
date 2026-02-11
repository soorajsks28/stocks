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
st.set_page_config(page_title="Market Master Pro Mobile", layout="wide")

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
    .big-price { font-size: 32px; font-weight: bold; color: #ffffff; }
    .math-text { font-family: 'Courier New', monospace; font-size: 13px; color: #bbb; }
    
    /* Mobile Button Adjustment */
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; height: 45px; }
    
    /* Graph Container */
    .js-plotly-plot { margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Market Master AI: Mobile Pro")

# --- HELPER: TIME STRATEGY ---
def get_time_strategy():
    IST = pytz.timezone('Asia/Kolkata')
    now = datetime.now(IST)
    
    if now.hour < 10: 
        return "☕ **Morning (9:15-10:00):** Volatility High hai. Wait karein."
    elif now.hour < 13: 
        return "✅ **Prime Time:** Entry lene ka best time."
    elif now.hour >= 13 and now.hour < 14:
        return "⏸️ **Lunch:** Volume low ho sakta hai."
    elif now.hour >= 14: 
        return "🚀 **2:00 PM Move:** Trend fast ho sakta hai."
    else:
        return "🌙 **After Market:** Analysis mode."

# --- HELPER: CALCULATE DATA ---
def process_stock_data(symbol):
    try:
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

# --- HELPER: MOBILE OPTIMIZED CHART (UPDATED) ---
def plot_chart(df, symbol):
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='Price'))
    
    # Indicators
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='orange', width=1), name='50 SMA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='BB Top'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='BB Bot'))
    
    # MOBILE LAYOUT FIXES
    fig.update_layout(
        title=dict(text=f"{symbol} Chart", font=dict(size=14)),
        template="plotly_dark", 
        height=400, # Fixed height specifically for mobile
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=20), # Tight margins for small screens
        legend=dict(
            orientation="h", # Horizontal Legend (Saves vertical space)
            yanchor="bottom", y=1.02, 
            xanchor="right", x=1
        ),
        dragmode="pan" # Better for touch scrolling
    )
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
                <h3 style="margin:0;">{ticker}</h3>
                <span class="big-price">₹{curr_price:.2f}</span>
            </div>
            <div style="text-align:right;">
                <h4 style="color:{color_code}; margin:0;">{signal}</h4>
                <p style="margin:0; color:#aaa; font-size:12px;">RSI: {rsi:.1f}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. CHART SECTION (MOBILE CONFIG ADDED)
    st.subheader("📊 Technical Chart")
    # Config disables scroll zoom so page doesn't get stuck
    st.plotly_chart(
        plot_chart(df, ticker), 
        use_container_width=True,
        config={'displayModeBar': False, 'scrollZoom': False} 
    )
    
    # 3. MATHEMATICAL EQUATIONS SECTION
    st.markdown("### 🧮 Math Logic")
    
    sl_value = curr_price - (atr * 1.5)
    tgt_value = curr_price + (atr * 2.5)
    
    if curr_price > 0: qty = int(budget // curr_price)
    else: qty = 0
    
    col_math1, col_math2 = st.columns(2)
    with col_math1:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="font-size:16px;">📉 Indicators</h4>
            <ul style="padding-left:15px; font-size:13px; color:#ddd;">
                <li><b>RSI:</b> {rsi:.1f} <br><span style="color:#888;">(30=Cheap, 70=Exp)</span></li>
                <li><b>Trend:</b> {'>' if curr_price > sma50 else '<'} SMA50</li>
                <li><b>Vol (ATR):</b> ₹{atr:.2f}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_math2:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="font-size:16px;">🎯 Targets</h4>
            <p class="math-text">
            <b>SL:</b> {curr_price:.0f} - {atr*1.5:.0f} = <b style="color:#ff5252;">₹{sl_value:.0f}</b>
            </p>
            <p class="math-text">
            <b>Tgt:</b> {curr_price:.0f} + {atr*2.5:.0f} = <b style="color:#00e676;">₹{tgt_value:.0f}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 4. PRO TIPS SECTION
    st.subheader("💡 Action Plan")
    time_advice = get_time_strategy()
    
    st.markdown(f"""
    <div class="success-card">
        <p><b>1. Buy Qty:</b> {qty} Shares (₹{qty*curr_price:.0f})</p>
        <p><b>2. Time:</b> {time_advice}</p>
        <p><b>3. Exit Rules:</b></p>
        <ul style="font-size:14px;">
            <li>Profit: <b>₹{tgt_value:.2f}</b></li>
            <li>Loss: <b>₹{sl_value:.2f}</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# --- MAIN APP LOGIC ---
tab1, tab2 = st.tabs(["🏠 Search", "⚡ Scanner"])

# TAB 1: SEARCH
with tab1:
    st.header("🔎 Manual Check")
    tick = st.text_input("Stock Symbol", "RELIANCE")
    if st.button("Analyze Stock"):
        with st.spinner('Checking...'):
            d = process_stock_data(tick.upper())
            if d is not None:
                display_deep_dive(tick.upper(), d, 25000, "Intraday")
            else:
                st.error("Stock Not Found")

# TAB 2: SCANNER
with tab2:
    st.header("🤖 AI Scanner")
    
    c1, c2 = st.columns(2)
    with c1: budget = st.number_input("Capital", value=50000, step=5000)
    with c2: mode = st.selectbox("Mode", ["Intraday", "Delivery"])
        
    if st.button("🔄 Scan Market"):
        
        all_stocks = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'TATAMOTORS', 'ITC', 
            'LT', 'BAJFINANCE', 'MARUTI', 'TITAN', 'ULTRACEMCO', 'SUNPHARMA', 'ADANIENT', 
            'JINDALSTEL', 'DLF', 'ZOMATO', 'PAYTM', 'WIPRO'
        ]
        
        safe_picks = []
        risky_picks = []
        
        bar = st.progress(0, "Scanning...")
        
        for i, s in enumerate(all_stocks):
            bar.progress((i+1)/len(all_stocks))
            try:
                df = process_stock_data(s)
                if df is None: continue
                
                price = df['Close'].iloc[-1]
                rsi = df['RSI'].iloc[-1]
                atr = df['ATR'].iloc[-1]
                volatility_pct = (atr / price) * 100
                
                if volatility_pct < 2.5 and (rsi < 40 or rsi > 60):
                    safe_picks.append({"Stock": s, "Price": price, "Data": df})
                elif volatility_pct >= 2.5: 
                    risky_picks.append({"Stock": s, "Price": price, "Data": df, "Vol": volatility_pct})
            except: continue
            
        bar.empty()
        st.session_state['safe_list'] = safe_picks[:7]
        st.session_state['risk_list'] = risky_picks[:6]
        st.session_state['scanned'] = True

    if st.session_state.get('scanned'):
        
        # SAFE STOCKS
        st.markdown("### 🛡️ Safe Picks")
        safe_list = st.session_state['safe_list']
        
        if safe_list:
            s_opt = st.selectbox("👇 Select Safe Stock:", [f"{x['Stock']} (₹{x['Price']:.0f})" for x in safe_list], key="safe_select")
            s_ticker = s_opt.split(" (")[0]
            s_data_obj = next(x for x in safe_list if x['Stock'] == s_ticker)
            st.markdown("---")
            display_deep_dive(s_ticker, s_data_obj['Data'], budget, mode)
        else:
            st.info("No safe picks found.")
            
        st.markdown("---")
        
        # HIGH RISK STOCKS
        st.markdown("### 🔥 High Risk (Jackpot)")
        risk_list = st.session_state['risk_list']
        
        if risk_list:
            r_opt = st.selectbox("👇 Select Risky Stock:", [f"{x['Stock']} (Vol: {x['Vol']:.1f}%)" for x in risk_list], key="risk_select")
            r_ticker = r_opt.split(" (")[0]
            r_data_obj = next(x for x in risk_list if x['Stock'] == r_ticker)
            st.markdown("---")
            st.markdown(f'<div class="risk-card"><h3 style="color:white; margin:0;">⚠️ HIGH RISK: {r_ticker}</h3></div>', unsafe_allow_html=True)
            display_deep_dive(r_ticker, r_data_obj['Data'], budget, mode)
        else:
            st.info("No high risk stocks found.")



#key
 #python -m streamlit run stocks.py 
  
    
 # WIPRO

 # 230.72
