import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from datetime import datetime
import pytz

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Market Master AI Pro", layout="wide")

# --- CUSTOM CSS (Styling) ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; background-color: #007bff; color: white; font-weight: bold; padding: 10px; }
    .metric-card { background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333; margin-bottom: 10px; }
    .success-box { padding: 15px; background-color: #d4edda; color: #155724; border-radius: 10px; border: 1px solid #c3e6cb; margin-bottom: 10px; }
    .warning-box { padding: 15px; background-color: #fff3cd; color: #856404; border-radius: 10px; border: 1px solid #ffeeba; margin-bottom: 10px; }
    .danger-box { padding: 15px; background-color: #f8d7da; color: #721c24; border-radius: 10px; border: 1px solid #f5c6cb; margin-bottom: 10px; }
    .pro-tip { padding: 15px; background-color: #e3f2fd; color: #0d47a1; border-radius: 10px; border-left: 5px solid #2196f3; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 AI Market Master (With Exit Strategy)")

# --- TABS ---
tab1, tab2 = st.tabs(["🔍 Analyze Single Stock", "💰 AI Wealth Manager (Scanner)"])

# --- HELPER FUNCTION: GET SELL TIME ---
def get_sell_advice():
    # Indian Standard Time (IST) fetch karna
    IST = pytz.timezone('Asia/Kolkata')
    now = datetime.now(IST)
    current_time = now.strftime("%H:%M")
    
    # Logic for Selling Time
    advice = ""
    if now.hour < 11:
        advice = "🕚 **Best Sell Time:** Subah 11:00 AM se pehle profit book kar lein. (Morning Volatility)"
    elif now.hour < 13:
        advice = "🕐 **Best Sell Time:** 1:30 PM tak hold kar sakte hain, uske baad European market khulta hai."
    else:
        advice = "⚠️ **URGENT:** 3:10 PM se pehle har haal mein sell karein (Auto-Square off se bachein)."
        
    return advice

# ==========================================
# TAB 1: SINGLE STOCK ANALYSIS
# ==========================================
with tab1:
    st.header("Search & Analyze")
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        ticker = st.text_input("Stock Name (e.g., RELIANCE, ADANIENT)", "RELIANCE")
    with col_btn:
        st.write("") 
        st.write("") 
        btn_analyze = st.button("Analyze Now 🚀")

    if btn_analyze:
        symbol = ticker.upper()
        if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
            symbol += ".NS"
        
        try:
            with st.spinner('Checking Data...'):
                data = yf.download(symbol, period="1y", interval="1d", progress=False)
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                if data.empty:
                    st.error("Stock nahi mila! Spelling check karein.")
                else:
                    close = data['Close']
                    current_price = float(close.iloc[-1])
                    rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
                    sma20 = SMAIndicator(close, window=20).sma_indicator().iloc[-1]
                    sma50 = SMAIndicator(close, window=50).sma_indicator().iloc[-1]

                    # Metrics
                    st.subheader(f"📊 Report: {symbol.replace('.NS', '')}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Current Price", f"₹{current_price:.2f}")
                    m2.metric("RSI Level", f"{rsi:.2f}")
                    trend = "UP 🟢" if current_price > sma50 else "DOWN 🔴"
                    m3.metric("Trend (50 Days)", trend)
                    
                    st.divider()

                    c1, c2 = st.columns(2)
                    
                    # Intraday Logic with PRO TIPS
                    with c1:
                        st.info("⚡ **INTRADAY (Aaj ke liye)**")
                        sell_time_advice = get_sell_advice()
                        
                        if rsi < 30:
                            st.markdown(f"""
                            <div class="success-box">
                                ✅ <b>STRONG BUY</b><br>Reason: Stock Oversold hai, bounce karega.
                            </div>
                            <div class="pro-tip">
                                💡 <b>Pro Tips (Exit Strategy):</b><br>
                                1. {sell_time_advice}<br>
                                2. <b>Target:</b> ₹{current_price * 1.015:.2f} (1.5% Profit)<br>
                                3. <b>Stop Loss:</b> ₹{current_price * 0.99:.2f} (1% Loss)
                            </div>
                            """, unsafe_allow_html=True)
                        elif rsi > 70:
                            st.markdown('<div class="danger-box">❌ <b>SELL / EXIT</b><br>Reason: Stock Mehenga hai.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="warning-box">⚠️ <b>WAIT</b><br>Reason: Market neutral hai.</div>', unsafe_allow_html=True)

                    # Delivery Logic
                    with c2:
                        st.info("📦 **DELIVERY (Investment)**")
                        if current_price > sma20 and sma20 > sma50:
                            st.markdown(f'<div class="success-box">✅ <b>BUY FOR HOLD</b><br>Target: 10-15% return in 3 months.</div>', unsafe_allow_html=True)
                        elif current_price < sma50:
                            st.markdown('<div class="danger-box">❌ <b>AVOID</b><br>Reason: Downtrend.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="warning-box">⚠️ <b>HOLD</b><br>Reason: Market sideways hai.</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# TAB 2: AI SCANNER
# ==========================================
with tab2:
    st.header("AI Wealth Manager (Auto-Scanner)")
    
    c_budget, c_mode = st.columns(2)
    with c_budget:
        budget = st.number_input("💰 Aapka Budget (Rs)", min_value=1000, value=10000, step=1000)
    with c_mode:
        mode = st.radio("📈 Trade Type", ["Intraday (Aaj ka Profit)", "Delivery (Investment)"])
    
    btn_scan = st.button("🔍 Best Stocks Dhundo")
    
    stock_list = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
        'SBIN.NS', 'TATAMOTORS.NS', 'ITC.NS', 'AXISBANK.NS', 'LT.NS', 
        'BAJFINANCE.NS', 'MARUTI.NS', 'WIPRO.NS', 'ASIANPAINT.NS'
    ]

    if btn_scan:
        recommendations = []
        progress_bar = st.progress(0)
        
        with st.spinner('Scanning Market...'):
            for i, s_symbol in enumerate(stock_list):
                progress_bar.progress((i + 1) / len(stock_list))
                try:
                    s_data = yf.download(s_symbol, period="1y", interval="1d", progress=False)
                    if isinstance(s_data.columns, pd.MultiIndex):
                        s_data.columns = s_data.columns.get_level_values(0)
                    
                    if s_data.empty: continue
                    
                    s_close = s_data['Close']
                    s_price = float(s_close.iloc[-1])
                    if s_price > budget: continue

                    s_rsi = RSIIndicator(s_close, window=14).rsi().iloc[-1]
                    s_sma20 = SMAIndicator(s_close, window=20).sma_indicator().iloc[-1]
                    s_sma50 = SMAIndicator(s_close, window=50).sma_indicator().iloc[-1]
                    
                    score = 0
                    reason = ""
                    
                    # Ranking
                    if "Intraday" in mode:
                        if s_rsi < 30:
                            score = 95
                            reason = "Oversold (RSI < 30)"
                        elif s_rsi < 40:
                            score = 80
                            reason = "Buying Zone"
                    else: 
                        if s_price > s_sma20 and s_sma20 > s_sma50:
                            score = 95
                            reason = "Strong Uptrend"
                        elif s_price > s_sma50:
                            score = 85
                            reason = "Positive Trend"

                    if score >= 80:
                        qty = int(budget // s_price)
                        recommendations.append({
                            "Stock": s_symbol.replace('.NS', ''),
                            "Price": s_price,
                            "Qty": qty,
                            "Total": qty * s_price,
                            "Reason": reason,
                            "Score": score
                        })
                except: continue
        
        progress_bar.empty()
        
        if not recommendations:
            st.warning("⚠️ Aaj market thanda hai. Koi trade mat lo.")
        else:
            recommendations.sort(key=lambda x: x['Score'], reverse=True)
            top = recommendations[0]
            sell_advice = get_sell_advice()
            
            st.success(f"🎉 Top Pick: {top['Stock']}")
            
            # Intraday specific advice in Scanner
            advice_html = ""
            if "Intraday" in mode:
                advice_html = f"""
                <div class="pro-tip">
                    <b>🕒 Sell Time:</b> {sell_advice}<br>
                    <b>🎯 Target Price:</b> ₹{top['Price'] * 1.015:.2f} (Sell here)<br>
                    <b>🛑 Stop Loss:</b> ₹{top['Price'] * 0.99:.2f} (Exit here)
                </div>
                """
            
            st.markdown(f"""
            <div class="success-box" style="border: 2px solid green; text-align: center;">
                <h2>🏆 Buy {top['Qty']} Shares of {top['Stock']}</h2>
                <h3>Price: ₹{top['Price']:.2f}</h3>
                <p>Invested: ₹{top['Total']:.2f}</p>
                <p>Reason: {top['Reason']}</p>
            </div>
            {advice_html}
            """, unsafe_allow_html=True)



#key
 #python -m streamlit run stocks.py 
  
    
 # WIPRO
 # 230.72