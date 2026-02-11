import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import math
import time
import requests
import warnings

# Suppress pandas warnings for cleaner UI
warnings.filterwarnings('ignore')

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Market Master Titan Infinity",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# --- 2. TITAN CSS ENGINE (Advanced Styling) ---
st.markdown("""
<style>
    /* Global Variables */
    :root {
        --primary: #00E676;
        --secondary: #2979FF;
        --danger: #FF1744;
        --warning: #FFC400;
        --bg-dark: #0A0E14;
        --card-bg: #131722;
        --text-primary: #FFFFFF;
        --text-secondary: #B2B5BE;
        --border-color: #2A2E39;
    }
    
    /* Main Background */
    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Titan Header Gradient */
    .titan-header {
        font-size: 48px;
        font-weight: 900;
        text-transform: uppercase;
        background: linear-gradient(90deg, #00E676 0%, #2979FF 50%, #FF1744 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        letter-spacing: 3px;
        text-shadow: 0px 0px 20px rgba(0, 230, 118, 0.3);
    }
    
    /* Glassmorphism Card */
    .stat-card {
        background: rgba(19, 23, 34, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 24px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--primary);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .stat-card:hover {
        border-color: var(--primary);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        transform: translateY(-5px);
    }
    
    .stat-card:hover::before {
        opacity: 1;
    }
    
    /* Typography */
    .ticker-symbol {
        font-size: 28px;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: 1px;
    }
    
    .ticker-price {
        font-size: 36px;
        font-weight: 700;
        color: var(--primary);
        font-variant-numeric: tabular-nums;
        margin: 10px 0;
    }
    
    /* Signal Badges */
    .signal-badge {
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .signal-buy { background: rgba(0, 230, 118, 0.2); color: #00E676; border: 1px solid #00E676; }
    .signal-sell { background: rgba(255, 23, 68, 0.2); color: #FF1744; border: 1px solid #FF1744; }
    .signal-hold { background: rgba(255, 196, 0, 0.2); color: #FFC400; border: 1px solid #FFC400; }
    
    /* Custom Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2962FF 0%, #1565C0 100%);
        color: white;
        border: none;
        padding: 14px 24px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 1px;
        transition: all 0.3s;
        text-transform: uppercase;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #448AFF 0%, #2962FF 100%);
        box-shadow: 0 8px 24px rgba(41, 98, 255, 0.5);
        transform: translateY(-2px);
    }
    
    /* Buy/Sell Specific Buttons */
    .buy-btn > button { background: linear-gradient(135deg, #00E676 0%, #00C853 100%) !important; }
    .sell-btn > button { background: linear-gradient(135deg, #FF1744 0%, #D50000 100%) !important; }
    
    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #1A1E29;
        color: white;
        border: 1px solid #333;
        border-radius: 8px;
        height: 50px;
        font-size: 18px;
        padding-left: 15px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
        padding-bottom: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255,255,255,0.03);
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 600;
        border: 1px solid transparent;
        transition: all 0.2s;
        padding: 0 20px;
    }
    
    .stTabs [aria-selected="true"] {
        color: #fff;
        background-color: rgba(41, 98, 255, 0.1);
        border-color: var(--secondary);
    }
    
    /* Dataframes */
    .stDataFrame { border: none !important; }
    .dataframe { background-color: var(--card-bg); font-family: 'Roboto Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# --- 3. MATH ENGINE (Core Calculation Logic) ---
class TechnicalIndicators:
    @staticmethod
    def sma(series, period):
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def wma(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def bollinger_bands(series, period=20, std_dev=2):
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def supertrend(high, low, close, period=10, multiplier=3):
        atr_val = TechnicalIndicators.atr(high, low, close, period)
        hl2 = (high + low) / 2
        up = hl2 - (multiplier * atr_val)
        down = hl2 + (multiplier * atr_val)
        st = pd.Series(0.0, index=close.index)
        trend = pd.Series(0, index=close.index)
        
        for i in range(1, len(close)):
            if close.iloc[i-1] > up.iloc[i-1]:
                up.iloc[i] = max(up.iloc[i], up.iloc[i-1])
            else:
                up.iloc[i] = up.iloc[i]
            
            if close.iloc[i-1] < down.iloc[i-1]:
                down.iloc[i] = min(down.iloc[i], down.iloc[i-1])
            else:
                down.iloc[i] = down.iloc[i]
            
            if close.iloc[i] > down.iloc[i-1]:
                trend.iloc[i] = 1
            elif close.iloc[i] < up.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]
                if trend.iloc[i] == 1:
                    up.iloc[i] = up.iloc[i-1]
                else:
                    down.iloc[i] = down.iloc[i-1]
            
            if trend.iloc[i] == 1:
                st.iloc[i] = up.iloc[i]
            else:
                st.iloc[i] = down.iloc[i]
        
        return st, trend

    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()
        k = 100 * ((close - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        return k, d

    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = TechnicalIndicators.atr(high, low, close, period)
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
        minus_di = 100 * (abs(minus_dm).ewm(alpha=1/period).mean() / tr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx_val = dx.ewm(alpha=1/period).mean()
        return adx_val, plus_di, minus_di

    @staticmethod
    def ichimoku(high, low, close):
        tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        chikou = close.shift(-26)
        return tenkan, kijun, senkou_a, senkou_b, chikou

    @staticmethod
    def vwap(high, low, close, volume):
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def cci(high, low, close, period=20):
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad)

# --- 4. MARKET ANALYZER (Data Processing) ---
class MarketAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = pd.DataFrame()
        
    def get_data(self, period="1y", interval="1d"):
        try:
            self.df = yf.download(self.ticker, period=period, interval=interval, progress=False)
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = self.df.columns.get_level_values(0)
            return not self.df.empty
        except:
            return False

    def get_live_quote(self):
        try:
            ticker_obj = yf.Ticker(self.ticker)
            data = ticker_obj.history(period="1d")
            if not data.empty:
                return data.iloc[-1]['Close'], data.iloc[-1]
            return None, None
        except:
            return None, None

    def process_technical_data(self):
        if self.df.empty: return
        
        h, l, c, v = self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume']
        
        # Moving Averages
        self.df['SMA_20'] = TechnicalIndicators.sma(c, 20)
        self.df['SMA_50'] = TechnicalIndicators.sma(c, 50)
        self.df['SMA_200'] = TechnicalIndicators.sma(c, 200)
        self.df['EMA_20'] = TechnicalIndicators.ema(c, 20)
        
        # Momentum
        self.df['RSI'] = TechnicalIndicators.rsi(c)
        self.df['Stoch_K'], self.df['Stoch_D'] = TechnicalIndicators.stochastic(h, l, c)
        self.df['CCI'] = TechnicalIndicators.cci(h, l, c)
        
        # Trend
        self.df['MACD'], self.df['MACD_Signal'], self.df['MACD_Hist'] = TechnicalIndicators.macd(c)
        self.df['ADX'], self.df['DI_Plus'], self.df['DI_Minus'] = TechnicalIndicators.adx(h, l, c)
        self.df['SuperTrend'], self.df['ST_Dir'] = TechnicalIndicators.supertrend(h, l, c)
        
        # Volatility
        self.df['BB_Up'], self.df['BB_Mid'], self.df['BB_Low'] = TechnicalIndicators.bollinger_bands(c)
        self.df['ATR'] = TechnicalIndicators.atr(h, l, c)
        
        # Advanced
        self.df['Tenkan'], self.df['Kijun'], self.df['SpanA'], self.df['SpanB'], self.df['Chikou'] = TechnicalIndicators.ichimoku(h, l, c)
        self.df['VWAP'] = TechnicalIndicators.vwap(h, l, c, v)

    def calculate_pivot_points(self):
        row = self.df.iloc[-1]
        h, l, c = row['High'], row['Low'], row['Close']
        p = (h + l + c) / 3
        return {
            "P": p, "R1": (2*p)-l, "S1": (2*p)-h, 
            "R2": p+(h-l), "S2": p-(h-l), 
            "R3": h+2*(p-l), "S3": l-2*(h-p)
        }

    def score_stock(self, mode):
        if self.df.empty: return 0, []
        
        score = 0
        reasons = []
        curr = self.df.iloc[-1]
        rsi = curr['RSI']
        close = curr['Close']
        
        if mode == "INTRADAY":
            if rsi > 50 and rsi < 70: score += 10; reasons.append("Momentum (RSI)")
            if rsi < 30: score += 20; reasons.append("Oversold (RSI < 30)")
            if curr['ST_Dir'] == 1: score += 25; reasons.append("SuperTrend Bullish")
            if close > curr['VWAP']: score += 15; reasons.append("Price > VWAP")
            if curr['MACD'] > curr['MACD_Signal']: score += 10; reasons.append("MACD Bull Cross")
            if curr['ADX'] > 25: score += 10; reasons.append("Strong Trend (ADX)")
            if close > curr['EMA_20']: score += 10; reasons.append("Above 20 EMA")
        else:
            if close > curr['SMA_200']: score += 30; reasons.append("Long Term Bull (>200 SMA)")
            if curr['SMA_50'] > curr['SMA_200']: score += 15; reasons.append("Golden Cross")
            if rsi > 40 and rsi < 65: score += 10; reasons.append("Stable RSI")
            if curr['SpanA'] > curr['SpanB']: score += 15; reasons.append("Ichimoku Cloud Green")
            if close < curr['BB_Low']: score += 20; reasons.append("Deep Value (Lower BB)")
            if curr['MACD'] > 0: score += 10; reasons.append("Positive Trend Cycle")
            
        return min(score, 100), reasons

# --- 5. CHART FACTORY (Visualizations) ---
class ChartFactory:
    @staticmethod
    def create_advanced_chart(df, ticker, mode):
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, 
            vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2]
        )
        
        # Price
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price'
        ), row=1, col=1)
        
        if mode == "INTRADAY":
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#FFD700', width=1), name='VWAP'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(color='gray', width=1, dash='dot'), name='BB'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', width=1, dash='dot'), showlegend=False), row=1, col=1)
            colors = ['#00E676' if x == 1 else '#FF1744' for x in df['ST_Dir']]
            fig.add_trace(go.Scatter(x=df.index, y=df['SuperTrend'], mode='markers', marker=dict(color=colors, size=3), name='SuperTrend'), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#2979FF', width=1), name='50 SMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='#AA00FF', width=2), name='200 SMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SpanA'], line=dict(color='rgba(0,230,118,0.2)'), fill=None, name='Cloud'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SpanB'], line=dict(color='rgba(255,23,68,0.2)'), fill='tonexty', showlegend=False), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#E040FB', width=2), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        colors = ['#00E676' if v >= 0 else '#FF1744' for v in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors, name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#2979FF'), name='MACD Line'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#FF9100'), name='Signal'), row=3, col=1)
        
        fig.update_layout(
            template="plotly_dark", height=700, margin=dict(l=0,r=0,t=0,b=0),
            xaxis_rangeslider_visible=False, showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    @staticmethod
    def create_gauge(score):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "AI CONFIDENCE", 'font': {'size': 14, 'color': "white"}},
            number={'font': {'size': 30, 'color': "white"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#2979FF"},
                'bgcolor': "black",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, 30], 'color': '#FF1744'},
                    {'range': [30, 70], 'color': '#FFC400'},
                    {'range': [70, 100], 'color': '#00E676'}
                ]
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=20), paper_bgcolor='rgba(0,0,0,0)')
        return fig

# --- 6. PORTFOLIO MANAGER ---
class PortfolioManager:
    @staticmethod
    def init():
        if 'portfolio' not in st.session_state:
            st.session_state['portfolio'] = {
                'balance': 1000000.0,
                'holdings': {},
                'history': []
            }

    @staticmethod
    def buy(ticker, qty, price):
        pf = st.session_state['portfolio']
        cost = qty * price
        if pf['balance'] >= cost:
            pf['balance'] -= cost
            if ticker in pf['holdings']:
                h = pf['holdings'][ticker]
                new_qty = h['qty'] + qty
                new_avg = ((h['qty'] * h['avg']) + cost) / new_qty
                pf['holdings'][ticker] = {'qty': new_qty, 'avg': new_avg}
            else:
                pf['holdings'][ticker] = {'qty': qty, 'avg': price}
            pf['history'].append({'type': 'BUY', 'ticker': ticker, 'qty': qty, 'price': price, 'date': datetime.now()})
            return True, "Order Executed Successfully"
        return False, "Insufficient Funds"

    @staticmethod
    def sell(ticker, qty, price):
        pf = st.session_state['portfolio']
        if ticker in pf['holdings']:
            h = pf['holdings'][ticker]
            if h['qty'] >= qty:
                revenue = qty * price
                pf['balance'] += revenue
                h['qty'] -= qty
                if h['qty'] == 0:
                    del pf['holdings'][ticker]
                pf['history'].append({'type': 'SELL', 'ticker': ticker, 'qty': qty, 'price': price, 'date': datetime.now()})
                return True, "Order Executed Successfully"
            return False, "Insufficient Holdings"
        return False, "Stock Not Found in Portfolio"

    @staticmethod
    def get_metrics():
        pf = st.session_state['portfolio']
        total_inv = 0
        curr_val = 0
        h_list = []
        
        for t, d in pf['holdings'].items():
            try:
                ltp = yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
            except:
                ltp = d['avg'] # Fallback
            
            val = d['qty'] * ltp
            inv = d['qty'] * d['avg']
            total_inv += inv
            curr_val += val
            
            h_list.append({
                'Ticker': t, 'Qty': d['qty'], 'Avg': d['avg'], 'LTP': ltp,
                'Invested': inv, 'Value': val, 'P&L': val - inv,
                'P&L %': ((val - inv) / inv) * 100 if inv > 0 else 0
            })
            
        return pf['balance'], total_inv, curr_val, h_list

PortfolioManager.init()

# --- 7. MAIN UI LAYOUT ---
st.markdown("<div class='titan-header'>Market Master Titan Infinity</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎮 SYSTEM CONTROL")
    capital_input = st.number_input("SIMULATION CAPITAL (₹)", 10000, 10000000, 100000, 10000)
    risk_tol = st.slider("RISK TOLERANCE", 0.5, 5.0, 2.0, 0.1)
    
    st.markdown("---")
    st.markdown("### 💼 PORTFOLIO LIVE")
    cash, inv, cur, h_list = PortfolioManager.get_metrics()
    st.metric("CASH BALANCE", f"₹{cash:,.2f}")
    st.metric("NET WORTH", f"₹{cash + cur:,.2f}", delta=f"{(cash+cur)-1000000:,.2f}")
    
    if st.button("🔴 RESET SIMULATION"):
        st.session_state['portfolio'] = {'balance': 1000000.0, 'holdings': {}, 'history': []}
        st.rerun()

# Tabs
main_tabs = st.tabs([
    "🔥 TITAN SCANNER", 
    "⚡ INSTANT TERMINAL", 
    "🔬 DEEP DIVE", 
    "🧠 KNOWLEDGE BASE", 
    "💎 PORTFOLIO HUB", 
    "⚖️ RISK CALCULATOR"
])

# --- TAB 1: SCANNER ---
with main_tabs[0]:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("### 📡 REAL-TIME MARKET SCANNER")
        mode = st.radio("SCAN ENGINE", ["INTRADAY", "DELIVERY"], horizontal=True)
    with c2:
        st.write("")
        st.write("")
        if st.button("🚀 INITIATE SCAN"):
            stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'TATAMOTORS', 'M&M', 'MARUTI', 'ADANIENT', 'SUNPHARMA', 'TITAN', 'BAJFINANCE', 'ULTRACEMCO', 'NTPC', 'POWERGRID', 'TATASTEEL', 'JSWSTEEL', 'COALINDIA', 'HINDALCO', 'GRASIM', 'CIPLA', 'WIPRO', 'DLF', 'ZOMATO', 'PAYTM', 'HAL', 'BEL', 'TRENT']
            
            res = []
            bar = st.progress(0)
            
            for i, s in enumerate(stocks):
                bar.progress((i+1)/len(stocks))
                try:
                    ma = MarketAnalyzer(s+".NS")
                    p = "5d" if mode == "INTRADAY" else "1y"
                    intr = "15m" if mode == "INTRADAY" else "1d"
                    
                    if ma.get_data(p, intr):
                        ma.process_technical_data()
                        score, rsn = ma.score_stock(mode)
                        
                        price = ma.df['Close'].iloc[-1]
                        atr = ma.df['ATR'].iloc[-1]
                        
                        if mode == "INTRADAY":
                            sl = price - (atr * 1.5)
                            tgt = price + (atr * 3)
                        else:
                            sl = price * 0.92
                            tgt = price * 1.20
                            
                        if score >= 40:
                            res.append({
                                "Symbol": s, "Price": price, "Score": score, "Reasons": rsn,
                                "SL": sl, "TGT": tgt, "ATR": atr, "DF": ma.df
                            })
                except: continue
            
            bar.empty()
            res.sort(key=lambda x: x['Score'], reverse=True)
            st.session_state['scan_results'] = res
            st.session_state['mode'] = mode

    if 'scan_results' in st.session_state:
        results = st.session_state['scan_results']
        if not results:
            st.warning("No high-probability setups found.")
        else:
            top_cols = st.columns(3)
            for idx in range(min(3, len(results))):
                item = results[idx]
                with top_cols[idx]:
                    st.markdown(f"""
                    <div class='stat-card'>
                        <div class='ticker-symbol'>{item['Symbol']}</div>
                        <div class='ticker-price'>₹{item['Price']:.2f}</div>
                        <div class='signal-badge signal-buy'>SCORE: {item['Score']}</div>
                        <div style='margin-top:10px; font-size:12px; color:#aaa;'>
                            TGT: <span style='color:#00E676'>₹{item['TGT']:.2f}</span> | SL: <span style='color:#FF1744'>₹{item['SL']:.2f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"OPEN DECK {item['Symbol']}", key=f"btn_{idx}"):
                        st.session_state['active_stock'] = item

            st.markdown("### 📋 FULL MARKET REPORT")
            for item in results:
                with st.expander(f"{item['Symbol']} | Score: {item['Score']} | ₹{item['Price']:.2f}"):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1: st.progress(item['Score']); st.write(f"Vol (ATR): {item['ATR']:.2f}")
                    with c2: 
                        for r in item['Reasons']: st.markdown(f"✅ {r}")
                    with c3:
                        if st.button(f"ANALYZE {item['Symbol']}", key=f"an_{item['Symbol']}"):
                            st.session_state['active_stock'] = item

# --- TAB 2: INSTANT TERMINAL (NEW FEATURE) ---
with main_tabs[1]:
    st.markdown("### ⚡ INSTANT TRADING TERMINAL")
    st.caption("Search ANY stock, view live data, and execute virtual trades instantly.")
    
    col_search, col_action = st.columns([3, 1])
    with col_search:
        search_sym = st.text_input("ENTER STOCK SYMBOL (e.g., ZOMATO, MRF, IDEA)", "RELIANCE")
    with col_action:
        st.write("")
        st.write("")
        btn_search = st.button("FETCH DATA")
        
    if btn_search or search_sym:
        sym_full = search_sym.upper() if search_sym.upper().endswith(".NS") else search_sym.upper() + ".NS"
        
        with st.spinner("Fetching Real-Time Data..."):
            ma_instant = MarketAnalyzer(sym_full)
            # Default to Delivery logic for instant check
            has_data = ma_instant.get_data("1y", "1d")
            
            if has_data:
                ma_instant.process_technical_data()
                score, reasons = ma_instant.score_stock("DELIVERY")
                curr_price = ma_instant.df['Close'].iloc[-1]
                
                # Layout
                st.markdown("---")
                header_col, score_col = st.columns([2, 1])
                
                with header_col:
                    st.markdown(f"<div class='ticker-symbol'>{search_sym.upper()}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='ticker-price'>₹{curr_price:.2f}</div>", unsafe_allow_html=True)
                
                with score_col:
                    st.metric("TITAN SCORE", f"{score}/100", delta="High Confidence" if score > 60 else "Low Confidence")
                
                # Chart
                st.plotly_chart(ChartFactory.create_advanced_chart(ma_instant.df, search_sym.upper(), "DELIVERY"), use_container_width=True)
                
                # Trading Interface
                st.markdown("### 🏦 EXECUTE ORDER")
                
                trade_c1, trade_c2 = st.columns(2)
                
                with trade_c1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4>🟢 BUY (LONG)</h4>
                        <p style="font-size:12px; color:#aaa;">Add to Portfolio using Cash Balance.</p>
                        <p>Available Cash: <span style="color:#00E676; font-weight:bold;">₹{st.session_state['portfolio']['balance']:,.2f}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    b_qty = st.number_input("Buy Quantity", 1, 100000, 10, key="b_qty")
                    st.markdown(f"**Total Cost:** ₹{b_qty * curr_price:,.2f}")
                    
                    if st.button("CONFIRM BUY ORDER", key="buy_btn"):
                        success, msg = PortfolioManager.buy(search_sym.upper(), b_qty, curr_price)
                        if success: st.success(msg)
                        else: st.error(msg)
                        
                with trade_c2:
                    st.markdown(f"""
                    <div class="stat-card" style="border-color:#FF1744;">
                        <h4>🔴 SELL (EXIT)</h4>
                        <p style="font-size:12px; color:#aaa;">Sell existing holdings to release Cash.</p>
                        <p>Current Holding: <span style="color:#FFC400; font-weight:bold;">{st.session_state['portfolio']['holdings'].get(search_sym.upper(), {'qty': 0})['qty']} Shares</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    s_qty = st.number_input("Sell Quantity", 1, 100000, 1, key="s_qty")
                    st.markdown(f"**Total Value:** ₹{s_qty * curr_price:,.2f}")
                    
                    if st.button("CONFIRM SELL ORDER", key="sell_btn"):
                        success, msg = PortfolioManager.sell(search_sym.upper(), s_qty, curr_price)
                        if success: st.success(msg)
                        else: st.error(msg)
                        
            else:
                st.error("Stock Not Found. Please check symbol (e.g., try 'TATASTEEL' instead of 'TATA').")

# --- TAB 3: DEEP DIVE ---
with main_tabs[2]:
    if 'active_stock' in st.session_state:
        data = st.session_state['active_stock']
        df = data['DF']
        symbol = data['Symbol']
        
        st.markdown(f"<div class='titan-header'>{symbol} INTELLIGENCE</div>", unsafe_allow_html=True)
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("LATEST PRICE", f"₹{data['Price']:.2f}")
        k2.metric("ALGO SCORE", f"{data['Score']}/100")
        k3.metric("R:R RATIO", f"1:{(data['TGT']-data['Price'])/(data['Price']-data['SL']):.2f}")
        k4.metric("VOLATILITY", f"{data['ATR']:.2f}")
        
        r1 = st.columns([3, 1])
        with r1[0]:
            st.plotly_chart(ChartFactory.create_advanced_chart(df, symbol, st.session_state['mode']), use_container_width=True)
            ma = MarketAnalyzer(symbol)
            ma.df = df
            lvls = ma.calculate_pivot_points()
            st.markdown("#### 🎯 INSTITUTIONAL PIVOTS")
            pc1, pc2, pc3, pc4, pc5 = st.columns(5)
            pc1.metric("S2", f"{lvls['S2']:.2f}"); pc2.metric("S1", f"{lvls['S1']:.2f}")
            pc3.metric("PIVOT", f"{lvls['P']:.2f}"); pc4.metric("R1", f"{lvls['R1']:.2f}"); pc5.metric("R2", f"{lvls['R2']:.2f}")
            
        with r1[1]:
            st.plotly_chart(ChartFactory.create_gauge(data['Score']), use_container_width=True)
            st.markdown(f"""
            <div class="stat-card" style="border-left: 4px solid #2979FF;">
                <h4>🏦 QUICK TRADE</h4>
                <div style="margin-bottom:10px;">
                    <span style="color:#FF1744; font-weight:bold;">STOP: ₹{data['SL']:.2f}</span><br>
                    <span style="color:#00E676; font-weight:bold;">TARGET: ₹{data['TGT']:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            risk_amt = capital_input * (risk_tol / 100)
            risk_per_share = data['Price'] - data['SL']
            rec_qty = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
            
            st.info(f"Safe Qty: {rec_qty}")
            qty_in = st.number_input("QTY", 1, 100000, rec_qty, key="dd_qty")
            
            if st.button(f"BUY {symbol}"):
                s, m = PortfolioManager.buy(symbol, qty_in, data['Price'])
                if s: st.success(m)
                else: st.error(m)
    else:
        st.info("👈 Select a stock from Scanner or use Instant Terminal.")

# --- TAB 4: KNOWLEDGE BASE ---
with main_tabs[3]:
    st.header("🧠 TITAN ACADEMY")
    kt1, kt2 = st.tabs(["INDICATORS", "STRATEGIES"])
    with kt1:
        with st.expander("Relative Strength Index (RSI)"): st.write("Momentum oscillator measuring speed/change of price movements.")
        with st.expander("MACD"): st.write("Trend-following momentum indicator showing relationship between two moving averages.")
        with st.expander("Bollinger Bands"): st.write("Volatility bands placed above and below a moving average.")
    with kt2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🚀 INTRADAY MOMENTUM")
            st.info("Entry: RSI > 50, Price > VWAP, SuperTrend Green.\nExit: RSI > 75 or SL Hit.\nRisk: 1.5x ATR.")
        with c2:
            st.markdown("#### 💎 GOLDEN CROSS")
            st.success("Entry: 50 SMA crosses above 200 SMA.\nExit: Close below 50 SMA.\nRisk: 10% Trailing.")

# --- TAB 5: PORTFOLIO HUB ---
with main_tabs[4]:
    st.header("💎 ASSET MANAGEMENT")
    cash, inv, cur, h_list = PortfolioManager.get_metrics()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TOTAL EQUITY", f"₹{cash+cur:,.2f}")
    m2.metric("CASH", f"₹{cash:,.2f}")
    m3.metric("INVESTED", f"₹{inv:,.2f}")
    m4.metric("UNREALIZED P&L", f"₹{cur-inv:,.2f}", delta_color="normal" if (cur-inv)==0 else ("inverse" if (cur-inv)<0 else "normal"))
    
    if not h_list:
        st.info("Portfolio is empty.")
    else:
        df_p = pd.DataFrame(h_list)
        def color_pnl(val): return f'color: {"#00E676" if val >= 0 else "#FF1744"}'
        st.dataframe(
            df_p.style.applymap(color_pnl, subset=['P&L', 'P&L %'])
            .format({"Value": "₹{:.2f}", "P&L": "₹{:.2f}", "P&L %": "{:.2f}%", "LTP": "₹{:.2f}"})
        )
        st.markdown("### 📜 HISTORY")
        if st.session_state['portfolio']['history']:
            st.dataframe(pd.DataFrame(st.session_state['portfolio']['history']).sort_values(by="date", ascending=False))

# --- TAB 6: RISK CALCULATOR ---
with main_tabs[5]:
    st.header("⚖️ RISK ENGINE")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.subheader("KELLY CRITERION")
        wp = st.slider("Win Probability (%)", 10, 90, 50, 5)
        wlr = st.number_input("Win/Loss Ratio", 0.5, 10.0, 2.0, 0.1)
        kp = ((wlr * (wp/100)) - (1 - (wp/100))) / wlr
        st.metric("Optimal Bet Size", f"{max(0, kp*100):.2f}%")
    with rc2:
        st.subheader("POSITION SIZER")
        ac = st.number_input("Account Size", value=100000)
        rp = st.number_input("Risk %", value=1.0)
        ep = st.number_input("Entry", value=100.0)
        sl = st.number_input("Stop", value=95.0)
        risk_share = ep - sl
        if risk_share > 0:
            shares = math.floor((ac * (rp/100)) / risk_share)
            st.success(f"Buy {shares} Shares")
            st.info(f"Total Cost: ₹{shares*ep:,.2f}")
        else: st.warning("Invalid Stop Loss")


#key
 #python -m streamlit run stocks.py 
  
    
 # WIPRO

 # 230.72

