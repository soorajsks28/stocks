import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sqlite3
import math
import time
import random
import warnings
from scipy.stats import norm

# -----------------------------------------------------------------------------
# 1. SYSTEM INITIALIZATION & CONFIGURATION
# -----------------------------------------------------------------------------
warnings.filterwarnings('ignore')
pd.set_option('mode.chained_assignment', None)

st.set_page_config(
    page_title="Titan Infinity Apex X",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="💠"
)

# -----------------------------------------------------------------------------
# 2. QUANTUM UI ENGINE (CSS STYLING)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* CORE THEME VARIABLES */
    :root {
        --titan-gold: #FFD700;
        --titan-cyan: #00F3FF;
        --titan-red: #FF0055;
        --titan-green: #00FF9D;
        --titan-dark: #02040a;
        --titan-card: #0e1117;
        --titan-border: rgba(255, 255, 255, 0.1);
        --glass-effect: rgba(20, 25, 35, 0.7);
    }

    /* APP BACKGROUND */
    .stApp {
        background-color: var(--titan-dark);
        font-family: 'SF Mono', 'Courier New', monospace;
    }

    /* HEADER TYPOGRAPHY */
    .sovereign-header {
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(180deg, #fff, #666);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        letter-spacing: 4px;
        margin-bottom: 10px;
        text-shadow: 0px 0px 30px rgba(0, 243, 255, 0.15);
    }
    
    .sovereign-sub {
        font-size: 14px;
        text-align: center;
        color: var(--titan-cyan);
        letter-spacing: 2px;
        margin-bottom: 40px;
        text-transform: uppercase;
    }

    /* CARDS */
    .titan-card {
        background: var(--titan-card);
        border: 1px solid var(--titan-border);
        border-radius: 4px;
        padding: 20px;
        margin-bottom: 15px;
        transition: transform 0.2s, border-color 0.2s;
        position: relative;
    }
    .titan-card:hover {
        border-color: var(--titan-cyan);
        transform: translateY(-2px);
    }
    
    /* SECTION HEADERS */
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #fff;
        border-left: 4px solid var(--titan-gold);
        padding-left: 15px;
        margin: 30px 0 20px 0;
        letter-spacing: 1px;
    }

    /* METRICS */
    .t-metric-label { font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .t-metric-val { font-size: 24px; font-weight: bold; color: #fff; font-variant-numeric: tabular-nums; }
    
    /* BUTTONS */
    .stButton > button {
        width: 100%;
        background: #0a0a0a;
        color: var(--titan-cyan);
        border: 1px solid var(--titan-cyan);
        border-radius: 0;
        padding: 16px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: var(--titan-cyan);
        color: #000;
        box-shadow: 0 0 25px var(--titan-cyan);
    }
    
    .buy-btn > button { border-color: var(--titan-green) !important; color: var(--titan-green) !important; }
    .buy-btn > button:hover { background: var(--titan-green) !important; color: black !important; }
    
    .sell-btn > button { border-color: var(--titan-red) !important; color: var(--titan-red) !important; }
    .sell-btn > button:hover { background: var(--titan-red) !important; color: black !important; }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        background: #0a0a0a;
        padding: 5px;
        gap: 0px;
        border-bottom: 1px solid #333;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: transparent;
        color: #666;
        border: none;
        border-radius: 0;
        font-size: 12px;
        font-weight: bold;
        flex: 1;
    }
    .stTabs [aria-selected="true"] {
        background: #1a1a1a;
        color: var(--titan-cyan);
        border-bottom: 2px solid var(--titan-cyan);
    }

    /* INPUTS */
    .stTextInput input, .stNumberInput input {
        background: #0a0a0a !important;
        border: 1px solid #333 !important;
        color: white !important;
        border-radius: 0 !important;
        padding: 12px !important;
    }
    
    /* DATAFRAME */
    .dataframe {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        background: #000;
    }
    
    /* ALERTS */
    .stAlert {
        background: #111;
        border: 1px solid #333;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. DATABASE ENGINE (PERSISTENCE LAYER)
# -----------------------------------------------------------------------------
class DatabaseManager:
    def __init__(self):
        self.db_path = 'titan_apex_final.db'
        self._initialize_schema()

    def _get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _initialize_schema(self):
        conn = self._get_connection()
        c = conn.cursor()
        
        # Portfolio Table
        c.execute('''CREATE TABLE IF NOT EXISTS portfolio (
            symbol TEXT PRIMARY KEY, 
            quantity REAL, 
            avg_price REAL
        )''')
        
        # Wallet Table
        c.execute('''CREATE TABLE IF NOT EXISTS wallet (
            id INTEGER PRIMARY KEY, 
            balance REAL
        )''')
        
        # Transaction History
        c.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, 
            symbol TEXT, 
            side TEXT, 
            quantity REAL, 
            price REAL, 
            value REAL
        )''')
        
        # Seed Wallet if new
        c.execute("SELECT count(*) FROM wallet")
        if c.fetchone()[0] == 0:
            c.execute("INSERT INTO wallet (id, balance) VALUES (1, 100000.0)")
            
        conn.commit()
        conn.close()

    def get_wallet_balance(self):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT balance FROM wallet WHERE id=1")
        bal = c.fetchone()[0]
        conn.close()
        return bal

    def update_wallet(self, amount):
        conn = self._get_connection()
        c = conn.cursor()
        current = self.get_wallet_balance()
        new_bal = current + amount
        c.execute("UPDATE wallet SET balance=? WHERE id=1", (new_bal,))
        conn.commit()
        conn.close()
        return new_bal

    def get_portfolio_position(self, symbol):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT quantity, avg_price FROM portfolio WHERE symbol=?", (symbol,))
        row = c.fetchone()
        conn.close()
        return row if row else (0.0, 0.0)

    def execute_transaction(self, symbol, qty, price, side):
        conn = self._get_connection()
        c = conn.cursor()
        
        curr_q, curr_avg = self.get_portfolio_position(symbol)
        total_val = qty * price
        
        if side == "BUY":
            new_q = curr_q + qty
            new_avg = ((curr_q * curr_avg) + total_val) / new_q
            
            if curr_q == 0:
                c.execute("INSERT INTO portfolio VALUES (?, ?, ?)", (symbol, new_q, new_avg))
            else:
                c.execute("UPDATE portfolio SET quantity=?, avg_price=? WHERE symbol=?", (new_q, new_avg, symbol))
                
        elif side == "SELL":
            new_q = curr_q - qty
            if new_q <= 0:
                c.execute("DELETE FROM portfolio WHERE symbol=?", (symbol,))
            else:
                c.execute("UPDATE portfolio SET quantity=? WHERE symbol=?", (new_q, symbol))
        
        # Log entry
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO trades (timestamp, symbol, side, quantity, price, value) VALUES (?,?,?,?,?,?)",
                  (ts, symbol, side, qty, price, total_val))
        
        conn.commit()
        conn.close()

    def fetch_all_positions(self):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM portfolio")
        data = c.fetchall()
        conn.close()
        return data

    def fetch_trade_logs(self):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 100")
        data = c.fetchall()
        conn.close()
        return data

    def factory_reset(self):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("DELETE FROM portfolio")
        c.execute("DELETE FROM trades")
        c.execute("UPDATE wallet SET balance=100000.0 WHERE id=1")
        conn.commit()
        conn.close()

# Initialize Database System
db_engine = DatabaseManager()

# -----------------------------------------------------------------------------
# 4. MATH UTILITIES (CRASH PREVENTION)
# -----------------------------------------------------------------------------
class SafeMath:
    @staticmethod
    def to_float(val):
        """Forces value to float, handling Series/DataFrames safely."""
        try:
            if isinstance(val, pd.DataFrame):
                return float(val.iloc[-1, 0])
            if isinstance(val, pd.Series):
                return float(val.iloc[-1])
            if isinstance(val, (np.ndarray, list)):
                return float(val[-1])
            return float(val)
        except:
            return 0.0

# -----------------------------------------------------------------------------
# 5. TECHNICAL INDICATORS (MODULAR CLASSES)
# -----------------------------------------------------------------------------

class TrendIndicators:
    @staticmethod
    def sma(series, period):
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def supertrend(df, period=10, multiplier=3):
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Calculate ATR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Basic Bands
        hl2 = (high + low) / 2
        up = hl2 - (multiplier * atr)
        dn = hl2 + (multiplier * atr)
        
        st = pd.Series(0.0, index=df.index)
        trend = pd.Series(0, index=df.index)
        
        # Iterative Logic
        for i in range(1, len(df)):
            curr_c = SafeMath.to_float(close.iloc[i])
            prev_c = SafeMath.to_float(close.iloc[i-1])
            
            curr_up = SafeMath.to_float(up.iloc[i])
            prev_up = SafeMath.to_float(up.iloc[i-1])
            curr_dn = SafeMath.to_float(dn.iloc[i])
            prev_dn = SafeMath.to_float(dn.iloc[i-1])
            
            if prev_c > prev_up: up.iloc[i] = max(curr_up, prev_up)
            else: up.iloc[i] = curr_up
                
            if prev_c < prev_dn: dn.iloc[i] = min(curr_dn, prev_dn)
            else: dn.iloc[i] = curr_dn
            
            prev_trend = trend.iloc[i-1]
            if curr_c > dn.iloc[i-1]: trend.iloc[i] = 1
            elif curr_c < up.iloc[i-1]: trend.iloc[i] = -1
            else: trend.iloc[i] = prev_trend
            
            st.iloc[i] = dn.iloc[i] if trend.iloc[i] == 1 else up.iloc[i]
            
        return st, trend

    @staticmethod
    def ichimoku(df):
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        chikou = close.shift(-26)
        
        return tenkan, kijun, senkou_a, senkou_b, chikou

class MomentumIndicators:
    @staticmethod
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(series):
        k = series.ewm(span=12, adjust=False).mean()
        d = series.ewm(span=26, adjust=False).mean()
        macd = k - d
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal, macd - signal
    
    @staticmethod
    def stochastic(high, low, close, k=14, d=3):
        lowest_low = low.rolling(k).min()
        highest_high = high.rolling(k).max()
        k_line = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_line = k_line.rolling(d).mean()
        return k_line, d_line

class VolatilityIndicators:
    @staticmethod
    def bollinger_bands(series, period=20, std=2):
        ma = series.rolling(period).mean()
        sigma = series.rolling(period).std()
        upper = ma + (sigma * std)
        lower = ma - (sigma * std)
        return upper, ma, lower
    
    @staticmethod
    def atr(df, period=14):
        h, l, c = df['High'], df['Low'], df['Close']
        tr1 = h - l
        tr2 = (h - c.shift()).abs()
        tr3 = (l - c.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

# -----------------------------------------------------------------------------
# 6. PATTERN RECOGNITION (ADVANCED)
# -----------------------------------------------------------------------------
class PatternEngine:
    @staticmethod
    def scan_patterns(df):
        if len(df) < 5: return []
        patterns = []
        
        # Helper to get scalar values safely
        def val(col, idx): return SafeMath.to_float(df[col].iloc[-1-idx])
        
        c0, c1, c2 = val('Close',0), val('Close',1), val('Close',2)
        o0, o1, o2 = val('Open',0), val('Open',1), val('Open',2)
        h0, h1 = val('High',0), val('High',1)
        l0, l1 = val('Low',0), val('Low',1)
        
        body = abs(c0 - o0)
        rng = h0 - l0
        if rng == 0: rng = 0.0001
        
        # 1. Doji
        if body <= 0.05 * rng: patterns.append("Doji")
        
        # 2. Hammer
        if (h0 - max(c0, o0)) < (0.1 * body) and (min(c0, o0) - l0) > (2 * body):
            patterns.append("Hammer")
            
        # 3. Shooting Star
        if (h0 - max(c0, o0)) > (2 * body) and (min(c0, o0) - l0) < (0.1 * body):
            patterns.append("Shooting Star")
            
        # 4. Bullish Engulfing
        if c0 > o0 and c1 < o1 and c0 > o1 and o0 < c1:
            patterns.append("Bullish Engulfing")
            
        # 5. Bearish Engulfing
        if c0 < o0 and c1 > o1 and c0 < o1 and o0 > c1:
            patterns.append("Bearish Engulfing")
            
        # 6. Marubozu
        if body > 0.8 * rng:
            patterns.append("Marubozu")
            
        # 7. Three White Soldiers
        if c0 > o0 and c1 > o1 and c2 > o2:
            if c0 > c1 and c1 > c2:
                patterns.append("Three White Soldiers")
                
        return patterns

# -----------------------------------------------------------------------------
# 7. DATA FEED & PROCESSING
# -----------------------------------------------------------------------------
class MarketData:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = pd.DataFrame()
        self.valid = False
        
    def fetch_data(self, period="1y", interval="1d"):
        # Smart Symbol Resolution
        candidates = [self.ticker, f"{self.ticker}.NS", f"{self.ticker}.BO"]
        if "USD" in self.ticker: candidates = [self.ticker]
        
        for sym in candidates:
            try:
                raw = yf.download(sym, period=period, interval=interval, progress=False)
                
                # Flatten MultiIndex Columns (Fixes AttributeError)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                
                # Deduplicate Columns
                raw = raw.loc[:, ~raw.columns.duplicated()]
                
                # Check Validity
                req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(c in raw.columns for c in req_cols) and len(raw) > 30:
                    self.df = raw
                    self.ticker = sym
                    self.valid = True
                    return True
            except: continue
        return False

    def compute_technicals(self):
        if not self.valid: return
        
        c = self.df['Close']
        h = self.df['High']
        l = self.df['Low']
        
        self.df['SMA_50'] = TrendIndicators.sma(c, 50)
        self.df['SMA_200'] = TrendIndicators.sma(c, 200)
        self.df['EMA_20'] = TrendIndicators.ema(c, 20)
        self.df['ST'], self.df['ST_DIR'] = TrendIndicators.supertrend(self.df)
        self.df['TK'], self.df['KJ'], self.df['SA'], self.df['SB'], self.df['CH'] = TrendIndicators.ichimoku(self.df)
        
        self.df['RSI'] = MomentumIndicators.rsi(c)
        self.df['MACD'], self.df['SIG'], self.df['HIST'] = MomentumIndicators.macd(c)
        self.df['K'], self.df['D'] = MomentumIndicators.stochastic(h, l, c)
        
        self.df['BB_UP'], self.df['BB_MID'], self.df['BB_LOW'] = VolatilityIndicators.bollinger_bands(c)
        self.df['ATR'] = VolatilityIndicators.atr(self.df)

    def generate_signal_score(self, mode):
        if not self.valid: return 0, []
        
        score = 0
        reasons = []
        
        # Get Values Safely
        c = SafeMath.to_float(self.df['Close'])
        rsi = SafeMath.to_float(self.df['RSI'])
        st_dir = SafeMath.to_float(self.df['ST_DIR'])
        
        if mode == "INTRADAY":
            if 50 < rsi < 70: score += 15; reasons.append("Strong Momentum")
            if rsi < 30: score += 20; reasons.append("Oversold (Dip)")
            if st_dir == 1: score += 25; reasons.append("SuperTrend Bullish")
            if c > SafeMath.to_float(self.df['EMA_20']): score += 10; reasons.append("Above 20 EMA")
            if SafeMath.to_float(self.df['MACD']) > SafeMath.to_float(self.df['SIG']): score += 10; reasons.append("MACD Cross")
            if c > SafeMath.to_float(self.df['BB_UP']): score += 10; reasons.append("Bollinger Breakout")
            
        else: # SWING
            if c > SafeMath.to_float(self.df['SMA_200']): score += 30; reasons.append("Long Term Uptrend")
            if SafeMath.to_float(self.df['SMA_50']) > SafeMath.to_float(self.df['SMA_200']): score += 20; reasons.append("Golden Cross")
            if 40 < rsi < 65: score += 10; reasons.append("Stable RSI")
            if SafeMath.to_float(self.df['SA']) > SafeMath.to_float(self.df['SB']): score += 15; reasons.append("Cloud Support")
            if c < SafeMath.to_float(self.df['BB_LOW']): score += 15; reasons.append("Value Buy Zone")
            
        return min(100, score), reasons

# -----------------------------------------------------------------------------
# 8. VISUALIZATION ENGINE
# -----------------------------------------------------------------------------
class ChartEngine:
    @staticmethod
    def build_chart(df, ticker, mode, interactive=False):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Price Action
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price'
        ), row=1, col=1)
        
        if mode == "INTRADAY":
            # Bollinger & SuperTrend
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_UP'], line=dict(color='rgba(255,255,255,0.3)', dash='dot'), name='BB'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOW'], line=dict(color='rgba(255,255,255,0.3)', dash='dot'), showlegend=False), row=1, col=1)
            
            st_cols = ['#00FF9D' if x==1 else '#FF0055' for x in df['ST_DIR']]
            fig.add_trace(go.Scatter(x=df.index, y=df['ST'], mode='markers', marker=dict(color=st_cols, size=2), name='ST'), row=1, col=1)
            
        else:
            # Moving Averages
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#00F3FF', width=1), name='50 SMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='#D500F9', width=2), name='200 SMA'), row=1, col=1)
            
            # Ichimoku Cloud
            fig.add_trace(go.Scatter(x=df.index, y=df['SA'], line=dict(color='rgba(0,255,157,0.1)'), fill=None, showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SB'], line=dict(color='rgba(255,0,85,0.1)'), fill='tonexty', name='Cloud'), row=1, col=1)

        # MACD on Row 2
        colors = ['#00FF9D' if v >= 0 else '#FF0055' for v in df['HIST']]
        fig.add_trace(go.Bar(x=df.index, y=df['HIST'], marker_color=colors, name='Momentum'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#00F3FF'), name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SIG'], line=dict(color='#FFD700'), name='Signal'), row=2, col=1)
        
        # Interaction Logic
        drag = 'pan' if interactive else False
        
        fig.update_layout(
            template="plotly_dark",
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            dragmode=drag,
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        return fig

# -----------------------------------------------------------------------------
# 9. OPTION GREEKS ENGINE ( RENAMED PD VARIABLE TO P_DELTA )
# -----------------------------------------------------------------------------
class GreeksEngine:
    @staticmethod
    def d1(S, K, T, r, sigma):
        return (math.log(S/K) + (r + sigma**2/2)*T) / (sigma*math.sqrt(T))
        
    @staticmethod
    def d2(d1, sigma, T):
        return d1 - sigma*math.sqrt(T)

    @staticmethod
    def calculate(S, K, T, r, sigma, option_type='call'):
        d1 = GreeksEngine.d1(S, K, T, r, sigma)
        d2 = GreeksEngine.d2(d1, sigma, T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            
        return price, delta

# -----------------------------------------------------------------------------
# 10. MAIN APP INTERFACE
# -----------------------------------------------------------------------------

# HEADER
st.markdown("<div class='sovereign-header'>TITAN INFINITY SOVEREIGN X</div>", unsafe_allow_html=True)
st.markdown("<div class='sovereign-sub'>INSTITUTIONAL GRADE ALGORITHMIC SUITE V11.0</div>", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("VAULT CONTROL")
    wallet_bal = db_engine.get_wallet_balance()
    st.metric("LIQUID FUNDS", f"₹{wallet_bal:,.2f}")
    
    fund_amt = st.number_input("DEPOSIT / WITHDRAW", value=0.0)
    if st.button("EXECUTE TRANSFER"):
        db_engine.update_wallet(fund_amt)
        st.rerun()
        
    st.markdown("---")
    if st.button("FACTORY RESET APP"):
        db_engine.factory_reset()
        st.rerun()

# NAVIGATION
tabs = st.tabs(["RADAR", "TERMINAL", "PORTFOLIO", "GREEKS", "SIMULATOR", "NEWS", "FUNDA"])

# --- TAB 1: RADAR ---
with tabs[0]:
    st.markdown("<div class='section-title'>MARKET SCANNER</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        mode_sel = st.radio("ENGINE MODE", ["INTRADAY", "SWING"], horizontal=True)
    with c2:
        st.write("")
        scan_trigger = st.button("INITIATE SCAN")
        
    if scan_trigger:
        # NIFTY 50 SUBSET (Enhanced List)
        symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'SBIN', 'ICICIBANK', 'TATAMOTORS', 'ITC', 'LT', 'ADANIENT', 'ZOMATO', 'PAYTM', 'BHEL', 'PNB', 'CANBK', 'VEDL', 'HINDALCO', 'NTPC', 'POWERGRID', 'ULTRACEMCO', 'JSWSTEEL', 'WIPRO', 'TECHM', 'HCLTECH', 'CIPLA', 'DRREDDY', 'EICHERMOT', 'M&M']
        
        results = []
        bar = st.progress(0)
        
        for i, s in enumerate(symbols):
            bar.progress((i+1)/len(symbols))
            md = MarketData(s)
            p, intr = ("5d", "15m") if mode_sel == "INTRADAY" else ("1y", "1d")
            
            if md.fetch_data(p, intr):
                md.compute_technicals()
                score, rsn = md.generate_signal_score(mode_sel)
                
                # ALWAYS ADD DATA (Filtering done in Display)
                results.append({
                    'Symbol': s,
                    'Price': SafeMath.to_float(md.df['Close']),
                    'Score': score,
                    'Reasons': rsn,
                    'ATR': SafeMath.to_float(md.df['ATR'])
                })
        
        bar.empty()
        # Sort by Score descending
        sorted_results = sorted(results, key=lambda x: x['Score'], reverse=True)
        # Take Top 10 to ensure user always sees suggestions
        st.session_state['scan_res'] = sorted_results[:10]

    if 'scan_res' in st.session_state:
        for r in st.session_state['scan_res']:
            col = "var(--titan-cyan)" if r['Score'] > 50 else "var(--titan-gold)"
            with st.expander(f"{r['Symbol']}  |  SCORE: {r['Score']}  |  ₹{r['Price']:.2f}"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"<h1 style='color:{col}'>{r['Score']}</h1>", unsafe_allow_html=True)
                    if st.button(f"LOAD {r['Symbol']}", key=f"btn_{r['Symbol']}"):
                        st.session_state['active_sym'] = r['Symbol']
                with c2:
                    sl = r['Price'] - (r['ATR'] * 1.5)
                    tg = r['Price'] + (r['ATR'] * 3)
                    st.write(f"🛑 STOP: ₹{sl:.2f}")
                    st.write(f"🎯 TARGET: ₹{tg:.2f}")
                    for x in r['Reasons']: st.markdown(f"`{x}`")

# --- TAB 2: TERMINAL ---
with tabs[1]:
    st.markdown("<div class='section-title'>TRADE TERMINAL</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input("SYMBOL", st.session_state.get('active_sym', 'RELIANCE'))
    with c2:
        st.write("")
        fetch = st.button("LOAD DATA")
        
    if fetch or query:
        eng = MarketData(query)
        if eng.fetch_data("1y", "1d"):
            eng.compute_technicals()
            
            # Metrics
            curr = SafeMath.to_float(eng.df['Close'])
            sc, _ = eng.generate_signal_score("SWING")
            
            st.markdown("---")
            h1, h2 = st.columns([2, 1])
            with h1:
                st.markdown(f"<div class='sovereign-sub' style='text-align:left; font-size:32px;'>{eng.ticker}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='t-metric-val' style='color:var(--titan-cyan)'>₹{curr:.2f}</div>", unsafe_allow_html=True)
            with h2:
                st.metric("TITAN SCORE", sc)
                
            # Chart
            inter = st.checkbox("UNLOCK CHART (ZOOM/PAN)", value=False)
            st.plotly_chart(ChartEngine.build_chart(eng.df, eng.ticker, "SWING", inter), use_container_width=True)
            
            # Patterns
            pats = PatternEngine.scan_patterns(eng.df)
            if pats: st.info(f"PATTERNS DETECTED: {', '.join(pats)}")
            
            # Execution
            st.markdown("### ORDER DECK")
            t1, t2 = st.columns(2)
            with t1:
                st.markdown("<div class='buy-btn'>", unsafe_allow_html=True)
                bq = st.number_input("BUY QTY", min_value=1, value=10, key='bq')
                if st.button("BUY ORDER"):
                    cost = bq * curr
                    if db_engine.get_wallet_balance() >= cost:
                        db_engine.update_wallet(-cost)
                        db_engine.execute_transaction(eng.ticker, bq, curr, "BUY")
                        st.success("FILLED")
                    else: st.error("NO FUNDS")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with t2:
                st.markdown("<div class='sell-btn'>", unsafe_allow_html=True)
                sq = st.number_input("SELL QTY", min_value=1, value=10, key='sq')
                if st.button("SELL ORDER"):
                    pos = db_engine.get_portfolio_position(eng.ticker)
                    if pos[0] >= sq:
                        val = sq * curr
                        db_engine.update_wallet(val)
                        db_engine.execute_transaction(eng.ticker, sq, curr, "SELL")
                        st.success("FILLED")
                    else: st.error("NO POSITION")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("SYMBOL NOT FOUND")

# --- TAB 3: PORTFOLIO ---
with tabs[2]:
    st.markdown("<div class='section-title'>HOLDINGS</div>", unsafe_allow_html=True)
    positions = db_engine.fetch_all_positions()
    
    if not positions:
        st.info("NO ACTIVE POSITIONS")
    else:
        tot_inv = 0
        tot_curr = 0
        
        for sym, qty, avg in positions:
            try: 
                ltp = SafeMath.to_float(yf.Ticker(sym).history(period="1d")['Close'])
            except: 
                ltp = avg
                
            val = qty * ltp
            cost = qty * avg
            pnl = val - cost
            pct = (pnl / cost) * 100 if cost > 0 else 0
            
            tot_inv += cost
            tot_curr += val
            
            col = "var(--titan-green)" if pnl >= 0 else "var(--titan-red)"
            
            st.markdown(f"""
            <div class='titan-card' style='border-left: 4px solid {col}'>
                <div style='display:flex; justify-content:space-between'>
                    <h3>{sym}</h3>
                    <h3>₹{val:,.2f}</h3>
                </div>
                <div style='display:flex; justify-content:space-between; font-size:12px; color:#aaa'>
                    <span>QTY: {qty}</span>
                    <span>AVG: {avg:.2f}</span>
                    <span style='color:{col}; font-weight:bold'> P&L: {pnl:,.2f} ({pct:.2f}%)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        c1, c2, c3 = st.columns(3)
        c1.metric("INVESTED", f"₹{tot_inv:,.2f}")
        c2.metric("CURRENT", f"₹{tot_curr:,.2f}")
        c3.metric("P&L", f"₹{tot_curr-tot_inv:,.2f}")

# --- TAB 4: GREEKS (FIXED VARIABLE NAMES) ---
with tabs[3]:
    st.markdown("<div class='section-title'>OPTION GREEKS</div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    spot_p = c1.number_input("SPOT", value=24500.0)
    strk_p = c2.number_input("STRIKE", value=24500.0)
    iv_val = c3.slider("IV", 10, 100, 15)
    expiry = st.slider("DAYS", 1, 30, 7)
    
    # Calculate - Using explicit variable names to avoid 'pd' conflict
    c_price, c_delta = GreeksEngine.calculate(spot_p, strk_p, expiry/365, 0.07, iv_val/100, 'call')
    p_price, p_delta = GreeksEngine.calculate(spot_p, strk_p, expiry/365, 0.07, iv_val/100, 'put')
    
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(f"""
        <div class='titan-card' style='border-color:var(--titan-green)'>
            <h4>CALL</h4>
            <h2>₹{c_price:.2f}</h2>
            <p>DELTA: {c_delta:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown(f"""
        <div class='titan-card' style='border-color:var(--titan-red)'>
            <h4>PUT</h4>
            <h2>₹{p_price:.2f}</h2>
            <p>DELTA: {p_delta:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 5: SIMULATOR ---
with tabs[4]:
    st.markdown("<div class='section-title'>MONTE CARLO ENGINE</div>", unsafe_allow_html=True)
    sim_tkr = st.text_input("TICKER", "RELIANCE")
    
    if st.button("RUN SIMULATION"):
        fed = MarketData(sim_tkr)
        if fed.fetch_data():
            ret = fed.df['Close'].pct_change().dropna()
            mu, sig = ret.mean(), ret.std()
            start = SafeMath.to_float(fed.df['Close'])
            
            fig = go.Figure()
            # 50 Sims
            for _ in range(50):
                path = [start]
                for _ in range(30):
                    shk = np.random.normal(0, 1)
                    nxt = path[-1] * np.exp((mu - 0.5*sig**2) + sig*shk)
                    path.append(nxt)
                fig.add_trace(go.Scatter(y=path, mode='lines', line=dict(width=1, color='rgba(0, 243, 255, 0.1)'), showlegend=False))
            
            fig.update_layout(template="plotly_dark", height=400, title="30-Day Future Projection")
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 6: NEWS ---
with tabs[5]:
    st.markdown("<div class='section-title'>MARKET INTEL</div>", unsafe_allow_html=True)
    headlines = [
        ("Central Bank holds rates steady", 0.6),
        ("Tech sector earnings beat estimates", 0.8),
        ("Oil supply concerns rise amid conflict", -0.4),
        ("Inflation data cools down globally", 0.5),
        ("Regulatory crackdown on crypto exchanges", -0.7)
    ]
    
    for h, s in headlines:
        col = "var(--titan-green)" if s > 0 else "var(--titan-red)"
        st.markdown(f"""
        <div class='titan-card' style='border-left: 3px solid {col}'>
            <div>{h}</div>
            <div style='font-size:10px; color:#666'>SENTIMENT SCORE: {s}</div>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 7: FUNDAMENTAL ---
with tabs[6]:
    st.markdown("<div class='section-title'>FUNDAMENTAL METRICS</div>", unsafe_allow_html=True)
    f_tkr = st.text_input("LOOKUP", "TCS")
    if st.button("GET FUNDAMENTALS"):
        try:
            inf = yf.Ticker(f_tkr+".NS").info
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**P/E:** {inf.get('trailingPE', 'N/A')}")
                st.write(f"**EPS:** {inf.get('trailingEps', 'N/A')}")
            with c2:
                st.write(f"**ROE:** {inf.get('returnOnEquity', 'N/A')}")
                st.write(f"**Debt/Eq:** {inf.get('debtToEquity', 'N/A')}")
        except:
            st.error("DATA UNAVAILABLE")

#key
 #python -m streamlit run stocks.py 
  
    
 # WIPRO

 # 230.72




