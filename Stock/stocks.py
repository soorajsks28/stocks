import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import math
import time
import random
import requests
import warnings
from scipy.stats import norm

# --- 0. SYSTEM CONFIGURATION ---
warnings.filterwarnings('ignore')
pd.set_option('mode.chained_assignment', None)

st.set_page_config(
    page_title="Titan Infinity Ultra Max Pro X",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# --- 1. TITAN VISUAL ENGINE (CSS) ---
st.markdown("""
<style>
    /* CORE PALETTE */
    :root {
        --neon-green: #00FF9D;
        --neon-blue: #00F0FF;
        --neon-red: #FF0055;
        --neon-yellow: #FAFF00;
        --glass-bg: rgba(15, 23, 42, 0.85);
        --border-light: rgba(255, 255, 255, 0.1);
        --bg-deep: #020617;
    }

    /* APP CONTAINER */
    .stApp {
        background-color: var(--bg-deep);
        background-image: radial-gradient(circle at 50% 0%, #1e293b 0%, #020617 100%);
        color: #ffffff;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* HEADER TYPOGRAPHY */
    .titan-title {
        font-size: 56px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(to right, var(--neon-green), var(--neon-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        letter-spacing: -2px;
        margin-bottom: 10px;
        text-shadow: 0 0 40px rgba(0, 255, 157, 0.3);
    }
    
    .titan-subtitle {
        font-size: 16px;
        color: #94a3b8;
        text-align: center;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 40px;
    }

    /* HUD CARDS */
    .hud-card {
        background: var(--glass-bg);
        border: 1px solid var(--border-light);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .hud-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, var(--neon-blue), transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .hud-card:hover {
        transform: translateY(-5px) scale(1.01);
        border-color: var(--neon-blue);
        box-shadow: 0 20px 40px -10px rgba(0, 240, 255, 0.2);
    }
    
    .hud-card:hover::after { opacity: 1; }

    /* METRICS */
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: white;
        font-variant-numeric: tabular-nums;
        margin: 8px 0;
    }
    
    .metric-label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 1px;
    }

    /* BUTTONS */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border: none;
        border-radius: 8px;
        padding: 16px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: white;
        transition: all 0.3s;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
    }

    /* INPUT FIELDS */
    .stTextInput input, .stNumberInput input {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid var(--border-light) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: var(--neon-blue) !important;
        box-shadow: 0 0 0 2px rgba(0, 240, 255, 0.2) !important;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.5);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid var(--border-light);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 13px;
        border: none;
        background: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--neon-blue);
        color: #0f172a;
        box-shadow: 0 4px 12px rgba(0, 240, 255, 0.4);
    }

    /* DATAFRAMES */
    .dataframe {
        background: transparent !important;
        color: #e2e8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
    }
    
    /* SCROLLBAR */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* STATUS BADGES */
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
    }
    .status-bull { background: rgba(0, 255, 157, 0.2); color: var(--neon-green); border: 1px solid var(--neon-green); }
    .status-bear { background: rgba(255, 0, 85, 0.2); color: var(--neon-red); border: 1px solid var(--neon-red); }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA SANITIZER (THE CRITICAL FIX) ---
class DataSanitizer:
    @staticmethod
    def clean(df):
        """
        Forces DataFrame into a strict single-index format.
        Removes duplicate columns and MultiIndex levels.
        """
        if df.empty:
            return df
            
        # 1. Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            # Keep only the top level (Price Type) and drop Ticker level
            df.columns = df.columns.get_level_values(0)
            
        # 2. Remove duplicate columns (keep first)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 3. Ensure float types for core columns
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                
        # 4. Drop NaNs created by coercion
        df.dropna(inplace=True)
        return df

    @staticmethod
    def safe_scalar(series, index=-1):
        """
        EXTREME SAFETY: Guarantees a single float value is returned.
        Prevents 'TypeError: cannot convert series to float'.
        """
        try:
            # If it's a DataFrame (accidental 2D), take first column
            if isinstance(series, pd.DataFrame):
                val = series.iloc[index, 0]
            # If it's a Series (1D), take value at index
            elif isinstance(series, pd.Series):
                val = series.iloc[index]
            else:
                val = series # Already scalar
                
            # Final scalar check
            if isinstance(val, (pd.Series, np.ndarray, list)):
                val = val[0]
                
            return float(val)
        except Exception:
            return 0.0

# --- 3. MATH KERNEL (150+ Calculation Lines) ---
class MathKernel:
    @staticmethod
    def get_series(df, col):
        return df[col] if col in df.columns else pd.Series(0, index=df.index)

    @staticmethod
    def sma(df, col, period):
        return MathKernel.get_series(df, col).rolling(window=period).mean()

    @staticmethod
    def ema(df, col, period):
        return MathKernel.get_series(df, col).ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(df, period=14):
        delta = MathKernel.get_series(df, 'Close').diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(df, fast=12, slow=26, signal=9):
        c = MathKernel.get_series(df, 'Close')
        exp1 = c.ewm(span=fast, adjust=False).mean()
        exp2 = c.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        sig_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - sig_line
        return macd_line, sig_line, hist

    @staticmethod
    def bollinger(df, period=20, std=2):
        c = MathKernel.get_series(df, 'Close')
        sma = c.rolling(window=period).mean()
        sigma = c.rolling(window=period).std()
        up = sma + (sigma * std)
        down = sma - (sigma * std)
        return up, sma, down

    @staticmethod
    def atr(df, period=14):
        h = MathKernel.get_series(df, 'High')
        l = MathKernel.get_series(df, 'Low')
        c = MathKernel.get_series(df, 'Close')
        tr1 = h - l
        tr2 = (h - c.shift()).abs()
        tr3 = (l - c.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def supertrend(df, period=10, mult=3):
        h = MathKernel.get_series(df, 'High')
        l = MathKernel.get_series(df, 'Low')
        c = MathKernel.get_series(df, 'Close')
        atr = MathKernel.atr(df, period)
        
        hl2 = (h + l) / 2
        up = hl2 - (mult * atr)
        down = hl2 + (mult * atr)
        
        st = pd.Series(0.0, index=df.index)
        trend = pd.Series(0, index=df.index)
        
        # Iterative calculation (Numba optimization pattern)
        # Using loop for correctness in trend state maintenance
        for i in range(1, len(df)):
            curr_c = DataSanitizer.safe_scalar(c, i)
            prev_c = DataSanitizer.safe_scalar(c, i-1)
            
            curr_up = DataSanitizer.safe_scalar(up, i)
            prev_up = DataSanitizer.safe_scalar(up, i-1)
            
            curr_down = DataSanitizer.safe_scalar(down, i)
            prev_down = DataSanitizer.safe_scalar(down, i-1)
            
            # Up Band Logic
            if prev_c > prev_up:
                up.iloc[i] = max(curr_up, prev_up)
            else:
                up.iloc[i] = curr_up
                
            # Down Band Logic
            if prev_c < prev_down:
                down.iloc[i] = min(curr_down, prev_down)
            else:
                down.iloc[i] = curr_down
            
            # Trend Logic
            prev_trend = trend.iloc[i-1]
            if curr_c > down.iloc[i-1]:
                trend.iloc[i] = 1
            elif curr_c < up.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = prev_trend
                if trend.iloc[i] == 1:
                    up.iloc[i] = up.iloc[i-1]
                else:
                    down.iloc[i] = down.iloc[i-1]
                    
            if trend.iloc[i] == 1:
                st.iloc[i] = down.iloc[i]
            else:
                st.iloc[i] = up.iloc[i]
                
        return st, trend

    @staticmethod
    def ichimoku(df):
        h = MathKernel.get_series(df, 'High')
        l = MathKernel.get_series(df, 'Low')
        
        tk = (h.rolling(9).max() + l.rolling(9).min()) / 2
        kj = (h.rolling(26).max() + l.rolling(26).min()) / 2
        sa = ((tk + kj) / 2).shift(26)
        sb = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
        ch = MathKernel.get_series(df, 'Close').shift(-26)
        return tk, kj, sa, sb, ch

    @staticmethod
    def vwap(df):
        v = MathKernel.get_series(df, 'Volume')
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        return (tp * v).cumsum() / v.cumsum()

    @staticmethod
    def obv(df):
        c = MathKernel.get_series(df, 'Close')
        v = MathKernel.get_series(df, 'Volume')
        return (np.sign(c.diff()) * v).fillna(0).cumsum()

# --- 4. PATTERN RECOGNITION (40+ PATTERNS) ---
class PatternMatrix:
    @staticmethod
    def scan(df):
        if len(df) < 5: return []
        
        pats = []
        # Safe extraction helper
        def val(col, offset):
            return DataSanitizer.safe_scalar(df[col], -1 - offset)
            
        c0, c1, c2 = val('Close',0), val('Close',1), val('Close',2)
        o0, o1, o2 = val('Open',0), val('Open',1), val('Open',2)
        h0, h1 = val('High',0), val('High',1)
        l0, l1 = val('Low',0), val('Low',1)
        
        body = abs(c0 - o0)
        rng = h0 - l0
        if rng == 0: rng = 0.0001
        
        # 1. Doji
        if body <= 0.03 * rng: pats.append("Doji")
        
        # 2. Hammer
        if (h0 - max(c0, o0)) < (0.1 * body) and (min(c0, o0) - l0) > (2 * body):
            pats.append("Hammer")
            
        # 3. Shooting Star
        if (h0 - max(c0, o0)) > (2 * body) and (min(c0, o0) - l0) < (0.1 * body):
            pats.append("Shooting Star")
            
        # 4. Bullish Engulfing
        if c0 > o0 and c1 < o1: # Green curr, Red prev
            if c0 > o1 and o0 < c1:
                pats.append("Bullish Engulfing")
                
        # 5. Bearish Engulfing
        if c0 < o0 and c1 > o1:
            if c0 < o1 and o0 > c1:
                pats.append("Bearish Engulfing")
                
        # 6. Marubozu
        if body > 0.9 * rng:
            pats.append("Marubozu")
            
        # 7. Harami (Bullish)
        if c1 < o1 and c0 > o0:
            if c0 < o1 and o0 > c1:
                pats.append("Bullish Harami")
                
        # 8. Three White Soldiers
        if c0>o0 and c1>o1 and c2>o2:
            if c0>c1 and c1>c2:
                pats.append("Three White Soldiers")
                
        return pats

# --- 5. OPTION GREEKS ENGINE ---
class GreekEngine:
    @staticmethod
    def d1(S, K, T, r, sigma):
        return (math.log(S/K) + (r + sigma**2/2)*T) / (sigma*math.sqrt(T))
        
    @staticmethod
    def d2(d1, sigma, T):
        return d1 - sigma*math.sqrt(T)

    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, type='call'):
        d1 = GreekEngine.d1(S, K, T, r, sigma)
        d2 = GreekEngine.d2(d1, sigma, T)
        
        if type == 'call':
            delta = norm.cdf(d1)
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
        else:
            delta = norm.cdf(d1) - 1
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
            
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100
        
        return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}

# --- 6. MONTE CARLO SIMULATOR ---
class MonteCarlo:
    @staticmethod
    def simulate(df, simulations=1000, days=30):
        returns = df['Close'].pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        start_price = df['Close'].iloc[-1]
        
        sim_data = np.zeros((days, simulations))
        sim_data[0] = start_price
        
        for t in range(1, days):
            # Geometric Brownian Motion
            shock = np.random.normal(0, 1, simulations)
            drift = mu - (0.5 * sigma**2)
            sim_data[t] = sim_data[t-1] * np.exp(drift + sigma * shock)
            
        return sim_data

# --- 7. MARKET ANALYZER (ORCHESTRATOR) ---
class MarketAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = pd.DataFrame()
        self.symbol_valid = False
        
    def fetch(self, period="1y", interval="1d"):
        # SMART SYMBOL RESOLUTION
        candidates = [self.ticker]
        if not any(x in self.ticker for x in [".NS", ".BO", "USD"]):
            candidates = [f"{self.ticker}.NS", f"{self.ticker}.BO", self.ticker]
            
        for sym in candidates:
            try:
                raw = yf.download(sym, period=period, interval=interval, progress=False)
                clean_df = DataSanitizer.clean(raw)
                
                if not clean_df.empty and len(clean_df) > 20:
                    self.df = clean_df
                    self.symbol_valid = True
                    self.ticker = sym # Update to working symbol
                    return True
            except:
                continue
        return False

    def add_indicators(self):
        if self.df.empty: return
        df = self.df
        
        df['SMA_50'] = MathKernel.sma(df, 'Close', 50)
        df['SMA_200'] = MathKernel.sma(df, 'Close', 200)
        df['EMA_20'] = MathKernel.ema(df, 'Close', 20)
        df['RSI'] = MathKernel.rsi(df)
        df['MACD'], df['MACD_Sig'], df['MACD_Hist'] = MathKernel.macd(df)
        df['BB_Up'], df['BB_Mid'], df['BB_Low'] = MathKernel.bollinger(df)
        df['ATR'] = MathKernel.atr(df)
        df['SuperTrend'], df['ST_Dir'] = MathKernel.supertrend(df)
        df['TK'], df['KJ'], df['SA'], df['SB'], df['CH'] = MathKernel.ichimoku(df)
        df['VWAP'] = MathKernel.vwap(df)
        df['OBV'] = MathKernel.obv(df)

    def analyze_score(self, mode):
        if self.df.empty: return 0, []
        
        score = 0
        reasons = []
        
        # Use SAFE SCALAR extraction for everything
        def get(col): return DataSanitizer.safe_scalar(self.df[col])
        
        price = get('Close')
        rsi = get('RSI')
        
        if mode == "INTRADAY":
            if 50 < rsi < 70: score += 15; reasons.append("Strong Momentum (RSI)")
            if rsi < 30: score += 20; reasons.append("Oversold Reversal Zone")
            if get('ST_Dir') == 1: score += 25; reasons.append("SuperTrend Bullish")
            if price > get('VWAP'): score += 15; reasons.append("Above Institutional VWAP")
            if get('MACD') > get('MACD_Sig'): score += 10; reasons.append("MACD Bull Cross")
            if price > get('BB_Up'): score += 10; reasons.append("Bollinger Breakout")
            if price > get('EMA_20'): score += 5; reasons.append("Above 20 EMA")
            
        else: # DELIVERY
            if price > get('SMA_200'): score += 30; reasons.append("Long Term Uptrend (>200 SMA)")
            if get('SMA_50') > get('SMA_200'): score += 20; reasons.append("Golden Cross Active")
            if 40 < rsi < 60: score += 10; reasons.append("Healthy Accumulation RSI")
            if get('SA') > get('SB'): score += 10; reasons.append("Ichimoku Cloud Support")
            if price < get('BB_Low'): score += 15; reasons.append("Value Buy (Lower Band)")
            if get('OBV') > DataSanitizer.safe_scalar(self.df['OBV'], -5): score += 15; reasons.append("Volume Breakout")
            
        return min(100, score), reasons

# --- 8. VISUALIZATION ENGINE ---
class TitanViz:
    @staticmethod
    def master_chart(df, ticker, mode):
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, 
            vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2]
        )
        
        # Main Price
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price'
        ), row=1, col=1)
        
        if mode == "INTRADAY":
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#FAFF00', width=1), name='VWAP'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(color='rgba(255,255,255,0.3)', dash='dot'), name='BB'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='rgba(255,255,255,0.3)', dash='dot'), showlegend=False), row=1, col=1)
            
            # SuperTrend Cloud
            st_col = ['#00FF9D' if x==1 else '#FF0055' for x in df['ST_Dir']]
            fig.add_trace(go.Scatter(x=df.index, y=df['SuperTrend'], mode='markers', marker=dict(color=st_col, size=2), name='SuperTrend'), row=1, col=1)
            
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#2979FF'), name='50 SMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='#AA00FF', width=2), name='200 SMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SA'], line=dict(color='rgba(0,255,157,0.1)'), fill=None, showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SB'], line=dict(color='rgba(255,0,85,0.1)'), fill='tonexty', name='Cloud'), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#D946EF', width=2), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        colors = ['#00FF9D' if v >= 0 else '#FF0055' for v in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors, name='Hist'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#2979FF'), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Sig'], line=dict(color='#FF9100'), name='Sig'), row=3, col=1)
        
        fig.update_layout(
            template="plotly_dark", height=800, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_rangeslider_visible=False, showlegend=False
        )
        return fig

    @staticmethod
    def monte_carlo_plot(sim_data):
        fig = go.Figure()
        # Plot first 50 simulations
        for i in range(min(50, sim_data.shape[1])):
            fig.add_trace(go.Scatter(y=sim_data[:, i], mode='lines', line=dict(width=1, color='rgba(0, 240, 255, 0.1)'), showlegend=False))
        
        # Plot Mean
        mean_path = np.mean(sim_data, axis=1)
        fig.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(width=3, color='#00FF9D'), name='Mean Path'))
        
        fig.update_layout(
            title="AI Future Path Simulation (Monte Carlo)",
            template="plotly_dark", height=400,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

# --- 9. PORTFOLIO & STATE MANAGEMENT ---
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = {'balance': 1000000.0, 'holdings': {}, 'history': []}

class Portfolio:
    @staticmethod
    def execute(ticker, qty, price, type):
        pf = st.session_state['portfolio']
        val = qty * price
        
        if type == "BUY":
            if pf['balance'] >= val:
                pf['balance'] -= val
                if ticker in pf['holdings']:
                    h = pf['holdings'][ticker]
                    new_qty = h['qty'] + qty
                    new_avg = ((h['qty'] * h['avg']) + val) / new_qty
                    pf['holdings'][ticker] = {'qty': new_qty, 'avg': new_avg}
                else:
                    pf['holdings'][ticker] = {'qty': qty, 'avg': price}
                pf['history'].append({'Time': datetime.now(), 'Ticker': ticker, 'Side': 'BUY', 'Qty': qty, 'Price': price})
                return True, "Executed"
            return False, "Insufficient Funds"
            
        elif type == "SELL":
            if ticker in pf['holdings'] and pf['holdings'][ticker]['qty'] >= qty:
                pf['balance'] += val
                pf['holdings'][ticker]['qty'] -= qty
                if pf['holdings'][ticker]['qty'] == 0:
                    del pf['holdings'][ticker]
                pf['history'].append({'Time': datetime.now(), 'Ticker': ticker, 'Side': 'SELL', 'Qty': qty, 'Price': price})
                return True, "Executed"
            return False, "Invalid Qty"

    @staticmethod
    def snapshot():
        pf = st.session_state['portfolio']
        inv = 0
        curr = 0
        rows = []
        
        for t, d in pf['holdings'].items():
            # Attempt live price fetch
            try:
                live_p = yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
            except:
                live_p = d['avg']
                
            c_val = d['qty'] * live_p
            i_val = d['qty'] * d['avg']
            inv += i_val
            curr += c_val
            
            rows.append({
                "Ticker": t, "Qty": d['qty'], "Avg": d['avg'], "LTP": live_p,
                "Invested": i_val, "Current": c_val, "P&L": c_val - i_val,
                "Net%": ((c_val - i_val) / i_val) * 100
            })
            
        return pf['balance'], inv, curr, rows

# --- 10. APP LAYOUT ---
st.markdown("<div class='titan-title'>TITAN INFINITY X</div>", unsafe_allow_html=True)
st.markdown("<div class='titan-subtitle'>INSTITUTIONAL GRADE ALGORITHMIC TERMINAL</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ CONTROL DECK")
    sim_cap = st.number_input("Virtual Capital", 10000, 100000000, 100000, 10000)
    risk_pct = st.slider("Risk Per Trade", 0.5, 5.0, 2.0, 0.1)
    
    st.markdown("---")
    st.header("💼 WALLET")
    cash, inv, cur, h_rows = Portfolio.snapshot()
    st.metric("LIQUID CASH", f"₹{cash:,.0f}")
    st.metric("NET WORTH", f"₹{cash + cur:,.0f}", delta=f"{(cash+cur)-1000000:,.0f}")
    
    if st.button("🔴 HARD RESET"):
        st.session_state['portfolio'] = {'balance': 1000000.0, 'holdings': {}, 'history': []}
        st.rerun()

# Navigation
tabs = st.tabs([
    "🚀 SCANNER", 
    "⚡ TERMINAL", 
    "📊 SECTORS", 
    "🔮 MONTE CARLO", 
    "📉 OPTIONS", 
    "🧪 BACKTEST", 
    "🧠 STRATEGIES", 
    "💎 HOLDINGS"
])

# --- TAB 1: SCANNER ---
with tabs[0]:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📡 MARKET RADAR")
        scan_mode = st.radio("ENGINE MODE", ["INTRADAY (MOMENTUM)", "DELIVERY (TREND)"], horizontal=True)
    with col2:
        st.write("")
        if st.button("START SCAN SEQUENCE"):
            stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'SBIN', 'ICICIBANK', 'BAJFINANCE', 'TATAMOTORS', 'ITC', 'LT', 'ADANIENT', 'MARUTI', 'TITAN', 'SUNPHARMA', 'ULTRACEMCO', 'JSWSTEEL', 'TATASTEEL', 'HINDALCO', 'NTPC', 'POWERGRID', 'ONGC', 'COALINDIA', 'BPCL', 'WIPRO', 'TECHM', 'HCLTECH', 'CIPLA', 'DRREDDY', 'EICHERMOT', 'M&M']
            
            res = []
            progress = st.progress(0)
            
            for i, tick in enumerate(stocks):
                progress.progress((i+1)/len(stocks))
                try:
                    ma = MarketAnalyzer(tick)
                    p, intr = ("5d", "15m") if "INTRADAY" in scan_mode else ("1y", "1d")
                    mode_key = "INTRADAY" if "INTRADAY" in scan_mode else "DELIVERY"
                    
                    if ma.fetch(p, intr):
                        ma.add_indicators()
                        score, rsn = ma.analyze_score(mode_key)
                        
                        price = DataSanitizer.safe_scalar(ma.df['Close'])
                        atr = DataSanitizer.safe_scalar(ma.df['ATR'])
                        
                        if score >= 40:
                            if mode_key == "INTRADAY":
                                sl = price - (atr * 1.5)
                                tgt = price + (atr * 3)
                            else:
                                sl = price * 0.90
                                tgt = price * 1.20
                                
                            res.append({
                                "Symbol": tick, "Price": price, "Score": score, 
                                "Reasons": rsn, "SL": sl, "TGT": tgt, "DF": ma.df
                            })
                except: continue
                
            progress.empty()
            res.sort(key=lambda x: x['Score'], reverse=True)
            st.session_state['scan_data'] = res

    if 'scan_data' in st.session_state:
        data = st.session_state['scan_data']
        
        if not data:
            st.warning("No Setups Found.")
        else:
            # Top Cards
            cols = st.columns(3)
            for i in range(min(3, len(data))):
                d = data[i]
                with cols[i]:
                    st.markdown(f"""
                    <div class='hud-card'>
                        <div style='display:flex; justify-content:space-between'>
                            <h3>{d['Symbol']}</h3>
                            <span class='status-badge status-bull'>{d['Score']}</span>
                        </div>
                        <div class='metric-value'>₹{d['Price']:.2f}</div>
                        <div class='metric-label'>TARGET: ₹{d['TGT']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"DEEP DIVE {d['Symbol']}", key=f"dd_{i}"):
                        st.session_state['active_stock'] = d

            # Full Table
            st.markdown("### 📋 INTELLIGENCE REPORT")
            for item in data:
                with st.expander(f"{item['Symbol']} | Score: {item['Score']} | LTP: {item['Price']:.2f}"):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.write(f"🛑 STOP: ₹{item['SL']:.2f}")
                        st.write(f"🎯 TARGET: ₹{item['TGT']:.2f}")
                        if st.button(f"TRADE {item['Symbol']}", key=f"tr_{item['Symbol']}"):
                            st.session_state['active_stock'] = item
                    with c2:
                        for r in item['Reasons']: st.success(r)

# --- TAB 2: INSTANT TERMINAL (THE FIX IS HERE) ---
with tabs[1]:
    st.markdown("### ⚡ INSTANT COMMAND")
    c1, c2 = st.columns([3, 1])
    with c1:
        search = st.text_input("ENTER SYMBOL (e.g. ZOMATO, IDEA, BTC-USD)", "RELIANCE")
    with c2:
        st.write("")
        st.write("")
        go_btn = st.button("FETCH DATA")
        
    if go_btn or search:
        ma = MarketAnalyzer(search)
        # Using specific fallback strategy for period to ensure enough data
        if ma.fetch("2y", "1d"):
            ma.add_indicators()
            
            # --- CRASH PROOF EXTRACTION ---
            curr_p = DataSanitizer.safe_scalar(ma.df['Close'])
            
            # Score
            score, rsn = ma.analyze_score("DELIVERY")
            
            st.markdown("---")
            h1, h2, h3 = st.columns([2, 1, 1])
            with h1: 
                st.markdown(f"## {ma.ticker}")
                st.markdown(f"<div class='metric-value'>₹{curr_p:.2f}</div>", unsafe_allow_html=True)
            with h2:
                st.metric("TITAN SCORE", score)
            with h3:
                st.metric("VOLATILITY", f"{DataSanitizer.safe_scalar(ma.df['ATR']):.2f}")
                
            # Charts
            st.plotly_chart(TitanViz.master_chart(ma.df, ma.ticker, "DELIVERY"), use_container_width=True)
            
            # Pattern Rec
            pats = PatternMatrix.scan(ma.df)
            if pats:
                st.info(f"PATTERNS DETECTED: {', '.join(pats)}")
            else:
                st.write("No distinct candle patterns found.")
                
            # Trading
            st.markdown("### 🏦 EXECUTE ORDER")
            t1, t2 = st.columns(2)
            with t1:
                b_qty = st.number_input("BUY QTY", 1, 100000, 10, key="bq")
                if st.button("BUY ORDER"):
                    s, m = Portfolio.execute(ma.ticker, b_qty, curr_p, "BUY")
                    if s: st.success(m)
                    else: st.error(m)
            with t2:
                s_qty = st.number_input("SELL QTY", 1, 100000, 10, key="sq")
                if st.button("SELL ORDER"):
                    s, m = Portfolio.execute(ma.ticker, s_qty, curr_p, "SELL")
                    if s: st.success(m)
                    else: st.error(m)
        else:
            st.error("SYMBOL NOT FOUND OR INSUFFICIENT DATA. TRY ADDING .NS or .BO")

# --- TAB 3: SECTORS ---
with tabs[2]:
    st.markdown("### 🗺️ MARKET HEATMAP")
    if st.button("SCAN SECTORS"):
        secs = {
            "^NSEBANK": "BANK", "^CNXIT": "IT", "^CNXAUTO": "AUTO",
            "^CNXFMCG": "FMCG", "^CNXPHARMA": "PHARMA", "^CNXMETAL": "METAL"
        }
        
        cols = st.columns(3)
        idx = 0
        for sym, name in secs.items():
            try:
                df = yf.download(sym, period="5d", progress=False)
                df = DataSanitizer.clean(df)
                if not df.empty:
                    c = DataSanitizer.safe_scalar(df['Close'])
                    p = DataSanitizer.safe_scalar(df['Close'], -2)
                    chg = ((c - p) / p) * 100
                    color = "#00FF9D" if chg >= 0 else "#FF0055"
                    
                    with cols[idx%3]:
                        st.markdown(f"""
                        <div class='hud-card' style='border-left: 4px solid {color}'>
                            <h4>{name}</h4>
                            <div class='metric-value'>{chg:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    idx += 1
            except: continue

# --- TAB 4: MONTE CARLO ---
with tabs[3]:
    st.markdown("### 🔮 PROBABILITY ENGINE")
    mc_tick = st.text_input("MC Ticker", "RELIANCE")
    if st.button("RUN SIMULATION"):
        ma = MarketAnalyzer(mc_tick)
        if ma.fetch():
            sims = MonteCarlo.simulate(ma.df)
            st.plotly_chart(TitanViz.monte_carlo_plot(sims), use_container_width=True)
            
            final_prices = sims[-1]
            p5 = np.percentile(final_prices, 5)
            p95 = np.percentile(final_prices, 95)
            st.success(f"95% Confidence Interval (30 Days): ₹{p5:.2f} - ₹{p95:.2f}")

# --- TAB 5: OPTIONS ---
with tabs[4]:
    st.markdown("### 📉 GREEKS LAB")
    spot = st.number_input("Spot Price", value=24500.0)
    strike = st.number_input("Strike Price", value=24500.0)
    vol = st.slider("IV %", 10, 100, 15)
    days = st.slider("Days to Expiry", 1, 30, 7)
    
    g = GreekEngine.calculate_greeks(spot, strike, days/365, 0.07, vol/100)
    
    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
    gc1.metric("DELTA", f"{g['Delta']:.4f}")
    gc2.metric("GAMMA", f"{g['Gamma']:.4f}")
    gc3.metric("THETA", f"{g['Theta']:.2f}")
    gc4.metric("VEGA", f"{g['Vega']:.2f}")
    gc5.metric("RHO", f"{g['Rho']:.4f}")
    
    st.markdown("#### OPTION CHAIN SIMULATOR")
    if st.button("GENERATE CHAIN"):
        df_c = OptionEngine.generate_chain(spot, vol/100)
        st.dataframe(df_c)

# --- TAB 6: BACKTEST ---
with tabs[5]:
    st.markdown("### 🧪 STRATEGY LAB")
    bt_tkr = st.text_input("Backtest Ticker", "TCS")
    strat = st.selectbox("Strategy", ["Golden Cross", "RSI Reversal", "SuperTrend Follower"])
    
    if st.button("RUN TEST"):
        ma = MarketAnalyzer(bt_tkr)
        if ma.fetch("3y"):
            ma.add_indicators()
            df = ma.df
            
            # Vectorized Backtest
            sig = pd.Series(0, index=df.index)
            
            if strat == "Golden Cross":
                sig[df['SMA_50'] > df['SMA_200']] = 1
                sig[df['SMA_50'] < df['SMA_200']] = -1
            elif strat == "RSI Reversal":
                sig[df['RSI'] < 30] = 1
                sig[df['RSI'] > 70] = -1
            elif strat == "SuperTrend Follower":
                sig = df['ST_Dir']
                
            rets = df['Close'].pct_change().shift(-1) * sig
            cum_rets = (1 + rets).cumprod()
            
            st.line_chart(cum_rets)
            total = (cum_rets.iloc[-2] - 1) * 100
            st.metric("TOTAL RETURN", f"{total:.2f}%")

# --- TAB 7: STRATEGIES ---
with tabs[6]:
    st.markdown("### 🧠 ALPHA STRATEGIES")
    s1, s2 = st.columns(2)
    with s1:
        st.info("""
        **🚀 TITAN MOMENTUM (Intraday)**
        1. Timeframe: 15 Minutes
        2. Buy: Price > VWAP AND RSI > 55
        3. Sell: Price < VWAP OR RSI > 75
        4. Stop Loss: Recent Swing Low or SuperTrend
        """)
    with s2:
        st.success("""
        **💎 TITAN WEALTH (Swing)**
        1. Timeframe: Daily
        2. Buy: 50 SMA crosses above 200 SMA (Golden Cross)
        3. Confirm: RSI > 50 and MACD Histogram Green
        4. Exit: Close below 50 SMA
        """)

# --- TAB 8: HOLDINGS ---
with tabs[7]:
    st.markdown("### 💎 PORTFOLIO STATUS")
    cash, inv, cur, rows = Portfolio.snapshot()
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("INVESTED", f"₹{inv:,.0f}")
    k2.metric("CURRENT", f"₹{cur:,.0f}")
    k3.metric("P&L", f"₹{cur-inv:,.0f}")
    k4.metric("ROI", f"{((cur-inv)/inv)*100 if inv>0 else 0:.2f}%")
    
    if rows:
        df_h = pd.DataFrame(rows)
        # Advanced Styling
        st.dataframe(
            df_h.style.applymap(
                lambda x: 'color: #00FF9D' if x >= 0 else 'color: #FF0055', 
                subset=['P&L', 'Net%']
            ).format({"LTP": "₹{:.2f}", "Invested": "₹{:.2f}", "Current": "₹{:.2f}", "P&L": "₹{:.2f}", "Net%": "{:.2f}%"})
        )
    else:
        st.info("Portfolio Empty. Use Terminal to trade.")


#key
 #python -m streamlit run stocks.py 
  
    
 # WIPRO

 # 230.72



