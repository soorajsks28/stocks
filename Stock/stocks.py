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
import random
import requests
import warnings

# --- 0. INITIAL SETUP & CONFIGURATION ---
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Titan Infinity Ultra",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="⚡"
)

# --- 1. RESPONSIVE CSS ENGINE (MOBILE FIX) ---
st.markdown("""
<style>
    /* GLOBAL RESET & VARIABLES */
    :root {
        --primary: #00E676;
        --primary-dim: rgba(0, 230, 118, 0.1);
        --secondary: #2979FF;
        --secondary-dim: rgba(41, 98, 255, 0.1);
        --danger: #FF1744;
        --warning: #FFC400;
        --bg-dark: #0A0E14;
        --card-bg: #131722;
        --text-primary: #FFFFFF;
        --text-secondary: #B2B5BE;
        --border-color: #2A2E39;
    }
    
    .stApp {
        background-color: var(--bg-dark);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    /* --- MOBILE RESPONSIVE HEADER FIX --- */
    .titan-header {
        font-weight: 900;
        text-transform: uppercase;
        background: linear-gradient(90deg, #00E676 0%, #2979FF 50%, #FF1744 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        letter-spacing: 1px;
        text-align: center;
        
        /* Auto-Scale Font Size */
        font-size: clamp(28px, 6vw, 52px); 
        line-height: 1.2;
    }
    
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: var(--text-primary);
        margin-top: 20px;
        margin-bottom: 10px;
        border-left: 4px solid var(--secondary);
        padding-left: 10px;
    }

    /* --- HORIZONTAL SCROLLING TABS (FIX FOR TABS CUTTING OFF) --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 5px;
        
        /* Force Horizontal Scroll on Mobile */
        flex-wrap: nowrap !important;
        white-space: nowrap !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none; /* Hide scrollbar Firefox */
    }
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none; /* Hide scrollbar Chrome/Safari */
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: nowrap;
        background-color: rgba(255,255,255,0.03);
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 600;
        font-size: 13px;
        border: 1px solid var(--border-color);
        transition: all 0.2s;
        padding: 0 20px;
        min-width: fit-content; /* Prevents squishing */
        flex-shrink: 0; /* Prevents shrinking */
    }

    .stTabs [aria-selected="true"] {
        color: #fff;
        background-color: var(--secondary-dim);
        border-color: var(--secondary);
        box-shadow: 0 0 15px rgba(41, 98, 255, 0.3);
    }

    /* --- CARD DESIGN --- */
    .stat-card {
        background: rgba(19, 23, 34, 0.95);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .stat-card-hover:hover {
        border-color: var(--primary);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 230, 118, 0.15);
    }

    .ticker-symbol {
        font-size: 24px;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: 0.5px;
    }
    
    .ticker-price {
        font-size: 32px;
        font-weight: 700;
        color: var(--primary);
        font-variant-numeric: tabular-nums;
        margin: 5px 0;
    }

    /* --- BADGES --- */
    .badge {
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: inline-block;
    }
    .badge-buy { background: var(--primary-dim); color: var(--primary); border: 1px solid var(--primary); }
    .badge-sell { background: rgba(255, 23, 68, 0.1); color: var(--danger); border: 1px solid var(--danger); }
    .badge-neutral { background: rgba(255, 196, 0, 0.1); color: var(--warning); border: 1px solid var(--warning); }

    /* --- BUTTONS --- */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2962FF 0%, #0039CB 100%);
        color: white;
        border: none;
        padding: 12px 15px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #448AFF 0%, #2962FF 100%);
        box-shadow: 0 6px 15px rgba(41, 98, 255, 0.5);
    }

    /* --- DATAFRAME FIX --- */
    .dataframe { font-size: 12px !important; background-color: var(--card-bg); }
    
    /* --- INPUT FIELDS --- */
    .stTextInput > div > div > input {
        background-color: #1A1E29;
        color: white;
        border: 1px solid #333;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. ADVANCED MATH ENGINE (15+ INDICATORS) ---
class MathEngine:
    @staticmethod
    def sma(series, period):
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

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
        atr_val = MathEngine.atr(high, low, close, period)
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
        tr = MathEngine.atr(high, low, close, period)
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
        minus_di = 100 * (abs(minus_dm).ewm(alpha=1/period).mean() / tr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx_val = dx.ewm(alpha=1/period).mean()
        return adx_val

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
    
    @staticmethod
    def williams_r(high, low, close, period=14):
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))

    @staticmethod
    def obv(close, volume):
        return (np.sign(close.diff()) * volume).fillna(0).cumsum()

# --- 3. BACKTEST ENGINE (NEW FEATURE) ---
class BacktestEngine:
    def __init__(self, df, initial_capital=100000):
        self.df = df
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.positions = 0
        self.equity_curve = []
        self.trades = []

    def run_strategy(self, strategy_type="CROSSOVER"):
        # Reset
        self.balance = self.initial_capital
        self.positions = 0
        self.trades = []
        self.equity_curve = [self.initial_capital]
        
        df = self.df.copy()
        
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            price = curr['Close']
            date = df.index[i]
            
            signal = 0 # 0: Hold, 1: Buy, -1: Sell
            
            if strategy_type == "CROSSOVER":
                # Golden Cross Strategy
                if curr['SMA_50'] > curr['SMA_200'] and prev['SMA_50'] <= prev['SMA_200']:
                    signal = 1
                elif curr['SMA_50'] < curr['SMA_200'] and prev['SMA_50'] >= prev['SMA_200']:
                    signal = -1
                    
            elif strategy_type == "MOMENTUM":
                # RSI Strategy
                if curr['RSI'] < 30 and prev['RSI'] >= 30:
                    signal = 1
                elif curr['RSI'] > 70 and prev['RSI'] <= 70:
                    signal = -1
            
            elif strategy_type == "SUPERTREND":
                if curr['ST_Dir'] == 1 and prev['ST_Dir'] != 1:
                    signal = 1
                elif curr['ST_Dir'] != 1 and prev['ST_Dir'] == 1:
                    signal = -1

            # Execution Logic
            if signal == 1 and self.balance > 0:
                # Buy Max
                qty = math.floor(self.balance / price)
                if qty > 0:
                    cost = qty * price
                    self.balance -= cost
                    self.positions += qty
                    self.trades.append({
                        'Type': 'BUY', 'Price': price, 'Date': date, 
                        'Qty': qty, 'Balance': self.balance
                    })
            
            elif signal == -1 and self.positions > 0:
                # Sell All
                revenue = self.positions * price
                self.balance += revenue
                self.trades.append({
                    'Type': 'SELL', 'Price': price, 'Date': date, 
                    'Qty': self.positions, 'Balance': self.balance
                })
                self.positions = 0
            
            # Track Equity
            current_equity = self.balance + (self.positions * price)
            self.equity_curve.append(current_equity)
            
        return self.equity_curve, self.trades

    def get_metrics(self):
        if not self.equity_curve: return {}
        start = self.equity_curve[0]
        end = self.equity_curve[-1]
        
        ret = ((end - start) / start) * 100
        max_drawdown = 0
        peak = start
        
        for val in self.equity_curve:
            if val > peak: peak = val
            dd = (peak - val) / peak
            if dd > max_drawdown: max_drawdown = dd
            
        win_trades = 0
        loss_trades = 0
        
        # Calculate PnL per trade cycle
        trade_pnls = []
        last_buy_price = 0
        
        for t in self.trades:
            if t['Type'] == 'BUY':
                last_buy_price = t['Price']
            elif t['Type'] == 'SELL' and last_buy_price > 0:
                pnl = (t['Price'] - last_buy_price) / last_buy_price
                trade_pnls.append(pnl)
                if pnl > 0: win_trades += 1
                else: loss_trades += 1
                
        total_trades = win_trades + loss_trades
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            "Total Return": ret,
            "Max Drawdown": max_drawdown * 100,
            "Trades Executed": total_trades,
            "Win Rate": win_rate,
            "Final Equity": end
        }

# --- 4. MARKET ANALYZER ---
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

    def process_technical_data(self):
        if self.df.empty: return
        
        c = self.df['Close']
        h = self.df['High']
        l = self.df['Low']
        v = self.df['Volume']
        
        # Trend
        self.df['SMA_20'] = MathEngine.sma(c, 20)
        self.df['SMA_50'] = MathEngine.sma(c, 50)
        self.df['SMA_200'] = MathEngine.sma(c, 200)
        self.df['EMA_20'] = MathEngine.ema(c, 20)
        
        # Momentum
        self.df['RSI'] = MathEngine.rsi(c)
        self.df['Stoch_K'], self.df['Stoch_D'] = MathEngine.stochastic(h, l, c)
        self.df['CCI'] = MathEngine.cci(h, l, c)
        self.df['WilliamsR'] = MathEngine.williams_r(h, l, c)
        
        # Complex
        self.df['MACD'], self.df['MACD_Signal'], self.df['MACD_Hist'] = MathEngine.macd(c)
        self.df['ADX'] = MathEngine.adx(h, l, c)
        self.df['SuperTrend'], self.df['ST_Dir'] = MathEngine.supertrend(h, l, c)
        
        # Bands
        self.df['BB_Up'], self.df['BB_Mid'], self.df['BB_Low'] = MathEngine.bollinger_bands(c)
        self.df['ATR'] = MathEngine.atr(h, l, c)
        
        # Advanced
        self.df['Tenkan'], self.df['Kijun'], self.df['SpanA'], self.df['SpanB'], self.df['Chikou'] = MathEngine.ichimoku(h, l, c)
        self.df['VWAP'] = MathEngine.vwap(h, l, c, v)
        self.df['OBV'] = MathEngine.obv(c, v)

    def calculate_pivot_points(self):
        row = self.df.iloc[-1]
        h, l, c = row['High'], row['Low'], row['Close']
        p = (h + l + c) / 3
        return {
            "P": p, "R1": (2*p)-l, "S1": (2*p)-h, 
            "R2": p+(h-l), "S2": p-(h-l), 
            "R3": h+2*(p-l), "S3": l-2*(h-p)
        }
    
    def calculate_fibonacci(self):
        # Lookback 1 year
        max_h = self.df['High'].max()
        min_l = self.df['Low'].min()
        diff = max_h - min_l
        return {
            "0%": max_h,
            "23.6%": max_h - 0.236 * diff,
            "38.2%": max_h - 0.382 * diff,
            "50%": max_h - 0.5 * diff,
            "61.8%": max_h - 0.618 * diff,
            "100%": min_l
        }

    def detect_candlestick_patterns(self):
        if self.df.empty: return []
        row = self.df.iloc[-1]
        prev = self.df.iloc[-2]
        patterns = []
        
        body = abs(row['Close'] - row['Open'])
        full_range = row['High'] - row['Low']
        
        if full_range == 0: return []
        
        # Doji
        if body <= 0.1 * full_range:
            patterns.append("Doji (Indecision)")
            
        # Hammer
        if (row['High'] - max(row['Open'], row['Close'])) < (0.2 * body) and \
           (min(row['Open'], row['Close']) - row['Low']) > (2 * body):
            patterns.append("Hammer (Bullish)")
            
        # Engulfing
        if row['Close'] > row['Open'] and prev['Close'] < prev['Open']:
            if row['Close'] > prev['Open'] and row['Open'] < prev['Close']:
                patterns.append("Bullish Engulfing")
        elif row['Close'] < row['Open'] and prev['Close'] > prev['Open']:
            if row['Close'] < prev['Open'] and row['Open'] > prev['Close']:
                patterns.append("Bearish Engulfing")
                
        return patterns

    def score_stock(self, mode):
        if self.df.empty: return 0, []
        
        score = 0
        reasons = []
        curr = self.df.iloc[-1]
        rsi = curr['RSI']
        close = curr['Close']
        
        if mode == "INTRADAY":
            # Momentum Factors
            if rsi > 50 and rsi < 70: score += 10; reasons.append("Strong Momentum")
            if rsi < 30: score += 15; reasons.append("Oversold (Bounce Likely)")
            
            # Trend Factors
            if curr['ST_Dir'] == 1: score += 20; reasons.append("SuperTrend Bullish")
            if close > curr['VWAP']: score += 15; reasons.append("Price > VWAP")
            if curr['MACD'] > curr['MACD_Signal']: score += 10; reasons.append("MACD Crossover")
            
            # Strength Factors
            if curr['ADX'] > 25: score += 10; reasons.append("Trend Strength High")
            if close > curr['EMA_20']: score += 10; reasons.append("Above 20 EMA")
            if curr['CCI'] > 100: score += 5; reasons.append("CCI Breakout")
            if curr['Stoch_K'] < 20: score += 5; reasons.append("Stoch Oversold")

        else: # DELIVERY
            # Long Term Trend
            if close > curr['SMA_200']: score += 25; reasons.append("Bull Market (>200 SMA)")
            if curr['SMA_50'] > curr['SMA_200']: score += 15; reasons.append("Golden Cross Active")
            
            # Value/Stability
            if rsi > 40 and rsi < 65: score += 10; reasons.append("Stable RSI Zone")
            if curr['SpanA'] > curr['SpanB']: score += 10; reasons.append("Ichimoku Cloud Green")
            if close < curr['BB_Low']: score += 20; reasons.append("Deep Value (Lower BB)")
            
            # Volume & Cycles
            if curr['MACD'] > 0: score += 10; reasons.append("Positive Cycle")
            if curr['OBV'] > self.df['OBV'].iloc[-20]: score += 10; reasons.append("Volume Accumulation")
            
        return min(score, 100), reasons

# --- 5. CHART FACTORY ---
class ChartFactory:
    @staticmethod
    def create_advanced_chart(df, ticker, mode):
        # 3 Rows: Price, RSI, MACD
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, 
            vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2]
        )
        
        # 1. Price Chart
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price'
        ), row=1, col=1)
        
        if mode == "INTRADAY":
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#FFD700', width=1), name='VWAP'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Up'], line=dict(color='gray', width=1, dash='dot'), name='BB'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', width=1, dash='dot'), showlegend=False), row=1, col=1)
            
            # SuperTrend Markers
            colors = ['#00E676' if x == 1 else '#FF1744' for x in df['ST_Dir']]
            fig.add_trace(go.Scatter(x=df.index, y=df['SuperTrend'], mode='markers', marker=dict(color=colors, size=3), name='SuperTrend'), row=1, col=1)
            
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#2979FF', width=1), name='50 SMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='#AA00FF', width=2), name='200 SMA'), row=1, col=1)
            # Ichimoku
            fig.add_trace(go.Scatter(x=df.index, y=df['SpanA'], line=dict(color='rgba(0,230,118,0.2)'), fill=None, name='Cloud'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SpanB'], line=dict(color='rgba(255,23,68,0.2)'), fill='tonexty', showlegend=False), row=1, col=1)

        # 2. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#E040FB', width=2), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # 3. MACD
        hist_colors = ['#00E676' if v >= 0 else '#FF1744' for v in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=hist_colors, name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#2979FF'), name='Line'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#FF9100'), name='Sig'), row=3, col=1)
        
        # Mobile Optimization Config
        fig.update_layout(
            template="plotly_dark", height=600, margin=dict(l=0,r=0,t=0,b=0),
            xaxis_rangeslider_visible=False, showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            dragmode='pan'
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
        fig.update_layout(height=180, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor='rgba(0,0,0,0)')
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
            return True, f"BOUGHT {qty} {ticker}"
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
                return True, f"SOLD {qty} {ticker}"
            return False, "Not enough shares"
        return False, "Stock not owned"

    @staticmethod
    def get_metrics():
        pf = st.session_state['portfolio']
        total_inv = 0
        curr_val = 0
        h_list = []
        
        for t, d in pf['holdings'].items():
            # Try fetch live price, else fallback
            try:
                ltp = yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
            except:
                ltp = d['avg']
            
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

# --- 7. UI: TITLE & HEADER ---
st.markdown("<div class='titan-header'>Titan Infinity Ultra</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("CONTROLS")
    sim_cap = st.number_input("CAPITAL (₹)", 10000, 10000000, 100000, 10000)
    risk_factor = st.slider("RISK %", 0.5, 5.0, 2.0, 0.1)
    
    st.markdown("---")
    st.markdown("### PORTFOLIO")
    cash, inv, cur, h_list = PortfolioManager.get_metrics()
    st.metric("CASH", f"₹{cash:,.2f}")
    st.metric("EQUITY", f"₹{cash + cur:,.2f}", delta=f"{(cash+cur)-1000000:,.2f}")
    
    if st.button("RESET ALL"):
        st.session_state['portfolio'] = {'balance': 1000000.0, 'holdings': {}, 'history': []}
        st.rerun()

# --- 8. TABS (MAIN NAVIGATION) ---
# Shorter names for mobile compatibility
tabs = st.tabs(["🚀 SCAN", "⚡ TERM", "🔬 DEEP", "🧠 INFO", "💎 PORT", "⚖️ RISK", "🔄 TEST"])

# --- TAB 1: SCANNER ---
with tabs[0]:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("<div class='section-header'>MARKET RADAR</div>", unsafe_allow_html=True)
        mode = st.radio("ENGINE", ["INTRADAY", "DELIVERY"], horizontal=True, label_visibility="collapsed")
    with c2:
        st.write("")
        if st.button("INITIATE SCAN"):
            # Nifty 50 Subset
            stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'SBIN', 'BHARTIARTL', 'ITC', 'LT', 'TATAMOTORS', 'M&M', 'MARUTI', 'ADANIENT', 'SUNPHARMA', 'TITAN', 'BAJFINANCE', 'ULTRACEMCO', 'NTPC', 'POWERGRID', 'TATASTEEL', 'JSWSTEEL', 'COALINDIA', 'HINDALCO', 'GRASIM', 'CIPLA', 'WIPRO', 'DLF', 'ZOMATO', 'PAYTM', 'HAL', 'BEL', 'TRENT']
            
            res = []
            bar = st.progress(0)
            
            for i, s in enumerate(stocks):
                bar.progress((i+1)/len(stocks))
                try:
                    ma = MarketAnalyzer(s+".NS")
                    # Timeframes
                    if mode == "INTRADAY": p, intr = "5d", "15m"
                    else: p, intr = "1y", "1d"
                    
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
            st.session_state['scan_res'] = res
            st.session_state['scan_mode'] = mode

    if 'scan_res' in st.session_state:
        results = st.session_state['scan_res']
        if not results:
            st.warning("No opportunities found.")
        else:
            # Top 3 Cards
            cols = st.columns(3)
            for i in range(min(3, len(results))):
                item = results[i]
                with cols[i]:
                    st.markdown(f"""
                    <div class='stat-card stat-card-hover'>
                        <div class='ticker-symbol'>{item['Symbol']}</div>
                        <div class='ticker-price'>₹{item['Price']:.2f}</div>
                        <span class='badge badge-buy'>SCORE: {item['Score']}</span>
                        <div style='margin-top:10px; font-size:11px; color:#aaa'>
                            TGT: <span style='color:#00E676'>₹{item['TGT']:.2f}</span><br>
                            SL: <span style='color:#FF1744'>₹{item['SL']:.2f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"OPEN {item['Symbol']}", key=f"b_{i}"):
                        st.session_state['active_stock'] = item

            st.markdown("### 📋 FULL REPORT")
            for item in results:
                with st.expander(f"{item['Symbol']} | {item['Score']}% | ₹{item['Price']:.2f}"):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1: st.progress(item['Score']); st.write(f"ATR: {item['ATR']:.2f}")
                    with c2: 
                        for r in item['Reasons']: st.markdown(f"✅ {r}")
                    with c3:
                        if st.button(f"ANALYZE {item['Symbol']}", key=f"an_{item['Symbol']}"):
                            st.session_state['active_stock'] = item

# --- TAB 2: INSTANT TERMINAL ---
with tabs[1]:
    st.markdown("<div class='section-header'>INSTANT TERMINAL</div>", unsafe_allow_html=True)
    c_src, c_btn = st.columns([3, 1])
    with c_src:
        u_tick = st.text_input("ENTER SYMBOL (e.g. ZOMATO)", "RELIANCE")
    with c_btn:
        st.write("")
        st.write("")
        fetch_btn = st.button("FETCH")
        
    if fetch_btn or u_tick:
        sym = u_tick.upper() + ".NS" if not u_tick.endswith(".NS") else u_tick.upper()
        ma = MarketAnalyzer(sym)
        if ma.get_data("1y", "1d"):
            ma.process_technical_data()
            sc, rs = ma.score_stock("DELIVERY")
            
            curr = ma.df['Close'].iloc[-1]
            
            st.markdown("---")
            col_inf, col_sc = st.columns([2, 1])
            with col_inf:
                st.markdown(f"<div class='ticker-symbol'>{u_tick.upper()}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='ticker-price'>₹{curr:.2f}</div>", unsafe_allow_html=True)
            with col_sc:
                st.metric("TITAN SCORE", f"{sc}/100")
            
            st.plotly_chart(ChartFactory.create_advanced_chart(ma.df, u_tick, "DELIVERY"), use_container_width=True)
            
            # Trading
            t1, t2 = st.columns(2)
            with t1:
                st.markdown("#### 🟢 BUY")
                qty_b = st.number_input("Qty Buy", 1, 100000, 10, key="qb")
                if st.button("CONFIRM BUY"):
                    s, m = PortfolioManager.buy(u_tick.upper(), qty_b, curr)
                    if s: st.success(m)
                    else: st.error(m)
            with t2:
                st.markdown("#### 🔴 SELL")
                qty_s = st.number_input("Qty Sell", 1, 100000, 10, key="qs")
                if st.button("CONFIRM SELL"):
                    s, m = PortfolioManager.sell(u_tick.upper(), qty_s, curr)
                    if s: st.success(m)
                    else: st.error(m)
        else:
            st.error("Stock not found. Try NSE symbol.")

# --- TAB 3: DEEP DIVE ---
with tabs[2]:
    if 'active_stock' in st.session_state:
        data = st.session_state['active_stock']
        df = data['DF']
        tkr = data['Symbol']
        
        st.markdown(f"<div class='section-header'>{tkr} DEEP DIVE</div>", unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PRICE", f"₹{data['Price']:.2f}")
        m2.metric("SCORE", f"{data['Score']}/100")
        m3.metric("TARGET", f"₹{data['TGT']:.2f}")
        m4.metric("ATR", f"{data['ATR']:.2f}")
        
        c_main, c_side = st.columns([3, 1])
        with c_main:
            st.plotly_chart(ChartFactory.create_advanced_chart(df, tkr, st.session_state.get('scan_mode', 'DELIVERY')), use_container_width=True)
            
            ma = MarketAnalyzer(tkr)
            ma.df = df
            pivs = ma.calculate_pivot_points()
            st.markdown("#### 🎯 PIVOT LEVELS")
            pc1, pc2, pc3, pc4, pc5 = st.columns(5)
            pc1.metric("S2", f"{pivs['S2']:.2f}")
            pc2.metric("S1", f"{pivs['S1']:.2f}")
            pc3.metric("PIVOT", f"{pivs['P']:.2f}")
            pc4.metric("R1", f"{pivs['R1']:.2f}")
            pc5.metric("R2", f"{pivs['R2']:.2f}")
            
        with c_side:
            st.plotly_chart(ChartFactory.create_gauge(data['Score']), use_container_width=True)
            
            # Quick Trade Box
            st.markdown(f"""
            <div class='stat-card' style='border-left:4px solid #2979FF'>
                <h4>🏦 QUICK TRADE</h4>
                <div style='margin:10px 0;'>
                    <span style='color:#FF1744; font-weight:bold'>SL: ₹{data['SL']:.2f}</span><br>
                    <span style='color:#00E676; font-weight:bold'>TGT: ₹{data['TGT']:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            risk_amt = sim_cap * (risk_factor/100)
            risk_share = data['Price'] - data['SL']
            rec_q = int(risk_amt/risk_share) if risk_share > 0 else 0
            
            st.info(f"Risk-Safe Qty: {rec_q}")
            q_in = st.number_input("Order Qty", 1, 100000, rec_q, key="ddq")
            if st.button(f"BUY {tkr}"):
                s, m = PortfolioManager.buy(tkr, q_in, data['Price'])
                if s: st.success(m)
                else: st.error(m)
                
        # Patterns
        st.markdown("#### 🕯️ PATTERNS")
        pats = ma.detect_candlestick_patterns()
        if pats:
            for p in pats: st.write(f"• {p}")
        else: st.write("No major patterns detected.")

    else:
        st.info("Select stock from Scanner first.")

# --- TAB 4: KNOWLEDGE ---
with tabs[3]:
    st.markdown("<div class='section-header'>TITAN ACADEMY</div>", unsafe_allow_html=True)
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("#### INDICATORS")
        with st.expander("RSI (Relative Strength)"): st.write("Momentum oscillator. >70 Overbought, <30 Oversold.")
        with st.expander("MACD"): st.write("Trend indicator. Line crossing Signal line indicates momentum shift.")
        with st.expander("SuperTrend"): st.write("Excellent for trailing stop losses in trending markets.")
    with k2:
        st.markdown("#### STRATEGIES")
        st.info("**INTRADAY:** Use 15m timeframe. Buy when Price > VWAP and RSI > 50.")
        st.success("**DELIVERY:** Use Daily timeframe. Buy on Golden Cross (50 SMA > 200 SMA).")

# --- TAB 5: PORTFOLIO ---
with tabs[4]:
    st.markdown("<div class='section-header'>PORTFOLIO HUB</div>", unsafe_allow_html=True)
    cash, inv, cur, h_list = PortfolioManager.get_metrics()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TOTAL EQUITY", f"₹{cash+cur:,.2f}")
    m2.metric("CASH", f"₹{cash:,.2f}")
    m3.metric("INVESTED", f"₹{inv:,.2f}")
    m4.metric("P&L", f"₹{cur-inv:,.2f}")
    
    if h_list:
        df_p = pd.DataFrame(h_list)
        def color_pnl(val): return f'color: {"#00E676" if val >= 0 else "#FF1744"}'
        st.dataframe(
            df_p.style.applymap(color_pnl, subset=['P&L', 'P&L %'])
            .format({"Value": "₹{:.2f}", "P&L": "₹{:.2f}", "P&L %": "{:.2f}%"})
        )
    else:
        st.info("No active positions.")

# --- TAB 6: RISK ---
with tabs[5]:
    st.markdown("<div class='section-header'>RISK CALCULATOR</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Kelly Criterion")
        wp = st.slider("Win Rate (%)", 10, 90, 50)
        rr = st.number_input("Risk:Reward", 0.5, 10.0, 2.0)
        kp = ((rr * (wp/100)) - (1 - (wp/100))) / rr
        st.metric("Optimal Bet %", f"{max(0, kp*100):.2f}%")
    with c2:
        st.subheader("Position Sizing")
        acc = st.number_input("Account", value=100000)
        rp = st.number_input("Risk %", value=1.0)
        ep = st.number_input("Entry", value=100.0)
        sl = st.number_input("Stop", value=95.0)
        risk_per_share = ep - sl
        if risk_per_share > 0:
            qty = math.floor((acc * (rp/100)) / risk_per_share)
            st.success(f"Position Size: {qty} Shares")

# --- TAB 7: BACKTEST (NEW) ---
with tabs[6]:
    st.markdown("<div class='section-header'>STRATEGY BACKTESTER</div>", unsafe_allow_html=True)
    
    bt_col1, bt_col2 = st.columns([1, 3])
    with bt_col1:
        bt_ticker = st.text_input("Ticker", "RELIANCE")
        bt_strat = st.selectbox("Strategy", ["CROSSOVER", "MOMENTUM", "SUPERTREND"])
        if st.button("RUN BACKTEST"):
            ma = MarketAnalyzer(bt_ticker+".NS")
            if ma.get_data("2y", "1d"):
                ma.process_technical_data()
                be = BacktestEngine(ma.df, 100000)
                eq, tr = be.run_strategy(bt_strat)
                st.session_state['bt_res'] = (eq, tr, be.get_metrics())
            else:
                st.error("Data fetch failed")
                
    with bt_col2:
        if 'bt_res' in st.session_state:
            eq, tr, met = st.session_state['bt_res']
            
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Return", f"{met['Total Return']:.2f}%")
            m2.metric("Win Rate", f"{met['Win Rate']:.2f}%")
            m3.metric("Trades", met['Trades Executed'])
            m4.metric("Drawdown", f"{met['Max Drawdown']:.2f}%")
            
            # Equity Curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=eq, mode='lines', name='Equity', line=dict(color='#00E676')))
            fig.update_layout(title="Equity Curve", template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Trades Table
            if tr:
                st.write("Recent Trades")
                st.dataframe(pd.DataFrame(tr).tail())


#key
 #python -m streamlit run stocks.py 
  
    
 # WIPRO

 # 230.72


