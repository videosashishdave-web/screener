import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import json
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="VSN Stock Screener", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 🔐 LOGIN SYSTEM
# ==========================================
# તમે અહીં તમારા Username અને Password બદલી શકો છો
USERS = {
    "admin": "vsn123",      # Username: admin, Password: vsn123
    "user1": "pass123"      # બીજા કોઈ મિત્ર માટે અલગ પાસવર્ડ આપવો હોય તો
}

def login_screen():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🔐 VSN Screener Login")
            st.info("કૃપા કરીને આગળ વધવા માટે Login કરો.")
            username = st.text_input("Username").strip()
            password = st.text_input("Password", type="password").strip()
            
            if st.button("Login", type="primary", use_container_width=True):
                if username in USERS and USERS[username] == password:
                    st.session_state["logged_in"] = True
                    st.success("Login સફળ થયું!")
                    st.rerun()
                else:
                    st.error("ખોટો Username અથવા Password!")
        return False
    return True

# જો Login ન થયેલું હોય તો સ્કેનર આગળ નહીં વધે
if not login_screen():
    st.stop()

# Logout બટન સાઈડબારમાં
if st.sidebar.button("🚪 Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

# ==========================================
# Helpers & Data Fetching
# ==========================================
FAV_FILE = "favorites.json"

def load_favorites():
    if os.path.exists(FAV_FILE):
        try:
            with open(FAV_FILE, "r") as f:
                return json.load(f)
        except:
            return ["ITC", "SBIN", "BHARTIARTL", "RELIANCE", "LT"]
    return ["ITC", "SBIN", "BHARTIARTL", "RELIANCE", "LT"]

def save_favorites(fav_list):
    with open(FAV_FILE, "w") as f:
        json.dump(fav_list, f)

@st.cache_data(ttl=86400)
def get_all_nse_stocks():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            return df['SYMBOL'].dropna().astype(str).tolist()
    except Exception:
        pass
    return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "SUZLON", "IDEA"]

def format_custom_date(dt_obj):
    if not hasattr(dt_obj, 'day'):
        return str(dt_obj)[:10]

    day = dt_obj.day
    suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return f"{day}{suffix} {dt_obj.strftime('%b %Y')}"

def convert_to_heikin_ashi(df):
    if df.empty:
        return df
    ha_df = df.copy()
    ha_df['Real_Close'] = df['Close']
    ha_df['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    ha_open = np.zeros(len(df))
    ha_open[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_df['Close'].iloc[i-1]) / 2
    ha_df['Open'] = ha_open
    
    ha_df['High'] = ha_df[['High', 'Open', 'Close']].max(axis=1)
    ha_df['Low'] = ha_df[['Low', 'Open', 'Close']].min(axis=1)
    return ha_df

def calculate_slow_stochastic_k(df, k_period=14, smooth_k=4):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    fast_k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    return fast_k.rolling(window=smooth_k).mean()

def calculate_ema(df, period=100):
    return df['Close'].ewm(span=period, adjust=False).mean()

def fetch_yfinance_data(symbol, timeframe, use_lookback, lookback_days, start_date, end_date):
    tf_map = {"5 Min": "5m", "15 Min": "15m", "30 Min": "30m", "1 Hour": "60m", "1 Day": "1d", "1 Week": "1wk"}
    yf_interval = tf_map.get(timeframe, "5m")
    
    if symbol == "NIFTY 50":
        ticker = "^NSEI"
    elif symbol == "SENSEX":
        ticker = "^BSESN"
    else:
        ticker = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol

    try:
        if use_lookback:
            period_str = "1mo" if timeframe in ["5 Min", "15 Min", "30 Min", "1 Hour"] else "6mo"
            df = yf.download(ticker, period=period_str, interval=yf_interval, progress=False)
        else:
            df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'), interval=yf_interval, progress=False)
        
        if df.empty:
            return pd.DataFrame()
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.reset_index(inplace=True)
        date_col = 'Datetime' if 'Datetime' in df.columns else ('Date' if 'Date' in df.columns else df.columns[0])
        df.rename(columns={date_col: 'Date'}, inplace=True)
        
        for col in ['Open', 'Close', 'High', 'Low']:
            df[col] = pd.to_numeric(df[col].squeeze(), errors='coerce')

        if pd.api.types.is_datetime64_any_dtype(df['Date']) and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_convert('Asia/Kolkata')
                
        return df
    except Exception:
        return pd.DataFrame()

# ==========================================
# Single Stock Processing Logic
# ==========================================
def process_stock(symbol, timeframe, use_lookback, lookback_days, start_date, end_date, params):
    local_bullish, local_bearish = [], []
    
    df = fetch_yfinance_data(symbol, timeframe, use_lookback, lookback_days, start_date, end_date)
    min_rows = params['stoch_k'] + params['smooth']
    
    if df.empty or len(df) < min_rows:
        return local_bullish, local_bearish

    # Unclosed Candle Filter
    if timeframe in ["5 Min", "15 Min", "30 Min", "1 Hour"]:
        tf_mins = {"5 Min": 5, "15 Min": 15, "30 Min": 30, "1 Hour": 60}[timeframe]
        last_dt = df.iloc[-1]['Date']
        if hasattr(last_dt, 'to_pydatetime'):
            last_dt = last_dt.to_pydatetime().replace(tzinfo=None)
        if datetime.now() < last_dt + timedelta(minutes=tf_mins):
            df = df.iloc[:-1]

    if len(df) < min_rows:
        return local_bullish, local_bearish

    if params['use_heikin_ashi']:
        df = convert_to_heikin_ashi(df)

    df['Stoch_K'] = calculate_slow_stochastic_k(df, params['stoch_k'], params['smooth'])
    if params['use_ema']:
        df['EMA'] = calculate_ema(df, params['ema_period'])

    allowed_dates = None
    if use_lookback:
        unique_dates = sorted(list(df['Date'].dt.date.unique())) if hasattr(df['Date'].dt, 'date') else sorted(list(set([d.date() for d in df['Date']])))
        allowed_dates = set(unique_dates[-lookback_days:])

    for i in range(1, len(df)):
        prev_k, curr_k = df.iloc[i - 1]['Stoch_K'], df.iloc[i]['Stoch_K']
        if pd.isna(prev_k) or pd.isna(curr_k):
            continue

        curr_row = df.iloc[i]
        dt_obj = curr_row['Date']
        row_date = dt_obj.date() if hasattr(dt_obj, 'date') else dt_obj

        if allowed_dates and row_date not in allowed_dates:
            continue

        price = float(curr_row['Real_Close']) if 'Real_Close' in curr_row else float(curr_row['Close'])
        if params['use_price_filter'] and not (params['min_price'] <= price <= params['max_price']):
            continue

        date_str = format_custom_date(dt_obj)
        time_str = dt_obj.strftime('%H:%M') if hasattr(dt_obj, 'strftime') else str(dt_obj)[11:16]

        curr_ema = curr_row['EMA'] if params['use_ema'] else None
        is_above_ema = True if not params['use_ema'] else (price > curr_ema)
        is_below_ema = True if not params['use_ema'] else (price < curr_ema)

        if prev_k < params['below'] and curr_k >= params['below'] and is_above_ema:
            local_bullish.append({'date': date_str, 'time': time_str, 'symbol': symbol, 'price': price, 'dt': dt_obj})
        elif prev_k > params['above'] and curr_k <= params['above'] and is_below_ema:
            local_bearish.append({'date': date_str, 'time': time_str, 'symbol': symbol, 'price': price, 'dt': dt_obj})

    return local_bullish, local_bearish

# ==========================================
# Sidebar UI Controls
# ==========================================
st.sidebar.title("🔍 Screener Settings")

timeframe = st.sidebar.selectbox("Select Timeframe", ["5 Min", "15 Min", "30 Min", "1 Hour", "1 Day", "1 Week"])
use_ha = st.sidebar.checkbox("📊 Use Heikin-Ashi Candles", value=False)

date_mode = st.sidebar.radio("Date Mode", ["Quick Lookback", "Custom Date Range"])
if date_mode == "Quick Lookback":
    use_lookback = True
    lookback_days = st.sidebar.slider("Lookback Days", 1, 20, 3)
    start_date, end_date = None, None
else:
    use_lookback = False
    lookback_days = 3
    start_date = st.sidebar.date_input("From Date", datetime.now() - timedelta(days=30))
    end_date = st.sidebar.date_input("To Date", datetime.now())

st.sidebar.subheader("Watchlists")
selected_watchlists = []
if st.sidebar.checkbox("My Favorites", value=True): selected_watchlists.append("My Favorites")
if st.sidebar.checkbox("Nifty50 Stocks"): selected_watchlists.append("Nifty50 Stocks")
if st.sidebar.checkbox("Sensex Stocks"): selected_watchlists.append("Sensex Stocks")
if st.sidebar.checkbox("Nifty50 Index"): selected_watchlists.append("Nifty50 Index")
if st.sidebar.checkbox("Sensex Index"): selected_watchlists.append("Sensex Index")
if st.sidebar.checkbox("All NSE Stocks"): selected_watchlists.append("All NSE Stocks")

# Manage Favorites Expander
with st.sidebar.expander("⚙ Manage Favorites"):
    fav_list = load_favorites()
    new_fav = st.text_input("Add Stock Symbol").upper().strip()
    if st.button("Add Stock") and new_fav:
        if new_fav not in fav_list:
            fav_list.append(new_fav)
            save_favorites(fav_list)
            st.rerun()
    selected_to_remove = st.selectbox("Remove Stock", ["None"] + fav_list)
    if st.button("Remove Selected") and selected_to_remove != "None":
        fav_list.remove(selected_to_remove)
        save_favorites(fav_list)
        st.rerun()

st.sidebar.subheader("Parameters")
stoch_k = st.sidebar.number_input("Slow Stoch K Period", value=14)
smooth_k = st.sidebar.number_input("Smooth K", value=4)
cross_below_val = st.sidebar.number_input("Crossed Above (Bullish Threshold)", value=22)
cross_above_val = st.sidebar.number_input("Crossed Below (Bearish Threshold)", value=76)

use_ema = st.sidebar.checkbox("Enable EMA Filter", value=True)
ema_period = st.sidebar.selectbox("EMA Period", [20, 50, 100, 200], index=2) if use_ema else 100

use_price_filter = st.sidebar.checkbox("Enable Price Filter", value=False)
min_p, max_p = 5.0, 25000.0
if use_price_filter:
    min_p = st.sidebar.number_input("Min Price ₹", value=5.0)
    max_p = st.sidebar.number_input("Max Price ₹", value=25000.0)

# ==========================================
# Main Dashboard UI
# ==========================================
st.title("📈 VSN Info - Slow Stochastic & EMA Screener")

if st.button("🚀 Run Market Scan", type="primary", use_container_width=True):
    stocks = set()
    if "All NSE Stocks" in selected_watchlists: stocks.update(get_all_nse_stocks())
    if "Nifty50 Stocks" in selected_watchlists: stocks.update(["RELIANCE", "BHARTIARTL", "GRASIM", "M&M", "LT", "ITC", "INFY", "TCS", "HDFCBANK", "ICICIBANK"])
    if "Sensex Stocks" in selected_watchlists: stocks.update(["ADANIPORTS", "HDFCLIFE", "SBIN", "AXISBANK", "KOTAKBANK"])
    if "Nifty50 Index" in selected_watchlists: stocks.add("NIFTY 50")
    if "Sensex Index" in selected_watchlists: stocks.add("SENSEX")
    if "My Favorites" in selected_watchlists: stocks.update(load_favorites())

    if not stocks:
        st.warning("કૃપા કરીને સાઇડબારમાંથી ઓછામાં ઓછી ૧ Watchlist સિલેક્ટ કરો.")
    else:
        params = {
            'stoch_k': stoch_k, 'smooth': smooth_k, 'below': cross_below_val,
            'above': cross_above_val, 'use_ema': use_ema, 'ema_period': ema_period,
            'use_price_filter': use_price_filter, 'min_price': min_p, 'max_price': max_p,
            'use_heikin_ashi': use_ha
        }

        bullish, bearish = [], []
        progress_bar = st.progress(0)
        status_text = st.empty()

        with ThreadPoolExecutor(max_workers=8) as executor:
            future_map = {
                executor.submit(process_stock, symbol, timeframe, use_lookback, lookback_days, start_date, end_date, params): symbol 
                for symbol in stocks
            }
            total = len(future_map)
            completed = 0
            
            for future in as_completed(future_map):
                completed += 1
                progress_bar.progress(completed / total)
                status_text.text(f"Scanning ({completed}/{total})...")
                try:
                    b_list, s_list = future.result()
                    bullish.extend(b_list)
                    bearish.extend(s_list)
                except Exception:
                    pass

        status_text.success("Scan Completed!")

        bullish.sort(key=lambda x: str(x['dt']), reverse=True)
        bearish.sort(key=lambda x: str(x['dt']), reverse=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("▲ Bullish Signals")
            if bullish:
                df_bull = pd.DataFrame(bullish)[['date', 'time', 'symbol', 'price']]
                df_bull.columns = ['Date', 'Time', 'Stock', 'Price ₹']
                st.dataframe(df_bull, use_container_width=True, hide_index=True)
            else:
                st.info("No Bullish Signals Found.")

        with col2:
            st.subheader("▼ Bearish Signals")
            if bearish:
                df_bear = pd.DataFrame(bearish)[['date', 'time', 'symbol', 'price']]
                df_bear.columns = ['Date', 'Time', 'Stock', 'Price ₹']
                st.dataframe(df_bear, use_container_width=True, hide_index=True)
            else:
                st.info("No Bearish Signals Found.")
