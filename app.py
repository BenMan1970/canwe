import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import os
from ta.trend import adx

st.set_page_config(page_title="Scanner Multi-Timeframe", layout="wide")
st.title("⭐ Signaux 5 & 6 Étoiles - Multi Timeframe")

# 🔐 Clés API depuis Streamlit Secrets
try:
    api_key = st.secrets["alpaca"]["alpaca_api_key"]
    secret_key = st.secrets["alpaca"]["alpaca_secret_key"]
except:
    api_key = os.getenv("alpaca_api_key", "")
    secret_key = os.getenv("alpaca_secret_key", "")

headers = {
    "Apca-Api-Key-Id": api_key,
    "Apca-Api-Secret-Key": secret_key
}

# Paramètres des indicateurs
hma_length = 20
adx_threshold = 20
rsi_length = 10
adx_length = 14
ichimoku_len = 9

symbols = [
    'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD',
    'USDJPY', 'USDCAD', 'USDCHF',
    'XAUUSD', 'US30', 'NAS100', 'SPX'
]

# Mapping des timeframes
timeframes = {
    'Daily': '1Day',
    'H4': '4Hour', 
    'H1': '1Hour'
}

@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def fetch_bars(symbol, timeframe='1Hour', limit=100):
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    params = {'timeframe': timeframe, 'limit': limit}
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'bars' in data and data['bars']:
                return pd.DataFrame(data['bars'])
        return None
    except Exception as e:
        return None

def hma(values, length):
    if len(values) < length:
        return np.full(len(values), np.nan)
    
    half_length = max(1, int(length / 2))
    sqrt_length = max(1, int(np.sqrt(length)))
    
    series = pd.Series(values)
    wma_half = series.rolling(half_length).mean()
    wma_full = series.rolling(length).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_result = raw_hma.rolling(sqrt_length).mean()
    
    return hma_result.fillna(method='bfill').values

def calculate_indicators(df):
    if len(df) < 60:
        return None
        
    # Assurer que les colonnes sont numériques
    for col in ['close', 'high', 'low', 'open']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    if len(df) < 60:
        return None
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values

    signals = {}

    # HMA - Tendance
    try:
        hma_values = hma(close, hma_length)
        hma_slope = 1 if len(hma_values) >= 2 and hma_values[-1] > hma_values[-2] else -1
        signals['hma_slope'] = hma_slope
    except:
        signals['hma_slope'] = 0

    # Heiken Ashi
    try:
        ha_close = (open_ + high + low + close) / 4
        ha_open = np.zeros_like(open_)
        ha_open[0] = (open_[0] + close[0]) / 2
        for i in range(1, len(ha_open)):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
        signals['haSignal'] = 1 if ha_close[-1] > ha_open[-1] else -1
    except:
        signals['haSignal'] = 0

    # Smoothed Heiken Ashi
    try:
        def ema(x, n): 
            return pd.Series(x).ewm(span=n, adjust=False).mean().values
        o = ema(open_, 10)
        c = ema(close, 10)
        h = ema(high, 10)
        l = ema(low, 10)
        haclose = (o + h + l + c) / 4
        haopen = np.zeros_like(o)
        haopen[0] = (o[0] + c[0]) / 2
        for i in range(1, len(haopen)):
            haopen[i] = (haopen[i - 1] + haclose[i - 1]) / 2
        ema_haopen = ema(haopen, 10)
        ema_haclose = ema(haclose, 10)
        signals['smoothedHaSignal'] = 1 if ema_haopen[-1] < ema_haclose[-1] else -1
    except:
        signals['smoothedHaSignal'] = 0

    # RSI
    try:
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(rsi_length).mean()
        avg_loss = loss.rolling(rsi_length).mean()
        rs = avg_gain / avg_loss
        rsi_values = 100 - (100 / (1 + rs))
        signals['rsiSignal'] = 1 if rsi_values.iloc[-1] > 50 else -1
    except:
        signals['rsiSignal'] = 0

    # ADX
    try:
        adx_values = adx(pd.Series(high), pd.Series(low), pd.Series(close), window=adx_length)
        signals['adxHasMomentum'] = 1 if adx_values.iloc[-1] >= adx_threshold else 0
    except:
        signals['adxHasMomentum'] = 0

    # Ichimoku
    try:
        tenkan = (pd.Series(high).rolling(ichimoku_len).max() + pd.Series(low).rolling(ichimoku_len).min()) / 2
        kijun = (pd.Series(high).rolling(26).max() + pd.Series(low).rolling(26).min()) / 2
        senkouA = (tenkan + kijun) / 2
        senkouB = (pd.Series(high).rolling(52).max() + pd.Series(low).rolling(52).min()) / 2
        
        if close[-1] > senkouA.iloc[-1]:
            signals['ichimokuSignal'] = 1
        elif close[-1] < senkouB.iloc[-1]:
            signals['ichimokuSignal'] = -1
        else:
            signals['ichimokuSignal'] = 0
    except:
        signals['ichimokuSignal'] = 0

    return signals

def analyze_symbol_multi_timeframe(symbol):
    """Analyse un symbole sur plusieurs timeframes"""
    timeframe_results = {}
    
    for tf_name, tf_code in timeframes.items():
        data = fetch_bars(symbol, tf_code, 100)
        if data is not None and len(data) > 60:
            signals = calculate_indicators(data)
            if signals:
                timeframe_results[tf_name] = signals
    
    return timeframe_results

def check_trend_alignment(tf_results):
    """Vérifie si la tendance principale (HMA) est alignée sur tous les timeframes"""
    if len(tf_results) < 3:  # Besoin des 3 timeframes
        return False, "Données insuffisantes"
    
    hma_signals = []
    for tf in ['Daily', 'H4', 'H1']:
        if tf in tf_results and 'hma_slope' in tf_results[tf]:
            hma_signals.append(tf_results[tf]['hma_slope'])
    
    if len(hma_signals) < 3:
        return False, "HMA manquant"
    
    # Vérifier si tous les HMA vont dans la même direction
    all_bullish = all(signal == 1 for signal in hma_signals)
    all_bearish = all(signal == -1 for signal in hma_signals)
    
    if all_bullish:
        return True, "Tendance HAUSSIÈRE alignée"
    elif all_bearish:
        return True, "Tendance BAISSIÈRE alignée"
    else:
        return False, f"Tendances mixtes: D:{hma_signals[0]} H4:{hma_signals[1]} H1:{hma_signals[2]}"

def scan_market_multi_timeframe():
    """Scanner le marché avec analyse multi-timeframe"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols):
        status_text.text(f'📊 Analyse multi-timeframe de {symbol}...')
        progress_bar.progress((i + 1) / len(symbols))
        
        # Analyser sur tous les timeframes
        tf_results = analyze_symbol_multi_timeframe(symbol)
        
        if len(tf_results) >= 3:  # Au moins les 3 timeframes
            # Vérifier l'alignement des tendances
            trend_aligned, trend_msg = check_trend_alignment(tf_results)
            
            if trend_aligned:
                # Calculer le score sur H1 (timeframe d'entrée)
                h1_signals = tf_results.get('H1', {})
                signal_columns = ['hma_slope', 'haSignal', 'smoothedHaSignal', 'rsiSignal', 'adxHasMomentum', 'ichimokuSignal']
                
                bull = sum(1 for col in signal_columns if h1_signals.get(col, 0) == 1)
                bear = sum(1 for col in signal_columns if h1_signals.get(col, 0) == -1)
                conf = max(bull, bear)
                
                if conf >= 5:
                    direction = '🟢 ↑ ACHAT' if bull >= bear else '🔴 ↓ VENTE'
                    
                    results.append({
                        'Symbole': symbol,
                        'Direction': direction,
                        'Étoiles': '⭐' * conf,
                        'Score': conf,
                        'Tendance': trend_msg,
                        'Daily': get_tf_summary(tf_results.get('Daily', {})),
                        'H4': get_tf_summary(tf_results.get('H4', {})),
                        'H1': get_tf_summary(tf_results.get('H1', {}))
                    })
    
    progress_bar.empty()
    status_text.empty()
    return results

def get_tf_summary(signals):
    """Résumé des signaux pour un timeframe"""
    if not signals:
        return "❌"
    
    signal_columns = ['hma_slope', 'haSignal', 'smoothedHaSignal', 'rsiSignal', 'adxHasMomentum', 'ichimokuSignal']
    bull = sum(1 for col in signal_columns if signals.get(col, 0) == 1)
    bear = sum(1 for col in signal_columns if signals.get(col, 0) == -1)
    
    if bull > bear:
        return f"🟢 {bull}/6"
    elif bear > bull:
        return f"🔴 {bear}/6"
    else:
        return f"⚪ {max(bull, bear)}/6"

# Interface principale
st.markdown("""
### 📈 Scanner Multi-Timeframe
**Conditions pour un signal valide :**
1. ✅ Tendance HMA alignée sur Daily, H4 et H1
2. ✅ Score minimum de 5/6 indicateurs sur H1
3. ✅ Confirmation multi-timeframe
""")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔄 Scanner Multi-Timeframe", type="primary", use_container_width=True):
        with st.spinner("🔍 Analyse multi-timeframe en cours..."):
            signals = scan_market_multi_timeframe()
            
            if signals:
                st.success(f"✅ {len(signals)} signaux VALIDÉS détectés!")
                
                # Afficher sous forme de tableau détaillé
                df_results = pd.DataFrame(signals)
                df_results = df_results.sort_values('Score', ascending=False)
                
                st.dataframe(
                    df_results,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Symbole": st.column_config.TextColumn("🎯 Symbole", width="small"),
                        "Direction": st.column_config.TextColumn("📊 Direction", width="medium"),
                        "Étoiles": st.column_config.TextColumn("⭐ Score", width="small"),
                        "Tendance": st.column_config.TextColumn("📈 Tendance Multi-TF", width="large"),
                        "Daily": st.column_config.TextColumn("📅 Daily", width="small"),
                        "H4": st.column_config.TextColumn("🕐 H4", width="small"),
                        "H1": st.column_config.TextColumn("⏰ H1", width="small")
                    }
                )
                
                # Signaux détaillés
                st.write("### 🎯 Signaux Détaillés:")
                for result in signals:
                    with st.expander(f"📊 {result['Symbole']} - {result['Direction']} {result['Étoiles']}"):
                        st.write(f"**Tendance Multi-Timeframe:** {result['Tendance']}")
                        st.write(f"**Daily:** {result['Daily']}")
                        st.write(f"**H4:** {result['H4']}")
                        st.write(f"**H1:** {result['H1']} ← *Timeframe d'entrée*")
            else:
                st.info("🔍 Aucun signal multi-timeframe validé détecté actuellement.")
                st.write("*Les signaux doivent avoir une tendance HMA alignée sur Daily, H4 et H1 avec un score ≥ 5/6*")

with col2:
    auto_refresh = st.checkbox("🔄 Auto-refresh (5 min)")
    if auto_refresh:
        st.info("🔄 Refresh automatique activé")

# Informations détaillées
with st.expander("ℹ️ Méthodologie Multi-Timeframe"):
    st.write("""
    **🎯 Processus de Validation:**
    
    **1. Analyse de Tendance (Multi-TF):**
    - Daily : Tendance principale long terme
    - H4 : Tendance intermédiaire  
    - H1 : Tendance court terme (entrée)
    - ✅ **Requis:** HMA aligné sur les 3 timeframes
    
    **2. Indicateurs Analysés (sur H1):**
    - **HMA** : Tendance (Hull Moving Average)
    - **Heiken Ashi** : Direction du mouvement
    - **Smoothed HA** : Tendance lissée
    - **RSI** : Force relative (>50 = haussier)
    - **ADX** : Momentum (≥20 = tendance forte)
    - **Ichimoku** : Support/Résistance cloud
    
    **3. Scoring:**
    - 🟢 Signal haussier = indicateur positif
    - 🔴 Signal baissier = indicateur négatif  
    - ⭐ Score = nombre d'indicateurs alignés
    - 🎯 **Minimum requis:** 5/6 indicateurs
    
    **4. Validation Finale:**
    - Tendance multi-TF alignée ✅
    - Score H1 ≥ 5/6 ✅
    - Direction confirmée ✅
    """)

# Auto-refresh
if auto_refresh:
    time.sleep(5)  # 5 secondes pour les tests (300 pour 5 minutes en prod)
    st.rerun()
