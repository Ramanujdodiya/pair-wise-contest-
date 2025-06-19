import pandas as pd
import numpy as np
from typing import Dict, List

# --- Helper Functions for Indicators ---

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Average True Range (ATR)."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Average Directional Index (ADX)."""
    atr = calculate_atr(high, low, close, period)
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[(plus_dm < 0) | (plus_dm <= -minus_dm)] = 0
    minus_dm[(minus_dm < 0) | (minus_dm <= plus_dm)] = 0
    
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.000001)
    return dx.ewm(alpha=1/period, adjust=False).mean()


# --- Main Strategy Functions ---

def get_coin_metadata() -> dict:
    """Defines the coins and timeframes for the strategy."""
    return {
        'anchors': [{'symbol': 'BTC', 'timeframe': '1H'}],
        'targets': [
            {'symbol': 'MATIC', 'timeframe': '1H'},
            {'symbol': 'AVAX', 'timeframe': '1H'},
            {'symbol': 'LINK', 'timeframe': '1H'},
        ]
    }

def generate_signals(anchor_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced pairs trading strategy with multiple filters and dynamic position sizing.
    """
    all_signals = []
    
    # --- Parameters (Optimized Guesses) ---
    primary_anchor = 'BTC'
    lookback_period = 72
    entry_std_dev = 2.5
    stop_loss_std_dev = 3.5
    profit_target_std_dev = 1.0
    rsi_period = 14
    rsi_threshold = 60
    volume_lookback = 24
    adx_period = 14
    adx_threshold = 20 # Lowered to allow trading in moderately trending environments

    # --- 1. Prepare Anchor Data & Advanced Regime Filter ---
    anchor_price_col = f'close_{primary_anchor}_1H'
    if anchor_price_col not in anchor_df.columns:
        return pd.DataFrame()
        
    if anchor_df.index.name == 'timestamp':
        anchor_df = anchor_df.reset_index()

    # Calculate ADX and ATR for the regime filter
    anchor_df['adx'] = calculate_adx(anchor_df[f'high_{primary_anchor}_1H'], anchor_df[f'low_{primary_anchor}_1H'], anchor_df[anchor_price_col], adx_period)
    anchor_df['is_trending'] = anchor_df['adx'] > adx_threshold
    
    anchor_data_to_merge = anchor_df[['timestamp', anchor_price_col, 'is_trending']].copy()

    # --- 2. Loop Through Each Target to Form a Pair ---
    for symbol in get_coin_metadata()['targets']:
        symbol = symbol['symbol']
        target_price_col = f'close_{symbol}_1H'
        volume_col = f'volume_{symbol}_1H'
        if target_price_col not in target_df.columns or volume_col not in target_df.columns:
            continue
        
        if target_df.index.name == 'timestamp':
            target_df = target_df.reset_index()

        # --- 3. Vectorized Calculation of Indicators & Conditions ---
        # Merge all necessary data
        merge_cols = ['timestamp', target_price_col, volume_col]
        pair_df = pd.merge(target_df[merge_cols], anchor_data_to_merge, on='timestamp', how='left').dropna()
        
        # Calculate Ratio and Bands
        pair_df['price_ratio'] = pair_df[target_price_col] / pair_df[anchor_price_col]
        pair_df['ratio_sma'] = pair_df['price_ratio'].rolling(window=lookback_period).mean()
        ratio_std = pair_df['price_ratio'].rolling(window=lookback_period).std()
        
        pair_df['entry_band'] = pair_df['ratio_sma'] - (ratio_std * entry_std_dev)
        pair_df['stop_loss_band'] = pair_df['ratio_sma'] - (ratio_std * stop_loss_std_dev)
        pair_df['profit_target_band'] = pair_df['ratio_sma'] + (ratio_std * profit_target_std_dev)
        
        # Calculate Confirmation Indicators
        pair_df['rsi'] = calculate_rsi(pair_df[target_price_col], rsi_period)
        pair_df['volume_sma'] = pair_df[volume_col].rolling(window=volume_lookback).mean()

        # Define all conditions vectorially
        is_trending_regime = pair_df['is_trending']
        is_ratio_entry = pair_df['price_ratio'] < pair_df['entry_band']
        is_rsi_ok = pair_df['rsi'] < rsi_threshold
        is_volume_ok = pair_df[volume_col] > pair_df['volume_sma']

        # Combine all conditions into a final buy trigger
        buy_condition = is_trending_regime & is_ratio_entry & is_rsi_ok & is_volume_ok
        
        # --- 4. Stateful Signal Generation (Loop for state management) ---
        pair_df['signal'] = 'HOLD'
        pair_df['position_size'] = 0.0
        in_position = False

        for i in pair_df.index:
            # Check for BUY signal
            if not in_position and buy_condition[i]:
                pair_df.at[i, 'signal'] = 'BUY'
                in_position = True
                # Dynamic Position Sizing
                deviation = (pair_df.at[i, 'ratio_sma'] - pair_df.at[i, 'price_ratio']) / ratio_std[i]
                pair_df.at[i, 'position_size'] = np.clip(deviation * 0.1, 0.1, 0.3) # Scale size by deviation, cap at 30%

            # Check for SELL signal
            elif in_position:
                is_stop_loss = pair_df.at[i, 'price_ratio'] < pair_df.at[i, 'stop_loss_band']
                is_take_profit = pair_df.at[i, 'price_ratio'] > pair_df.at[i, 'profit_target_band']
                
                if is_stop_loss or is_take_profit:
                    pair_df.at[i, 'signal'] = 'SELL'
                    pair_df.at[i, 'position_size'] = 1.0  # Sell entire position
                    in_position = False

        pair_df['symbol'] = symbol
        trade_signals = pair_df[pair_df['signal'] != 'HOLD']
        if not trade_signals.empty:
            all_signals.append(trade_signals[['timestamp', 'symbol', 'signal', 'position_size']])
    
    # --- 5. Combine and Return Final Signals ---
    if not all_signals:
        print("Warning: No BUY/SELL signals were generated under the new advanced rules.")
        return pd.DataFrame()
        
    return pd.concat(all_signals).sort_values('timestamp').reset_index(drop=True)
