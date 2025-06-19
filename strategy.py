import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Helper function to calculate RSI
def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_coin_metadata() -> dict:
    """
    Defines the coins and timeframes for the strategy.
    """
    return {
        'anchors': [
            {'symbol': 'BTC', 'timeframe': '1H'},
            {'symbol': 'ETH', 'timeframe': '1H'},
        ],
        'targets': [
            {'symbol': 'MATIC', 'timeframe': '1H'},
            {'symbol': 'AVAX', 'timeframe': '1H'},
            {'symbol': 'LINK', 'timeframe': '1H'},
        ]
    }

def generate_signals(anchor_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Further improved pairs trading strategy with an RSI confirmation filter.
    
    - Market Filter: Only trades when BTC is above its 200-hour SMA.
    - Confirmation Filter: Only enters if the target asset's RSI is not overbought.
    - Entry: Buys when the price ratio drops 2 std deviations below its mean.
    - Take Profit & Stop-Loss: Based on the price ratio's standard deviation.
    """
    anchor_df = anchor_df.copy()
    target_df = target_df.copy()
    all_signals = []
    
    # --- Parameters ---
    primary_anchor = 'BTC'
    lookback_period = 48
    entry_std_dev = 2.0
    stop_loss_std_dev = 3.0
    profit_target_std_dev = 0.5
    rsi_period = 14
    rsi_overbought_threshold = 65 # New parameter for confirmation filter

    # --- 1. Prepare Anchor Data & Market Regime Filter ---
    anchor_price_col = f'close_{primary_anchor}_1H'
    if anchor_price_col not in anchor_df.columns:
        print("Error: Could not find anchor price column for BTC.")
        return pd.DataFrame()
        
    if anchor_df.index.name == 'timestamp':
        anchor_df = anchor_df.reset_index()

    anchor_df['regime_sma'] = anchor_df[anchor_price_col].rolling(window=200).mean()
    anchor_df['is_bull_regime'] = anchor_df[anchor_price_col] > anchor_df['regime_sma']
    
    anchor_prices_df = anchor_df[['timestamp', anchor_price_col, 'is_bull_regime']].copy()
    anchor_prices_df[anchor_price_col] = anchor_prices_df[anchor_price_col].ffill()

    # --- 2. Loop Through Each Target to Form a Pair ---
    target_symbols = [t['symbol'] for t in get_coin_metadata()['targets']]

    for symbol in target_symbols:
        target_price_col = f'close_{symbol}_1H'
        if target_price_col not in target_df.columns:
            continue
        
        if target_df.index.name == 'timestamp':
            target_df = target_df.reset_index()

        # --- NEW: Calculate RSI for the target asset ---
        target_df[f'rsi_{symbol}'] = calculate_rsi(target_df[target_price_col], rsi_period)

        # Merge all necessary data together
        merge_cols = ['timestamp', target_price_col, f'rsi_{symbol}']
        pair_df = pd.merge(target_df[merge_cols], anchor_prices_df, on='timestamp', how='left')
        pair_df.rename(columns={target_price_col: 'target_price', anchor_price_col: 'anchor_price'}, inplace=True)
        pair_df.dropna(inplace=True)

        if pair_df.empty: continue

        # --- 3. Calculate Ratio and Dynamic Bands ---
        pair_df['price_ratio'] = pair_df['target_price'] / pair_df['anchor_price']
        pair_df['ratio_sma'] = pair_df['price_ratio'].rolling(window=lookback_period).mean()
        ratio_std = pair_df['price_ratio'].rolling(window=lookback_period).std()
        
        pair_df['entry_band'] = pair_df['ratio_sma'] - (ratio_std * entry_std_dev)
        pair_df['stop_loss_band'] = pair_df['ratio_sma'] - (ratio_std * stop_loss_std_dev)
        pair_df['profit_target_band'] = pair_df['ratio_sma'] + (ratio_std * profit_target_std_dev)
        
        # --- 4. Generate Signals with New Rules ---
        pair_df['signal'] = 'HOLD'
        pair_df['position_size'] = 0.0
        in_position = False

        for i in pair_df.index:
            is_bull_regime = pair_df.at[i, 'is_bull_regime']
            current_ratio = pair_df.at[i, 'price_ratio']
            current_rsi = pair_df.at[i, f'rsi_{symbol}'] # Get current RSI
            
            entry_band = pair_df.at[i, 'entry_band']
            stop_loss_band = pair_df.at[i, 'stop_loss_band']
            profit_target_band = pair_df.at[i, 'profit_target_band']
            
            if pd.isna(entry_band) or pd.isna(stop_loss_band) or pd.isna(profit_target_band) or pd.isna(current_rsi):
                continue
            
            # --- UPDATED BUY Signal with RSI confirmation ---
            is_entry_signal = current_ratio < entry_band
            is_rsi_ok = current_rsi < rsi_overbought_threshold # New condition
            
            if not in_position and is_bull_regime and is_entry_signal and is_rsi_ok:
                pair_df.at[i, 'signal'] = 'BUY'
                pair_df.at[i, 'position_size'] = 0.2
                in_position = True
            
            elif in_position:
                is_stop_loss = current_ratio < stop_loss_band
                is_take_profit = current_ratio > profit_target_band
                
                if is_stop_loss or is_take_profit:
                    pair_df.at[i, 'signal'] = 'SELL'
                    pair_df.at[i, 'position_size'] = 1.0
                    in_position = False

        pair_df['symbol'] = symbol
        trade_signals = pair_df[pair_df['signal'] != 'HOLD']
        if not trade_signals.empty:
            all_signals.append(trade_signals[['timestamp', 'symbol', 'signal', 'position_size']])
    
    if not all_signals:
        print("Warning: No BUY/SELL signals were generated under the new rules.")
        return pd.DataFrame()
        
    full_signals_df = pd.concat(all_signals).sort_values('timestamp').reset_index(drop=True)
    return full_signals_df
