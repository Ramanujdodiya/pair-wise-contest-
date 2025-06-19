import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

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
    This is a hybrid strategy combining a Pairs Trading entry with robust risk management.

    - Entry: Buy when the target/anchor price ratio drops significantly below its average.
    - Take Profit: Sell when the price ratio reverts back to its historical average.
    - Stop-Loss: Sell if the asset's price drops by a fixed percentage from the entry price.
    """
    anchor_df = anchor_df.copy()
    target_df = target_df.copy()
    all_signals = []
    
    # --- Parameters ---
    primary_anchor = 'BTC'
    lookback_period = 48
    std_dev_multiplier = 2.0
    stop_loss_pct = 0.03

    # --- 1. Prepare Anchor Data ---
    anchor_price_col = f'close_{primary_anchor}_1H'
    if anchor_price_col not in anchor_df.columns:
        print(f"Warning: Primary anchor '{primary_anchor}' not found, falling back to ETH.")
        primary_anchor = 'ETH'
        anchor_price_col = f'close_{primary_anchor}_1H'
        if anchor_price_col not in anchor_df.columns:
            print("Error: Could not find anchor price columns for BTC or ETH.")
            return pd.DataFrame() # Return empty DataFrame
    
    # Ensure timestamp is a column for merging
    if anchor_df.index.name == 'timestamp':
        anchor_df = anchor_df.reset_index()
    
    anchor_prices_df = anchor_df[['timestamp', anchor_price_col]].copy()
    anchor_prices_df[anchor_price_col] = anchor_prices_df[anchor_price_col].ffill()


    # --- 2. Loop Through Each Target to Form a Pair ---
    target_symbols = [t['symbol'] for t in get_coin_metadata()['targets']]

    for symbol in target_symbols:
        target_price_col = f'close_{symbol}_1H'
        if target_price_col not in target_df.columns:
            print(f"Info: Skipping {symbol} as its price column '{target_price_col}' was not found in target_df.")
            continue
        
        # Ensure timestamp is a column for merging
        if target_df.index.name == 'timestamp':
            target_df = target_df.reset_index()

        # Create a DataFrame for the specific pair
        pair_df = pd.merge(
            target_df[['timestamp', target_price_col]], 
            anchor_prices_df, 
            on='timestamp', 
            how='left'
        )
        pair_df.rename(columns={target_price_col: 'target_price', anchor_price_col: 'anchor_price'}, inplace=True)
        pair_df.dropna(inplace=True)

        if pair_df.empty:
            continue

        # --- 3. Calculate Ratio and Bands ---
        pair_df['price_ratio'] = pair_df['target_price'] / pair_df['anchor_price']
        pair_df['ratio_sma'] = pair_df['price_ratio'].rolling(window=lookback_period).mean()
        pair_df['lower_band'] = pair_df['ratio_sma'] - (pair_df['price_ratio'].rolling(window=lookback_period).std() * std_dev_multiplier)
        
        # --- 4. Generate Signals ---
        pair_df['signal'] = 'HOLD'
        pair_df['position_size'] = 0.0
        in_position = False
        entry_price = 0 

        for i in pair_df.index:
            current_ratio = pair_df.at[i, 'price_ratio']
            current_price = pair_df.at[i, 'target_price']
            lower_band = pair_df.at[i, 'lower_band']
            mean_ratio = pair_df.at[i, 'ratio_sma']
            
            if pd.isna(lower_band) or pd.isna(mean_ratio) or pd.isna(current_price):
                continue
            
            if not in_position and current_ratio < lower_band:
                pair_df.at[i, 'signal'] = 'BUY'
                pair_df.at[i, 'position_size'] = 0.2
                in_position = True
                entry_price = current_price
            
            elif in_position:
                # Check for stop-loss or take-profit conditions
                is_stop_loss = (current_price - entry_price) / entry_price <= -stop_loss_pct
                is_take_profit = current_ratio >= mean_ratio
                
                if is_stop_loss or is_take_profit:
                    pair_df.at[i, 'signal'] = 'SELL'
                    pair_df.at[i, 'position_size'] = 1.0
                    in_position = False
                    entry_price = 0

        pair_df['symbol'] = symbol
        # Filter for only rows where a signal exists
        trade_signals = pair_df[pair_df['signal'] != 'HOLD']
        if not trade_signals.empty:
            all_signals.append(trade_signals[['timestamp', 'symbol', 'signal', 'position_size']])
    
    # --- 5. Combine and Return Final Signals ---
    if not all_signals:
        print("Warning: No BUY/SELL signals were generated for any target symbols.")
        return pd.DataFrame()
        
    full_signals_df = pd.concat(all_signals).sort_values('timestamp').reset_index(drop=True)
    return full_signals_df
