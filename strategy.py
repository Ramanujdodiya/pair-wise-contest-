



import pandas as pd
import numpy as np

def get_coin_metadata() -> dict:
    """
    Defines the coins and timeframes for the strategy.
    The user's selected coins are kept.
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
            {'symbol': 'ADA', 'timeframe': '1H'},
            {'symbol': 'DOT', 'timeframe': '1H'}
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
    std_dev_multiplier = 2.0 # Widened the band slightly to look for more significant deviations
    stop_loss_pct = 0.03   # 3% price-based stop-loss

    # --- 1. Prepare Anchor Data ---
    anchor_price_col = f'close_{primary_anchor}_1H'
    if anchor_price_col not in anchor_df.columns:
        primary_anchor = 'ETH'
        anchor_price_col = f'close_{primary_anchor}_1H'
        if anchor_price_col not in anchor_df.columns:
            return pd.DataFrame()
    
    anchor_prices = anchor_df[anchor_price_col].ffill()

    # --- 2. Loop Through Each Target to Form a Pair ---
    target_symbols = [t['symbol'] for t in get_coin_metadata()['targets']]

    for symbol in target_symbols:
        target_price_col = f'close_{symbol}_1H'
        if target_price_col not in target_df.columns:
            continue

        pair_df = pd.DataFrame({
            'timestamp': target_df['timestamp'],
            'target_price': target_df[target_price_col],
            'anchor_price': anchor_prices
        })

        # --- 3. Calculate Ratio and Bands ---
        pair_df['price_ratio'] = pair_df['target_price'] / pair_df['anchor_price']
        pair_df['ratio_sma'] = pair_df['price_ratio'].rolling(window=lookback_period).mean()
        pair_df['lower_band'] = pair_df['ratio_sma'] - (pair_df['price_ratio'].rolling(window=lookback_period).std() * std_dev_multiplier)
        
        # --- 4. Generate Signals ---
        pair_df['signal'] = 'HOLD'
        pair_df['position_size'] = 0.0
        in_position = False
        entry_price = 0 

        for i in range(1, len(pair_df)):
            current_ratio = pair_df.at[i, 'price_ratio']
            current_price = pair_df.at[i, 'target_price']
            lower_band = pair_df.at[i, 'lower_band']
            mean_ratio = pair_df.at[i, 'ratio_sma']
            
            if pd.isna(lower_band) or pd.isna(mean_ratio):
                continue
                
            # BUY Signal: Ratio drops below the lower band
            if not in_position and current_ratio < lower_band:
                pair_df.at[i, 'signal'] = 'BUY'
                pair_df.at[i, 'position_size'] = 0.2
                in_position = True
                entry_price = current_price # Store the asset's price at entry
            
            # --- CORRECTED HYBRID SELL LOGIC ---
            elif in_position and pd.notna(current_price):
                # SELL Condition 1: Take profit if ratio reverts to mean
                if current_ratio >= mean_ratio:
                    pair_df.at[i, 'signal'] = 'SELL'
                    pair_df.at[i, 'position_size'] = 1.0
                    in_position = False
                    entry_price = 0
                # SELL Condition 2: Stop-loss if asset price drops 3%
                elif (current_price - entry_price) / entry_price <= -stop_loss_pct:
                    pair_df.at[i, 'signal'] = 'SELL'
                    pair_df.at[i, 'position_size'] = 1.0
                    in_position = False
                    entry_price = 0
                # If no sell condition is met, continue to HOLD
                else:
                    pair_df.at[i, 'signal'] = 'HOLD'
                    pair_df.at[i, 'position_size'] = 0.2
            
            # If in position but price is missing, just HOLD
            elif in_position:
                pair_df.at[i, 'signal'] = 'HOLD'
                pair_df.at[i, 'position_size'] = 0.2

        pair_df['symbol'] = symbol
        all_signals.append(pair_df[['timestamp', 'symbol', 'signal', 'position_size']])
    
    # --- 5. Combine and Return Final Signals ---
    if not all_signals:
        return pd.DataFrame()
        
    full_signals_df = pd.concat(all_signals).sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    return full_signals_df


