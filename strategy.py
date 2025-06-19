import pandas as pd
import numpy as np

def get_coin_metadata():
    """
    Define the coins and timeframes for the strategy.
    Anchors: Major coins that lead the market
    Targets: Coins we trade based on anchor signals
    """
    return {
        'anchors': [
            {'symbol': 'BTC', 'timeframe': '1H'},
            {'symbol': 'ETH', 'timeframe': '1H'},
            {'symbol': 'SOL', 'timeframe': '1H'}
        ],
        'targets': [
            {'symbol': 'MATIC', 'timeframe': '1H'},
            {'symbol': 'AVAX', 'timeframe': '1H'},
            {'symbol': 'LINK', 'timeframe': '1H'},
            {'symbol': 'ADA', 'timeframe': '1H'},
            {'symbol': 'DOT', 'timeframe': '1H'}
        ]
    }

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(anchor_df, target_df):
    """
    Generate trading signals based on lead-lag relationships between anchors and targets.
    """
    signals = []
    
    # Strategy Parameters
    LAG_WINDOW = 6  # hours to look back for momentum
    MOMENTUM_THRESHOLD = 0.025  # 2.5% move threshold
    POSITION_SIZE = 0.18  # 18% per position
    MAX_POSITIONS = 4
    
    # Calculate anchor momentum signals
    anchor_momentum_cols = []
    
    for anchor in ['BTC', 'ETH', 'SOL']:
        close_col = f'close_{anchor}_1H'
        if close_col in anchor_df.columns:
            # Calculate momentum over lag window
            anchor_df.loc[:, f'{anchor}_momentum'] = anchor_df[close_col].pct_change(LAG_WINDOW)
            
            # Calculate volatility for normalization
            anchor_df[f'{anchor}_vol'] = anchor_df[close_col].pct_change().rolling(24).std()
            
            # Volatility-adjusted momentum
            anchor_df[f'{anchor}_adj_momentum'] = (
                anchor_df[f'{anchor}_momentum'] / anchor_df[f'{anchor}_vol']
            ).fillna(0)
            
            anchor_momentum_cols.append(f'{anchor}_momentum')
    
    # Calculate composite anchor signals
    if anchor_momentum_cols:
        anchor_df['composite_momentum'] = anchor_df[anchor_momentum_cols].mean(axis=1)
    else:
        anchor_df['composite_momentum'] = 0
    
    adj_momentum_cols = [col for col in anchor_df.columns if '_adj_momentum' in col]
    if adj_momentum_cols:
        anchor_df['composite_adj_momentum'] = anchor_df[adj_momentum_cols].mean(axis=1)
    else:
        anchor_df['composite_adj_momentum'] = 0
    
    # Track position states
    position_tracker = {}
    
    # Generate signals for each target coin
    for target in ['MATIC', 'AVAX', 'LINK', 'ADA', 'DOT']:
        close_col = f'close_{target}_1H'
        
        if close_col not in target_df.columns:
            continue
        
        # Initialize position tracker for this target
        position_tracker[target] = {'has_position': False, 'entry_time': None}
        
        # Merge anchor signals with target data
        merged_df = pd.merge(
            target_df[['timestamp', close_col]], 
            anchor_df[['timestamp', 'composite_momentum', 'composite_adj_momentum']], 
            on='timestamp', 
            how='left'
        )
        
        # Calculate target-specific indicators
        merged_df['target_momentum'] = merged_df[close_col].pct_change(3)
        merged_df['target_rsi'] = calculate_rsi(merged_df[close_col])
        
        # Generate signals for each timestamp
        for i, row in merged_df.iterrows():
            if pd.isna(row['composite_momentum']) or pd.isna(row['target_momentum']):
                continue
            
            timestamp = row['timestamp']
            composite_mom = row['composite_momentum']
            composite_adj_mom = row['composite_adj_momentum']
            target_mom = row['target_momentum']
            target_rsi = row['target_rsi']
            
            # Calculate signal strength
            recent_adj_std = merged_df['composite_adj_momentum'].rolling(48).std().iloc[i]
            if pd.isna(recent_adj_std) or recent_adj_std == 0:
                signal_strength = 1.0
            else:
                signal_strength = abs(composite_adj_mom) / recent_adj_std
            
            signal = 'HOLD'
            position_size = 0
            
            current_position = position_tracker[target]['has_position']
            
            # Entry Logic: Strong anchor momentum + target lag + high confidence
            if not current_position:
                # Long entry: Positive anchor momentum, target hasn't moved
                if (composite_mom > MOMENTUM_THRESHOLD and 
                    abs(target_mom) < MOMENTUM_THRESHOLD/2 and 
                    target_rsi < 70 and 
                    signal_strength > 1.3):
                    
                    signal = 'BUY'
                    position_size = POSITION_SIZE
                    position_tracker[target]['has_position'] = True
                    position_tracker[target]['entry_time'] = timestamp
                
                # Short entry: Negative anchor momentum, target hasn't dropped
                elif (composite_mom < -MOMENTUM_THRESHOLD and 
                      abs(target_mom) < MOMENTUM_THRESHOLD/2 and 
                      target_rsi > 30 and 
                      signal_strength > 1.3):
                    
                    # For backtester, we'll use BUY with negative position_size to indicate short
                    signal = 'BUY'
                    position_size = -POSITION_SIZE  # Negative for short
                    position_tracker[target]['has_position'] = True
                    position_tracker[target]['entry_time'] = timestamp
            
            # Exit Logic: Close positions when signals fade
            else:
                entry_time = position_tracker[target]['entry_time']
                hours_held = (timestamp - entry_time).total_seconds() / 3600 if entry_time else 0
                
                # Exit conditions
                should_exit = (
                    abs(composite_mom) < MOMENTUM_THRESHOLD/3 or  # Momentum fading
                    signal_strength < 0.8 or  # Low confidence
                    hours_held >= 48 or  # Max hold time
                    (composite_mom > 0 and target_mom > MOMENTUM_THRESHOLD) or  # Target caught up (long)
                    (composite_mom < 0 and target_mom < -MOMENTUM_THRESHOLD)  # Target caught up (short)
                )
                
                if should_exit:
                    signal = 'SELL'
                    position_size = 1.0  # Sell all
                    position_tracker[target]['has_position'] = False
                    position_tracker[target]['entry_time'] = None
            
            # Add signal to results
            signals.append({
                'timestamp': timestamp,
                'symbol': target,
                'signal': signal,
                'position_size': position_size,
                'anchor_momentum': composite_mom,
                'target_momentum': target_mom,
                'signal_strength': signal_strength,
                'target_rsi': target_rsi
            })
    
    return pd.DataFrame(signals)