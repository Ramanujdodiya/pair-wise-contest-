import pandas as pd

def get_coin_metadata() -> dict:
    """
    STEP 1: Define your trading pairs and timeframes
    
    Configure which coins you want to trade (targets) and which coins 
    you want to use for market analysis (anchors).
    
    Rules:
    - Max 3 target coins, 5 anchor coins
    - Timeframes: 1H, 2H, 4H, 12H, 1D
    - All symbols must be available on Binance as USDT pairs
    """
    return {
        "targets": [
            {"symbol": "BONK", "timeframe": "1H"},  # The coin you want to trade
            # {"symbol": "PEPE", "timeframe": "2H"},  # Add more targets if needed
        ],
        "anchors": [
            {"symbol": "BTC", "timeframe": "4H"},   # Major market indicator
            {"symbol": "ETH", "timeframe": "4H"},   # Another market indicator
            # {"symbol": "SOL", "timeframe": "1D"},   # Add more anchors if needed
        ]
    }

def generate_signals(anchor_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals.
    Must return: ['timestamp', 'symbol', 'signal', 'position_size']
    """
    
    # Your strategy logic here
    
    # Must return this exact structure
    result_df = pd.DataFrame({
        'timestamp': target_df['timestamp'],  # Use all timestamps
        'symbol': 'BONK',                     # Your target symbol
        'signal': 'HOLD',                     # Your signals: BUY/SELL/HOLD
        'position_size': 0.0                  # Your position size: 0.0 to 1.0
    })
    
    # Minimum 2 buy-sell pairs required for validation
      
    return result_df