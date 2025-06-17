import pandas as pd
import numpy as np

def get_coin_metadata() -> dict:
    return {
        "targets": [
            {"symbol": "LDO", "timeframe": "1H"},
            {"symbol": "PEPE", "timeframe": "1H"},
            {"symbol": "BONK", "timeframe": "1H"}
        ],
        "anchors": [
            {"symbol": "BTC", "timeframe": "4H"},
            {"symbol": "ETH", "timeframe": "4H"},
            {"symbol": "SOL", "timeframe": "1D"}
        ]
    }

def generate_signals(anchor_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    result = []

    # Detect all target symbols and iterate
    target_symbols = set(col.split('_')[1] for col in target_df.columns if col.startswith('close_'))

    for symbol in target_symbols:
        close_col = f'close_{symbol}_1H'
        if close_col not in target_df.columns:
            continue

        df = target_df[['timestamp', close_col]].copy()
        df.rename(columns={close_col: 'close'}, inplace=True)

        # Calculate SMAs
        df['sma_fast'] = df['close'].rolling(window=8).mean()
        df['sma_slow'] = df['close'].rolling(window=21).mean()

        # Entry/exit signals
        df['signal'] = 'HOLD'
        df['position_size'] = 0.0

        in_position = False
        entry_price = None

        for i in range(len(df)):
            price = df['close'].iloc[i]
            fast = df['sma_fast'].iloc[i]
            slow = df['sma_slow'].iloc[i]

            if pd.isna(fast) or pd.isna(slow):
                continue

            # Buy signal: fast crosses above slow
            if not in_position and fast > slow and df['sma_fast'].iloc[i - 1] <= df['sma_slow'].iloc[i - 1]:
                df.at[i, 'signal'] = 'BUY'
                df.at[i, 'position_size'] = 0.7
                in_position = True
                entry_price = price

            # Sell signal: 5% profit or 3% stop-loss
            elif in_position and pd.notna(price) and entry_price:
                pnl = (price - entry_price) / entry_price
                if pnl >= 0.05 or pnl <= -0.03:
                    df.at[i, 'signal'] = 'SELL'
                    df.at[i, 'position_size'] = 0.0
                    in_position = False
                    entry_price = None
                else:
                    df.at[i, 'signal'] = 'HOLD'
                    df.at[i, 'position_size'] = 0.7
            elif in_position:
                df.at[i, 'signal'] = 'HOLD'
                df.at[i, 'position_size'] = 0.7

        df['symbol'] = symbol
        result.append(df[['timestamp', 'symbol', 'signal', 'position_size']])

    full_signals = pd.concat(result).sort_values(['timestamp', 'symbol']).reset_index(drop=True)

    # Constraint check: ensure all signals are valid
    valid_signals = {"BUY", "SELL", "HOLD"}
    for idx, row in full_signals.iterrows():
        if row['signal'] not in valid_signals:
            raise ValueError(f"Invalid signal '{row['signal']}' at {row['timestamp']}")
        if not (0.0 <= row['position_size'] <= 1.0):
            raise ValueError(f"Invalid position_size {row['position_size']} at {row['timestamp']}")
        if row['signal'] == "BUY" and row['position_size'] == 0.0:
            raise ValueError(f"BUY signal with zero position_size at {row['timestamp']}")
        if row['signal'] == "SELL" and row['position_size'] != 0.0:
            raise ValueError(f"SELL signal must have position_size 0.0 at {row['timestamp']}")

    return full_signals

# def generate_signals(anchor_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
#     result = []
#     anchor_df = anchor_df.copy()
#     target_df = target_df.copy()
#     # First, forward-fill the prices and SMAs to handle different timeframes
#     btc_price = anchor_df['close_BTC_4H'].ffill()
#     eth_price = anchor_df['close_ETH_4H'].ffill()
#     btc_sma = anchor_df['close_BTC_4H'].rolling(10).mean().ffill()
#     eth_sma = anchor_df['close_ETH_4H'].rolling(10).mean().ffill()

# # Now, create the trend signal. This will give us a valid True/False for every hour.
#     anchor_df['anchor_trend'] = (btc_price > btc_sma) | (eth_price > eth_sma)


#     target_symbols = set(col.split('_')[1] for col in target_df.columns if col.startswith('close_'))

#     for symbol in target_symbols:
#         close_col = f'close_{symbol}_1H'
#         if close_col not in target_df.columns:
#             continue

#         # Merge with anchor trend
#         df = pd.merge(
#             target_df[['timestamp', close_col]].rename(columns={close_col: 'close'}),
#             anchor_df[['timestamp', 'anchor_trend']],
#             on='timestamp',
#             how='left'
#         ).sort_values('timestamp').reset_index(drop=True)

#         # Calculate SMAs for the target
#         df['sma_fast'] = df['close'].rolling(window=8).mean()
#         df['sma_slow'] = df['close'].rolling(window=21).mean()

#         # Initialize trading logic
#         df['signal'] = 'HOLD'
#         df['position_size'] = 0.0
#         in_position = False
#         entry_price = None

#         for i in range(len(df)):
#             price = df['close'].iloc[i]
#             fast = df['sma_fast'].iloc[i]
#             slow = df['sma_slow'].iloc[i]
#             anchor_ok = df['anchor_trend'].iloc[i]

#             if pd.isna(fast) or pd.isna(slow) or pd.isna(anchor_ok):
#                 continue

#             # Buy only if crossover + anchor trend is bullish
#             if (
#                 not in_position and
#                 fast > slow and
#                 df['sma_fast'].iloc[i - 1] <= df['sma_slow'].iloc[i - 1] and
#                 anchor_ok
#             ):
#                 df.at[i, 'signal'] = 'BUY'
#                 df.at[i, 'position_size'] = 0.7
#                 in_position = True
#                 entry_price = price

#             # Sell if profit > 5% or loss > 3%
#             elif in_position and pd.notna(price) and entry_price:
#                 pnl = (price - entry_price) / entry_price
#                 if pnl >= 0.05 or pnl <= -0.03:
#                     df.at[i, 'signal'] = 'SELL'
#                     df.at[i, 'position_size'] = 0.0
#                     in_position = False
#                     entry_price = None
#                 else:
#                     df.at[i, 'signal'] = 'HOLD'
#                     df.at[i, 'position_size'] = 0.7
#             elif in_position:
#                 df.at[i, 'signal'] = 'HOLD'
#                 df.at[i, 'position_size'] = 0.7

#         df['symbol'] = symbol
#         result.append(df[['timestamp', 'symbol', 'signal', 'position_size']])

#     full_signals = pd.concat(result).sort_values(['timestamp', 'symbol']).reset_index(drop=True)

#     # Final constraint check
#     valid_signals = {"BUY", "SELL", "HOLD"}
#     for idx, row in full_signals.iterrows():
#         if row['signal'] not in valid_signals:
#             raise ValueError(f"Invalid signal '{row['signal']}' at {row['timestamp']}")
#         if not (0.0 <= row['position_size'] <= 1.0):
#             raise ValueError(f"Invalid position_size {row['position_size']} at {row['timestamp']}")
#         if row['signal'] == "BUY" and row['position_size'] == 0.0:
#             raise ValueError(f"BUY signal with zero position_size at {row['timestamp']}")
#         if row['signal'] == "SELL" and row['position_size'] != 0.0:
#             raise ValueError(f"SELL signal must have position_size 0.0 at {row['timestamp']}")

#     return full_signals