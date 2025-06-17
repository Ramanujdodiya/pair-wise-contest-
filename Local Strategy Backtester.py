import pandas as pd
import numpy as np
from data_download_manager import CryptoDataManager
import strategy # This imports your strategy.py file

def run_backtest():
    """
    Runs a full backtest simulation of the strategy defined in strategy.py.
    It fetches data, generates signals, simulates trades, and calculates
    performance metrics against the contest's minimum cutoffs.
    """
    print("--- Starting Local Backtest Simulation ---")

    # --- 1. Get Strategy Metadata and Market Data ---
    print("\n[Step 1/5] Fetching market data based on your strategy's coin selection...")
    try:
        metadata = strategy.get_coin_metadata()
        data_manager = CryptoDataManager()
        all_configs = metadata.get('targets', []) + metadata.get('anchors', [])
        
        if not all_configs:
            print("  ERROR: No targets or anchors defined in get_coin_metadata().")
            return

        full_df = data_manager.get_market_data(all_configs)
        
        # Separate anchor and target dataframes as required by the strategy
        anchor_cols = ['timestamp'] + [col for col in full_df.columns if any(f"_{coin['symbol']}_{coin['timeframe']}" in col for coin in metadata.get('anchors', []))]
        target_cols = ['timestamp'] + [col for col in full_df.columns if any(f"_{coin['symbol']}_{coin['timeframe']}" in col for coin in metadata.get('targets', []))]
        anchor_df = full_df[list(dict.fromkeys(anchor_cols))]
        target_df = full_df[list(dict.fromkeys(target_cols))]
        print("  Data fetched successfully.")
    except Exception as e:
        print(f"  ERROR in data fetching: {e}")
        return

    # --- 2. Generate Trading Signals ---
    print("\n[Step 2/5] Generating trading signals from your strategy...")
    try:
        signals_df = strategy.generate_signals(anchor_df, target_df)
        print("  Signals generated successfully.")
    except Exception as e:
        print(f"  ERROR in generate_signals: {e}")
        return
        
    # --- 3. Prepare Data for Simulation ---
    print("\n[Step 3/5] Preparing data for trade simulation...")
    # We need the 1H close price of each target coin to execute trades
    price_cols_to_merge = ['timestamp']
    for coin in metadata.get('targets', []):
        # Assumes target timeframe is 1H for price execution as per contest structure
        price_col = f"close_{coin['symbol']}_1H" 
        if price_col in full_df.columns:
            price_cols_to_merge.append(price_col)
    
    sim_df = pd.merge(signals_df, full_df[price_cols_to_merge], on='timestamp', how='left')
    print("  Simulation data prepared.")

    # --- 4. Run the Trade Simulation ---
    print("\n[Step 4/5] Simulating trades hour-by-hour...")
    initial_capital = 10000.0
    cash = initial_capital
    portfolio_value = initial_capital
    positions = {target['symbol']: {'shares': 0, 'avg_cost': 0} for target in metadata.get('targets', [])}
    portfolio_history = []
    fee_pct = 0.001 # 0.1% trading fee, as per simulator rules

    for i, row in sim_df.iterrows():
        # Update portfolio value at the start of each hour BEFORE any trades
        current_holdings_value = 0
        for pos_symbol, pos_data in positions.items():
            price_col = f"close_{pos_symbol}_1H"
            if price_col in sim_df.columns:
                close_price = sim_df.at[i, price_col]
                if pd.notna(close_price):
                    current_holdings_value += pos_data['shares'] * close_price
        
        portfolio_value = cash + current_holdings_value
        portfolio_history.append(portfolio_value)

        # Execute trades for the current hour
        symbol = row['symbol']
        signal = row['signal']
        pos_size = row['position_size']
        price_col = f"close_{symbol}_1H"
        price = row[price_col] if price_col in row else np.nan

        if pd.isna(price):
            continue
        
        if signal == 'BUY' and cash > 1.0 and pos_size > 0:
            investment_amount = cash * pos_size
            fee = investment_amount * fee_pct
            actual_investment = investment_amount - fee
            shares_to_buy = actual_investment / price
            
            cash -= investment_amount
            old_shares = positions[symbol]['shares']
            old_cost = positions[symbol]['avg_cost']
            new_shares = old_shares + shares_to_buy
            if new_shares > 0:
                positions[symbol]['avg_cost'] = ((old_cost * old_shares) + (price * shares_to_buy)) / new_shares
            positions[symbol]['shares'] += shares_to_buy

        elif signal == 'SELL' and positions[symbol]['shares'] > 0:
            shares_to_sell = positions[symbol]['shares'] # Sells entire position
            sale_value = shares_to_sell * price
            fee = sale_value * fee_pct
            cash += (sale_value - fee)
            positions[symbol] = {'shares': 0, 'avg_cost': 0}

    print("  Simulation complete.")

    # --- 5. Calculate and Display Performance Metrics ---
    print("\n[Step 5/5] Calculating performance metrics...")
    returns = pd.Series(portfolio_history).pct_change().fillna(0)
    
    # Profitability
    final_value = portfolio_history[-1]
    total_return_pct = ((final_value / initial_capital) - 1) * 100
    
    # Sharpe Ratio (annualized for hourly data)
    trading_hours_per_year = 365 * 24
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(trading_hours_per_year) if returns.std() > 0 else 0

    # Max Drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown_pct = abs(drawdown.min() * 100)

    print("\n" + "="*40)
    print("---           BACKTEST RESULTS           ---")
    print("="*40)
    print(f"Initial Capital:       ${initial_capital:,.2f}")
    print(f"Final Portfolio Value:   ${final_value:,.2f}")
    
    # Check against cutoffs
    profit_check = "✅" if total_return_pct >= 5 else "❌"
    sharpe_check = "✅" if sharpe_ratio >= 0.5 else "❌"
    drawdown_check = "✅" if max_drawdown_pct <= 50 else "❌"
    
    print("\n--- Minimum Cutoff Requirements ---")
    print(f"Profitability: {total_return_pct:.2f}% {profit_check}  (Minimum: 5%)")
    print(f"Sharpe Ratio:    {sharpe_ratio:.2f} {sharpe_check}   (Minimum: 0.5)")
    print(f"Max Drawdown:  {max_drawdown_pct:.2f}% {drawdown_check}  (Maximum: 50%)")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_backtest()

