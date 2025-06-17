

## Trade Simulator Errors

### 1. Invalid BUY Signal: position_size is zero
- **Error:** `Invalid BUY signal: position_size is zero for {symbol} at {timestamp}.`
- **Why:** The BUY signal requested a position size of 0, which is not meaningful for a trade.

### 2. Cannot execute BUY: No cash available
- **Error:** `Cannot execute BUY: No cash available for {symbol} at {timestamp}.`
- **Why:** The simulator's cash balance is zero, so no purchase can be made.

### 3. Cannot execute BUY: Allocated cash is zero or negative
- **Error:** `Cannot execute BUY: Allocated cash is zero or negative for {symbol} at {timestamp}.`
- **Why:** The product of cash and position size is zero or negative, possibly due to a very small position size or zero cash.

### 4. Cannot execute BUY: Target amount after fees is zero or negative
- **Error:** `Cannot execute BUY: Target amount after fees is zero or negative for {symbol} at {timestamp}.`
- **Why:** Trading fees are so high relative to the allocated cash that nothing remains to buy assets.

### 5. Insufficient funds
- **Error:** `Insufficient funds: Need ${total_cost:.5f} but only ${self.cash:.5f} available`
- **Why:** The total cost of the intended purchase exceeds the available cash.

### 6. Cannot execute BUY: Price is None or not finite
- **Error:** `Cannot execute BUY: Price is None or not finite for {symbol} at {timestamp}.`
- **Why:** The price data is missing or not a valid number (NaN, inf, etc.).

### 7. Cannot execute BUY: Price is negative
- **Error:** `Cannot execute BUY: Price is negative for {symbol} at {timestamp}.`
- **Why:** The price data is negative, which is invalid for trading.

### 8. Cannot execute BUY: Shares to buy is zero or negative
- **Error:** `Cannot execute BUY: Shares to buy is zero or negative for {symbol} at {timestamp}.`
- **Why:** After all calculations, the number of shares to buy is zero or negative, possibly due to high price, low cash, or high fees.

### 9. Division by zero : total_shares=0
- **Error:** `Division by zero: Attempted to update position for {symbol} with zero total shares.`
- **Why:** Both old and new shares are zero, so updating the position would require dividing by zero.

### 10. Cannot sell: No position found
- **Error:** `Cannot sell {symbol}: No position found`
- **Why:** A SELL signal was received for a symbol with no holdings.

### 11. Cannot execute SELL: Price is None or not finite
- **Error:** `Cannot execute SELL: Price is None or not finite for {symbol} at {timestamp}.`
- **Why:** The price data is missing or not a valid number (NaN, inf, etc.).

### 12. Cannot execute SELL: Price is negative
- **Error:** `Cannot execute SELL: Price is negative for {symbol} at {timestamp}.`
- **Why:** The price data is negative, which is invalid for trading.

### 13. Cannot execute SELL: Shares to sell is zero or negative
- **Error:** `Cannot execute SELL: Shares to sell is zero or negative for {symbol} at {timestamp}.`
- **Why:** The calculated number of shares to sell is zero or negative, possibly due to a very small position size or no holdings.

### 14. Cannot execute SELL: Gross proceeds is zero or negative
- **Error:** `Cannot execute SELL: Gross proceeds is zero or negative for {symbol} at {timestamp}.`
- **Why:** The product of shares to sell and price is zero or negative, possibly due to zero price or shares.

### 15. Cannot execute SELL: Net proceeds after fees is zero or negative
- **Error:** `Cannot execute SELL: Net proceeds after fees is zero or negative for {symbol} at {timestamp}.`
- **Why:** Trading fees are so high relative to the gross proceeds that nothing remains after the sale.

### 16. Division by zero: Average cost is zero for SELL
- **Error:** `Division by zero: Average cost is zero for {symbol} at {timestamp} during SELL.`
- **Why:** The average cost basis for the position is zero, which is not valid for a real trade and would cause a division by zero in P&L calculation.

### 17. No valid price data found for symbol
- **Error:** `No valid price data found for {symbol}`
- **Why:** The price lookup failed for all timeframes and fallback options.

### 18. Missing columns in signals/candles
- **Error:** `Missing columns in signals: {missing_cols}` or `Missing 'timestamp' column in candles`
- **Why:** The input DataFrames are missing required columns for simulation.

### 19. Invalid signals
- **Error:** `Invalid signals: {invalid_signals}. Valid: {valid_signals}`
- **Why:** The signals DataFrame contains values outside the allowed set (BUY, SELL, HOLD).

### 20. Invalid position_size in signals
- **Error:** `position_size must be between 0.0 and 1.0 (0% to 100%)`
- **Why:** The signals DataFrame contains position sizes outside the allowed range.

---

## General Processin Errors

### 1. Strategy validation failed
- **Error:** `Strategy validation failed: {error_details} [submission_id: {submission_id}]`
- **Why:** The strategy did not pass validation checks (syntax, required functions, etc.).

### 2. generate_signals must return a pandas DataFrame
- **Error:** `generate_signals must return a pandas DataFrame [submission_id: {submission_id}]`
- **Why:** The strategy's `generate_signals` function did not return a DataFrame.

### 3. signals DataFrame missing required columns
- **Error:** `signals DataFrame missing required columns: {missing_columns} [submission_id: {submission_id}]`
- **Why:** The signals DataFrame is missing one or more of the required columns: timestamp, symbol, signal, position_size.

### 4. position_size must be between 0.0 and 1.0
- **Error:** `position_size must be between 0.0 and 1.0 [submission_id: {submission_id}]`
- **Why:** The signals DataFrame contains position_size values outside the allowed range.

### 5. Invalid signal values
- **Error:** `Invalid signal values: {invalid_signals}. Valid: {valid_signals} [submission_id: {submission_id}]`
- **Why:** The signals DataFrame contains values for 'signal' that are not in the allowed set (BUY, SELL, HOLD).

### 6. Strategy failed to generate signals
- **Error:** `Strategy failed to generate signals: {str(e)} [submission_id: {submission_id}]`
- **Why:** An exception occurred during signal generation (e.g., runtime error in user code).

### 7. Trade simulation failed
- **Error:** `Trade simulation failed: {str(e)} [submission_id: {submission_id}]`
- **Why:** The trade simulation raised an exception (e.g., from the simulator, invalid data, or logic error).

### 8. No trades were executed during simulation
- **Error:** `No trades were executed during simulation [submission_id: {submission_id}]`
- **Why:** The simulation completed but did not execute any trades (possibly due to all HOLD signals or invalid signals).

### 9. Trade log missing required columns
- **Error:** `Trade log missing required columns: {missing_columns} [submission_id: {submission_id}]`
- **Why:** The trade log DataFrame is missing one or more of the required columns: timestamp, action, symbol, cash, portfolio_value.

### 10. Invalid cash/portfolio_value data detected (NaN values)
- **Error:** `Invalid cash data detected (NaN values) [submission_id: {submission_id}]` or `Invalid portfolio_value data detected (NaN values) [submission_id: {submission_id}]`
- **Why:** The trade log contains NaN values in critical columns, indicating a data integrity issue.

### 11. Metrics calculation failed
- **Error:** `Metrics calculation failed: {str(e)} [submission_id: {submission_id}]`
- **Why:** An exception occurred during the calculation of performance metrics.

---

