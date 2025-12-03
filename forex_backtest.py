"""
Forex Backtesting Engine
Command-line tool for running multiple strategies across different years and currency pairs
Exports results to CSV for comparison and analysis

Usage:
python3 backtest_engine.py --years 2021,2022,2023 --pairs EUR/USD,GBP/USD --strategies ma,meanrev --output results.csv
"""

import argparse
from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import sys

class BacktestEngine:
    def __init__(self, initial_capital=10000, max_positions=3, stop_loss_pct=3):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.position_size_per_trade = initial_capital / max_positions
        
        # Available currency pairs
        self.all_pairs = {
            'EUR/USD': {'symbol': 'EUR', 'currency': 'USD'},
            'USD/CAD': {'symbol': 'USD', 'currency': 'CAD'},
            'GBP/USD': {'symbol': 'GBP', 'currency': 'USD'},
            'AUD/USD': {'symbol': 'AUD', 'currency': 'USD'},
            'USD/JPY': {'symbol': 'USD', 'currency': 'JPY'},
            'USD/CHF': {'symbol': 'USD', 'currency': 'CHF'},
            'NZD/USD': {'symbol': 'NZD', 'currency': 'USD'},
            'EUR/GBP': {'symbol': 'EUR', 'currency': 'GBP'},
            'EUR/JPY': {'symbol': 'EUR', 'currency': 'JPY'},
            'GBP/JPY': {'symbol': 'GBP', 'currency': 'JPY'},
            'USD/MXN': {'symbol': 'USD', 'currency': 'MXN'},
            'USD/ZAR': {'symbol': 'USD', 'currency': 'ZAR'},
            'AUD/CAD': {'symbol': 'AUD', 'currency': 'CAD'},
            'NZD/CAD': {'symbol': 'NZD', 'currency': 'CAD'},
            'EUR/CHF': {'symbol': 'EUR', 'currency': 'CHF'},
            'GBP/CHF': {'symbol': 'GBP', 'currency': 'CHF'}
        }
        
        self.ib = None
        
    def connect_tws(self, port=7497):
        """Connect to TWS"""
        self.ib = IB()
        try:
            self.ib.connect('127.0.0.1', port, clientId=1)
            print(f"✓ Connected to TWS on port {port}")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from TWS"""
        if self.ib:
            self.ib.disconnect()
    
    def fetch_data(self, pair_name, year):
        """Fetch hourly data for a specific year"""
        if pair_name not in self.all_pairs:
            return None
        
        pair_info = self.all_pairs[pair_name]
        
        try:
            contract = Forex(pair_info['symbol'] + pair_info['currency'])
            self.ib.qualifyContracts(contract)
            
            # Set end date to November 1st of the following year to get full year
            end_date = f"{year+1}1101 23:59:59"
            
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr='1 Y',
                barSizeSetting='1 hour',
                whatToShow='MIDPOINT',
                useRTH=False,
                formatDate=1
            )
            
            df = util.df(bars)
            return df
            
        except Exception as e:
            print(f"  ✗ {pair_name} ({year}): Error - {e}")
            return None
    
    # ==================== STRATEGY IMPLEMENTATIONS ====================
    
    def calculate_ma_crossover_signals(self, df, fast_period=50, slow_period=200):
        """Moving Average Crossover Strategy"""
        df['MA_fast'] = df['close'].rolling(window=fast_period).mean()
        df['MA_slow'] = df['close'].rolling(window=slow_period).mean()
        
        df['signal'] = 0
        df.loc[df['MA_fast'] > df['MA_slow'], 'signal'] = 1   # Long
        df.loc[df['MA_fast'] < df['MA_slow'], 'signal'] = -1  # Short
        
        df['position_change'] = df['signal'].diff()
        
        return df
    
    def calculate_rsi(self, series, period=14):
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_mean_reversion_signals(self, df, rsi_period=14, bb_period=20, bb_std=2):
        """Mean Reversion Strategy with RSI + Bollinger Bands"""
        # RSI
        df['RSI'] = self.calculate_rsi(df['close'], period=rsi_period)
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=bb_period).mean()
        rolling_std = df['close'].rolling(window=bb_period).std()
        df['BB_upper'] = df['BB_middle'] + (rolling_std * bb_std)
        df['BB_lower'] = df['BB_middle'] - (rolling_std * bb_std)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle'] * 100
        
        # Signals
        df['signal'] = 0
        df['near_lower_bb'] = (df['close'] <= df['BB_lower'] * 1.001)
        df['near_upper_bb'] = (df['close'] >= df['BB_upper'] * 0.999)
        df['at_middle_bb'] = (df['close'] >= df['BB_middle'] * 0.998) & (df['close'] <= df['BB_middle'] * 1.002)
        
        # Entry signals
        df.loc[(df['RSI'] < 30) & df['near_lower_bb'] & (df['BB_width'] > 0.5), 'signal'] = 1
        df.loc[(df['RSI'] > 70) & df['near_upper_bb'] & (df['BB_width'] > 0.5), 'signal'] = -1
        
        df['position_change'] = df['signal'].diff()
        
        return df
    
    # ==================== BACKTEST EXECUTION ====================
    
    def run_strategy_backtest(self, df, strategy_name, pair_name):
        """Run backtest for a single strategy on a single pair"""
        capital = self.initial_capital
        positions = {}
        trades = []
        
        # Determine starting index based on strategy
        if strategy_name == 'ma':
            start_idx = 200  # Need 200 periods for slow MA
        elif strategy_name == 'meanrev':
            start_idx = 20   # Need 20 periods for BB
        else:
            start_idx = 200
        
        for i in range(start_idx, len(df)):
            current_row = df.iloc[i]
            date = current_row['date']
            price = current_row['close']
            
            # Check stop loss for existing position
            if pair_name in positions:
                pos = positions[pair_name]
                stop_hit = False
                
                if pos['type'] == 'long':
                    stop_price = pos['entry_price'] * (1 - self.stop_loss_pct / 100)
                    if price <= stop_price:
                        stop_hit = True
                elif pos['type'] == 'short':
                    stop_price = pos['entry_price'] * (1 + self.stop_loss_pct / 100)
                    if price >= stop_price:
                        stop_hit = True
                
                if stop_hit:
                    pnl = self.close_position(positions, pair_name, price, date, 'Stop Loss', trades)
                    capital += pnl
            
            # Check for mean reversion exit (if applicable)
            if strategy_name == 'meanrev' and pair_name in positions:
                if current_row['at_middle_bb']:
                    pnl = self.close_position(positions, pair_name, price, date, 'Mean Reversion', trades)
                    capital += pnl
            
            # Check for new signals
            if pair_name not in positions and len(positions) < self.max_positions:
                if current_row['position_change'] == 2:  # Signal changed to 1 (golden cross or oversold)
                    positions[pair_name] = {
                        'type': 'long',
                        'entry_price': price,
                        'entry_date': date,
                        'size': 10000
                    }
                    
                elif current_row['position_change'] == -2:  # Signal changed to -1 (death cross or overbought)
                    positions[pair_name] = {
                        'type': 'short',
                        'entry_price': price,
                        'entry_date': date,
                        'size': 10000
                    }
            
            # Check for signal reversals
            elif pair_name in positions:
                if current_row['position_change'] == 2 and positions[pair_name]['type'] == 'short':
                    pnl = self.close_position(positions, pair_name, price, date, 'Signal Reversal', trades)
                    capital += pnl
                    # Open new long position
                    positions[pair_name] = {
                        'type': 'long',
                        'entry_price': price,
                        'entry_date': date,
                        'size': 10000
                    }
                    
                elif current_row['position_change'] == -2 and positions[pair_name]['type'] == 'long':
                    pnl = self.close_position(positions, pair_name, price, date, 'Signal Reversal', trades)
                    capital += pnl
                    # Open new short position
                    positions[pair_name] = {
                        'type': 'short',
                        'entry_price': price,
                        'entry_date': date,
                        'size': 10000
                    }
        
        # Close any remaining positions
        if pair_name in positions:
            final_price = df.iloc[-1]['close']
            final_date = df.iloc[-1]['date']
            pnl = self.close_position(positions, pair_name, final_price, final_date, 'End of Period', trades)
            capital += pnl
        
        # Calculate metrics
        return self.calculate_metrics(trades, capital)
    
    def close_position(self, positions, pair_name, price, date, reason, trades):
        """Close a position and record the trade"""
        pos = positions[pair_name]
        
        if pos['type'] == 'long':
            pnl = (price - pos['entry_price']) * pos['size']
        else:  # short
            pnl = (pos['entry_price'] - price) * pos['size']
        
        trades.append({
            'pair': pair_name,
            'entry_date': pos['entry_date'],
            'exit_date': date,
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'type': pos['type'],
            'pnl': pnl,
            'reason': reason
        })
        
        del positions[pair_name]
        return pnl
    
    def calculate_metrics(self, trades, final_capital):
        """Calculate performance metrics"""
        if not trades:
            return {
                'final_capital': final_capital,
                'total_return': 0,
                'total_return_pct': 0,
                'num_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0
            }
        
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        total_wins = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
        total_losses = sum([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(total_wins / total_losses) if total_losses != 0 else 0
        
        return {
            'final_capital': final_capital,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'num_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(trades) * 100) if trades else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': 0  # TODO: Calculate proper drawdown
        }
    
    # ==================== MAIN EXECUTION ====================
    
    def run_backtest_suite(self, years, pairs, strategies):
        """Run backtests for all combinations of years, pairs, and strategies"""
        results = []
        
        total_tests = len(years) * len(pairs) * len(strategies)
        current_test = 0
        
        print(f"\n{'='*70}")
        print(f"BACKTESTING ENGINE - Running {total_tests} tests")
        print(f"{'='*70}\n")
        
        for year in years:
            print(f"\n--- YEAR: {year} ---")
            
            # Fetch data for all pairs for this year
            year_data = {}
            for pair in pairs:
                print(f"  Fetching {pair}...", end=' ')
                df = self.fetch_data(pair, year)
                if df is not None and len(df) > 200:
                    year_data[pair] = df
                    print(f"✓ ({len(df)} bars)")
                else:
                    print(f"✗ Failed")
            
            if not year_data:
                print(f"  ✗ No data available for {year}")
                continue
            
            # Run each strategy on each pair
            for strategy in strategies:
                print(f"\n  Strategy: {strategy.upper()}")
                
                for pair in year_data.keys():
                    current_test += 1
                    print(f"    [{current_test}/{total_tests}] {pair}...", end=' ')
                    
                    # Apply strategy signals
                    df = year_data[pair].copy()
                    if strategy == 'ma':
                        df = self.calculate_ma_crossover_signals(df)
                    elif strategy == 'meanrev':
                        df = self.calculate_mean_reversion_signals(df)
                    
                    # Run backtest
                    metrics = self.run_strategy_backtest(df, strategy, pair)
                    
                    # Store results
                    result = {
                        'year': year,
                        'pair': pair,
                        'strategy': strategy,
                        **metrics
                    }
                    results.append(result)
                    
                    # Print quick summary
                    print(f"Return: {metrics['total_return_pct']:+.2f}% | Trades: {metrics['num_trades']}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Forex Backtesting Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test MA crossover on EUR/USD for 2022
  python3 backtest_engine.py --years 2022 --pairs EUR/USD --strategies ma
  
  # Test both strategies on multiple pairs across multiple years
  python3 backtest_engine.py --years 2021,2022,2023 --pairs EUR/USD,GBP/USD,USD/JPY --strategies ma,meanrev
  
  # Test all available pairs
  python3 backtest_engine.py --years 2022 --pairs all --strategies ma,meanrev --output results.csv
        """
    )
    
    parser.add_argument('--years', required=True, 
                       help='Comma-separated years (e.g., 2021,2022,2023)')
    parser.add_argument('--pairs', required=True,
                       help='Comma-separated currency pairs (e.g., EUR/USD,GBP/USD) or "all"')
    parser.add_argument('--strategies', required=True,
                       help='Comma-separated strategies: ma (moving average) and/or meanrev (mean reversion)')
    parser.add_argument('--output', default='backtest_results.csv',
                       help='Output CSV filename (default: backtest_results.csv)')
    parser.add_argument('--capital', type=float, default=10000,
                       help='Initial capital (default: 10000)')
    parser.add_argument('--max-positions', type=int, default=3,
                       help='Max concurrent positions (default: 3)')
    parser.add_argument('--stop-loss', type=float, default=3,
                       help='Stop loss percentage (default: 3)')
    
    args = parser.parse_args()
    
    # Parse arguments
    years = [int(y.strip()) for y in args.years.split(',')]
    
    engine = BacktestEngine(
        initial_capital=args.capital,
        max_positions=args.max_positions,
        stop_loss_pct=args.stop_loss
    )
    
    if args.pairs.lower() == 'all':
        pairs = list(engine.all_pairs.keys())
    else:
        pairs = [p.strip() for p in args.pairs.split(',')]
    
    strategies = [s.strip().lower() for s in args.strategies.split(',')]
    
    # Validate strategies
    valid_strategies = {'ma', 'meanrev'}
    for strat in strategies:
        if strat not in valid_strategies:
            print(f"Error: Invalid strategy '{strat}'. Must be 'ma' or 'meanrev'")
            sys.exit(1)
    
    # Connect to TWS
    if not engine.connect_tws():
        sys.exit(1)
    
    try:
        # Run backtests
        results = engine.run_backtest_suite(years, pairs, strategies)
        
        # Export to CSV
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(args.output, index=False)
            print(f"\n{'='*70}")
            print(f"✓ Results exported to {args.output}")
            print(f"{'='*70}\n")
            
            # Print summary table
            print("SUMMARY BY STRATEGY:")
            summary = df_results.groupby('strategy').agg({
                'total_return_pct': 'mean',
                'num_trades': 'sum',
                'win_rate': 'mean'
            }).round(2)
            print(summary)
            
            print("\nTOP 5 PERFORMERS:")
            top5 = df_results.nlargest(5, 'total_return_pct')[['year', 'pair', 'strategy', 'total_return_pct', 'num_trades']]
            print(top5.to_string(index=False))
            
        else:
            print("\n✗ No results generated")
    
    finally:
        engine.disconnect()
        print("\n✓ Disconnected from TWS")


if __name__ == "__main__":
    main()