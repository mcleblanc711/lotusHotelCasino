"""
Mean Reversion Portfolio Backtesting Script
Strategy: RSI + Bollinger Bands Mean Reversion
Assets: 12 Forex Pairs (Majors, Crosses, Emerging Markets)
Period: November 1, 2021 to November 1, 2022
Stop Loss: 3%
Entry: RSI < 30 + Price at lower BB (BUY) OR RSI > 70 + Price at upper BB (SHORT)
Exit: Price reaches middle BB OR opposite signal OR stop loss
Position Sizing: Max 3 positions, ~$3,333 per trade
"""

from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MeanReversionBacktester:
    def __init__(self, initial_capital=10000, max_positions=3):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size_per_trade = initial_capital / max_positions
        self.capital = initial_capital
        self.positions = {}  # {asset_name: {'type': 'long'/'short', 'entry_price': price, 'size': units}}
        self.trades = []
        self.equity_curve = []
        self.ib = None
        
        # 12 forex pairs - majors, crosses, and emerging markets
        self.assets = {
            # Majors
            'EUR/USD': {'symbol': 'EUR', 'currency': 'USD'},
            'USD/CAD': {'symbol': 'USD', 'currency': 'CAD'},
            'GBP/USD': {'symbol': 'GBP', 'currency': 'USD'},
            'AUD/USD': {'symbol': 'AUD', 'currency': 'USD'},
            'USD/JPY': {'symbol': 'USD', 'currency': 'JPY'},
            'USD/CHF': {'symbol': 'USD', 'currency': 'CHF'},
            'NZD/USD': {'symbol': 'NZD', 'currency': 'USD'},
            # Crosses
            'EUR/GBP': {'symbol': 'EUR', 'currency': 'GBP'},
            'EUR/JPY': {'symbol': 'EUR', 'currency': 'JPY'},
            'GBP/JPY': {'symbol': 'GBP', 'currency': 'JPY'},
            # Emerging Markets
            'USD/MXN': {'symbol': 'USD', 'currency': 'MXN'},
            'USD/ZAR': {'symbol': 'USD', 'currency': 'ZAR'}
        }
        
    def connect_tws(self, port=7497):
        """Connect to TWS paper trading"""
        self.ib = IB()
        try:
            self.ib.connect('127.0.0.1', port, clientId=1)
            print(f"Connected to TWS on port {port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def fetch_historical_data(self, asset_name):
        """Fetch hourly data for Nov 2021 - Nov 2022"""
        asset_info = self.assets[asset_name]
        
        try:
            contract = Forex(asset_info['symbol'] + asset_info['currency'])
            self.ib.qualifyContracts(contract)
            
            end_date = '20221101 23:59:59'  # November 1, 2022
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
            print(f"  {asset_name}: Fetched {len(df)} bars")
            return df
            
        except Exception as e:
            print(f"  {asset_name}: Error fetching data - {e}")
            return None
    
    def calculate_rsi(self, df, period=14):
        """Calculate RSI (Relative Strength Index)"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        df['BB_middle'] = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        df['BB_upper'] = df['BB_middle'] + (rolling_std * std_dev)
        df['BB_lower'] = df['BB_middle'] - (rolling_std * std_dev)
        
        return df
    
    def calculate_signals(self, df):
        """Calculate mean reversion signals using RSI + Bollinger Bands"""
        # Calculate indicators
        df['RSI'] = self.calculate_rsi(df, period=14)
        df = self.calculate_bollinger_bands(df, period=20, std_dev=2)
        
        # Calculate BB width for filtering (avoid trading in super tight ranges)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle'] * 100
        
        # Generate signals
        df['signal'] = 0
        
        # BUY signal: RSI < 30 (oversold) AND price near/below lower BB
        # We'll say "near" means within 0.1% of lower BB
        df['near_lower_bb'] = (df['close'] <= df['BB_lower'] * 1.001)
        df.loc[(df['RSI'] < 30) & df['near_lower_bb'] & (df['BB_width'] > 0.5), 'signal'] = 1
        
        # SHORT signal: RSI > 70 (overbought) AND price near/above upper BB
        df['near_upper_bb'] = (df['close'] >= df['BB_upper'] * 0.999)
        df.loc[(df['RSI'] > 70) & df['near_upper_bb'] & (df['BB_width'] > 0.5), 'signal'] = -1
        
        # Exit signals: price crosses middle BB (mean reversion achieved)
        df['at_middle_bb'] = (df['close'] >= df['BB_middle'] * 0.998) & (df['close'] <= df['BB_middle'] * 1.002)
        
        df['position_change'] = df['signal'].diff()
        
        return df
    
    def check_stop_loss(self, asset_name, current_price):
        """Check if 3% stop loss is hit"""
        if asset_name not in self.positions:
            return False
            
        position = self.positions[asset_name]
        
        if position['type'] == 'long':
            stop_price = position['entry_price'] * 0.97
            if current_price <= stop_price:
                return True
        elif position['type'] == 'short':
            stop_price = position['entry_price'] * 1.03
            if current_price >= stop_price:
                return True
        return False
    
    def check_mean_reversion_exit(self, asset_name, current_row):
        """Check if price has reverted to middle BB"""
        if asset_name not in self.positions:
            return False
        
        position = self.positions[asset_name]
        
        # Exit long if price reaches middle BB
        if position['type'] == 'long' and current_row['at_middle_bb']:
            return True
        # Exit short if price reaches middle BB
        elif position['type'] == 'short' and current_row['at_middle_bb']:
            return True
        
        return False
    
    def calculate_position_size(self, price, asset_name):
        """Calculate position size - mini lots for all forex pairs"""
        return 10000  # Mini lot (10,000 units)
    
    def execute_trade(self, date, asset_name, price, action, reason):
        """Execute a trade"""
        trade_record = {
            'date': date,
            'asset': asset_name,
            'action': action,
            'price': price,
            'reason': reason
        }
        
        if action == 'BUY':
            if len(self.positions) >= self.max_positions:
                return  # Max positions reached
            
            size = self.calculate_position_size(price, asset_name)
            self.positions[asset_name] = {
                'type': 'long',
                'entry_price': price,
                'size': size
            }
            trade_record['size'] = size
            print(f"{date} | {asset_name}: BUY at {price:.5f} - {reason}")
            
        elif action == 'SELL':
            if asset_name in self.positions and self.positions[asset_name]['type'] == 'long':
                position = self.positions[asset_name]
                pnl = (price - position['entry_price']) * position['size']
                self.capital += pnl
                trade_record['pnl'] = pnl
                trade_record['return_pct'] = (price / position['entry_price'] - 1) * 100
                print(f"{date} | {asset_name}: SELL at {price:.5f} - {reason} | P&L: ${pnl:.2f} ({trade_record['return_pct']:.2f}%)")
                del self.positions[asset_name]
            
        elif action == 'SHORT':
            if len(self.positions) >= self.max_positions:
                return
            
            size = self.calculate_position_size(price, asset_name)
            self.positions[asset_name] = {
                'type': 'short',
                'entry_price': price,
                'size': size
            }
            trade_record['size'] = size
            print(f"{date} | {asset_name}: SHORT at {price:.5f} - {reason}")
            
        elif action == 'COVER':
            if asset_name in self.positions and self.positions[asset_name]['type'] == 'short':
                position = self.positions[asset_name]
                pnl = (position['entry_price'] - price) * position['size']
                self.capital += pnl
                trade_record['pnl'] = pnl
                trade_record['return_pct'] = (position['entry_price'] / price - 1) * 100
                print(f"{date} | {asset_name}: COVER at {price:.5f} - {reason} | P&L: ${pnl:.2f} ({trade_record['return_pct']:.2f}%)")
                del self.positions[asset_name]
        
        self.trades.append(trade_record)
    
    def run_backtest(self, asset_data):
        """Run mean reversion backtest across all assets"""
        print("\n" + "="*70)
        print("STARTING MEAN REVERSION BACKTEST")
        print("="*70)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Max Positions: {self.max_positions}")
        print(f"Position Size: ${self.position_size_per_trade:,.2f} per trade")
        print(f"Assets: {len(asset_data)} pairs")
        print(f"Strategy: RSI (14) + Bollinger Bands (20, 2σ)")
        print(f"Entry: RSI<30 at lower BB (buy) | RSI>70 at upper BB (short)")
        print(f"Exit: Price reverts to middle BB OR stop loss (3%)")
        print("="*70 + "\n")
        
        # Find common date range across all assets
        min_length = min(len(df) for df in asset_data.values())
        
        # Process bar by bar across all assets
        for i in range(20, min_length):  # Start after 20 periods for BB calculation
            current_date = None
            
            # First, check stop losses and mean reversion exits for all existing positions
            for asset_name in list(self.positions.keys()):
                df = asset_data[asset_name]
                if i < len(df):
                    current_row = df.iloc[i]
                    current_date = current_row['date']
                    price = current_row['close']
                    
                    # Check stop loss
                    if self.check_stop_loss(asset_name, price):
                        if self.positions[asset_name]['type'] == 'long':
                            self.execute_trade(current_date, asset_name, price, 'SELL', 'Stop Loss')
                        else:
                            self.execute_trade(current_date, asset_name, price, 'COVER', 'Stop Loss')
                    
                    # Check mean reversion exit
                    elif self.check_mean_reversion_exit(asset_name, current_row):
                        if self.positions[asset_name]['type'] == 'long':
                            self.execute_trade(current_date, asset_name, price, 'SELL', 'Mean Reversion - Middle BB')
                        else:
                            self.execute_trade(current_date, asset_name, price, 'COVER', 'Mean Reversion - Middle BB')
            
            # Then check for new signals across all assets
            for asset_name, df in asset_data.items():
                if i >= len(df):
                    continue
                    
                current_row = df.iloc[i]
                current_date = current_row['date']
                price = current_row['close']
                
                # Skip if we already have a position in this asset
                if asset_name in self.positions:
                    continue
                
                # Check for entry signals
                if current_row['signal'] == 1:  # Oversold - BUY signal
                    self.execute_trade(current_date, asset_name, price, 'BUY', 'RSI Oversold at Lower BB')
                        
                elif current_row['signal'] == -1:  # Overbought - SHORT signal
                    self.execute_trade(current_date, asset_name, price, 'SHORT', 'RSI Overbought at Upper BB')
            
            # Track equity
            if current_date:
                self.equity_curve.append({'date': current_date, 'equity': self.capital})
        
        # Close all open positions at end
        for asset_name in list(self.positions.keys()):
            df = asset_data[asset_name]
            final_row = df.iloc[-1]
            final_price = final_row['close']
            final_date = final_row['date']
            
            if self.positions[asset_name]['type'] == 'long':
                self.execute_trade(final_date, asset_name, final_price, 'SELL', 'End of Period')
            else:
                self.execute_trade(final_date, asset_name, final_price, 'COVER', 'End of Period')
    
    def print_results(self):
        """Print comprehensive results"""
        print("\n" + "="*70)
        print("MEAN REVERSION BACKTEST RESULTS")
        print("="*70)
        
        total_return = self.capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        print(f"\nFinal Capital: ${self.capital:,.2f}")
        print(f"Total Return: ${total_return:,.2f} ({total_return_pct:.2f}%)")
        
        # Overall trade statistics
        completed_trades = [t for t in self.trades if 'pnl' in t]
        if completed_trades:
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            losing_trades = [t for t in completed_trades if t['pnl'] <= 0]
            
            print(f"\nTotal Completed Trades: {len(completed_trades)}")
            print(f"Winning Trades: {len(winning_trades)}")
            print(f"Losing Trades: {len(losing_trades)}")
            print(f"Win Rate: {(len(winning_trades) / len(completed_trades) * 100):.1f}%")
            
            if winning_trades:
                avg_win = np.mean([t['pnl'] for t in winning_trades])
                print(f"Average Win: ${avg_win:.2f}")
            if losing_trades:
                avg_loss = np.mean([t['pnl'] for t in losing_trades])
                print(f"Average Loss: ${avg_loss:.2f}")
            
            if winning_trades and losing_trades:
                profit_factor = abs(sum([t['pnl'] for t in winning_trades]) / sum([t['pnl'] for t in losing_trades]))
                print(f"Profit Factor: {profit_factor:.2f}")
            
            # Performance by asset
            print("\n" + "-"*70)
            print("PERFORMANCE BY ASSET")
            print("-"*70)
            for asset_name in self.assets.keys():
                asset_trades = [t for t in completed_trades if t['asset'] == asset_name]
                if asset_trades:
                    asset_pnl = sum([t['pnl'] for t in asset_trades])
                    asset_wins = len([t for t in asset_trades if t['pnl'] > 0])
                    asset_win_rate = (asset_wins / len(asset_trades) * 100) if asset_trades else 0
                    print(f"{asset_name:12} | Trades: {len(asset_trades):3} | P&L: ${asset_pnl:9.2f} | Win Rate: {asset_win_rate:5.1f}%")
                else:
                    print(f"{asset_name:12} | No trades")
            
            # Max drawdown
            if self.equity_curve:
                equity_values = [e['equity'] for e in self.equity_curve]
                peak = equity_values[0]
                max_dd = 0
                for equity in equity_values:
                    if equity > peak:
                        peak = equity
                    dd = (peak - equity) / peak * 100
                    if dd > max_dd:
                        max_dd = dd
                print(f"\nMax Drawdown: {max_dd:.2f}%")
        else:
            print("\nNo completed trades during the backtest period.")
        
        print("="*70)
    
    def disconnect(self):
        """Disconnect from TWS"""
        if self.ib:
            self.ib.disconnect()
            print("\nDisconnected from TWS")


def main():
    backtester = MeanReversionBacktester(initial_capital=10000, max_positions=3)
    
    if not backtester.connect_tws(port=7497):
        return
    
    try:
        # Fetch data for all assets
        print("\nFetching historical data for all 12 forex pairs...")
        asset_data = {}
        
        for asset_name in backtester.assets.keys():
            df = backtester.fetch_historical_data(asset_name)
            if df is not None and len(df) > 20:
                df = backtester.calculate_signals(df)
                asset_data[asset_name] = df
        
        if not asset_data:
            print("\nNo data available. Check TWS connection and data subscriptions.")
            return
        
        print(f"\nSuccessfully loaded {len(asset_data)} assets")
        
        # Run backtest
        backtester.run_backtest(asset_data)
        
        # Print results
        backtester.print_results()
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        backtester.disconnect()


if __name__ == "__main__":
    main()