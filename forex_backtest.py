"""
Multi-Asset Portfolio Backtesting Script - PURE CROSSOVER MODE
Assets: EUR/USD, USD/CAD, GBP/USD, AUD/USD, USD/JPY
Strategy: 50/200 Hour MA Crossover - NO FILTERS
Period: November 1, 2021 to November 1, 2022
Stop Loss: 3%
NO ADX FILTER - NO MA SPREAD FILTER - FULL YOLO
Position Sizing: Max 3 positions, ~$3,333 per trade
"""

from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PortfolioBacktester:
    def __init__(self, initial_capital=10000, max_positions=3):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size_per_trade = initial_capital / max_positions
        self.capital = initial_capital
        self.positions = {}  # {asset_name: {'type': 'long'/'short', 'entry_price': price, 'size': units}}
        self.trades = []
        self.equity_curve = []
        self.ib = None
        
        # Asset definitions - forex pairs only
        self.assets = {
            'EUR/USD': {'type': 'forex', 'symbol': 'EUR', 'currency': 'USD', 'pip_value': 1},
            'USD/CAD': {'type': 'forex', 'symbol': 'USD', 'currency': 'CAD', 'pip_value': 1},
            'GBP/USD': {'type': 'forex', 'symbol': 'GBP', 'currency': 'USD', 'pip_value': 1},
            'AUD/USD': {'type': 'forex', 'symbol': 'AUD', 'currency': 'USD', 'pip_value': 1},
            'USD/JPY': {'type': 'forex', 'symbol': 'USD', 'currency': 'JPY', 'pip_value': 1}
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
    
    def calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = abs(df['high'] - df['close'].shift(1))
        df['L-PC'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        df['DMplus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                                df['high'] - df['high'].shift(1), 0)
        df['DMplus'] = np.where(df['DMplus'] < 0, 0, df['DMplus'])
        
        df['DMminus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                                 df['low'].shift(1) - df['low'], 0)
        df['DMminus'] = np.where(df['DMminus'] < 0, 0, df['DMminus'])
        
        df['TR_smooth'] = df['TR'].rolling(window=period).sum()
        df['DMplus_smooth'] = df['DMplus'].rolling(window=period).sum()
        df['DMminus_smooth'] = df['DMminus'].rolling(window=period).sum()
        
        df['DIplus'] = 100 * (df['DMplus_smooth'] / df['TR_smooth'])
        df['DIminus'] = 100 * (df['DMminus_smooth'] / df['TR_smooth'])
        
        df['DX'] = 100 * abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus'])
        df['ADX'] = df['DX'].rolling(window=period).mean()
        
        df.drop(['H-L', 'H-PC', 'L-PC', 'TR', 'DMplus', 'DMminus', 
                'TR_smooth', 'DMplus_smooth', 'DMminus_smooth', 
                'DIplus', 'DIminus', 'DX'], axis=1, inplace=True)
        
        return df
    
    def calculate_signals(self, df):
        """Calculate signals - PURE MA CROSSOVER, NO FILTERS"""
        df['MA_50'] = df['close'].rolling(window=50).mean()
        df['MA_200'] = df['close'].rolling(window=200).mean()
        
        # Pure signals - no ADX, no spread filter, just crossovers
        df['signal'] = 0
        df.loc[df['MA_50'] > df['MA_200'], 'signal'] = 1   # Golden cross - LONG
        df.loc[df['MA_50'] < df['MA_200'], 'signal'] = -1  # Death cross - SHORT
        
        df['position_change'] = df['signal'].diff()
        
        return df
    
    def check_stop_loss(self, asset_name, current_price):
        """Check if 3% stop loss is hit for a position"""
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
        """Run portfolio backtest across all assets"""
        print("\n" + "="*70)
        print("STARTING PORTFOLIO BACKTEST - PURE CROSSOVER MODE")
        print("="*70)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Max Positions: {self.max_positions}")
        print(f"Position Size: ${self.position_size_per_trade:,.2f} per trade")
        print(f"Assets: {', '.join(asset_data.keys())}")
        print(f"Strategy: 50/200 Hour MA Crossover - NO FILTERS")
        print(f"Stop Loss: 3%")
        print("="*70 + "\n")
        
        # Find common date range across all assets
        min_length = min(len(df) for df in asset_data.values())
        
        # Process bar by bar across all assets
        for i in range(200, min_length):  # Start after 200 MA period
            current_date = None
            
            # First, check stop losses for all existing positions
            for asset_name in list(self.positions.keys()):
                df = asset_data[asset_name]
                if i < len(df):
                    current_row = df.iloc[i]
                    current_date = current_row['date']
                    price = current_row['close']
                    
                    if self.check_stop_loss(asset_name, price):
                        if self.positions[asset_name]['type'] == 'long':
                            self.execute_trade(current_date, asset_name, price, 'SELL', 'Stop Loss')
                        else:
                            self.execute_trade(current_date, asset_name, price, 'COVER', 'Stop Loss')
            
            # Then check for new signals across all assets
            for asset_name, df in asset_data.items():
                if i >= len(df):
                    continue
                    
                current_row = df.iloc[i]
                current_date = current_row['date']
                price = current_row['close']
                
                # Check for crossover signals
                if current_row['position_change'] == 2:  # Golden cross
                    if asset_name in self.positions and self.positions[asset_name]['type'] == 'short':
                        self.execute_trade(current_date, asset_name, price, 'COVER', 'Signal Reversal')
                    if asset_name not in self.positions:
                        self.execute_trade(current_date, asset_name, price, 'BUY', 'Golden Cross')
                        
                elif current_row['position_change'] == -2:  # Death cross
                    if asset_name in self.positions and self.positions[asset_name]['type'] == 'long':
                        self.execute_trade(current_date, asset_name, price, 'SELL', 'Signal Reversal')
                    if asset_name not in self.positions:
                        self.execute_trade(current_date, asset_name, price, 'SHORT', 'Death Cross')
            
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
        print("PORTFOLIO BACKTEST RESULTS")
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
                    print(f"{asset_name:12} | Trades: {len(asset_trades):2} | P&L: ${asset_pnl:8.2f} | Win Rate: {asset_win_rate:5.1f}%")
            
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
    backtester = PortfolioBacktester(initial_capital=10000, max_positions=3)
    
    if not backtester.connect_tws(port=7497):
        return
    
    try:
        # Fetch data for all assets
        print("\nFetching historical data for all assets...")
        asset_data = {}
        
        for asset_name in backtester.assets.keys():
            df = backtester.fetch_historical_data(asset_name)
            if df is not None and len(df) > 200:
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