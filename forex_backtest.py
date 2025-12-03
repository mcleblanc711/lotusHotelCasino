"""
Forex Backtesting Script - EUR/USD Golden/Death Cross Strategy
20-hour / 50-hour Moving Average Crossover
Period: November 1, 2023 to November 1, 2024
Stop Loss: 2%
"""

from ib_insync import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ForexBacktester:
    def __init__(self, initial_capital=10000, position_size=10000):  # mini lot = 10,000 units
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.capital = initial_capital
        self.position = None  # None, 'long', or 'short'
        self.entry_price = None
        self.trades = []
        self.equity_curve = []
        
    def connect_tws(self, port=7497):
        """Connect to TWS paper trading (default port 7497)"""
        self.ib = IB()
        try:
            self.ib.connect('127.0.0.1', port, clientId=1)
            print(f"Connected to TWS on port {port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            print("Make sure TWS is running and API connections are enabled")
            print("Paper trading typically uses port 7497")
            return False
    
    def fetch_historical_data(self, symbol='EUR', currency='USD'):
        """Fetch hourly data from Nov 1, 2023 to Nov 1, 2024"""
        forex = Forex(symbol + currency)
        self.ib.qualifyContracts(forex)
        
        # IBKR limits historical data requests, so we'll fetch in chunks
        print("Fetching historical data... this may take a moment")
        
        end_date = '20241101 23:59:59'
        bars = self.ib.reqHistoricalData(
            forex,
            endDateTime=end_date,
            durationStr='1 Y',  # 1 year of data
            barSizeSetting='1 hour',
            whatToShow='MIDPOINT',
            useRTH=False,  # Include all hours (24/5 forex)
            formatDate=1
        )
        
        df = util.df(bars)
        print(f"Fetched {len(df)} hourly bars")
        return df
    
    def calculate_signals(self, df):
        """Calculate 20/50 MA and generate crossover signals"""
        df['MA_20'] = df['close'].rolling(window=20).mean()
        df['MA_50'] = df['close'].rolling(window=50).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['MA_20'] > df['MA_50'], 'signal'] = 1  # Golden cross - go long
        df.loc[df['MA_20'] < df['MA_50'], 'signal'] = -1  # Death cross - go short
        
        # Detect crossovers (when signal changes)
        df['position'] = df['signal'].diff()
        
        return df
    
    def check_stop_loss(self, current_price):
        """Check if 2% stop loss is hit"""
        if self.position == 'long':
            stop_price = self.entry_price * 0.98  # 2% below entry
            if current_price <= stop_price:
                return True
        elif self.position == 'short':
            stop_price = self.entry_price * 1.02  # 2% above entry
            if current_price >= stop_price:
                return True
        return False
    
    def execute_trade(self, date, price, action, reason):
        """Simulate trade execution"""
        trade_record = {
            'date': date,
            'action': action,
            'price': price,
            'reason': reason,
            'capital': self.capital
        }
        
        if action == 'BUY':
            self.position = 'long'
            self.entry_price = price
            print(f"{date}: BUY at {price:.5f} - {reason}")
            
        elif action == 'SELL':
            if self.position == 'long':
                # Close long position
                pnl = (price - self.entry_price) * self.position_size
                self.capital += pnl
                trade_record['pnl'] = pnl
                trade_record['return_pct'] = (price / self.entry_price - 1) * 100
                print(f"{date}: SELL at {price:.5f} - {reason} | P&L: ${pnl:.2f} ({trade_record['return_pct']:.2f}%)")
            self.position = None
            self.entry_price = None
            
        elif action == 'SHORT':
            self.position = 'short'
            self.entry_price = price
            print(f"{date}: SHORT at {price:.5f} - {reason}")
            
        elif action == 'COVER':
            if self.position == 'short':
                # Close short position
                pnl = (self.entry_price - price) * self.position_size
                self.capital += pnl
                trade_record['pnl'] = pnl
                trade_record['return_pct'] = (self.entry_price / price - 1) * 100
                print(f"{date}: COVER at {price:.5f} - {reason} | P&L: ${pnl:.2f} ({trade_record['return_pct']:.2f}%)")
            self.position = None
            self.entry_price = None
        
        self.trades.append(trade_record)
        self.equity_curve.append({'date': date, 'equity': self.capital})
    
    def run_backtest(self, df):
        """Run the backtest simulation"""
        print("\n" + "="*60)
        print("STARTING BACKTEST")
        print("="*60)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Position Size: {self.position_size:,} units (mini lot)")
        print(f"Strategy: 20/50 Hour MA Crossover")
        print(f"Stop Loss: 2%")
        print("="*60 + "\n")
        
        # Start after we have 50 periods for MA calculation
        for i in range(50, len(df)):
            current_row = df.iloc[i]
            date = current_row['date']
            price = current_row['close']
            
            # Check stop loss first if we have a position
            if self.position and self.check_stop_loss(price):
                if self.position == 'long':
                    self.execute_trade(date, price, 'SELL', 'Stop Loss Hit')
                elif self.position == 'short':
                    self.execute_trade(date, price, 'COVER', 'Stop Loss Hit')
            
            # Check for crossover signals
            if current_row['position'] == 2:  # Golden cross (signal went from -1 to 1)
                if self.position == 'short':
                    self.execute_trade(date, price, 'COVER', 'Death Cross -> Golden Cross')
                if self.position is None:
                    self.execute_trade(date, price, 'BUY', 'Golden Cross')
                    
            elif current_row['position'] == -2:  # Death cross (signal went from 1 to -1)
                if self.position == 'long':
                    self.execute_trade(date, price, 'SELL', 'Golden Cross -> Death Cross')
                if self.position is None:
                    self.execute_trade(date, price, 'SHORT', 'Death Cross')
        
        # Close any open position at the end
        if self.position:
            final_price = df.iloc[-1]['close']
            final_date = df.iloc[-1]['date']
            if self.position == 'long':
                self.execute_trade(final_date, final_price, 'SELL', 'End of backtest period')
            elif self.position == 'short':
                self.execute_trade(final_date, final_price, 'COVER', 'End of backtest period')
    
    def print_results(self):
        """Print backtest performance metrics"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        total_return = self.capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        print(f"\nFinal Capital: ${self.capital:,.2f}")
        print(f"Total Return: ${total_return:,.2f} ({total_return_pct:.2f}%)")
        
        # Trade statistics
        completed_trades = [t for t in self.trades if 'pnl' in t]
        if completed_trades:
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            losing_trades = [t for t in completed_trades if t['pnl'] <= 0]
            
            print(f"\nTotal Completed Trades: {len(completed_trades)}")
            print(f"Winning Trades: {len(winning_trades)}")
            print(f"Losing Trades: {len(losing_trades)}")
            
            if completed_trades:
                win_rate = (len(winning_trades) / len(completed_trades)) * 100
                print(f"Win Rate: {win_rate:.1f}%")
                
                avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
                print(f"\nAverage Winning Trade: ${avg_win:.2f}")
                print(f"Average Losing Trade: ${avg_loss:.2f}")
                
                if losing_trades:
                    profit_factor = abs(sum([t['pnl'] for t in winning_trades]) / sum([t['pnl'] for t in losing_trades]))
                    print(f"Profit Factor: {profit_factor:.2f}")
                
                # Max drawdown
                equity_values = [e['equity'] for e in self.equity_curve]
                peak = equity_values[0]
                max_dd = 0
                for equity in equity_values:
                    if equity > peak:
                        peak = equity
                    dd = (peak - equity) / peak * 100
                    if dd > max_dd:
                        max_dd = dd
                print(f"Max Drawdown: {max_dd:.2f}%")
        
        print("="*60)
    
    def disconnect(self):
        """Disconnect from TWS"""
        self.ib.disconnect()
        print("\nDisconnected from TWS")


def main():
    # Initialize backtester
    backtester = ForexBacktester(initial_capital=10000, position_size=10000)
    
    # Connect to TWS
    if not backtester.connect_tws(port=7497):
        return
    
    try:
        # Fetch historical data
        df = backtester.fetch_historical_data('EUR', 'USD')
        
        # Calculate signals
        df = backtester.calculate_signals(df)
        
        # Run backtest
        backtester.run_backtest(df)
        
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
