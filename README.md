# Forex Algorithmic Trading Backtesting Suite

A comprehensive Python-based backtesting framework for testing forex trading strategies using Interactive Brokers historical data. Built for rapid strategy testing and comparison across multiple currency pairs and time periods.

**Vibecoded in an evening with Claude Sonnet 4.5** - Anthropic's latest AI assistant for coding and analysis. (ie I'm a poor and can't afford Opus 4.5 for everything)

## 🏨 About the Name

This project is named after the **Lotus Hotel and Casino** from Percy Jackson and the Olympians - a place where time moves differently and you can get lost in endless possibilities. Just like the Lotus Hotel, this backtesting suite lets you explore countless trading scenarios across different time periods, where you might lose track of time testing "just one more strategy." 

(This is dedicated to my girlfriend; who may have came up with the name and has had to listen to far too many rants about algo trading)

## 🚀 Features

- **Interactive Command-Line Interface** - User-friendly prompts guide you through backtest configuration
- **Multiple Trading Strategies**
  - Moving Average Crossover (50/200 hour MA)
  - Mean Reversion (RSI + Bollinger Bands)
- **Portfolio Testing** - Test up to 16 currency pairs simultaneously
- **Comprehensive Results** - Export detailed performance metrics to CSV
- **Real Market Data** - Fetches hourly historical data from Interactive Brokers
- **Risk Management** - Configurable stop losses and position sizing

## 📋 Requirements

- Python 3.7+
- Interactive Brokers TWS (Trader Workstation) or IB Gateway
- Active IBKR account (paper trading works fine)

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/mcleblanc711/lotusHotelCasino.git
cd lotusHotelCasino
```

2. **Install dependencies**
```bash
pip install ib_insync pandas numpy --break-system-packages
```

3. **Set up Interactive Brokers TWS**
   - Download and install [TWS](https://www.interactivebrokers.com/en/trading/tws.php) (Requires InteractiveBrokers papertrading account)
   - Enable API connections: File → Global Configuration → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Paper trading uses port 7497 (default in scripts)

## 🎯 Quick Start

### Interactive Backtesting Engine (Main Tool)

Run the interactive backtesting engine for guided configuration:

```bash
python3 forex_backtest.py
```

You'll be prompted for:
- **Years to test** (e.g., `2021,2022,2023` or `all`)
- **Currency pairs** (e.g., `EUR/USD,GBP/USD` or `majors` or `all`)
- **Strategies** (`ma`, `meanrev`, or `both`)
- **Risk parameters** (capital, max positions, stop loss %)
- **Output filename** (CSV export)

## 📊 Available Strategies

### 1. Moving Average Crossover
- **Entry**: 50-hour MA crosses above 200-hour MA (long) or below (short)
- **Exit**: Opposite crossover or stop loss
- **Best for**: Strong trending markets
- **Example result**: +1,212% on USD/JPY in 2022

### 2. Mean Reversion (RSI + Bollinger Bands)
- **Entry**: RSI < 30 at lower BB (buy) or RSI > 70 at upper BB (short)
- **Exit**: Price reverts to middle BB or stop loss
- **Best for**: Ranging/choppy markets
- **Parameters**: RSI(14), BB(20, 2σ)

## 💹 Supported Currency Pairs

### Major Pairs
- EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF, NZD/USD

### Cross Pairs
- EUR/GBP, EUR/JPY, GBP/JPY, AUD/CAD, NZD/CAD, EUR/CHF, GBP/CHF

### Emerging Market Pairs
- USD/MXN, USD/ZAR

## 📁 Project Structure

```
lotusHotelCasino/
├── forex_backtest.py            # Interactive backtesting engine (main tool)
├── README.md                    # This file
└── backtest_results.csv         # Example output (generated after running)
```

## 📈 Output & Results

### CSV Export Columns
- `year` - Year tested
- `pair` - Currency pair
- `strategy` - Strategy used (ma/meanrev)
- `final_capital` - Ending capital
- `total_return` - Dollar return
- `total_return_pct` - Percentage return
- `num_trades` - Total trades executed
- `winning_trades` - Number of winning trades
- `losing_trades` - Number of losing trades
- `win_rate` - Win percentage
- `avg_win` - Average winning trade P&L
- `avg_loss` - Average losing trade P&L
- `profit_factor` - Ratio of gross profit to gross loss

### Console Output
The engine prints:
- Summary by strategy
- Summary by year
- Top 5 best performers
- Top 5 worst performers

## 🎓 Example Usage

### Example 1: Test Both Strategies on Major Pairs (2022)
```bash
python3 forex_backtest.py
```
When prompted:
- Years: `2022`
- Pairs: `majors`
- Strategy: `both`
- Press Enter for default risk parameters
- Output: `results_2022.csv`

### Example 2: Compare 2021 vs 2022 vs 2023 on EUR/USD
```bash
python3 forex_backtest.py
```
When prompted:
- Years: `2021,2022,2023`
- Pairs: `EUR/USD`
- Strategy: `both`
- Output: `eur_comparison.csv`

### Example 3: Test All Pairs, All Strategies, Multiple Years
```bash
python3 forex_backtest.py
```
When prompted:
- Years: `all`
- Pairs: `all`
- Strategy: `both`
- Output: `comprehensive_backtest.csv`

## ⚠️ Important Notes

### Data Limitations
- IBKR provides up to 1 year of hourly data per request
- Historical data availability varies by pair
- Emerging market pairs may have data gaps

### Backtesting Caveats
- **Past performance ≠ future results**
- No slippage or commission modeling (yet)
- Assumes perfect fills at close prices
- Stop losses execute at exact stop price (no slippage)
- Paper trading results shown - real trading has additional costs

### Risk Warning
These are **backtests on paper trading data**. Real trading involves:
- Slippage and commission costs
- Execution delays
- Liquidity constraints
- Psychological factors
- Margin requirements and potential liquidation

**Always test thoroughly in paper trading before risking real capital.**

## 🔬 Key Findings from Testing

### 2022 (Strong USD Trending Year)
- **MA Crossover**: +1,212% on portfolio (driven by USD/JPY)
- **Mean Reversion**: Expected to underperform in strong trends
- **Best pair**: USD/JPY (+$120,235 on $10k capital in 38 trades)
- **Worst pairs**: EUR/USD, GBP/USD (choppy consolidation)
- **Win rate**: 35.7% overall but average wins were 2.8x average losses

### 2023-2024 (Ranging Markets)
- **MA Crossover**: 0% with filters (no trades generated)
- **Mean Reversion**: Expected to outperform trend-following strategies

### Key Insights
1. **Strategy matters by market regime**: Trend-following dominates trending markets, mean reversion works in ranging markets
2. **Diversification helps**: Multiple pairs reduce dependence on single pair performance
3. **Filters are double-edged**: Prevent bad trades but also filter out good ones
4. **Risk management is crucial**: 185% max drawdown in 2022 would've required significant margin
5. **Low win rates can still profit**: With proper risk/reward ratios (2.8:1 in 2022 test)

## 🛠️ Customization

### Modify Strategy Parameters

**MA Crossover periods** (in `forex_backtest.py`):
```python
def calculate_ma_crossover_signals(self, df, fast_period=50, slow_period=200):
    # Change fast_period and slow_period here
```

**Mean Reversion thresholds** (in `forex_backtest.py`):
```python
# Change RSI thresholds (default: 30/70)
df.loc[(df['RSI'] < 30) & df['near_lower_bb'], 'signal'] = 1
df.loc[(df['RSI'] > 70) & df['near_upper_bb'], 'signal'] = -1
```

**Risk Parameters** (via interactive prompts):
- Initial capital (default: $10,000)
- Max concurrent positions (default: 3)
- Stop loss percentage (default: 3%)

## 📊 Analyzing Results

### Load Results in Python
```python
import pandas as pd

# Load backtest results
df = pd.read_csv('backtest_results.csv')

# Compare strategies
strategy_comparison = df.groupby('strategy')['total_return_pct'].mean()
print(strategy_comparison)

# Find best year for each pair
best_years = df.loc[df.groupby('pair')['total_return_pct'].idxmax()]
print(best_years[['pair', 'year', 'total_return_pct']])

# Plot performance
import matplotlib.pyplot as plt
df.pivot(index='pair', columns='strategy', values='total_return_pct').plot(kind='bar')
plt.ylabel('Return %')
plt.title('Strategy Performance by Pair')
plt.show()
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add more strategies (MACD, RSI divergence, Ichimoku, etc.)
- Implement commission and slippage modeling
- Add equity curve visualization
- Support for other asset classes (stocks, futures, options)
- Walk-forward optimization
- Monte Carlo simulation
- Real-time trading execution (currently backtest-only)

## 🤖 Built With AI

This project was developed in collaboration with **Claude Sonnet 4.5** by Anthropic. Claude assisted with:
- Strategy implementation and optimization
- Code architecture and best practices
- Documentation and examples
- Debugging and troubleshooting

## 📝 License

MIT License - feel free to use and modify for your own trading research.

## ⚡ Troubleshooting

### "Connection failed" error
- Ensure TWS is running
- Check API settings are enabled in TWS: File → Global Configuration → API → Settings
- Verify port 7497 (paper) or 7496 (live) is correct
- Restart TWS if connection hangs

### "No data available" error
- Some pairs have limited historical data on IBKR
- Try a different year or pair
- Check your IBKR account has market data subscriptions
- Ensure TWS is logged in and connected

### "Permission denied (publickey)" error (Git)
- Use HTTPS instead of SSH for GitHub
- `git remote set-url origin https://github.com/username/repo.git`

### Script runs but no trades generated
- This is expected if filters are too strict for the market conditions
- Try loosening filters (lower ADX threshold, reduce MA spread requirement)
- Test different time periods - some years are better for certain strategies
- Check the data actually loaded (script shows "Fetched X bars" messages)

**Disclaimer**: This software is for educational and research purposes only. Not financial advice. Trade at your own risk.
