import numpy as np
import pandas as pd
import yfinance as yf
import talib
from backtesting import Backtest, Strategy
 
# Keep these in sync with market_scanner.py
MIN_SCORE = 100
MIN_PRICE = 5.0
 
 
def rolling_high_20_prior(high: pd.Series) -> np.ndarray:
    """Max of the PRIOR 20 highs (excludes current bar) — mirrors np.max(high[-21:-1])."""
    return high.rolling(20).max().shift(1).values
 
 
class CatosBreakout(Strategy):
    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume.astype(float)
 
        # SMAs
        self.sma21 = self.I(talib.SMA, close, 21)
        self.sma55 = self.I(talib.SMA, close, 55)
        self.sma233 = self.I(talib.SMA, close, 233)
 
        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(talib.MACD, close, 12, 26, 9)
 
        # DMI / ADX
        self.adx = self.I(talib.ADX, high, low, close, 14)
        self.plus_di = self.I(talib.PLUS_DI, high, low, close, 14)
        self.minus_di = self.I(talib.MINUS_DI, high, low, close, 14)
 
        # StochRSI
        self.fastk, self.fastd = self.I(talib.STOCHRSI, close, 14, 14, 3, 0)
 
        # Volume SMA + prior-20-day high
        self.vol_sma = self.I(talib.SMA, volume, 20)
        self.high20 = self.I(rolling_high_20_prior, pd.Series(np.asarray(high)))
 
    def score(self) -> float:
        """Point-for-point replica of calculate_catos_score() on the current bar."""
        c = self.data.Close[-1]
        if c < MIN_PRICE:
            return 0.0
 
        s = 0.0
 
        # 1. TREND
        if c > self.sma21[-1] > self.sma55[-1] > self.sma233[-1]:
            s += 40
        elif self.sma21[-1] > self.sma55[-1] > self.sma233[-1]:
            s += 20
 
        # 2. MACD
        if self.macd[-1] > self.macd_signal[-1] and self.macd[-1] > 0:
            s += 20
        elif self.macd[-1] > self.macd_signal[-1]:
            s += 10
 
        # 3. DMI/ADX
        if self.plus_di[-1] > self.minus_di[-1] and self.adx[-1] > 20:
            s += 20
        elif self.plus_di[-1] > self.minus_di[-1]:
            s += 10
 
        # 4. StochRSI
        k, d = self.fastk[-1], self.fastd[-1]
        if k > d:
            if self.adx[-1] > 30:
                s += 20 if k > 50 else 15
            else:
                s += 5
 
        # 5. Volume / exhaustion
        rel_vol = (self.data.Volume[-1] / self.vol_sma[-1]) if self.vol_sma[-1] > 0 else 0.0
        if rel_vol > 1.2:
            s += 10
        elif rel_vol < 0.7:
            s -= 15  # exhaustion penalty (matches scanner v2)
 
        # 6. Breakout — volume-confirmed only (matches scanner v2)
        if c > self.high20[-1] and rel_vol >= 1.0:
            s += 10
 
        return s
 
    def next(self):
        if not self.position:
            if self.score() >= MIN_SCORE:
                self.buy()
        # EXIT LOGIC (Trailing Stop): sell if price closes BELOW the 21 SMA (trend break)
        elif self.data.Close[-1] < self.sma21[-1]:
            self.position.close()
 
 
def load_data(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """Download OHLCV and flatten yfinance's MultiIndex columns for backtesting.py."""
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data is None or data.empty:
        return None
    # yfinance returns MultiIndex columns even for a single ticker — flatten or backtesting.py crashes
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    data = data.dropna()
    return data if len(data) >= 250 else None
 
 
def run_test(ticker: str, cash: float = 10_000, commission: float = 0.002, verbose: bool = True):
    data = load_data(ticker)
    if data is None:
        print(f"⚠️ {ticker}: insufficient data, skipped")
        return None
 
    bt = Backtest(data, CatosBreakout, cash=cash, commission=commission, finalize_trades=True)
    stats = bt.run()
 
    if verbose:
        print(f"\n{'='*60}\n{ticker}\n{'='*60}")
        print(stats)
        # bt.plot()  # Uncomment to see the chart popup
 
    return stats
 
 
def run_basket(tickers: list[str]):
    """
    Run the strategy across a basket and report aggregate stats.
    A single-ticker test (especially NVDA) tells you nothing — a momentum
    strategy 'works' on the best momentum stock of the decade by construction.
    """
    rows = []
    for t in tickers:
        stats = run_test(t, verbose=False)
        if stats is None:
            continue
        rows.append({
            "Ticker": t,
            "Return [%]": stats["Return [%]"],
            "Buy&Hold [%]": stats["Buy & Hold Return [%]"],
            "# Trades": stats["# Trades"],
            "Win Rate [%]": stats["Win Rate [%]"],
            "Max DD [%]": stats["Max. Drawdown [%]"],
            "Sharpe": stats["Sharpe Ratio"],
        })
        print(f"  {t:<6} | Ret {stats['Return [%]']:>8.1f}% | B&H {stats['Buy & Hold Return [%]']:>8.1f}% | "
              f"Trades {stats['# Trades']:>3} | Win {stats['Win Rate [%]']:>5.1f}%")
 
    if not rows:
        print("No results.")
        return
 
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}\nAGGREGATE ({len(df)} tickers)\n{'='*60}")
    print(f"Mean strategy return : {df['Return [%]'].mean():>8.1f}%")
    print(f"Median strategy return: {df['Return [%]'].median():>7.1f}%")
    print(f"Mean buy & hold      : {df['Buy&Hold [%]'].mean():>8.1f}%")
    print(f"Mean win rate        : {df['Win Rate [%]'].mean():>8.1f}%")
    print(f"Mean max drawdown    : {df['Max DD [%]'].mean():>8.1f}%")
    print(f"Total trades         : {int(df['# Trades'].sum()):>8}")
    print(f"Beat buy & hold on   : {(df['Return [%]'] > df['Buy&Hold [%]']).sum()}/{len(df)} tickers")
 
 
if __name__ == "__main__":
    # A mixed basket: winners, losers, sideways names, different sectors.
    # Swap in your own — the point is NOT to cherry-pick momentum darlings.
    BASKET = [
        "NVDA", "AAPL", "MSFT", "AMD", "TSLA",
        "JPM", "XOM", "PFE", "KO", "DIS",
        "INTC", "F", "T", "NKE", "PYPL",
        "CAT", "UNH", "COST", "CRM", "BA",
    ]
    run_basket(BASKET)
 
    # Single-ticker deep dive (full stats printout):
    # run_test("NVDA")
 
