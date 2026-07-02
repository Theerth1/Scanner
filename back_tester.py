import yfinance as yf
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib

class Breakout(Strategy):
    # static
    n_sma21 = 21
    n_sma55 = 55
    n_sma233 = 233
    
    def init(self):
        # Pre-calculate indicators
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        # SMAs
        self.sma21 = self.I(talib.SMA, close, self.n_sma21)
        self.sma55 = self.I(talib.SMA, close, self.n_sma55)
        self.sma233 = self.I(talib.SMA, close, self.n_sma233)
        
        # MACD
        self.macd, self.signal, self.hist = self.I(talib.MACD, close, 12, 26, 9)
        
        # DMI / ADX
        self.adx = self.I(talib.ADX, high, low, close, 14)
        self.plus_di = self.I(talib.PLUS_DI, high, low, close, 14)
        self.minus_di = self.I(talib.MINUS_DI, high, low, close, 14)
        
        # StochRSI
        self.fastk, self.fastd = self.I(talib.STOCHRSI, close, 14, 14, 3, 0)
        
        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume.astype(float), 20)

    def next(self): 
        # 1. Price > 21 > 55 > 233
        trend_ok = (self.data.Close[-1] > self.sma21[-1] > self.sma55[-1] > self.sma233[-1])
        
        # 2. Momentum (MACD > Signal)
        macd_ok = (self.macd[-1] > self.signal[-1])
        
        # 3. Strength (ADX > 20 and +DI > -DI)
        dmi_ok = (self.plus_di[-1] > self.minus_di[-1] and self.adx[-1] > 20)
        
        # 4. StochRSI (Continuation: K > D and K > 50)
        stoch_ok = (self.fastk[-1] > self.fastd[-1] and self.fastk[-1] > 50)
        
        # 5. Volume (Vol > 1.2x Avg)
        # Note: Volume data can be noisy, so we sometimes relax this for backtesting
        vol_ok = (self.data.Volume[-1] > 1.2 * self.vol_sma[-1])

        # Combined Entry (If mostly true)
        # Treat this as "Score >= 80"
        if not self.position:
            if trend_ok and macd_ok and dmi_ok and stoch_ok:
                self.buy()

        # EXIT LOGIC (Trailing Stop): sell if Price closes BELOW the 21 EMA (Trend Break)
        elif self.position:
            if self.data.Close[-1] < self.sma21[-1]:
                self.position.close()

# RUN BACKTEST
def run_test(ticker):
    print(f"Testing {ticker}...")
    # Download 2 years of data
    data = yf.download(ticker, period="2y", progress=False)
    
    # Run Backtest
    bt = Backtest(data, Breakout, cash=10000, commission=.002)
    stats = bt.run()
    
    print(stats)
    # bt.plot() # Uncomment to see the chart popup

if __name__ == "__main__":
    # Test on a known momentum stock
    run_test("NVDA")
