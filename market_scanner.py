import yfinance as yf
import pandas as pd
import pandas_ta as ta
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import time
import requests
import io
import os

# --- CONFIGURATION ---
# These pull the "Secrets" you saved in GitHub Settings
GENAI_API_KEY = os.environ.get("AIzaSyBWTZ_IzOPJsg189w4jxnbsNnaVpPsdnPk")
EMAIL_SENDER = os.environ.get("coolcreeper6277@gmail.com")
EMAIL_PASSWORD = os.environ.get("cewe sxtj ztwy pixo") 
EMAIL_RECEIVER = os.environ.get("theerth.srinivasan@gmail.com")

# Safety check to ensure secrets are loaded
if not GENAI_API_KEY:
    print("❌ Error: GENAI_API_KEY not found in environment variables.")
if not EMAIL_PASSWORD:
    print("❌ Error: EMAIL_PASSWORD not found in environment variables.")

# Initialize Gemini
try:
    genai.configure(api_key=GENAI_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini: {e}")

# --- 1. GET ALL TICKERS (NASDAQ + NYSE) ---
def get_all_tickers():
    print("🌍 Fetching full market ticker list...")
    try:
        # Pulling a raw CSV of all US stocks from a public repo
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        s = requests.get(url).content
        tickers = pd.read_csv(io.StringIO(s.decode('utf-8')), header=None)[0].tolist()
        
        # Clean up: Remove test stocks, warrants (^), and preferreds (.)
        clean_tickers = [x for x in tickers if isinstance(x, str) and "^" not in x and "." not in x]
        
        print(f"✅ Found {len(clean_tickers)} total tickers.")
        return clean_tickers
    except Exception as e:
        print(f"Failed to get full list: {e}")
        return ['AAPL', 'NVDA', 'AMD', 'TSLA', 'MSFT'] # Fallback

# --- 2. BULK DATA DOWNLOADER ---
def get_data_bulk(tickers):
    # We download 2 years of data to ensure the 233 SMA is accurate
    try:
        data = yf.download(tickers, period="2y", group_by='ticker', progress=False, threads=True)
        return data
    except Exception:
        return pd.DataFrame()

# --- 3. THE "CATOS" STRATEGY ENGINE (Fibonacci + DMI) ---
def analyze_ticker(ticker, df):
    try:
        # Filter 1: Data Sufficiency (Need ~250 days for 233 SMA)
        if len(df) < 250: return 0, []

        # Filter 2: Price > $1.00 (Avoid penny stocks)
        last_price = df['Close'].iloc[-1]
        if last_price < 1.00: return 0, []

        # Drop NaN
        df = df.dropna()
        if df.empty: return 0, []

        # --- INDICATORS ---
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # 1. Fibonacci SMAs
        sma_21 = ta.sma(close, length=21)
        sma_55 = ta.sma(close, length=55)
        sma_233 = ta.sma(close, length=233)
        
        # 2. MACD (12, 26, 9)
        macd = ta.macd(close)
        hist = macd['MACDh_12_26_9']
        
        # 3. DMI / ADX (14)
        dmi = ta.adx(high, low, close, length=14)
        adx = dmi['ADX_14']
        pos_di = dmi['DMP_14']
        neg_di = dmi['DMN_14']

        # 4. StochRSI (Custom 14, 14, 7, 5)
        stoch_rsi = ta.stochrsi(close, length=14, rsi_length=14, k=7, d=5)
        srsi_k = stoch_rsi.iloc[:, 0]
        srsi_d = stoch_rsi.iloc[:, 1]
        
        # 5. Volume
        vol_sma = ta.sma(df['Volume'], length=20)

        # Current Values
        s21 = sma_21.iloc[-1]
        s55 = sma_55.iloc[-1]
        s233 = sma_233.iloc[-1]
        hist_curr = hist.iloc[-1]
        hist_prev = hist.iloc[-2]
        
        score = 0
        reasons = []

        # --- SCORING (Strict) ---

        # 1. FIBONACCI TREND (30 pts)
        if last_price > s21 > s55 > s233:
            score += 30
            reasons.append("✅ Perfect Fib Trend (Price>21>55>233)")
        elif last_price > s233 and s21 > s55:
            score += 15
            reasons.append("✅ Bullish Trend (Price > 233)")

        # 2. MACD MOMENTUM (20 pts)
        if hist_curr > 0 and hist_curr > hist_prev:
            score += 20
            reasons.append("✅ MACD Expanding")

        # 3. DMI CONFIRMATION (20 pts)
        if pos_di.iloc[-1] > neg_di.iloc[-1]:
            if adx.iloc[-1] > 25:
                score += 20
                reasons.append(f"✅ Strong ADX Trend ({int(adx.iloc[-1])})")
            elif adx.iloc[-1] > 20:
                score += 10
                reasons.append("✅ Positive DMI")

        # 4. STOCH RSI (15 pts) - Signal Line Cross
        if (srsi_k.iloc[-2] < srsi_d.iloc[-2]) and (srsi_k.iloc[-1] > srsi_d.iloc[-1]):
            score += 15
            reasons.append("✅ StochRSI Buy Cross")
        elif srsi_k.iloc[-1] < 20:
             score += 10
             reasons.append("✅ StochRSI Oversold")

        # 5. VOLUME (15 pts)
        if df['Volume'].iloc[-1] > (vol_sma.iloc[-1] * 1.2):
            score += 15
            reasons.append("✅ High Volume")

        return score, reasons

    except Exception:
        return 0, []

# --- 4. GEMINI FUNDAMENTAL CHECK ---
def get_gemini_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Helper to get data safely
        def get_val(key): return info.get(key, 'N/A')

        fund_data = {
            "Symbol": ticker,
            "Sector": get_val('sector'),
            "Forward PE": get_val('forwardPE'),
            "PEG Ratio": get_val('pegRatio'),
            "Profit Margins": get_val('profitMargins'),
            "Revenue Growth": get_val('revenueGrowth'),
            "Target Price": get_val('targetMeanPrice'),
            "Current Price": get_val('currentPrice')
        }

        prompt = f"""
        Act as a strict hedge fund manager. 
        I have a strong TECHNICAL BUY signal for {ticker} based on Fibonacci trends.
        
        Here is the fundamental data:
        {fund_data}
        
        Please analyze this stock in under 50 words.
        1. Is the valuation dangerous? (e.g. PEG > 3 or negative earnings)
        2. Is this a real company or a junk stock?
        3. Final Verdict: "CONVICTION BUY", "SPECULATIVE BUY", or "TRAP/AVOID".
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Retry logic
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception:
                time.sleep(2)
                continue
        return "AI Analysis Failed after 3 retries."
        
    except Exception as e:
        return f"AI Analysis Failed: {e}"

# --- 5. MAIN EXECUTION ---
def run_scanner():
    print("🚀 Starting UNRESTRICTED Market Scan...")
    
    all_tickers = get_all_tickers()
    tickers_to_scan = all_tickers # Scan everyone
    
    chunk_size = 100
    high_conviction_list = []
    
    print(f"📊 Scanning {len(tickers_to_scan)} stocks for >= 80% matches...")

    for i in range(0, len(tickers_to_scan), chunk_size):
        chunk = tickers_to_scan[i:i+chunk_size]
        
        data = get_data_bulk(chunk)
        if data.empty: continue
            
        for ticker in chunk:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    stock_df = data.xs(ticker, axis=1, level=1, drop_level=True)
                else:
                    stock_df = data
                
                score, reasons = analyze_ticker(ticker, stock_df)
                
                if score >= 70:
                    print(f"🌟 Match Found: {ticker} ({score}%)")
                    high_conviction_list.append({
                        "ticker": ticker,
                        "score": score,
                        "price": stock_df['Close'].iloc[-1],
                        "reasons": reasons
                    })
            except:
                continue

    match_count = len(high_conviction_list)
    print(f"\n🎯 Technical Scan Complete. Found {match_count} stocks with Score >= 70%.")
    print("🤖 Starting Gemini Analysis on ALL matches (Estimated time: " + str(match_count * 5) + " seconds)...")

    high_conviction_list.sort(key=lambda x: x['score'], reverse=True)

    email_body = f"☀️ HIGH CONVICTION REPORT: {datetime.date.today()}\n"
    email_body += f"Found {match_count} stocks with Technical Score >= 70%\n"
    email_body += "========================================\n\n"

    for i, stock in enumerate(high_conviction_list):
        print(f"   ({i+1}/{match_count}) Analyzing {stock['ticker']}...")
        
        ai_verdict = get_gemini_analysis(stock['ticker'])
        
        email_body += f"🚀 {stock['ticker']} (Score: {stock['score']}%)\n"
        email_body += f"   Price: ${stock['price']:.2f}\n"
        email_body += f"   Signals: {', '.join(stock['reasons'])}\n"
        email_body += f"   🧠 Gemini Verdict:\n   {ai_verdict}\n"
        email_body += "----------------------------------------\n\n"
        
        time.sleep(5) # Rate limit protection

    if not high_conviction_list:
        print("No stocks passed the 70% threshold today.")
        return

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = f"🚀 {match_count} High-Conviction Breakouts (>=70%)"
    msg.attach(MIMEText(email_body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("✅ Report Sent Successfully!")
    except Exception as e:
        print(f"❌ Email Failed: {e}")

if __name__ == "__main__":
    run_scanner()
