"""
Catos Method Stock Scanner - Continuation & Breakout Strategy
"""

import os
import time
import smtplib
import yfinance as yf
import pandas as pd
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import google.generativeai as genai
from typing import List, Dict, Tuple
import talib

# Configuration
CHUNK_SIZE = 30  
CHUNK_DELAY = 1.5
MIN_SCORE = 80
MIN_PRICE = 5.0

# --- UTILITIES ---
def get_all_tickers() -> List[str]:
    print("Fetching complete US stock list...")
    try:
        import requests
        import io
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            tickers_raw = response.text.strip().split('\n')
            cleaned = [t.strip().upper() for t in tickers_raw if t and not any(c in t for c in ['^', '.', '/', '='])]
            print(f"✅ Loaded {len(cleaned)} tickers")
            return sorted(list(set(cleaned)))
    except Exception as e:
        print(f"⚠️ Failed to fetch list: {e}. Using fallback.")
        return ['AAPL', 'NVDA', 'AMD', 'TSLA', 'MSFT'] # Fallback

def download_data_in_chunks(tickers: List[str]) -> Dict:
    print(f"\nDownloading {len(tickers)} tickers...")
    all_data = {}
    
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        try:
            # Auto-adjust=True is CRITICAL for correct indicator calc
            data = yf.download(chunk, period='1y', group_by='ticker', auto_adjust=True, progress=False, threads=True)
            
            if len(chunk) == 1:
                if not data.empty and len(data) >= 250: all_data[chunk[0]] = data
            else:
                for ticker in chunk:
                    try:
                        if ticker in data.columns.levels[0]:
                            td = data[ticker]
                            if not td.empty and len(td) >= 250: all_data[ticker] = td
                    except: continue
            time.sleep(CHUNK_DELAY)
        except:
            time.sleep(CHUNK_DELAY * 2)
            continue
            
    return all_data

# --- STRATEGY ENGINE (Continuation / Breakout) ---
def calculate_catos_score(df: pd.DataFrame) -> Tuple[float, Dict]:
    if len(df) < 250: return 0.0, {}
    
    try:
        # Data Prep
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        if close[-1] < MIN_PRICE: return 0.0, {}

        # 1. TREND (40 pts)
        # Logic: Price > 21 > 55 > 233 (Perfect Bullish Alignment)
        sma21 = talib.SMA(close, timeperiod=21)[-1]
        sma55 = talib.SMA(close, timeperiod=55)[-1]
        sma233 = talib.SMA(close, timeperiod=233)[-1]
        
        score = 0.0
        details = {'price': close[-1]}
        
        if close[-1] > sma21 > sma55 > sma233:
            score += 40
            details['trend'] = "Perfect Stack (Price > 21 > 55 > 233)"
        elif sma21 > sma55 > sma233:
            score += 20
            details['trend'] = "Strong Trend (21 > 55 > 233)"
            
        # 2. MOMENTUM - MACD (20 pts)
        # Logic: Signal > Line (Bullish) AND Line > 0 (Positive Trend)
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        if macd[-1] > signal[-1] and macd[-1] > 0:
            score += 20
            details['macd'] = "Bullish & Positive (Continuation)"
        elif macd[-1] > signal[-1]:
            score += 10
            details['macd'] = "Bullish Crossover"

        # 3. STRENGTH - DMI/ADX (20 pts)
        # Logic: +DI > -DI (Bulls control) AND ADX > 20 (Trend exists)
        adx = talib.ADX(high, low, close, timeperiod=14)[-1]
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)[-1]
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)[-1]
        
        if plus_di > minus_di and adx > 20:
            score += 20
            details['adx'] = f"Strong Trend (ADX {adx:.0f})"
        elif plus_di > minus_di:
            score += 10
            details['adx'] = "Bulls Leading (+DI > -DI)"

        # 4. ENTRY SIGNAL - STOCH RSI (20 pts)
        # Logic: K > D (Momentum up)
        # For Continuation: If K > 50, it means we are in the "Power Zone" pushing higher
        fastk, fastd = talib.STOCHRSI(close, timeperiod=14, fastk_period=14, fastd_period=3, fastd_matype=0)
        k = fastk[-1]
        d = fastd[-1]
        
        if k > d:
            if adx > 30:  # trend is strong
                if k > 50:  # strong stochastic
                    score += 20
                    details['stoch'] = "Power Zone (K > D & K > 50 & Strong Trend)"
                else:
                    score += 15
                    details['stoch'] = "Momentum Rising (Strong Trend)"
            else:
                score += 5
                details['stoch'] = "Momentum Rising (Weak Trend)"


        # 5. BREAKOUT BONUS (+10 pts)
        # Logic: Breaking 20 day high
        high_20 = np.max(high[-21:-1]) # Max of previous 20 days
        if close[-1] > high_20:
            score += 10 # Bonus points pushing over 100
            details['breakout'] = "🚨 NEW 20-DAY HIGH"

        return score, details

    except Exception as e:
        return 0.0, {'error': str(e)}

# --- AI VALIDATION ---
def validate_with_gemini(candidates: List[Dict], api_key: str) -> List[Dict]:
    print(f"\nValidating {len(candidates)} candidates with AI...")
    try:
        genai.configure(api_key=api_key)
        # Fallback list of models to try
        models = ['gemini-1.5-flash', 'gemini-pro', 'gemini-1.0-pro']
        
        active_model = None
        for m in models:
            try:
                model = genai.GenerativeModel(m)
                model.generate_content("test")
                active_model = model
                print(f"✅ Using AI Model: {m}")
                break
            except: continue
            
        if not active_model:
            print("⚠️ AI Unavailable (Check Key/Billing). Returning technicals only.")
            return candidates

        validated = []
        for c in candidates:
            ticker = c['ticker']
            prompt = f"""
            Analyze {ticker} for a MOMENTUM BREAKOUT trade.
            Tech Score: {c['score']}%
            Trend: {c['details'].get('trend')}
            Signal: {c['details'].get('breakout', 'Consolidating')}
            
            Briefly assess:
            1. Sector Strength
            2. Recent Catalyst/News
            3. Verdict: "BUY" or "WAIT"
            """
            try:
                response = active_model.generate_content(prompt)
                c['ai_analysis'] = response.text
                time.sleep(2) # Rate limit protection
            except:
                c['ai_analysis'] = "AI Rate Limit/Error"
            validated.append(c)
            
        return validated

    except Exception as e:
        print(f"AI System Error: {e}")
        return candidates

# --- REPORTING ---
def send_email_report(results: List[Dict], email_config: Dict):
    print("\nSending email...")
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"🚀 Breakout Scanner: {len(results)} Matches"
    msg['From'] = email_config['sender']
    msg['To'] = email_config['receiver']
    
    body = "<h2>Daily Breakout Report</h2>"
    if not results:
        body += "<p>No stocks met the strict continuation criteria today.</p>"
    else:
        for r in results:
            d = r['details']
            body += f"""
            <div style="padding:10px; border-bottom:1px solid #ccc;">
                <h3>{r['ticker']} (Score: {r['score']:.0f})</h3>
                <p><b>Trend:</b> {d.get('trend', 'N/A')}</p>
                <p><b>Breakout:</b> {d.get('breakout', 'None')}</p>
                <p><b>AI Verdict:</b><br><i>{r.get('ai_analysis', 'N/A')}</i></p>
            </div>
            """
    
    msg.attach(MIMEText(body, 'html'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(email_config['sender'], email_config['password'])
            s.send_message(msg)
        print("✅ Email sent successfully")
    except Exception as e:
        print(f"❌ Email failed: {e}")

def main():
    print("Starting Scan...")
    
    # 1. Credentials
    api_key = os.environ.get('GENAI_API_KEY')
    email_config = {
        'sender': os.environ.get('EMAIL_SENDER'),
        'password': os.environ.get('EMAIL_PASSWORD'),
        'receiver': os.environ.get('EMAIL_RECEIVER')
    }
    
    if not email_config['password']:
        print("❌ Secrets missing. Exiting.")
        return

    # 2. Scan
    tickers = get_all_tickers()
    # tickers = tickers[:200] # Uncomment for fast testing
    data = download_data_in_chunks(tickers)
    
    candidates = []
    for ticker, df in data.items():
        score, details = calculate_catos_score(df)
        if score >= MIN_SCORE:
            print(f"⭐ Match: {ticker} ({score})")
            candidates.append({'ticker': ticker, 'score': score, 'details': details})
            
    # 3. Validate & Send
    final_list = validate_with_gemini(candidates, api_key)
    send_email_report(final_list, email_config)

if __name__ == "__main__":
    main()
