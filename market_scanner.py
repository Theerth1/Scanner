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
MIN_SCORE = 110
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
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        if close[-1] < MIN_PRICE: return 0.0, {}

        # 1. TREND
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
        else:
            details['trend'] = "Weak Trend"
            
        # 2. MOMENTUM - MACD
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        # Store raw text for the report
        if macd[-1] > signal[-1] and macd[-1] > 0:
            score += 20
            details['macd_desc'] = "Bullish (Positive & Rising)"
        elif macd[-1] > signal[-1]:
            score += 10
            details['macd_desc'] = "Bullish Crossover"
        else:
            details['macd_desc'] = "Bearish"

        # 3. STRENGTH - DMI/ADX
        adx = talib.ADX(high, low, close, timeperiod=14)[-1]
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)[-1]
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)[-1]
        
        # Save exact figures for the report
        details['dmi_desc'] = f"{'Bullish' if plus_di > minus_di else 'Bearish'} (+DI {plus_di:.1f} / -DI {minus_di:.1f})"
        
        if plus_di > minus_di and adx > 20:
            score += 20
        elif plus_di > minus_di:
            score += 10

        # 4. STOCH RSI
        fastk, fastd = talib.STOCHRSI(close, timeperiod=14, fastk_period=14, fastd_period=3, fastd_matype=0)
        k = fastk[-1]
        d = fastd[-1]
        
        # Save exact figures
        details['stoch_desc'] = f"K={k:.1f}, D={d:.1f}"
        
        if k > d:
            if adx > 30:
                if k > 50: score += 20
                else: score += 15
            else: score += 5

        vol_sma = talib.SMA(volume, timeperiod=20)[-1]
        if volume[-1] > 1.2 * vol_sma:  # 20% above average
            score += 10
            details['vol'] = "High Volume (Confirmed)"
        elif volume[-1] > vol_sma:
            score += 5
            details['vol'] = "Slight Volume Increase"
        else:
            details['vol'] = "Low Volume (Caution)"


        # 5. BREAKOUT
        high_20 = np.max(high[-21:-1]) 
        if close[-1] > high_20:
            score += 10
            details['breakout'] = "YES (New 20-Day High)"
        else:
            details['breakout'] = "No"

        return score, details

    except Exception as e:
        return 0.0, {'error': str(e)}

# --- AI VALIDATION ---
# --- AI VALIDATION (FIXED) ---
def validate_with_gemini(candidates: List[Dict], api_key: str) -> List[Dict]:
    print(f"\nValidating top {len(candidates)} candidates with AI...")
    
    # Configure GenAI
    active_model = None
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Find a model that supports generation, prefer Flash
            all_models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            for m in all_models:
                if 'flash' in m.name.lower():
                    active_model = genai.GenerativeModel(m.name)
                    print(f"✅ AI Ready: {m.name}")
                    break
            if not active_model and all_models:
                active_model = genai.GenerativeModel(all_models[0].name)
                print(f"⚠️ AI Fallback: {all_models[0].name}")
        except:
            print("❌ AI Connection Failed")

    validated = []
    for i, c in enumerate(candidates):
        ticker = c['ticker']
        print(f"[{i+1}/{len(candidates)}] Processing {ticker}...")
        
        # 1. Fetch Sector/PE Data (Slow, so only doing for winners)
        try:
            stock_info = yf.Ticker(ticker).info
            c['details']['sector'] = stock_info.get('sector', 'N/A')
            c['details']['pe'] = stock_info.get('trailingPE', 'N/A')
        except:
            c['details']['sector'] = "N/A"
            c['details']['pe'] = "N/A"

        # 2. Run AI
        if active_model:
            prompt = f"Analyze {ticker}. Technical score {c['score']}/120. Trend: {c['details']['trend']}. Give a 1-sentence verdict if these stocks are bullish (breakout after consolidation/continuation trading) using fundamental analysis."
            try:
                response = active_model.generate_content(prompt)
                c['ai_analysis'] = response.text.strip()
                time.sleep(4) # Rate limit safety
            except:
                c['ai_analysis'] = "AI Unavailable"
        else:
            c['ai_analysis'] = "AI Not Configured"
            
        validated.append(c)
        
    return validated
# --- REPORTING ---
def send_email_report(results: List[Dict], email_config: Dict):
    print("\nSending email...")
    msg = MIMEMultipart('alternative')
    
    # --- RENAME HEADER HERE ---
    current_date = datetime.now().strftime("%Y-%m-%d")
    msg['Subject'] = f"Market Scan: {current_date} - {len(results)} Setup(s) Found"
    
    msg['From'] = email_config['sender']
    msg['To'] = email_config['receiver']
    
    body = """
    <html>
      <body style="font-family: Arial, sans-serif; color: #333;">
        <h2 style="background-color: #2c3e50; color: white; padding: 10px;">Daily Market Scans</h2>
    """
    
    if not results:
        body += "<p>No setups met the criteria today.</p>"
    else:
        for r in results:
            d = r['details']
            
            # Format P/E nicely if it's a number
            pe_str = f"{d.get('pe', 'N/A'):.2f}" if isinstance(d.get('pe'), (int, float)) else "N/A"
            
            body += f"""
            <div style="border: 1px solid #ddd; margin-bottom: 20px; padding: 15px; border-radius: 8px;">
                <h3 style="margin-top: 0; color: #0056b3;">{r['ticker']} - {r['score']:.1f}%</h3>
                
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td style="padding: 3px 0;"><b>Price:</b></td><td>${d.get('price', 0):.2f}</td></tr>
                    <tr><td style="padding: 3px 0;"><b>Setup:</b></td><td>{d.get('trend', 'N/A')}</td></tr>
                    <tr><td style="padding: 3px 0;"><b>MACD:</b></td><td>{d.get('macd_desc', 'N/A')}</td></tr>
                    <tr><td style="padding: 3px 0;"><b>DMI:</b></td><td>{d.get('dmi_desc', 'N/A')}</td></tr>
                    <tr><td style="padding: 3px 0;"><b>StochRSI:</b></td><td>{d.get('stoch_desc', 'N/A')}</td></tr>
                    <tr><td style="padding: 3px 0;"><b>Sector:</b></td><td>{d.get('sector', 'N/A')}</td></tr>
                    <tr><td style="padding: 3px 0;"><b>P/E:</b></td><td>{pe_str}</td></tr>
                </table>
                
                <div style="margin-top: 10px; background-color: #f0f4f8; padding: 10px; border-radius: 4px;">
                    <b>AI Analysis:</b><br>
                    <i style="color: #444;">{r.get('ai_analysis', 'N/A')}</i>
                </div>
            </div>
            """
            
    body += "</body></html>"
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
