"""
Catos Method Stock Scanner - Simplified with talib
No pandas-ta complexity - just clean, working code
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
MIN_SCORE = 70
LOOKBACK_DAYS = 250


def get_all_tickers() -> List[str]:
    """Get major tickers to scan"""
    print("Fetching ticker list...")
    
    tickers = set()
    
    try:
        # S&P 500
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers.update(sp500['Symbol'].tolist())
        
        # NASDAQ 100
        nasdaq = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[4]
        tickers.update(nasdaq['Ticker'].tolist())
    except:
        pass
    
    # Add major stocks
    major = ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','AMD','NFLX','INTC']
    tickers.update(major)
    
    cleaned = [t.strip().upper() for t in tickers if t and '^' not in str(t)]
    print(f"Scanning {len(cleaned)} tickers")
    return sorted(list(cleaned))


def download_data_in_chunks(tickers: List[str]) -> Dict:
    """Download in chunks to avoid bans"""
    print(f"\nDownloading data...")
    all_data = {}
    
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        print(f"Chunk {i//CHUNK_SIZE + 1}/{(len(tickers)-1)//CHUNK_SIZE + 1}")
        
        try:
            data = yf.download(chunk, period='1y', group_by='ticker', progress=False, threads=True)
            
            if len(chunk) == 1:
                if not data.empty:
                    all_data[chunk[0]] = data
            else:
                for ticker in chunk:
                    try:
                        if ticker in data.columns.levels[0]:
                            ticker_data = data[ticker]
                            if not ticker_data.empty and len(ticker_data) > 50:
                                all_data[ticker] = ticker_data
                    except:
                        continue
            
            time.sleep(CHUNK_DELAY)
        except Exception as e:
            print(f"Chunk error: {e}")
            continue
    
    print(f"Downloaded {len(all_data)} stocks")
    return all_data


def calculate_catos_score(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    Catos Method scoring using talib
    
    Scoring:
    - SMA Alignment (40 max): Price > 21 > 55 > 233 = 40pts, 21 > 55 > 233 = 30pts, 55 > 233 = 15pts
    - MACD (20 max): Histogram positive & rising
    - DMI (15 max): +DI > -DI
    - StochRSI (25 max): Bullish signals
    """
    if len(df) < 250:
        return 0.0, {"error": "Insufficient data"}
    
    score = 0.0
    details = {}
    
    try:
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        current_price = close[-1]
        
        # 1. SMAs (40 points)
        sma21 = talib.SMA(close, timeperiod=21)[-1]
        sma55 = talib.SMA(close, timeperiod=55)[-1]
        sma233 = talib.SMA(close, timeperiod=233)[-1]
        
        details['price'] = current_price
        details['sma_21'] = sma21
        details['sma_55'] = sma55
        details['sma_233'] = sma233
        
        if current_price > sma21 > sma55 > sma233:
            score += 40
            details['sma_setup'] = "Perfect Stack (Price > 21 > 55 > 233)"
        elif sma21 > sma55 > sma233:
            score += 30
            details['sma_setup'] = "Strong Trend (21 > 55 > 233)"
        elif sma55 > sma233:
            score += 15
            details['sma_setup'] = "Emerging Trend (55 > 233)"
        else:
            details['sma_setup'] = "No alignment"
        
        # 2. MACD (20 points)
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        current_hist = hist[-1]
        prev_hist = hist[-2]
        
        if current_hist > 0 and current_hist > prev_hist:
            score += 20
            details['macd_signal'] = "Bullish (Positive & Rising)"
        else:
            details['macd_signal'] = f"{'Pos' if current_hist > 0 else 'Neg'}, {'Rising' if current_hist > prev_hist else 'Falling'}"
        
        # 3. DMI (15 points)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)[-1]
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)[-1]
        
        if plus_di > minus_di:
            score += 15
            details['dmi_signal'] = f"Bullish (+DI {plus_di:.1f} > -DI {minus_di:.1f})"
        else:
            details['dmi_signal'] = f"Bearish (+DI {plus_di:.1f} < -DI {minus_di:.1f})"
        
        # 4. StochRSI (25 points)
        fastk, fastd = talib.STOCHRSI(close, timeperiod=14, fastk_period=7, fastd_period=5)
        current_k = fastk[-1]
        prev_k = fastk[-2]
        current_d = fastd[-1]
        prev_d = fastd[-2]
        
        details['stochrsi_k'] = current_k
        details['stochrsi_d'] = current_d
        
        # Bullish cross below 80
        if prev_k <= prev_d and current_k > current_d and current_k < 80:
            score += 25
            details['stochrsi_signal'] = "Bullish Cross"
        # Rising momentum above 20
        elif current_k > prev_k and current_d > prev_d and current_k > 20:
            score += 15
            details['stochrsi_signal'] = "Rising Momentum"
        else:
            details['stochrsi_signal'] = f"K={current_k:.1f}, D={current_d:.1f}"
        
    except Exception as e:
        details['error'] = str(e)
        return 0.0, details
    
    return score, details


def get_fundamental_data(ticker: str) -> Dict:
    """Fetch fundamentals"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'peg_ratio': info.get('pegRatio', 'N/A'),
            'profit_margin': info.get('profitMargins', 'N/A'),
        }
    except:
        return {}


def validate_with_gemini(candidates: List[Dict], api_key: str) -> List[Dict]:
    """AI validation with error handling"""
    print(f"\nValidating {len(candidates)} candidates with AI...")
    
    genai.configure(api_key=api_key)
    
    # Try models in order
    model = None
    ai_available = False
    for model_name in ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']:
        try:
            model = genai.GenerativeModel(model_name)
            # Test with actual generation
            test_response = model.generate_content("Say 'OK'")
            if test_response and test_response.text:
                print(f"Using: {model_name}")
                ai_available = True
                break
        except Exception as e:
            print(f"Model {model_name} failed: {str(e)[:50]}")
            continue
    
    if not ai_available:
        print("⚠️ AI unavailable - returning technical results only")
        for candidate in candidates:
            candidate['ai_analysis'] = "AI unavailable on this runner"
            candidate['fundamentals'] = get_fundamental_data(candidate['ticker'])
        return candidates
    
    validated = []
    for candidate in candidates:
        ticker = candidate['ticker']
        fundamentals = get_fundamental_data(ticker)
        
        prompt = f"""Analyze {ticker} for breakout potential:
Score: {candidate['score']:.1f}%
Setup: {candidate['details'].get('sma_setup')}
Sector: {fundamentals.get('sector')}
P/E: {fundamentals.get('pe_ratio')}

Rate 1-10 with brief reasoning."""

        try:
            response = model.generate_content(prompt)
            candidate['ai_analysis'] = response.text
            candidate['fundamentals'] = fundamentals
            validated.append(candidate)
            time.sleep(1)
        except Exception as e:
            if '429' in str(e):
                print("Rate limit - sleeping 60s...")
                time.sleep(60)
                try:
                    response = model.generate_content(prompt)
                    candidate['ai_analysis'] = response.text
                except:
                    candidate['ai_analysis'] = "Rate limited"
            else:
                candidate['ai_analysis'] = f"Error: {str(e)[:50]}"
            
            candidate['fundamentals'] = fundamentals
            validated.append(candidate)
    
    return validated


def send_email_report(results: List[Dict], email_config: Dict):
    """Send email report"""
    print("\nSending email...")
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"Morning Scanner Report - {datetime.now().strftime('%Y-%m-%d')}"
    msg['From'] = email_config['sender']
    msg['To'] = email_config['receiver']
    
    if not results:
        body = f"""<html><body>
<h2>Morning Scanner Report</h2>
<p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
<p>Status: ✅ Scan completed</p>
<p>Results: No stocks met criteria (Score > {MIN_SCORE}%)</p>
</body></html>"""
    else:
        stocks_html = ""
        for r in results:
            f = r.get('fundamentals', {})
            d = r['details']
            stocks_html += f"""
<div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0;">
<h3>{r['ticker']} - {r['score']:.1f}%</h3>
<p><b>Price:</b> ${d.get('price', 0):.2f}</p>
<p><b>Setup:</b> {d.get('sma_setup')}</p>
<p><b>MACD:</b> {d.get('macd_signal')}</p>
<p><b>DMI:</b> {d.get('dmi_signal')}</p>
<p><b>StochRSI:</b> {d.get('stochrsi_signal')}</p>
<hr>
<p><b>Sector:</b> {f.get('sector', 'N/A')}</p>
<p><b>P/E:</b> {f.get('pe_ratio', 'N/A')}</p>
<hr>
<p><b>AI Analysis:</b> {r.get('ai_analysis', 'N/A')}</p>
</div>"""
        
        body = f"""<html><body>
<h2>Morning Scanner Report</h2>
<p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
<p>Matches: {len(results)}</p>
<hr>{stocks_html}</body></html>"""
    
    msg.attach(MIMEText(body, 'html'))
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_config['sender'], email_config['password'])
            smtp.send_message(msg)
        print("✅ Email sent")
    except Exception as e:
        print(f"❌ Email failed: {e}")


def main():
    print("=" * 60)
    print("CATOS METHOD SCANNER")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    
    # Load credentials
    api_key = os.environ.get('GENAI_API_KEY')
    email_config = {
        'sender': os.environ.get('EMAIL_SENDER'),
        'password': os.environ.get('EMAIL_PASSWORD'),
        'receiver': os.environ.get('EMAIL_RECEIVER')
    }
    
    if not all([email_config['sender'], email_config['password'], email_config['receiver']]):
        print("❌ Missing email credentials")
        return
    
    # Get tickers
    tickers = get_all_tickers()
    
    # Download data
    stock_data = download_data_in_chunks(tickers)
    
    # Calculate scores
    print("\nCalculating scores...")
    candidates = []
    
    for ticker, df in stock_data.items():
        try:
            score, details = calculate_catos_score(df)
            if score >= MIN_SCORE:
                print(f"✅ {ticker}: {score:.1f}%")
                candidates.append({'ticker': ticker, 'score': score, 'details': details})
        except Exception as e:
            continue
    
    print(f"\nFound {len(candidates)} candidates")
    
    # AI validation
    if candidates and api_key:
        validated = validate_with_gemini(candidates, api_key)
    else:
        validated = candidates
    
    # Send report
    send_email_report(validated, email_config)
    
    print("\n" + "=" * 60)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
