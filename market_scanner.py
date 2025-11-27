"""
Catos Method Stock Scanner - Based on Actual TradingView Setup
Designed for GitHub Actions with robust error handling
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
import pandas_ta_classic as ta
import traceback

# Configuration
CHUNK_SIZE = 30
CHUNK_DELAY = 1.5
MIN_SCORE = 70
LOOKBACK_DAYS = 250  # ~1 year of data


def get_all_tickers() -> List[str]:
    """
    Get all NYSE and NASDAQ tickers.
    Returns a filtered list of valid stock symbols.
    """
    print("Fetching ticker list...")
    
    # Get tickers from major indices as a robust starting point
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    nasdaq_url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    
    tickers = set()
    
    try:
        # S&P 500
        sp500_table = pd.read_html(sp500_url)[0]
        tickers.update(sp500_table['Symbol'].tolist())
        
        # NASDAQ 100
        nasdaq_table = pd.read_html(nasdaq_url)[4]
        nasdaq_tickers = nasdaq_table['Ticker'].tolist()
        tickers.update(nasdaq_tickers)
        
    except Exception as e:
        print(f"Warning: Could not fetch from Wikipedia: {e}")
    
    # Add additional common tickers
    additional = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
        'V', 'UNH', 'XOM', 'JNJ', 'WMT', 'JPM', 'MA', 'PG', 'AVGO', 'HD',
        'CVX', 'MRK', 'ABBV', 'COST', 'PEP', 'KO', 'LLY', 'ADBE', 'TMO',
        'CSCO', 'MCD', 'ACN', 'NKE', 'DHR', 'ABT', 'VZ', 'TXN', 'ORCL',
        'AMD', 'NFLX', 'INTC', 'QCOM', 'CRM', 'WFC', 'CMCSA', 'INTU',
        'IBM', 'BA', 'CAT', 'GS', 'HON', 'SBUX', 'MMM', 'AXP', 'GE'
    ]
    tickers.update(additional)
    
    # Clean up tickers
    cleaned = []
    for t in tickers:
        t = t.strip().upper()
        if t and not any(char in t for char in ['/', '^', '=']):
            cleaned.append(t)
    
    print(f"Total tickers to scan: {len(cleaned)}")
    return sorted(list(cleaned))


def download_data_in_chunks(tickers: List[str]) -> pd.DataFrame:
    """
    Download stock data in chunks to avoid Yahoo Finance bans.
    Handles MultiIndex columns from yfinance.
    """
    print(f"\nDownloading data in chunks of {CHUNK_SIZE}...")
    all_data = {}
    
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        print(f"Chunk {i//CHUNK_SIZE + 1}/{(len(tickers)-1)//CHUNK_SIZE + 1}: {len(chunk)} tickers")
        
        try:
            data = yf.download(
                chunk,
                period='1y',
                interval='1d',
                group_by='ticker',
                auto_adjust=True,
                progress=False,
                threads=True
            )
            
            # Handle both single and multiple tickers
            if len(chunk) == 1:
                ticker = chunk[0]
                if not data.empty:
                    all_data[ticker] = data
            else:
                # MultiIndex: (Ticker, OHLCV)
                for ticker in chunk:
                    try:
                        if ticker in data.columns.levels[0]:
                            ticker_data = data[ticker]
                            if not ticker_data.empty and len(ticker_data) > 50:
                                all_data[ticker] = ticker_data
                    except (KeyError, AttributeError):
                        continue
            
            time.sleep(CHUNK_DELAY)
            
        except Exception as e:
            print(f"Error downloading chunk: {e}")
            continue
    
    print(f"Successfully downloaded data for {len(all_data)} tickers")
    return all_data


def calculate_catos_score(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    Calculate the Catos Method score based on TradingView setup.
    
    Scoring:
    - SMA Alignment (40 max):
      * Price > SMA21 > SMA55 > SMA233: 40 points (strongest)
      * SMA21 > SMA55 > SMA233: 30 points
      * SMA55 > SMA233: 15 points
    - MACD (20 max): Histogram positive & rising
    - DMI (15 max): +DI > -DI
    - StochRSI (25 max): Bullish signals
    
    Returns: (score, details_dict)
    """
    if len(df) < 250:
        return 0.0, {"error": "Insufficient data"}
    
    score = 0.0
    details = {}
    
    try:
        # Get current values
        close = df['Close'].iloc[-1]
        
        # 1. SMA Alignment (40 points max)
        sma_21 = df['Close'].rolling(21).mean().iloc[-1]
        sma_55 = df['Close'].rolling(55).mean().iloc[-1]
        sma_233 = df['Close'].rolling(233).mean().iloc[-1]
        
        details['price'] = close
        details['sma_21'] = sma_21
        details['sma_55'] = sma_55
        details['sma_233'] = sma_233
        
        # Check alignment - prioritize price above all SMAs
        if close > sma_21 > sma_55 > sma_233:
            score += 40
            details['sma_setup'] = "Perfect Stack (Price > 21 > 55 > 233)"
        elif sma_21 > sma_55 > sma_233:
            score += 30
            details['sma_setup'] = "Strong Trend (21 > 55 > 233)"
        elif sma_55 > sma_233:
            score += 15
            details['sma_setup'] = "Emerging Trend (55 > 233)"
        else:
            details['sma_setup'] = "No trend alignment"
        
        # 2. MACD (20 points)
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd is not None and len(macd) >= 2:
            macd_hist = macd['MACDh_12_26_9']
            current_hist = macd_hist.iloc[-1]
            prev_hist = macd_hist.iloc[-2]
            
            macd_positive = current_hist > 0
            macd_rising = current_hist > prev_hist
            
            if macd_positive and macd_rising:
                score += 20
                details['macd_signal'] = "Bullish (Positive & Rising)"
            else:
                details['macd_signal'] = f"{'Positive' if macd_positive else 'Negative'}, {'Rising' if macd_rising else 'Falling'}"
        
        # 3. DMI (15 points)
        adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx_data is not None:
            plus_di = adx_data['DMP_14'].iloc[-1]
            minus_di = adx_data['DMN_14'].iloc[-1]
            
            if plus_di > minus_di:
                score += 15
                details['dmi_signal'] = f"Bullish (+DI {plus_di:.1f} > -DI {minus_di:.1f})"
            else:
                details['dmi_signal'] = f"Bearish (+DI {plus_di:.1f} < -DI {minus_di:.1f})"
        
        # 4. StochRSI (25 points) - Settings from your chart: (14, 80, 20, with K=7, D=5)
        try:
            stochrsi = ta.stochrsi(df['Close'], length=14, rsi_length=14, k=7, d=5)
            if stochrsi is not None and len(stochrsi) >= 2:
                k_line = stochrsi['STOCHRSIk_14_14_7_5']
                d_line = stochrsi['STOCHRSId_14_14_7_5']
                
                current_k = k_line.iloc[-1]
                prev_k = k_line.iloc[-2]
                current_d = d_line.iloc[-1]
                prev_d = d_line.iloc[-2]
                
                details['stochrsi_k'] = current_k
                details['stochrsi_d'] = current_d
                
                # Bullish cross: K crosses above D below 80
                if prev_k <= prev_d and current_k > current_d and current_k < 80:
                    score += 25
                    details['stochrsi_signal'] = "Bullish Cross (K crossed above D)"
                # Rising momentum above 20
                elif current_k > prev_k and current_d > prev_d and current_k > 20 and current_d > 20:
                    score += 15
                    details['stochrsi_signal'] = "Rising Momentum"
                else:
                    details['stochrsi_signal'] = f"K={current_k:.1f}, D={current_d:.1f}"
        except Exception as e:
            details['stochrsi_signal'] = f"Calculation error: {str(e)[:50]}"
        
    except Exception as e:
        details['error'] = str(e)
        return 0.0, details
    
    return score, details


def get_fundamental_data(ticker: str) -> Dict:
    """
    Fetch fundamental data for AI analysis.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'peg_ratio': info.get('pegRatio', 'N/A'),
            'profit_margin': info.get('profitMargins', 'N/A'),
            'operating_margin': info.get('operatingMargins', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'revenue_growth': info.get('revenueGrowth', 'N/A')
        }
    except:
        return {}


def validate_with_gemini(candidates: List[Dict], api_key: str) -> List[Dict]:
    """
    Validate technical winners using Gemini AI.
    Handles rate limits and model fallback.
    """
    print(f"\nValidating {len(candidates)} candidates with Gemini AI...")
    
    genai.configure(api_key=api_key)
    
    # Try primary model first, fallback to gemini-pro
    models_to_try = ['gemini-1.5-flash', 'gemini-pro']
    model = None
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            # Test the model
            test_response = model.generate_content("test")
            print(f"Using model: {model_name}")
            break
        except Exception as e:
            print(f"Model {model_name} unavailable: {e}")
            continue
    
    if model is None:
        print("Warning: No Gemini model available. Returning candidates without AI validation.")
        return candidates
    
    validated = []
    
    for candidate in candidates:
        ticker = candidate['ticker']
        fundamentals = get_fundamental_data(ticker)
        
        if not fundamentals:
            continue
        
        prompt = f"""Analyze this stock for a breakout or trend continuation trade:

Ticker: {ticker}
Technical Score: {candidate['score']:.1f}%
SMA Setup: {candidate['details'].get('sma_setup', 'N/A')}
Sector: {fundamentals.get('sector')}
Industry: {fundamentals.get('industry')}
P/E Ratio: {fundamentals.get('pe_ratio')}
PEG Ratio: {fundamentals.get('peg_ratio')}
Profit Margin: {fundamentals.get('profit_margin')}

Rate this stock's breakout/continuation potential on a scale of 1-10 and provide 2-3 sentence reasoning focusing on technical momentum and fundamental strength.
Format: Score: X/10 | Reasoning: [your analysis]"""

        try:
            response = model.generate_content(prompt)
            ai_analysis = response.text
            
            candidate['ai_analysis'] = ai_analysis
            candidate['fundamentals'] = fundamentals
            validated.append(candidate)
            
            time.sleep(1)  # Rate limit protection
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle rate limits
            if '429' in error_msg or 'quota' in error_msg.lower():
                print(f"Rate limit hit. Sleeping 60 seconds...")
                time.sleep(60)
                try:
                    response = model.generate_content(prompt)
                    candidate['ai_analysis'] = response.text
                    candidate['fundamentals'] = fundamentals
                    validated.append(candidate)
                except:
                    candidate['ai_analysis'] = "Rate limit exceeded"
                    candidate['fundamentals'] = fundamentals
                    validated.append(candidate)
            else:
                print(f"Error analyzing {ticker}: {e}")
                candidate['ai_analysis'] = f"Analysis failed: {error_msg[:100]}"
                candidate['fundamentals'] = fundamentals
                validated.append(candidate)
    
    return validated


def send_email_report(results: List[Dict], email_config: Dict):
    """
    Send email report via Gmail SMTP.
    Always sends an email (even if no results) to confirm the scan ran.
    """
    print("\nSending email report...")
    
    sender = email_config['sender']
    password = email_config['password']
    receiver = email_config['receiver']
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"Catos Scanner Results - {datetime.now().strftime('%Y-%m-%d')}"
    msg['From'] = sender
    msg['To'] = receiver
    
    # Build HTML report
    if len(results) == 0:
        html_body = f"""
        <html>
        <body>
            <h2>Catos Method Daily Scanner Report</h2>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
            <p><strong>Status:</strong> ✅ Scan completed successfully</p>
            <p><strong>Results:</strong> No stocks met the criteria today (Score > {MIN_SCORE}%)</p>
            <p>The scanner is working correctly. Check back tomorrow!</p>
        </body>
        </html>
        """
    else:
        stocks_html = ""
        for r in results:
            fundamentals = r.get('fundamentals', {})
            details = r['details']
            stocks_html += f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">
                <h3>{r['ticker']} - Score: {r['score']:.1f}%</h3>
                <p><strong>Price:</strong> ${details.get('price', 'N/A'):.2f}</p>
                <p><strong>SMA Setup:</strong> {details.get('sma_setup', 'N/A')}</p>
                <p><strong>MACD:</strong> {details.get('macd_signal', 'N/A')}</p>
                <p><strong>DMI:</strong> {details.get('dmi_signal', 'N/A')}</p>
                <p><strong>StochRSI:</strong> {details.get('stochrsi_signal', 'N/A')}</p>
                <hr>
                <p><strong>Sector:</strong> {fundamentals.get('sector', 'N/A')}</p>
                <p><strong>P/E:</strong> {fundamentals.get('pe_ratio', 'N/A')}</p>
                <p><strong>PEG:</strong> {fundamentals.get('peg_ratio', 'N/A')}</p>
                <hr>
                <p><strong>AI Analysis:</strong></p>
                <p>{r.get('ai_analysis', 'N/A')}</p>
            </div>
            """
        
        html_body = f"""
        <html>
        <body>
            <h2>Catos Method Daily Scanner Report</h2>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
            <p><strong>Matches Found:</strong> {len(results)}</p>
            <hr>
            {stocks_html}
        </body>
        </html>
        """
    
    msg.attach(MIMEText(html_body, 'html'))
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print("✅ Email sent successfully")
    except Exception as e:
        print(f"❌ Email failed: {e}")


def main():
    """
    Main execution flow.
    """
    print("=" * 60)
    print("CATOS METHOD STOCK SCANNER")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    
    # Load credentials
    api_key = os.environ.get('GENAI_API_KEY')
    email_config = {
        'sender': os.environ.get('EMAIL_SENDER'),
        'password': os.environ.get('EMAIL_PASSWORD'),
        'receiver': os.environ.get('EMAIL_RECEIVER')
    }
    
    if not all([api_key, email_config['sender'], email_config['password'], email_config['receiver']]):
        print("❌ Missing required environment variables")
        return
    
    # Step 1: Get tickers
    tickers = get_all_tickers()
    
    # Step 2: Download data
    stock_data = download_data_in_chunks(tickers)
    
    # Step 3: Calculate scores
    print("\nCalculating Catos scores...")
    candidates = []
    
    for ticker, df in stock_data.items():
        try:
            score, details = calculate_catos_score(df)
            
            if score >= MIN_SCORE:
                print(f"✅ {ticker}: {score:.1f}% - {details.get('sma_setup', '')}")
                candidates.append({
                    'ticker': ticker,
                    'score': score,
                    'details': details
                })
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            continue
    
    print(f"\nFound {len(candidates)} candidates with score >= {MIN_SCORE}%")
    
    # Step 4: AI validation
    if len(candidates) > 0 and api_key:
        validated = validate_with_gemini(candidates, api_key)
    else:
        validated = candidates
    
    # Step 5: Send report (always)
    send_email_report(validated, email_config)
    
    print("\n" + "=" * 60)
    print(f"Scan completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
