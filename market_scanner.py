import os
import time
import smtplib
import yfinance as yf
import pandas as pd
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from google import genai
from google.genai import types
from typing import List, Dict, Tuple
import talib
 
# Configuration
PREFILTER_CHUNK_SIZE = 100   # cheap 5-day pass can use big chunks
CHUNK_SIZE = 30              # full 1y history pass
CHUNK_DELAY = 1.5
MIN_SCORE = 100
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 5_000_000  # 20-day avg dollar volume floor (liquidity filter)
 
# Gemini
GEMINI_MODEL = "gemini-2.5-flash"  # pinned — no list_models() roulette
GEMINI_DELAY = 6                   # free tier ~10 RPM, 6s keeps us safely under
 
 
def get_all_tickers() -> List[str]:
    print("Fetching complete US stock list...")
    try:
        import requests
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            tickers_raw = response.text.strip().split('\n')
            cleaned = [t.strip().upper() for t in tickers_raw if t and not any(c in t for c in ['^', '.', '/', '='])]
            print(f"✅ Loaded {len(cleaned)} tickers")
            return sorted(list(set(cleaned)))
    except Exception as e:
        print(f"⚠️ Failed to fetch list: {e}. Using fallback.")
        return ['AAPL', 'NVDA', 'AMD', 'TSLA', 'MSFT']  # Fallback
 
 
def prefilter_tickers(tickers: List[str]) -> List[str]:
    """
    PASS 1 (cheap): download only 5 days of data in large chunks and throw out
    anything that fails the price floor or the average-dollar-volume liquidity
    floor. This typically cuts ~7,000 tickers down to a few hundred, so the
    expensive 1-year download only runs on names we might actually trade.
    """
    print(f"\n[Pass 1] Liquidity prefilter on {len(tickers)} tickers...")
    survivors = []
 
    for i in range(0, len(tickers), PREFILTER_CHUNK_SIZE):
        chunk = tickers[i:i + PREFILTER_CHUNK_SIZE]
        try:
            data = yf.download(chunk, period='5d', group_by='ticker',
                               auto_adjust=True, progress=False, threads=True)
            if data.empty:
                time.sleep(CHUNK_DELAY)
                continue
 
            for ticker in chunk:
                try:
                    td = data[ticker] if len(chunk) > 1 else data
                    td = td.dropna()
                    if td.empty:
                        continue
                    last_close = td['Close'].iloc[-1]
                    avg_dollar_vol = (td['Close'] * td['Volume']).mean()
                    if last_close >= MIN_PRICE and avg_dollar_vol >= MIN_DOLLAR_VOLUME:
                        survivors.append(ticker)
                except Exception:
                    continue
            time.sleep(CHUNK_DELAY)
        except Exception:
            time.sleep(CHUNK_DELAY * 2)
            continue
 
    print(f"✅ {len(survivors)} tickers passed price/liquidity filters")
    return survivors
 
 
def download_data_in_chunks(tickers: List[str]) -> Dict:
    """PASS 2: full 1-year history, but only for prefilter survivors."""
    print(f"\n[Pass 2] Downloading 1y history for {len(tickers)} tickers...")
    all_data = {}
 
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        try:
            # Auto-adjust=True is CRITICAL for correct indicator calc
            data = yf.download(chunk, period='1y', group_by='ticker',
                               auto_adjust=True, progress=False, threads=True)
 
            if len(chunk) == 1:
                td = data.dropna()
                if not td.empty and len(td) >= 250:
                    all_data[chunk[0]] = td
            else:
                for ticker in chunk:
                    try:
                        if ticker in data.columns.levels[0]:
                            td = data[ticker].dropna()  # dropna kills all-NaN frames from failed tickers
                            if not td.empty and len(td) >= 250:
                                all_data[ticker] = td
                    except Exception:
                        continue
            time.sleep(CHUNK_DELAY)
        except Exception:
            time.sleep(CHUNK_DELAY * 2)
            continue
 
    return all_data
 
 
# --- STRATEGY ENGINE (Continuation / Breakout) ---
def calculate_catos_score(df: pd.DataFrame) -> Tuple[float, Dict]:
    if len(df) < 250:
        return 0.0, {}
 
    try:
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
 
        if close[-1] < MIN_PRICE:
            return 0.0, {}
 
        # LIQUIDITY GATE (belt-and-suspenders; prefilter already applied it)
        avg_dollar_vol = np.mean(close[-20:] * volume[-20:])
        if avg_dollar_vol < MIN_DOLLAR_VOLUME:
            return 0.0, {}
 
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
 
        details['dmi_desc'] = f"{'Bullish' if plus_di > minus_di else 'Bearish'} (+DI {plus_di:.1f} / -DI {minus_di:.1f})"
 
        if plus_di > minus_di and adx > 20:
            score += 20
        elif plus_di > minus_di:
            score += 10
 
        # 4. STOCH RSI
        fastk, fastd = talib.STOCHRSI(close, timeperiod=14, fastk_period=14, fastd_period=3, fastd_matype=0)
        k = fastk[-1]
        d = fastd[-1]
 
        details['stoch_desc'] = f"K={k:.1f}, D={d:.1f}"
 
        if k > d:
            if adx > 30:
                if k > 50:
                    score += 20
                else:
                    score += 15
            else:
                score += 5
 
        # 5. VOLUME / EXHAUSTION CHECK
        vol_sma = talib.SMA(volume.astype(float), timeperiod=20)[-1]
        rel_vol = volume[-1] / vol_sma if vol_sma > 0 else 0.0
        details['vol_desc'] = f"{rel_vol:.1f}x Avg"
 
        if rel_vol > 1.2:  # 20% above average
            score += 10
            details['vol_desc'] += " (Strong)"
        elif rel_vol < 0.7:  # 30% below average
            score -= 15  # PENALIZED now, not just labeled — dead volume near highs is the failed-breakout signature
            details['vol_desc'] += " (Exhaustion Risk, -15)"
        else:
            details['vol_desc'] += " (Normal)"
 
        # 6. BREAKOUT — only rewarded if volume confirms (rel_vol >= 1.0)
        high_20 = np.max(high[-21:-1])
        if close[-1] > high_20:
            if rel_vol >= 1.0:
                score += 10
                details['breakout'] = "YES (New 20-Day High, volume-confirmed)"
            else:
                details['breakout'] = "YES but UNCONFIRMED (below-avg volume, no points)"
        else:
            details['breakout'] = "No"
 
        return score, details
 
    except Exception as e:
        return 0.0, {'error': str(e)}
 
 
# --- AI VALIDATION (google-genai SDK + Google Search grounding) ---
def _fmt(val, pct=False):
    """Format yfinance info fields safely."""
    if val is None or val == "N/A":
        return "N/A"
    try:
        return f"{val * 100:.1f}%" if pct else f"{val:,.2f}"
    except (TypeError, ValueError):
        return str(val)
 
 
def validate_with_gemini(candidates: List[Dict], api_key: str) -> List[Dict]:
    print(f"\nValidating top {len(candidates)} candidates with AI...")
 
    client = None
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            print(f"✅ AI Ready: {GEMINI_MODEL} (with Google Search grounding)")
        except Exception as e:
            print(f"❌ AI Connection Failed: {e}")
 
    # This tools config is what gives the model live internet access
    grounding_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
 
    validated = []
    for i, c in enumerate(candidates):
        ticker = c['ticker']
        print(f"[{i+1}/{len(candidates)}] Processing {ticker}...")
 
        # 1. Fetch fundamentals — and actually feed them to the model this time
        info = {}
        try:
            info = yf.Ticker(ticker).info
        except Exception as e:
            print(f"  ⚠️ yfinance info failed for {ticker}: {e}")
 
        c['details']['sector'] = info.get('sector', 'N/A')
        c['details']['pe'] = info.get('trailingPE', 'N/A')
 
        fundamentals_block = (
            f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}\n"
            f"Market Cap: {_fmt(info.get('marketCap'))}\n"
            f"Trailing P/E: {_fmt(info.get('trailingPE'))} | Forward P/E: {_fmt(info.get('forwardPE'))}\n"
            f"Revenue Growth (YoY): {_fmt(info.get('revenueGrowth'), pct=True)} | "
            f"Earnings Growth: {_fmt(info.get('earningsGrowth'), pct=True)}\n"
            f"Profit Margin: {_fmt(info.get('profitMargins'), pct=True)} | "
            f"Operating Margin: {_fmt(info.get('operatingMargins'), pct=True)}\n"
            f"Debt/Equity: {_fmt(info.get('debtToEquity'))} | "
            f"Free Cash Flow: {_fmt(info.get('freeCashflow'))}\n"
            f"Short % of Float: {_fmt(info.get('shortPercentOfFloat'), pct=True)}"
        )
 
        # 2. Run AI (grounded)
        if client:
            prompt = (
                f"You are validating a technical breakout screen for {ticker} "
                f"({info.get('longName', ticker)}).\n\n"
                f"TECHNICAL CONTEXT (already computed, do not re-derive):\n"
                f"Score: {c['score']}/120 | Trend: {c['details'].get('trend', 'N/A')} | "
                f"MACD: {c['details'].get('macd_desc', 'N/A')} | "
                f"Volume: {c['details'].get('vol_desc', 'N/A')} | "
                f"Breakout: {c['details'].get('breakout', 'N/A')}\n\n"
                f"FUNDAMENTAL SNAPSHOT (from Yahoo Finance):\n{fundamentals_block}\n\n"
                f"TASK: Use Google Search to check for recent news, earnings results/guidance, "
                f"upcoming earnings dates, and any catalysts or red flags for {ticker} from the "
                f"last 30 days. Combine that with the fundamental snapshot above to form a "
                f"fundamental verdict. Then reconcile it against the technical setup and give a "
                f"final call: BULLISH BUY, DO NOT BUY, or CAUTIOUS BUY. "
                f"Keep the entire response to 3-4 sentences. Flag explicitly if earnings are "
                f"within the next 5 trading days."
            )
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=grounding_config,
                )
                c['ai_analysis'] = (response.text or "").strip() or "Empty response"
            except Exception as e:
                print(f"  ❌ Gemini error for {ticker}: {e}")
                c['ai_analysis'] = "AI validation unavailable (error logged in CI output)"
            time.sleep(GEMINI_DELAY)  # Rate limit safety (~10 RPM free tier)
        else:
            c['ai_analysis'] = "AI Not Configured"
 
        validated.append(c)
 
    return validated
 
 
# --- REPORTING ---
def send_email_report(results: List[Dict], email_config: Dict):
    print("\nSending email...")
    msg = MIMEMultipart('alternative')
 
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
                <h3 style="margin-top: 0; color: #0056b3;">{r['ticker']} - {r['score']:.1f}</h3>
 
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td style="padding: 3px 0;"><b>Price:</b></td><td>${d.get('price', 0):.2f}</td></tr>
                    <tr><td style="padding: 3px 0;"><b>Setup:</b></td><td>{d.get('trend', 'N/A')}</td></tr>
                    <tr><td style="padding: 3px 0;"><b>Volume:</b></td><td>{d.get('vol_desc', 'N/A')}</td></tr>
                    <tr><td style="padding: 3px 0;"><b>Breakout:</b></td><td>{d.get('breakout', 'N/A')}</td></tr>
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
 
    # 2. Scan (two-pass: cheap liquidity prefilter, then full history)
    tickers = get_all_tickers()
    liquid_tickers = prefilter_tickers(tickers)
    data = download_data_in_chunks(liquid_tickers)
 
    candidates = []
    for ticker, df in data.items():
        score, details = calculate_catos_score(df)
        if score >= MIN_SCORE:
            print(f"⭐ Match: {ticker} ({score})")
            candidates.append({'ticker': ticker, 'score': score, 'details': details})
 
    # Sort by Score (Highest first) and keep only Top 20
    candidates.sort(key=lambda x: x['score'], reverse=True)
    final_candidates = candidates[:20]
    print(f"\nSending top {len(final_candidates)} of {len(candidates)} matches to AI...")
 
    # 3. Validate & Send
    final_list = validate_with_gemini(final_candidates, api_key)
    send_email_report(final_list, email_config)
 
 
if __name__ == "__main__":
    main()
