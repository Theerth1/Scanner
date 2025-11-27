#!/usr/bin/env python3
"""
market_scanner.py

Updated: raises technical threshold to 70, adds Gemini fallback model (gemini-pro),
and implements an AI fail-safe: after 5 consecutive AI failures, stop calling the AI
and send the technical-only report.

Reads secrets from environment:
  - GENAI_API_KEY
  - EMAIL_SENDER
  - EMAIL_PASSWORD
  - EMAIL_RECEIVER

Assumes requirements.txt includes:
  yfinance
  pandas
  pandas_ta
  google-generativeai>=0.8.3
  requests
"""

import os
import time
import datetime
import traceback
import io
import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------------- CONFIG ----------------
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

# Scanner tuning
CHUNK_SIZE = 30
SLEEP_BETWEEN_CHUNKS = 1.5
SLEEP_BETWEEN_AI = 5
TECHNICAL_SCORE_THRESHOLD = 70  # <-- raised from 10 to 70
MIN_DATA_ROWS = 250

# AI behavior
AI_PRIMARY_MODEL = "gemini-1.5-flash"
AI_FALLBACK_MODEL = "gemini-pro"
AI_MAX_RETRIES = 3
AI_MAX_CONSECUTIVE_FAILURES = 5  # stop calling AI after this many consecutive failures

# ---------------- Setup ----------------
if GENAI_API_KEY:
    try:
        genai.configure(api_key=GENAI_API_KEY)
    except Exception as e:
        print(f"⚠️ Warning configuring Gemini client: {e}")

def send_email(subject: str, body: str) -> bool:
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER or "unknown"
    msg['To'] = EMAIL_RECEIVER or "unknown"
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER]):
            raise ValueError("Missing EMAIL_SENDER/EMAIL_PASSWORD/EMAIL_RECEIVER environment variables.")
        server = smtplib.SMTP('smtp.gmail.com', 587, timeout=60)
        server.ehlo()
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("✅ Email sent.")
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False

# ---------------- Utilities ----------------
def get_all_tickers():
    try:
        print("🌍 Fetching full ticker list...")
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        s = requests.get(url, timeout=30).content
        tickers = pd.read_csv(io.StringIO(s.decode('utf-8')), header=None)[0].tolist()
        clean = [t for t in tickers if isinstance(t, str) and "^" not in t and "." not in t]
        print(f"✅ {len(clean)} tickers found.")
        return clean
    except Exception as e:
        print(f"⚠️ Couldn't fetch tickers list: {e}. Using fallback sample.")
        return ['AAPL', 'NVDA', 'AMD', 'TSLA', 'MSFT']

def get_data_bulk(tickers, period="2y", max_retries=3):
    """
    Download bulk data with retries. If chunk downloads always fail, fallback to per-ticker downloads.
    Returns (dataframe, failed_list)
    """
    failed = []
    attempt = 0
    while attempt < max_retries:
        try:
            print(f"   ↳ Downloading chunk size={len(tickers)}, attempt {attempt+1}")
            data = yf.download(tickers, period=period, group_by='ticker', progress=False, threads=True)
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                raise ValueError("Empty result from yfinance")
            return data, failed
        except Exception as e:
            print(f"   ⚠️ Chunk download attempt {attempt+1} failed: {e}")
            attempt += 1
            time.sleep(1 + attempt)
    # fallback to individual downloads
    print("   ↳ Chunk downloads failed after retries; falling back to single-ticker downloads.")
    frames = []
    for t in tickers:
        try:
            single = yf.download(t, period=period, progress=False, threads=False)
            if single is None or single.empty:
                failed.append(t)
                continue
            frames.append((t, single))
            time.sleep(1.0)
        except Exception:
            failed.append(t)
    if not frames:
        return pd.DataFrame(), failed
    assembled = pd.concat([df for (_t, df) in frames], axis=1, keys=[_t for _t, df in frames])
    return assembled, failed

def extract_stock_df_from_bulk(data: pd.DataFrame, ticker: str):
    """
    Robustly extract a per-ticker DataFrame from bulk yfinance download, handling MultiIndex shapes.
    """
    if data is None or data.empty:
        raise KeyError("No data")
    if isinstance(data.columns, pd.MultiIndex):
        # Try ticker in level 0
        if ticker in data.columns.levels[0]:
            return data.xs(ticker, axis=1, level=0, drop_level=True)
        # Try ticker in level 1
        if ticker in data.columns.levels[1]:
            return data.xs(ticker, axis=1, level=1, drop_level=True)
        # Last resort: search levels for str equality
        for lvl in range(len(data.columns.levels)):
            if any(str(x) == ticker for x in data.columns.levels[lvl]):
                return data.xs(ticker, axis=1, level=lvl, drop_level=True)
        raise KeyError(f"{ticker} not found in MultiIndex columns")
    else:
        # single-ticker download
        return data

# ---------------- Strategy / Indicators ----------------
def analyze_ticker(ticker: str, df: pd.DataFrame):
    """
    Returns (score:int, reasons:list). Defensive: returns 0 + reason if any issue.
    """
    try:
        df = df.copy()
        # flatten multiindex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.dropna()
        if len(df) < MIN_DATA_ROWS:
            return 0, [f"Insufficient data (<{MIN_DATA_ROWS})"]

        # ensure required columns
        for col in ['Close', 'High', 'Low', 'Volume']:
            if col not in df.columns:
                return 0, [f"Missing column {col}"]

        last_price = float(df['Close'].iloc[-1])
        if last_price < 1.00:
            return 0, ["Penny stock (<$1)"]

        close = df['Close']
        high = df['High']
        low = df['Low']

        # indicators
        sma_21 = ta.sma(close, length=21)
        sma_55 = ta.sma(close, length=55)
        sma_233 = ta.sma(close, length=233)
        macd = ta.macd(close)
        hist = macd.get('MACDh_12_26_9') if isinstance(macd, dict) else macd['MACDh_12_26_9']
        dmi = ta.adx(high, low, close, length=14)
        pos_di = dmi['DMP_14']
        neg_di = dmi['DMN_14']

        score = 0
        reasons = []

        # Perfect fib trend grants 30 points
        try:
            if last_price > sma_21.iloc[-1] > sma_55.iloc[-1] > sma_233.iloc[-1]:
                score += 30
                reasons.append("Perfect Fib Trend")
            elif last_price > sma_233.iloc[-1]:
                score += 10
                reasons.append("Above 233 SMA")
        except Exception:
            reasons.append("SMA calc issue")

        # MACD rising grants 20
        try:
            if hist is not None and len(hist) >= 2:
                if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]:
                    score += 20
                    reasons.append("MACD Rising")
        except Exception:
            reasons.append("MACD calc issue")

        # Positive DMI grants 20
        try:
            if pos_di.iloc[-1] > neg_di.iloc[-1]:
                score += 20
                reasons.append("Positive DMI")
        except Exception:
            reasons.append("DMI calc issue")

        print(f"🔍 {ticker} Score: {score} | Price: {last_price:.2f} | Reasons: {reasons}")
        return score, reasons
    except Exception as e:
        print(f"❌ Analyzer crash for {ticker}: {e}\n{traceback.format_exc()}")
        return 0, [f"Analyzer crash: {e}"]

# ---------------- Gemini / AI ----------------
def get_gemini_analysis(ticker, active_model=AI_PRIMARY_MODEL):
    """
    Attempts to obtain a short Gemini analysis for the ticker.
    Will try 'active_model' (string). Caller should implement retries and fallback logic.
    Returns (success:bool, text:str)
    """
    if not GENAI_API_KEY:
        return False, "Gemini API key missing; skipping AI analysis."

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        def g(k): return info.get(k, 'N/A')
        fund_data = {
            "Symbol": ticker,
            "Sector": g('sector'),
            "Forward PE": g('forwardPE'),
            "PEG Ratio": g('pegRatio'),
            "Profit Margins": g('profitMargins'),
            "Revenue Growth": g('revenueGrowth'),
            "Target Price": g('targetMeanPrice'),
            "Current Price": g('currentPrice')
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

        model = genai.GenerativeModel(active_model)
        response = model.generate_content(prompt)
        text = response.text.strip() if hasattr(response, "text") else str(response).strip()
        return True, text
    except Exception as e:
        return False, f"AI error with model {active_model}: {e}"

# ---------------- Main ----------------
def run_scanner():
    started_at = datetime.datetime.utcnow().isoformat() + "Z"
    print(f"🚀 Market scan starting at {started_at}")

    tickers = get_all_tickers()
    total = len(tickers)
    high_conviction = []
    failed_downloads = set()
    failed_analysis = set()

    # AI state
    consecutive_ai_failures = 0
    ai_disabled = False

    print(f"📊 Scanning {total} tickers in chunks of {CHUNK_SIZE} (threshold={TECHNICAL_SCORE_THRESHOLD})")
    for i in range(0, total, CHUNK_SIZE):
        chunk = tickers[i:i+CHUNK_SIZE]
        print(f"\n--- chunk {i//CHUNK_SIZE + 1} / {((total-1)//CHUNK_SIZE)+1} (size {len(chunk)}) ---")
        data, failed = get_data_bulk(chunk)
        failed_downloads.update(failed)
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            print("   ⚠️ Empty chunk; marking chunk tickers as failed and continuing.")
            failed_downloads.update(chunk)
            time.sleep(SLEEP_BETWEEN_CHUNKS)
            continue

        for ticker in chunk:
            try:
                try:
                    stock_df = extract_stock_df_from_bulk(data, ticker)
                except KeyError as e:
                    failed_downloads.add(ticker)
                    print(f"   ⚠️ {ticker} missing in data: {e}")
                    continue

                score, reasons = analyze_ticker(ticker, stock_df)
                if score >= TECHNICAL_SCORE_THRESHOLD:
                    price = 'N/A'
                    try:
                        price = float(stock_df['Close'].iloc[-1])
                    except Exception:
                        pass
                    high_conviction.append({
                        "ticker": ticker,
                        "score": score,
                        "price": price,
                        "reasons": reasons
                    })
            except Exception as e:
                failed_analysis.add(ticker)
                print(f"   ❌ Error while processing {ticker}: {e}\n{traceback.format_exc()}")
                continue

        time.sleep(SLEEP_BETWEEN_CHUNKS)

    match_count = len(high_conviction)
    print(f"\n🎯 Scan complete. {match_count} stocks met the threshold >= {TECHNICAL_SCORE_THRESHOLD}.")

    # Sort by score desc
    high_conviction.sort(key=lambda x: x['score'], reverse=True)

    # Build email body
    body = []
    body.append(f"HIGH CONVICTION REPORT: {datetime.date.today()}")
    body.append(f"Started: {started_at} (UTC)")
    body.append(f"Scanned {total} tickers in chunks of {CHUNK_SIZE}")
    body.append(f"Threshold: {TECHNICAL_SCORE_THRESHOLD}")
    body.append(f"Matches found: {match_count}")
    body.append("\n========================================\n")

    # If there are matches and AI is enabled, attempt Gemini analysis with fallback
    ai_used = False
    if match_count > 0 and not ai_disabled:
        for idx, s in enumerate(high_conviction, start=1):
            ticker = s['ticker']
            body.append(f"{idx}. {ticker} (Score: {s['score']})")
            body.append(f"   Price: {s['price']}")
            body.append(f"   Signals: {', '.join(s['reasons'])}")

            # If AI has been disabled due to many failures, skip
            if consecutive_ai_failures >= AI_MAX_CONSECUTIVE_FAILURES:
                ai_disabled = True
                body.append("   AI disabled due to repeated failures. Skipping Gemini analysis.\n")
                continue

            # Try primary model, then fallback model if primary fails
            success = False
            ai_text = "AI not configured."
            for model_name in (AI_PRIMARY_MODEL, AI_FALLBACK_MODEL):
                for attempt in range(AI_MAX_RETRIES):
                    ok, txt = get_gemini_analysis(ticker, active_model=model_name)
                    if ok:
                        success = True
                        ai_text = txt
                        break
                    else:
                        print(f"   ⚠️ Gemini {model_name} attempt {attempt+1} failed for {ticker}: {txt}")
                        time.sleep(1 + attempt)
                if success:
                    ai_used = True
                    break
                else:
                    print(f"   ↳ Model {model_name} failed for {ticker}; trying next model if available.")

            if not success:
                consecutive_ai_failures += 1
                body.append(f"   🧠 Gemini analysis failed for {ticker} (consecutive AI failures: {consecutive_ai_failures}).\n")
                if consecutive_ai_failures >= AI_MAX_CONSECUTIVE_FAILURES:
                    body.append("   🛑 Reached maximum consecutive AI failures; future tickers will skip AI.\n")
            else:
                consecutive_ai_failures = 0
                body.append("   🧠 Gemini Verdict:\n")
                # indent AI text
                for line in ai_text.splitlines():
                    body.append(f"      {line}")
                body.append("")

            time.sleep(SLEEP_BETWEEN_AI)

    # If no matches or AI disabled/skipped, still show technical list
    if match_count == 0:
        body.append("No matches passed the technical threshold today.\n")

    # Diagnostics
    body.append("\n--- DIAGNOSTICS ---")
    body.append(f"Failed downloads / missing tickers: {len(failed_downloads)}")
    if failed_downloads:
        body.append(", ".join(sorted(list(failed_downloads))[:200]) + ("" if len(failed_downloads) <= 200 else f"... (+{len(failed_downloads)-200} more)"))
    body.append(f"Tickers that crashed during analysis: {len(failed_analysis)}")
    if failed_analysis:
        body.append(", ".join(sorted(list(failed_analysis))[:200]) + ("" if len(failed_analysis) <= 200 else f"... (+{len(failed_analysis)-200} more)"))
    body.append(f"AI used: {'Yes' if ai_used else 'No'}")
    body.append(f"AI disabled due to consecutive failures: {'Yes' if consecutive_ai_failures >= AI_MAX_CONSECUTIVE_FAILURES else 'No'}")

    final_body = "\n".join(body)
    subject = f"🚀 {match_count} High-Conviction Breakouts (>= {TECHNICAL_SCORE_THRESHOLD})" if match_count else f"⚠️ MARKET SCAN: 0 Matches ({datetime.date.today()})"

    # Send final report
    sent = send_email(subject, final_body)
    if not sent:
        print("❌ Final report failed to send; please check EMAIL_* secrets and Action logs.")
    else:
        print("✅ Final report sent (or at least an attempt was made).")

if __name__ == "__main__":
    run_scanner()
