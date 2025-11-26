#!/usr/bin/env python3
"""
market_scanner.py
Improved version: safer secrets, startup email, robust yfinance handling, chunking/retries,
and guaranteed email on no matches. Designed to run from GitHub Actions (cron).
"""

import os
import time
import datetime
import requests
import io
import traceback
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ------------- CONFIG (read from env / secrets) -------------
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

# Basic checks
missing = []
for name, val in [
    ("GENAI_API_KEY", GENAI_API_KEY),
    ("EMAIL_SENDER", EMAIL_SENDER),
    ("EMAIL_PASSWORD", EMAIL_PASSWORD),
    ("EMAIL_RECEIVER", EMAIL_RECEIVER),
]:
    if not val:
        missing.append(name)

if missing:
    print(f"❌ Missing environment variables: {missing}")
    # still continue so the startup email attempt will show failure if SMTP creds missing
else:
    print("✅ All required env vars appear present.")

# Configure Gemini (if key present)
try:
    if GENAI_API_KEY:
        genai.configure(api_key=GENAI_API_KEY)
except Exception as e:
    print(f"⚠️ Warning: couldn't configure Gemini: {e}")

# ------------- Utilities -------------
def send_email(subject: str, body: str):
    """
    Sends an email via Gmail SMTP. Returns True on success, False on failure.
    """
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER or "unknown"
    msg['To'] = EMAIL_RECEIVER or "unknown"
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
            raise ValueError("Missing email configuration (sender/password/receiver).")
        server = smtplib.SMTP('smtp.gmail.com', 587, timeout=60)
        server.ehlo()
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("✅ Email Sent.")
        return True
    except Exception as e:
        print(f"❌ Email Failed: {e}")
        return False

# ------------- 1. Get Tickers -------------
def get_all_tickers():
    print("🌍 Fetching full market ticker list...")
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        s = requests.get(url, timeout=30).content
        tickers = pd.read_csv(io.StringIO(s.decode('utf-8')), header=None)[0].tolist()
        # filter out test tickers, preferreds, and anything with '^' or '.'
        clean_tickers = [x for x in tickers if isinstance(x, str) and "^" not in x and "." not in x]
        print(f"✅ Found {len(clean_tickers)} tickers.")
        return clean_tickers
    except Exception as e:
        print(f"⚠️ Failed to fetch tickers list: {e}. Falling back to sample list.")
        return ['AAPL', 'NVDA', 'AMD', 'TSLA', 'MSFT']

# ------------- 2. Bulk downloader with retries and fallbacks -------------
def get_data_bulk(tickers, period="2y", max_retries=3):
    """
    Attempts to download historical data for a list of tickers.
    On persistent failure of the whole chunk, attempts per-ticker fetch (slower).
    Returns a tuple (dataframe, failed_tickers_list).
    """
    failed = []
    try:
        attempt = 0
        while attempt < max_retries:
            try:
                print(f"   ↳ Downloading chunk (size={len(tickers)}), attempt {attempt+1}")
                data = yf.download(tickers, period=period, group_by='ticker', progress=False, threads=True)
                if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                    # Sometimes yfinance returns empty; treat as failure and retry
                    raise ValueError("Empty dataframe returned")
                return data, failed
            except Exception as e:
                print(f"   ⚠️ Chunk download attempt {attempt+1} failed: {e}")
                attempt += 1
                time.sleep(2 + attempt)  # incremental backoff
        # If chunk still fails, fall back to individual fetches
        print("   ↳ Chunk downloads failed after retries; falling back to per-ticker download.")
        all_frames = []
        for t in tickers:
            try:
                single = yf.download(t, period=period, progress=False, threads=False)
                if single is None or single.empty:
                    failed.append(t)
                    continue
                # rename columns to MultiIndex similar shape for downstream logic, but easier to store as dict
                # We'll create a panel-like structure by prefixing columns with ticker when needed.
                # For simplicity, store single-frame keyed externally; caller will detect non-multiindex case.
                all_frames.append((t, single))
                time.sleep(1.0)  # be gentle on YF
            except Exception as e:
                failed.append(t)
        if not all_frames:
            return pd.DataFrame(), failed
        # Convert list of (ticker, df) into a multiindex-like DataFrame where necessary
        # We'll use pd.concat with keys -> results in MultiIndex columns (ticker, field)
        assembled = pd.concat([df.rename(columns=lambda c: c) for (_t, df) in all_frames], axis=1, keys=[_t for _t, df in all_frames])
        return assembled, failed
    except Exception as e:
        print(f"   ❌ Unexpected downloader error: {e}")
        return pd.DataFrame(), tickers  # everything failed

# ------------- 3. Strategy Engine (analyze ticker) -------------
def extract_stock_df_from_bulk(data: pd.DataFrame, ticker: str):
    """
    Given the bulk yfinance output and a ticker, try robust extraction for:
      - MultiIndex with ticker at level 0 or level 1
      - Single-level DataFrame (single ticker download)
    Returns a DataFrame for that ticker or raises KeyError.
    """
    if data is None or data.empty:
        raise KeyError(f"No data available for {ticker}")

    # If multiindex columns:
    if isinstance(data.columns, pd.MultiIndex):
        # Find which level contains the ticker
        # Two common shapes:
        # - level 0: ticker, level 1: fields (('AAPL','Close'))
        # - level 1: ticker, level 0: fields (('Close','AAPL'))
        levels = [list(level) for level in data.columns.levels]
        # Try common arrangement: level 0 == tickers
        if ticker in levels[0]:
            try:
                df = data.xs(ticker, axis=1, level=0, drop_level=True)
                return df
            except Exception:
                pass
        # Try ticker in level 1
        if ticker in levels[1]:
            try:
                df = data.xs(ticker, axis=1, level=1, drop_level=True)
                return df
            except Exception:
                pass
        # If not found, try to detect where ticker text appears
        for lvl in range(len(levels)):
            if any(str(x) == ticker for x in levels[lvl]):
                df = data.xs(ticker, axis=1, level=lvl, drop_level=True)
                return df
        # didn't find it
        raise KeyError(f"Ticker {ticker} not present in MultiIndex columns.")
    else:
        # single dataframe (either single-ticker download or one-field)
        return data

def analyze_ticker(ticker: str, df: pd.DataFrame):
    """
    Returns (score:int, reasons:list)
    Keep analysis defensive and log errors for GitHub logs.
    """
    try:
        # Defensive copy
        df = df.copy()
        # Make sure columns are simple names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        # Remove NA rows
        df = df.dropna()
        if len(df) < 250:
            return 0, ["Insufficient data (<250 rows)"]

        # Ensure existence of columns
        for needed in ['Close', 'High', 'Low', 'Volume']:
            if needed not in df.columns:
                return 0, [f"Missing column {needed}"]

        last_price = float(df['Close'].iloc[-1])
        if last_price < 1.00:
            return 0, ["Penny stock (<$1)"]

        close = df['Close']
        high = df['High']
        low = df['Low']

        # Indicators
        sma_21 = ta.sma(close, length=21)
        sma_55 = ta.sma(close, length=55)
        sma_233 = ta.sma(close, length=233)
        macd = ta.macd(close)
        hist = macd.get('MACDh_12_26_9') if isinstance(macd, dict) else macd['MACDh_12_26_9']
        dmi = ta.adx(high, low, close, length=14)
        adx = dmi['ADX_14']
        pos_di = dmi['DMP_14']
        neg_di = dmi['DMN_14']

        # Score
        score = 0
        reasons = []

        try:
            if last_price > sma_21.iloc[-1] > sma_55.iloc[-1] > sma_233.iloc[-1]:
                score += 30
                reasons.append("Perfect Fib Trend")
            elif last_price > sma_233.iloc[-1]:
                score += 10
                reasons.append("Above 233 SMA")
        except Exception:
            reasons.append("SMA calc issue")

        try:
            if hist is not None and len(hist) >= 2:
                if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]:
                    score += 20
                    reasons.append("MACD Rising")
        except Exception:
            reasons.append("MACD calc issue")

        try:
            if pos_di.iloc[-1] > neg_di.iloc[-1]:
                score += 20
                reasons.append("Positive DMI")
        except Exception:
            reasons.append("DMI calc issue")

        print(f"🔍 {ticker} Score: {score} | Price: {last_price:.2f} | Reasons: {reasons}")
        return score, reasons
    except Exception as e:
        print(f"❌ CRASH on {ticker}: {e}\n{traceback.format_exc()}")
        return 0, [f"Analyzer crashed: {e}"]

# ------------- 4. Gemini fundamental analysis -------------
def get_gemini_analysis(ticker):
    """
    Returns the string response from Gemini or an error message.
    Keeps retry logic and defensive behavior.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        def get_val(k): return info.get(k, 'N/A')

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
        if not GENAI_API_KEY:
            return "Gemini API key missing; skipping AI analysis."

        model = genai.GenerativeModel('gemini-1.5-flash')
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"   ⚠️ Gemini attempt {attempt+1} failed: {e}")
                time.sleep(2)
        return "AI Analysis Failed after 3 retries."
    except Exception as e:
        return f"AI Analysis Failed: {e}"

# ------------- 5. Main Execution -------------
def run_scanner():
    started_at = datetime.datetime.utcnow().isoformat() + "Z"
    print(f"🚀 Starting market scan at {started_at}")
    # 1) Send immediate startup email to verify SMTP and runner
    startup_subject = f"[Scanner Startup] {datetime.datetime.utcnow().date()} - Market Scanner starting"
    startup_body = f"Market scanner started at UTC {started_at}.\nIf you see this email, SMTP worked from this runner.\n\nHostname/Runner logs are available in GitHub Actions logs."
    send_email(startup_subject, startup_body)

    all_tickers = get_all_tickers()
    # For initial testing, you may restrict tickers_to_scan to a smaller slice,
    # then uncomment full list after you confirm startup works.
    tickers_to_scan = all_tickers  # change if you want a subset for testing

    # Tune these for rate limiting
    chunk_size = 30             # reduce from 100 to avoid Yahoo rate limit
    sleep_between_chunks = 1.5  # seconds to sleep between chunk downloads
    sleep_between_ai = 5        # sleep between Gemini calls to be gentle
    high_conviction_list = []
    failed_downloads = set()
    failed_analysis = set()

    total = len(tickers_to_scan)
    print(f"📊 Scanning {total} stocks in chunks of {chunk_size}...")

    # iterate chunks
    for i in range(0, total, chunk_size):
        chunk = tickers_to_scan[i:i+chunk_size]
        print(f"\n--- Processing chunk {i//chunk_size + 1} / {((total-1)//chunk_size)+1} (size {len(chunk)}) ---")
        data, failed = get_data_bulk(chunk)
        # record any failed tickers
        failed_downloads.update(failed)
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            print("   ⚠️ Chunk returned no data; marking all chunk tickers as failed and continuing.")
            failed_downloads.update(chunk)
            time.sleep(sleep_between_chunks)
            continue

        # analyze each ticker in chunk
        for ticker in chunk:
            try:
                try:
                    stock_df = extract_stock_df_from_bulk(data, ticker)
                except KeyError as e:
                    # not present in bulk result
                    failed_downloads.add(ticker)
                    print(f"   ⚠️ {ticker} missing from bulk data: {e}")
                    continue

                score, reasons = analyze_ticker(ticker, stock_df)
                if score >= 10:
                    price = None
                    try:
                        price = float(stock_df['Close'].iloc[-1])
                    except Exception:
                        price = 'N/A'
                    high_conviction_list.append({
                        "ticker": ticker,
                        "score": score,
                        "price": price,
                        "reasons": reasons
                    })
            except Exception as e:
                print(f"   ❌ Unexpected error while processing {ticker}: {e}\n{traceback.format_exc()}")
                failed_analysis.add(ticker)
                continue

        # be gentle to avoid rate limit
        time.sleep(sleep_between_chunks)

    match_count = len(high_conviction_list)
    print(f"\n🎯 Technical Scan Complete. Found {match_count} stocks with Score >= 10%.")

    # sort results
    high_conviction_list.sort(key=lambda x: x['score'], reverse=True)

    # Build email body
    email_body = f"☀️ HIGH CONVICTION REPORT: {datetime.date.today()}\n"
    email_body += f"Started at (UTC): {started_at}\n"
    email_body += f"Scanned {total} tickers in chunks of {chunk_size}.\n"
    email_body += f"Found {match_count} stocks with Technical Score >= 10%\n"
    email_body += "\n========================================\n\n"

    if match_count == 0:
        email_body += "No matches found today.\n\n"

    # Gemini analysis for each match
    for i, stock in enumerate(high_conviction_list):
        print(f"   ({i+1}/{match_count}) Analyzing {stock['ticker']} fundamentals via Gemini...")
        ai_verdict = get_gemini_analysis(stock['ticker'])
        email_body += f"🚀 {stock['ticker']} (Score: {stock['score']})\n"
        email_body += f"   Price: {stock['price']}\n"
        email_body += f"   Signals: {', '.join(stock['reasons'])}\n"
        email_body += f"   🧠 Gemini Verdict:\n   {ai_verdict}\n"
        email_body += "----------------------------------------\n\n"
        time.sleep(sleep_between_ai)

    # Append diagnostics
    email_body += "\n\nDIAGNOSTICS / FAILED ITEMS\n"
    email_body += "-------------------------\n"
    email_body += f"Failed chunk downloads or missing tickers: {len(failed_downloads)}\n"
    if failed_downloads:
        email_body += ", ".join(sorted(list(failed_downloads))[:200])  # cap long list
        if len(failed_downloads) > 200:
            email_body += f"... (+{len(failed_downloads)-200} more)\n"
    email_body += "\n\nTickers that crashed during analysis: {}\n".format(len(failed_analysis))
    if failed_analysis:
        email_body += ", ".join(sorted(list(failed_analysis))[:200])
        if len(failed_analysis) > 200:
            email_body += f"... (+{len(failed_analysis)-200} more)\n"

    # Subject mirrors matches
    subject = f"🚀 {match_count} High-Conviction Breakouts (>=10%)" if match_count else f"⚠️ MARKET SCAN: 0 Matches Found ({datetime.date.today()})"

    # final send
    send_ok = send_email(subject, email_body)
    if not send_ok:
        print("❌ Final report failed to send; check EMAIL credentials and logs.")

if __name__ == "__main__":
    run_scanner()
