#!/usr/bin/env python3
"""
market_scanner.py

Features:
- SMA-only stacking logic (SMA21 > SMA55 > SMA233 or SMA55 > SMA233)
- Robust yfinance chunking + retries + per-ticker fallback
- Defensive Gemini usage: detects model-not-found 404 and disables AI globally
- Limits number of Gemini analyses per run (AI_ANALYSIS_CAP)
- Sends a final email report (always)
"""

import os
import time
import datetime
import traceback
import io
import requests
import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
import google.generativeai as genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------------- CONFIG (env / tuning) ----------------
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 30))
SLEEP_BETWEEN_CHUNKS = float(os.environ.get("SLEEP_BETWEEN_CHUNKS", 1.5))
SLEEP_BETWEEN_AI = float(os.environ.get("SLEEP_BETWEEN_AI", 5))
MIN_DATA_ROWS = int(os.environ.get("MIN_DATA_ROWS", 250))

# Technical threshold (only SMAs + MACD/DMI scoring). Example set to 70 previously.
TECHNICAL_SCORE_THRESHOLD = int(os.environ.get("TECHNICAL_SCORE_THRESHOLD", 70))

# AI configuration
AI_PRIMARY_MODEL = os.environ.get("AI_PRIMARY_MODEL", "gemini-1.5-flash")
AI_FALLBACK_MODEL = os.environ.get("AI_FALLBACK_MODEL", "gemini-pro")
AI_MAX_RETRIES = int(os.environ.get("AI_MAX_RETRIES", 3))
AI_MAX_CONSECUTIVE_FAILURES = int(os.environ.get("AI_MAX_CONSECUTIVE_FAILURES", 5))

# IMPORTANT: cap how many matches will be sent to Gemini in one run to avoid long runtimes / rate limits
AI_ANALYSIS_CAP = int(os.environ.get("AI_ANALYSIS_CAP", 25))  # change to suit your quota

# ---------------- Initialize Gemini client if key present ----------------
if GENAI_API_KEY:
    try:
        genai.configure(api_key=GENAI_API_KEY)
    except Exception as e:
        print(f"⚠️ Warning configuring Gemini client: {e}")

# ---------------- Email helper ----------------
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

# ---------------- Tickers list ----------------
def get_all_tickers():
    print("🌍 Fetching full market ticker list...")
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        s = requests.get(url, timeout=30).content
        tickers = pd.read_csv(io.StringIO(s.decode('utf-8')), header=None)[0].tolist()
        clean = [t for t in tickers if isinstance(t, str) and "^" not in t and "." not in t]
        print(f"✅ Found {len(clean)} tickers.")
        return clean
    except Exception as e:
        print(f"⚠️ Failed to fetch tickers: {e}. Using fallback.")
        return ['AAPL','NVDA','AMD','TSLA','MSFT']

# ---------------- Bulk downloader ----------------
def get_data_bulk(tickers, period="2y", max_retries=3):
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

    # fallback to single-ticker downloads
    print("   ↳ Chunk retries exhausted; falling back to single-ticker downloads.")
    frames = []
    for t in tickers:
        try:
            single = yf.download(t, period=period, progress=False, threads=False)
            if single is None or single.empty:
                failed.append(t)
                continue
            frames.append((t, single))
            time.sleep(1.0)
        except Exception as e:
            print(f"   ⚠️ Single download failed for {t}: {e}")
            failed.append(t)
    if not frames:
        return pd.DataFrame(), failed
    assembled = pd.concat([df for (_t, df) in frames], axis=1, keys=[_t for _t, df in frames])
    return assembled, failed

# ---------------- Extract per-ticker DataFrame ----------------
def extract_stock_df_from_bulk(data: pd.DataFrame, ticker: str):
    if data is None or data.empty:
        raise KeyError("No data")
    if isinstance(data.columns, pd.MultiIndex):
        # Try common placements
        if ticker in data.columns.levels[0]:
            return data.xs(ticker, axis=1, level=0, drop_level=True)
        if ticker in data.columns.levels[1]:
            return data.xs(ticker, axis=1, level=1, drop_level=True)
        # search levels
        for lvl in range(len(data.columns.levels)):
            if any(str(x) == ticker for x in data.columns.levels[lvl]):
                return data.xs(ticker, axis=1, level=lvl, drop_level=True)
        raise KeyError(f"{ticker} not found in MultiIndex columns")
    else:
        return data

# ---------------- Strategy: analyze_ticker ----------------
def analyze_ticker(ticker: str, df: pd.DataFrame):
    """
    SMA-only ordering (no price check):
      - If SMA21 > SMA55 > SMA233 -> +30 (Perfect stack)
      - Elif SMA55 > SMA233 -> +10
    Plus MACD (+20) and DMI (+20).
    """
    try:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        # conservative approach: do not drop all NaNs (drop rows where close is NaN)
        df = df[df['Close'].notna()] if 'Close' in df.columns else df.dropna()
        if len(df) < MIN_DATA_ROWS:
            return 0, [f"Insufficient data (<{MIN_DATA_ROWS})"]

        for col in ['Close','High','Low','Volume']:
            if col not in df.columns:
                return 0, [f"Missing column {col}"]

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

        # Extract current SMA values safely
        s21 = sma_21.iloc[-1] if len(sma_21) > 0 else float('nan')
        s55 = sma_55.iloc[-1] if len(sma_55) > 0 else float('nan')
        s233 = sma_233.iloc[-1] if len(sma_233) > 0 else float('nan')

        score = 0
        reasons = []

        # SMA-only checks (price is NOT considered)
        if pd.notna(s21) and pd.notna(s55) and pd.notna(s233):
            if s21 > s55 > s233:
                score += 30
                reasons.append("Perfect SMA Stack (21 > 55 > 233)")
            elif s55 > s233:
                score += 10
                reasons.append("55 Above 233")
        else:
            missing = []
            if not pd.notna(s21): missing.append("SMA21")
            if not pd.notna(s55): missing.append("SMA55")
            if not pd.notna(s233): missing.append("SMA233")
            reasons.append(f"Missing SMAs: {', '.join(missing)}")

        # MACD
        try:
            if hist is not None and len(hist) >= 2:
                if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]:
                    score += 20
                    reasons.append("MACD Rising")
        except Exception:
            reasons.append("MACD calc issue")

        # DMI
        try:
            if pos_di.iloc[-1] > neg_di.iloc[-1]:
                score += 20
                reasons.append("Positive DMI")
        except Exception:
            reasons.append("DMI calc issue")

        print(f"🔍 {ticker} Score: {score} | Reasons: {reasons}")
        return score, reasons

    except Exception as e:
        print(f"❌ Analyzer crash for {ticker}: {e}\n{traceback.format_exc()}")
        return 0, [f"Analyzer crash: {e}"]

# ---------------- Gemini analysis (defensive) ----------------
def get_gemini_analysis(ticker):
    global AI_DISABLED
    if AI_DISABLED: return "AI Analysis Skipped (Too many errors)"

    try:
        # Try primary model (1.5 Flash)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Analyze {ticker} for a breakout or continuation trade. Verdict: BUY or AVOID? Keep it under 50 words."
        
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            # CATCH THE 404 ERROR
            if "404" in str(e) or "not found" in str(e):
                print("⚠️ 1.5 Flash not found, using fallback model...")
                try:
                    # Fallback to the older Pro model if Flash fails
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(prompt)
                    return response.text.strip()
                except:
                    return "AI Fallback Failed"
            raise e

    except Exception as e:
        print(f"⚠️ AI Error on {ticker}: {e}")
        return "AI Unavailable"
# ---------------- Main run ----------------
def run_scanner():
    started_at = datetime.datetime.utcnow().isoformat() + "Z"
    print(f"🚀 Market scan starting at {started_at}")

    all_tickers = get_all_tickers()
    total = len(all_tickers)
    high_conviction = []
    failed_downloads = set()
    failed_analysis = set()

    for i in range(0, total, CHUNK_SIZE):
        chunk = all_tickers[i:i+CHUNK_SIZE]
        print(f"\n--- chunk {i//CHUNK_SIZE + 1} / {((total-1)//CHUNK_SIZE)+1} (size {len(chunk)}) ---")
        data, failed = get_data_bulk(chunk)
        failed_downloads.update(failed)
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            print("   ⚠️ Chunk returned no data; marking them as failed.")
            failed_downloads.update(chunk)
            time.sleep(SLEEP_BETWEEN_CHUNKS)
            continue

        for ticker in chunk:
            try:
                try:
                    stock_df = extract_stock_df_from_bulk(data, ticker)
                except KeyError as e:
                    failed_downloads.add(ticker)
                    print(f"   ⚠️ {ticker} missing in bulk data: {e}")
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
                print(f"   ❌ Processing error for {ticker}: {e}\n{traceback.format_exc()}")
                continue

        time.sleep(SLEEP_BETWEEN_CHUNKS)

    match_count = len(high_conviction)
    print(f"\n🎯 Scan complete. {match_count} tickers met threshold >= {TECHNICAL_SCORE_THRESHOLD}.")

    high_conviction.sort(key=lambda x: x['score'], reverse=True)

    # Build email body
    lines = []
    lines.append(f"HIGH CONVICTION REPORT: {datetime.date.today()}")
    lines.append(f"Started: {started_at} (UTC)")
    lines.append(f"Scanned {total} tickers in chunks of {CHUNK_SIZE}")
    lines.append(f"Threshold: {TECHNICAL_SCORE_THRESHOLD}")
    lines.append(f"Matches found: {match_count}")
    lines.append("\n========================================\n")

    # AI variables
    ai_disabled = False
    consecutive_ai_failures = 0
    ai_used = False
    ai_count = 0

    # Iterate matches and attempt Gemini (limited by AI_ANALYSIS_CAP)
    for idx, s in enumerate(high_conviction, start=1):
        lines.append(f"{idx}. {s['ticker']} (Score: {s['score']})")
        lines.append(f"   Price: {s['price']}")
        lines.append(f"   Signals: {', '.join(s['reasons'])}")

        if ai_disabled:
            lines.append("   AI disabled; skipping Gemini analysis.\n")
            continue

        if ai_count >= AI_ANALYSIS_CAP:
            lines.append(f"   AI analysis cap reached ({AI_ANALYSIS_CAP}); skipping further AI analysis.\n")
            ai_disabled = True
            continue

        # Attempt primary then fallback model with retries; detect fatal model-not-found
        success = False
        fatal = False
        ai_text = ""
        for model in (AI_PRIMARY_MODEL, AI_FALLBACK_MODEL):
            for attempt in range(AI_MAX_RETRIES):
                ok, txt, is_fatal = get_gemini_analysis(s['ticker'], model)
                if is_fatal:
                    fatal = True
                    ai_text = txt
                    break
                if ok:
                    success = True
                    ai_text = txt
                    break
                else:
                    print(f"   ⚠️ {model} attempt {attempt+1} failed for {s['ticker']}: {txt}")
                    time.sleep(1 + attempt)
            if success or fatal:
                break

        if fatal:
            ai_disabled = True
            lines.append("   🛑 Gemini model not available on this runner (model-not-found). AI disabled for rest of run.\n")
            lines.append(f"   Diagnostic: {ai_text}\n")
            # don't increment ai_count; it's disabled now
            continue

        if not success:
            consecutive_ai_failures += 1
            lines.append(f"   🧠 Gemini analysis failed for {s['ticker']} (consecutive AI failures: {consecutive_ai_failures}).\n")
            if consecutive_ai_failures >= AI_MAX_CONSECUTIVE_FAILURES:
                ai_disabled = True
                lines.append("   🛑 Reached maximum consecutive AI failures; disabling AI for rest of run.\n")
            continue
        else:
            consecutive_ai_failures = 0
            ai_used = True
            ai_count += 1
            lines.append("   🧠 Gemini Verdict:")
            for L in ai_text.splitlines():
                lines.append(f"      {L}")
            lines.append("")
            time.sleep(SLEEP_BETWEEN_AI)

    if match_count == 0:
        lines.append("No matches passed the technical threshold today.\n")

    # Diagnostics summary
    lines.append("\n--- DIAGNOSTICS ---")
    lines.append(f"Failed downloads / missing tickers: {len(failed_downloads)}")
    if failed_downloads:
        lines.append(", ".join(sorted(list(failed_downloads))[:200]) + ("" if len(failed_downloads) <= 200 else f"... (+{len(failed_downloads)-200} more)"))
    lines.append(f"Tickers that crashed during analysis: {len(failed_analysis)}")
    if failed_analysis:
        lines.append(", ".join(sorted(list(failed_analysis))[:200]) + ("" if len(failed_analysis) <= 200 else f"... (+{len(failed_analysis)-200} more)"))
    lines.append(f"AI used: {'Yes' if ai_used else 'No'}")
    lines.append(f"AI disabled due to fatal model-not-found or failures: {'Yes' if ai_disabled else 'No'}")
    lines.append(f"AI analyses performed this run: {ai_count} (cap: {AI_ANALYSIS_CAP})")

    body = "\n".join(lines)
    subject = f"🚀 {match_count} High-Conviction Breakouts (>= {TECHNICAL_SCORE_THRESHOLD})" if match_count else f"⚠️ MARKET SCAN: 0 Matches ({datetime.date.today()})"
    sent = send_email(subject, body)
    if not sent:
        print("❌ Final report failed to send; check EMAIL secrets and logs.")
    else:
        print("✅ Final report sent or attempted.")

if __name__ == "__main__":
    run_scanner()
