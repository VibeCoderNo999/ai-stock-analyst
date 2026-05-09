import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json

# ─── Page Config ───
st.set_page_config(
    page_title="AI Stock Analyst",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@700;900&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 900;
        color: #F0E6D3;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #8a8578;
        font-size: 1.05rem;
        margin-top: -4px;
        margin-bottom: 28px;
    }
    .ticker-badge {
        display: inline-block;
        background: linear-gradient(135deg, #C9A84C, #E8D48B);
        color: #1a1714;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.4rem;
        padding: 6px 18px;
        border-radius: 6px;
        letter-spacing: 1px;
    }
    .section-label {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #F0E6D3;
        margin: 28px 0 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #C9A84C;
        display: inline-block;
    }
    .metric-row {
        background: linear-gradient(135deg, #1a1714, #22201b);
        border: 1px solid #33302a;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-name { color: #8a8578; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.2px; }
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        font-weight: 700;
        color: #F0E6D3;
    }
    .report-block {
        background: #1a1714;
        border: 1px solid #33302a;
        border-radius: 12px;
        padding: 28px 32px;
        line-height: 1.75;
        color: #d4cdc0;
        font-size: 0.95rem;
    }
    .report-block h1, .report-block h2, .report-block h3 {
        font-family: 'Playfair Display', serif;
        color: #F0E6D3;
        margin-top: 24px;
    }
    .report-block h2 { border-bottom: 1px solid #33302a; padding-bottom: 6px; }
    .report-block strong { color: #E8D48B; }
    .report-block table { width: 100%; border-collapse: collapse; margin: 12px 0; }
    .report-block th {
        background: #22201b;
        color: #C9A84C;
        padding: 8px 12px;
        text-align: left;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .report-block td {
        padding: 8px 12px;
        border-bottom: 1px solid #2a2722;
        color: #d4cdc0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    .pos { color: #6BCB77; }
    .neg { color: #FF6B6B; }
    .verdict-box {
        background: linear-gradient(135deg, #2a2520, #1a1714);
        border: 2px solid #C9A84C;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin: 20px 0;
    }
    .footer-text {
        text-align: center;
        color: #555;
        font-size: 0.7rem;
        margin-top: 40px;
        padding: 12px;
        border-top: 1px solid #22201b;
    }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
    .stTextInput > div > div > input {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ─── Gemini API ───
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

def call_gemini(prompt: str, data_context: str) -> str:
    """Call Gemini 2.0 Flash with the analysis prompt."""
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key not configured. Add GEMINI_API_KEY to your Streamlit secrets."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    system_prompt = """You are a senior equity research analyst with CFA credentials and 15+ years of experience at a top-tier investment bank. You produce institutional-quality research reports.

ANALYSIS FRAMEWORK — follow all of these rigorously:

1. COMPANY OVERVIEW
   - What the company does, its sector position, competitive moat
   - Key products/services and revenue segments

2. FINANCIAL HEALTH SCORECARD
   Rate each 1-10 with brief justification:
   - Profitability (margins, ROE, ROA)
   - Growth (revenue & earnings trajectory)
   - Balance Sheet Strength (debt levels, current ratio)
   - Cash Flow Quality (FCF generation, consistency)
   - Valuation (relative to peers and historical range)
   Give an overall score.

3. VALUATION ANALYSIS
   - Compare current P/E, P/B, EV/EBITDA to:
     a) Its own 5-year historical average
     b) Sector/industry median
     c) Direct competitors
   - Identify whether it's trading at a premium or discount and WHY
   - Estimate a fair value target price using relative valuation (peer P/E × estimated EPS)

4. GROWTH ANALYSIS
   - Revenue growth trajectory (accelerating or decelerating?)
   - Earnings growth vs revenue growth (margin expansion or compression?)
   - Forward catalysts that could drive growth
   - TAM (Total Addressable Market) considerations

5. TECHNICAL SIGNALS
   - Interpret the RSI, MACD, and moving average data provided
   - Is the stock overbought/oversold?
   - Key support/resistance levels from the price data
   - Trend direction (bullish, bearish, or consolidating)

6. COMPETITIVE POSITIONING
   - Create a comparison table with key metrics vs competitors
   - Identify competitive advantages and disadvantages
   - Market share trends if inferable

7. KEY RISKS
   - Company-specific risks (3-4 points)
   - Sector/macro risks (2-3 points)
   - Rate each risk as LOW / MEDIUM / HIGH severity

8. RECENT NEWS & CATALYSTS
   - Summarise any significant recent developments
   - Upcoming catalysts (earnings dates, product launches, regulatory decisions)

9. TARGET PRICE & VERDICT
   - 12-month target price with clear methodology
   - Bull case target / Base case target / Bear case target
   - Final verdict: STRONG BUY / BUY / HOLD / SELL / STRONG SELL
   - Confidence level: HIGH / MEDIUM / LOW
   - Key assumption that would change your view

FORMATTING RULES:
- Use markdown headers (##) for each section
- Use tables for comparative data (markdown table format)
- Bold key numbers and conclusions
- Be specific with numbers — never vague
- If data is missing, say so explicitly rather than guessing
- Write for a sophisticated investor audience — no fluff
- Every claim must reference the data provided"""

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{prompt}\n\n--- RAW FINANCIAL DATA ---\n{data_context}"}],
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 8000,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try again."
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"


# ─── Technical Indicators ───
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal
    return macd_line, signal, histogram

def calc_bollinger(series: pd.Series, period: int = 20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return sma, upper, lower


# ─── Data Collection ───
@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str) -> dict:
    """Fetch comprehensive stock data from yfinance."""
    stock = yf.Ticker(ticker)

    # Basic info
    try:
        info = stock.info
    except Exception:
        info = {}

    # Price history
    try:
        hist_1y = stock.history(period="1y")
        if isinstance(hist_1y.columns, pd.MultiIndex):
            hist_1y.columns = hist_1y.columns.get_level_values(0)
    except Exception:
        hist_1y = pd.DataFrame()

    try:
        hist_5y = stock.history(period="5y")
        if isinstance(hist_5y.columns, pd.MultiIndex):
            hist_5y.columns = hist_5y.columns.get_level_values(0)
    except Exception:
        hist_5y = pd.DataFrame()

    # Financials
    try:
        income = stock.financials
    except Exception:
        income = pd.DataFrame()

    try:
        balance = stock.balance_sheet
    except Exception:
        balance = pd.DataFrame()

    try:
        cashflow = stock.cashflow
    except Exception:
        cashflow = pd.DataFrame()

    # Recommendations
    try:
        recs = stock.recommendations
    except Exception:
        recs = pd.DataFrame()

    # News
    try:
        news = stock.news[:8] if stock.news else []
    except Exception:
        news = []

    return {
        "info": info,
        "hist_1y": hist_1y,
        "hist_5y": hist_5y,
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
        "recommendations": recs,
        "news": news,
    }


def build_data_context(data: dict, ticker: str) -> str:
    """Pre-process all data into a structured text context for the AI."""
    info = data["info"]
    hist = data["hist_1y"]
    sections = []

    # ── Company basics ──
    basics = {
        "Ticker": ticker.upper(),
        "Name": info.get("longName", "N/A"),
        "Sector": info.get("sector", "N/A"),
        "Industry": info.get("industry", "N/A"),
        "Country": info.get("country", "N/A"),
        "Employees": info.get("fullTimeEmployees", "N/A"),
        "Description": info.get("longBusinessSummary", "N/A")[:500],
    }
    sections.append("=== COMPANY BASICS ===\n" + "\n".join(f"{k}: {v}" for k, v in basics.items()))

    # ── Price & Valuation Metrics ──
    price_data = {
        "Current Price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
        "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52-Week Low": info.get("fiftyTwoWeekLow", "N/A"),
        "50-Day MA": info.get("fiftyDayAverage", "N/A"),
        "200-Day MA": info.get("twoHundredDayAverage", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "Enterprise Value": info.get("enterpriseValue", "N/A"),
        "Trailing P/E": info.get("trailingPE", "N/A"),
        "Forward P/E": info.get("forwardPE", "N/A"),
        "PEG Ratio": info.get("pegRatio", "N/A"),
        "Price/Book": info.get("priceToBook", "N/A"),
        "Price/Sales": info.get("priceToSalesTrailing12Months", "N/A"),
        "EV/Revenue": info.get("enterpriseToRevenue", "N/A"),
        "EV/EBITDA": info.get("enterpriseToEbitda", "N/A"),
    }
    sections.append("=== PRICE & VALUATION ===\n" + "\n".join(f"{k}: {v}" for k, v in price_data.items()))

    # ── Profitability ──
    profit_data = {
        "Gross Margin": info.get("grossMargins", "N/A"),
        "Operating Margin": info.get("operatingMargins", "N/A"),
        "Net Margin": info.get("profitMargins", "N/A"),
        "ROE": info.get("returnOnEquity", "N/A"),
        "ROA": info.get("returnOnAssets", "N/A"),
        "Revenue": info.get("totalRevenue", "N/A"),
        "Revenue Per Share": info.get("revenuePerShare", "N/A"),
        "Net Income": info.get("netIncomeToCommon", "N/A"),
        "EBITDA": info.get("ebitda", "N/A"),
        "EPS (Trailing)": info.get("trailingEps", "N/A"),
        "EPS (Forward)": info.get("forwardEps", "N/A"),
    }
    sections.append("=== PROFITABILITY ===\n" + "\n".join(f"{k}: {v}" for k, v in profit_data.items()))

    # ── Balance Sheet ──
    balance_data = {
        "Total Cash": info.get("totalCash", "N/A"),
        "Total Debt": info.get("totalDebt", "N/A"),
        "Debt/Equity": info.get("debtToEquity", "N/A"),
        "Current Ratio": info.get("currentRatio", "N/A"),
        "Quick Ratio": info.get("quickRatio", "N/A"),
        "Book Value/Share": info.get("bookValue", "N/A"),
    }
    sections.append("=== BALANCE SHEET ===\n" + "\n".join(f"{k}: {v}" for k, v in balance_data.items()))

    # ── Cash Flow ──
    cashflow_data = {
        "Operating Cash Flow": info.get("operatingCashflow", "N/A"),
        "Free Cash Flow": info.get("freeCashflow", "N/A"),
    }
    sections.append("=== CASH FLOW ===\n" + "\n".join(f"{k}: {v}" for k, v in cashflow_data.items()))

    # ── Dividends ──
    div_data = {
        "Dividend Rate": info.get("dividendRate", "N/A"),
        "Dividend Yield": info.get("dividendYield", "N/A"),
        "Payout Ratio": info.get("payoutRatio", "N/A"),
        "Ex-Dividend Date": info.get("exDividendDate", "N/A"),
    }
    sections.append("=== DIVIDENDS ===\n" + "\n".join(f"{k}: {v}" for k, v in div_data.items()))

    # ── Growth Rates (computed) ──
    income_stmt = data["income"]
    if not income_stmt.empty and "Total Revenue" in income_stmt.index:
        try:
            rev = income_stmt.loc["Total Revenue"].dropna().sort_index()
            if len(rev) >= 2:
                rev_growth = [(rev.iloc[i] - rev.iloc[i - 1]) / abs(rev.iloc[i - 1]) * 100 for i in range(1, len(rev))]
                sections.append(f"=== REVENUE GROWTH (YoY) ===\n{', '.join(f'{g:.1f}%' for g in rev_growth)} (most recent last)")
        except Exception:
            pass

    if not income_stmt.empty and "Net Income" in income_stmt.index:
        try:
            ni = income_stmt.loc["Net Income"].dropna().sort_index()
            if len(ni) >= 2:
                ni_growth = [(ni.iloc[i] - ni.iloc[i - 1]) / abs(ni.iloc[i - 1]) * 100 for i in range(1, len(ni))]
                sections.append(f"=== NET INCOME GROWTH (YoY) ===\n{', '.join(f'{g:.1f}%' for g in ni_growth)} (most recent last)")
        except Exception:
            pass

    # ── Technical Indicators (computed) ──
    if not hist.empty and "Close" in hist.columns:
        close = hist["Close"]
        rsi = calc_rsi(close)
        macd_line, macd_signal, macd_hist = calc_macd(close)
        sma20, bb_upper, bb_lower = calc_bollinger(close)
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        current = close.iloc[-1]
        tech = {
            "RSI (14)": f"{rsi.iloc[-1]:.1f}" if not pd.isna(rsi.iloc[-1]) else "N/A",
            "MACD Line": f"{macd_line.iloc[-1]:.4f}" if not pd.isna(macd_line.iloc[-1]) else "N/A",
            "MACD Signal": f"{macd_signal.iloc[-1]:.4f}" if not pd.isna(macd_signal.iloc[-1]) else "N/A",
            "MACD Histogram": f"{macd_hist.iloc[-1]:.4f}" if not pd.isna(macd_hist.iloc[-1]) else "N/A",
            "SMA 20": f"{sma20.iloc[-1]:.2f}" if not pd.isna(sma20.iloc[-1]) else "N/A",
            "SMA 50": f"{sma50.iloc[-1]:.2f}" if not pd.isna(sma50.iloc[-1]) else "N/A",
            "SMA 200": f"{sma200.iloc[-1]:.2f}" if not pd.isna(sma200.iloc[-1]) else "N/A",
            "Bollinger Upper": f"{bb_upper.iloc[-1]:.2f}" if not pd.isna(bb_upper.iloc[-1]) else "N/A",
            "Bollinger Lower": f"{bb_lower.iloc[-1]:.2f}" if not pd.isna(bb_lower.iloc[-1]) else "N/A",
            "Price vs SMA50": f"{'Above' if current > sma50.iloc[-1] else 'Below'}" if not pd.isna(sma50.iloc[-1]) else "N/A",
            "Price vs SMA200": f"{'Above' if current > sma200.iloc[-1] else 'Below'}" if not pd.isna(sma200.iloc[-1]) else "N/A",
        }
        sections.append("=== TECHNICAL INDICATORS ===\n" + "\n".join(f"{k}: {v}" for k, v in tech.items()))

        # Price performance
        perf = {}
        for label, days in [("1W", 5), ("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252)]:
            if len(close) > days:
                ret = (current - close.iloc[-days]) / close.iloc[-days] * 100
                perf[label] = f"{ret:.1f}%"
        if perf:
            sections.append("=== PRICE PERFORMANCE ===\n" + "\n".join(f"{k}: {v}" for k, v in perf.items()))

        # Volatility
        daily_returns = close.pct_change().dropna()
        ann_vol = daily_returns.std() * np.sqrt(252) * 100
        sections.append(f"=== VOLATILITY ===\nAnnualised Volatility: {ann_vol:.1f}%\nAvg Daily Volume: {info.get('averageVolume', 'N/A')}")

    # ── Analyst Recommendations ──
    recs = data["recommendations"]
    if recs is not None and not recs.empty:
        try:
            recent = recs.tail(10).to_string()
            sections.append(f"=== ANALYST RECOMMENDATIONS (recent) ===\n{recent}")
        except Exception:
            pass

    # Analyst targets
    target_data = {
        "Target Mean Price": info.get("targetMeanPrice", "N/A"),
        "Target High": info.get("targetHighPrice", "N/A"),
        "Target Low": info.get("targetLowPrice", "N/A"),
        "Target Median": info.get("targetMedianPrice", "N/A"),
        "Recommendation": info.get("recommendationKey", "N/A"),
        "Number of Analysts": info.get("numberOfAnalystOpinions", "N/A"),
    }
    sections.append("=== ANALYST TARGETS ===\n" + "\n".join(f"{k}: {v}" for k, v in target_data.items()))

    # ── News Headlines ──
    news = data["news"]
    if news:
        news_lines = []
        for n in news[:8]:
            title = n.get("title", "")
            publisher = n.get("publisher", "")
            if title:
                news_lines.append(f"- [{publisher}] {title}")
        if news_lines:
            sections.append("=== RECENT NEWS ===\n" + "\n".join(news_lines))

    return "\n\n".join(sections)


@st.cache_data(ttl=300)
def fetch_competitor_data(sector: str, industry: str, ticker: str) -> pd.DataFrame:
    """Fetch basic data for competitors in the same sector."""
    # Common peer mappings by sector — fallback to broad indices
    sector_peers = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSM", "ORCL"],
        "Financial Services": ["JPM", "BAC", "GS", "MS", "C", "WFC", "HSBC"],
        "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "TMO", "LLY"],
        "Energy": ["XOM", "CVX", "SHEL", "TTE", "COP", "BP", "SLB"],
        "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "TM"],
        "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "CMCSA", "T", "VZ"],
        "Industrials": ["CAT", "HON", "UPS", "BA", "GE", "MMM", "RTX"],
        "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST", "PM", "UL"],
        "Basic Materials": ["LIN", "APD", "ECL", "NEM", "FCX", "NUE", "DOW"],
        "Real Estate": ["PLD", "AMT", "CCI", "SPG", "EQIX", "O", "PSA"],
        "Utilities": ["NEE", "DUK", "SO", "AEP", "D", "EXC", "SRE"],
    }

    # IDX-specific peers
    idx_peers = {
        "BBRI.JK": ["BBCA.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK"],
        "BBCA.JK": ["BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK"],
        "TLKM.JK": ["EXCL.JK", "ISAT.JK", "TBIG.JK"],
        "ASII.JK": ["UNTR.JK", "SMSM.JK", "AUTO.JK"],
        "UNVR.JK": ["ICBP.JK", "INDF.JK", "MYOR.JK"],
        "GOTO.JK": ["BUKA.JK", "EMTK.JK"],
    }

    # Check IDX-specific first
    peers = idx_peers.get(ticker.upper(), [])

    # Then add sector peers (skip the ticker itself)
    if sector in sector_peers:
        for p in sector_peers[sector]:
            if p.upper() != ticker.upper() and p not in peers:
                peers.append(p)
            if len(peers) >= 5:
                break

    if not peers:
        return pd.DataFrame()

    rows = []
    for p in peers[:5]:
        try:
            pinfo = yf.Ticker(p).info
            rows.append({
                "Ticker": p,
                "Name": pinfo.get("shortName", p),
                "Market Cap": pinfo.get("marketCap", None),
                "P/E": pinfo.get("trailingPE", None),
                "P/B": pinfo.get("priceToBook", None),
                "ROE": pinfo.get("returnOnEquity", None),
                "Net Margin": pinfo.get("profitMargins", None),
                "Div Yield": pinfo.get("dividendYield", None),
                "Rev Growth": pinfo.get("revenueGrowth", None),
            })
        except Exception:
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─── UI ───
st.markdown('<p class="hero-title">AI Stock Analyst</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Institutional-grade equity research powered by AI · Enter any ticker worldwide</p>', unsafe_allow_html=True)

col_input, col_btn = st.columns([3, 1])
with col_input:
    ticker_input = st.text_input(
        "Ticker",
        placeholder="e.g. AAPL, BBRI.JK, TSLA, 0700.HK",
        label_visibility="collapsed",
    )
with col_btn:
    analyse_btn = st.button("🔍 Analyse", use_container_width=True, type="primary")

st.markdown("")

if analyse_btn and ticker_input.strip():
    ticker = ticker_input.strip().upper()

    # ── Fetch Data ──
    with st.status("Gathering data…", expanded=True) as status:
        st.write("📡 Fetching price & financial data…")
        data = fetch_stock_data(ticker)

        if not data["info"] or data["info"].get("regularMarketPrice") is None and data["info"].get("currentPrice") is None:
            st.error(f"❌ Could not find data for **{ticker}**. Check the ticker symbol and try again.")
            st.stop()

        info = data["info"]
        hist = data["hist_1y"]

        st.write("📊 Computing technical indicators…")
        data_context = build_data_context(data, ticker)

        st.write("🏢 Fetching competitor data…")
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        comp_df = fetch_competitor_data(sector, industry, ticker)

        if not comp_df.empty:
            data_context += "\n\n=== COMPETITOR COMPARISON ===\n" + comp_df.to_string(index=False)

        st.write("🤖 Running AI analysis (this takes ~15 seconds)…")
        prompt = f"Produce a comprehensive equity research report for {ticker} ({info.get('longName', ticker)}). Use ONLY the data provided below. Follow your analysis framework exactly."
        report = call_gemini(prompt, data_context)

        status.update(label="Analysis complete ✓", state="complete")

    # ── Display: Header Info ──
    name = info.get("longName", ticker)
    price = info.get("currentPrice", info.get("regularMarketPrice", 0))
    prev_close = info.get("previousClose", price)
    change = price - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0
    delta_color = "pos" if change >= 0 else "neg"
    sign = "+" if change >= 0 else ""

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 8px;">
        <span class="ticker-badge">{ticker}</span>
        <span style="font-size: 1.6rem; font-weight: 700; color: #F0E6D3; font-family: 'JetBrains Mono', monospace;">{price:,.2f}</span>
        <span class="{delta_color}" style="font-size: 1.1rem; font-family: 'JetBrains Mono', monospace;">{sign}{change:.2f} ({sign}{change_pct:.2f}%)</span>
    </div>
    <div style="color: #8a8578; font-size: 0.85rem; margin-bottom: 20px;">
        {name} · {info.get('sector', '')} · {info.get('industry', '')} · {info.get('country', '')}
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics Row ──
    metric_cols = st.columns(6)
    metrics = [
        ("Market Cap", f"${info.get('marketCap', 0)/1e9:,.1f}B" if info.get("marketCap") else "N/A"),
        ("P/E Ratio", f"{info.get('trailingPE', 0):.1f}" if info.get("trailingPE") else "N/A"),
        ("EPS", f"{info.get('trailingEps', 0):.2f}" if info.get("trailingEps") else "N/A"),
        ("Div Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get("dividendYield") else "N/A"),
        ("52W Range", f"{info.get('fiftyTwoWeekLow', 0):,.0f} – {info.get('fiftyTwoWeekHigh', 0):,.0f}"),
        ("Beta", f"{info.get('beta', 0):.2f}" if info.get("beta") else "N/A"),
    ]
    for i, (label, val) in enumerate(metrics):
        with metric_cols[i]:
            st.markdown(f'<div class="metric-row"><div class="metric-name">{label}</div><div class="metric-val">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Price Chart with Technical Indicators ──
    chart_col, ta_col = st.columns([2, 1])

    with chart_col:
        st.markdown('<div class="section-label">Price Chart (1Y)</div>', unsafe_allow_html=True)

        if not hist.empty and "Close" in hist.columns:
            close = hist["Close"]
            sma50 = close.rolling(50).mean()
            sma200 = close.rolling(200).mean()
            _, bb_upper, bb_lower = calc_bollinger(close)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist["Open"], high=hist["High"],
                low=hist["Low"], close=hist["Close"],
                increasing_line_color="#6BCB77", decreasing_line_color="#FF6B6B",
                name="Price",
            ), row=1, col=1)

            # Moving averages
            fig.add_trace(go.Scatter(x=hist.index, y=sma50, name="SMA 50", line=dict(color="#C9A84C", width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=sma200, name="SMA 200", line=dict(color="#8a8578", width=1, dash="dot")), row=1, col=1)

            # Bollinger Bands
            fig.add_trace(go.Scatter(x=hist.index, y=bb_upper, name="BB Upper", line=dict(color="rgba(201,168,76,0.3)", width=1), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=bb_lower, name="BB Lower", line=dict(color="rgba(201,168,76,0.3)", width=1), fill="tonexty", fillcolor="rgba(201,168,76,0.05)", showlegend=False), row=1, col=1)

            # Volume
            colors = ["#6BCB77" if c >= o else "#FF6B6B" for c, o in zip(hist["Close"], hist["Open"])]
            fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], marker_color=colors, name="Volume", opacity=0.5), row=2, col=1)

            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=450, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=10)),
                xaxis_rangeslider_visible=False,
                yaxis=dict(gridcolor="#22201b"), yaxis2=dict(gridcolor="#22201b"),
                xaxis=dict(gridcolor="#22201b"), xaxis2=dict(gridcolor="#22201b"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with ta_col:
        st.markdown('<div class="section-label">Technical Signals</div>', unsafe_allow_html=True)

        if not hist.empty and "Close" in hist.columns:
            close = hist["Close"]
            rsi = calc_rsi(close)
            macd_line, macd_signal, macd_hist_vals = calc_macd(close)

            rsi_val = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 0
            rsi_label = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
            rsi_color = "neg" if rsi_val > 70 else "pos" if rsi_val < 30 else ""

            macd_val = macd_hist_vals.iloc[-1] if not pd.isna(macd_hist_vals.iloc[-1]) else 0
            macd_label = "Bullish" if macd_val > 0 else "Bearish"
            macd_color = "pos" if macd_val > 0 else "neg"

            sma50_val = close.rolling(50).mean().iloc[-1]
            sma200_val = close.rolling(200).mean().iloc[-1]
            trend = "Bullish" if not pd.isna(sma50_val) and not pd.isna(sma200_val) and sma50_val > sma200_val else "Bearish"
            trend_color = "pos" if trend == "Bullish" else "neg"

            price_pos = (close.iloc[-1] - close.min()) / (close.max() - close.min()) * 100 if close.max() != close.min() else 50

            signals = [
                ("RSI (14)", f"{rsi_val:.1f}", rsi_label, rsi_color),
                ("MACD", f"{macd_val:.4f}", macd_label, macd_color),
                ("Trend (SMA)", "", trend, trend_color),
                ("52W Position", f"{price_pos:.0f}%", "of range", ""),
            ]

            for label, val, desc, color in signals:
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-name">{label}</div>
                    <div class="metric-val">{val} <span class="{color}" style="font-size:0.8rem;">{desc}</span></div>
                </div>""", unsafe_allow_html=True)

            # RSI gauge
            fig_rsi = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rsi_val,
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#8a8578"),
                    bar=dict(color="#C9A84C"),
                    bgcolor="#1a1714",
                    steps=[
                        dict(range=[0, 30], color="#1a3a1a"),
                        dict(range=[30, 70], color="#22201b"),
                        dict(range=[70, 100], color="#3a1a1a"),
                    ],
                    threshold=dict(line=dict(color="#FF6B6B", width=2), thickness=0.8, value=rsi_val),
                ),
                number=dict(font=dict(color="#F0E6D3", family="JetBrains Mono")),
            ))
            fig_rsi.update_layout(
                height=180, margin=dict(l=20, r=20, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#8a8578"),
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

    # ── Competitor Table ──
    if not comp_df.empty:
        st.markdown('<div class="section-label">Competitor Comparison</div>', unsafe_allow_html=True)
        display_df = comp_df.copy()
        for col in ["ROE", "Net Margin", "Div Yield", "Rev Growth"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
        if "Market Cap" in display_df.columns:
            display_df["Market Cap"] = display_df["Market Cap"].apply(lambda x: f"${x/1e9:.1f}B" if pd.notna(x) and x else "N/A")
        if "P/E" in display_df.columns:
            display_df["P/E"] = display_df["P/E"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        if "P/B" in display_df.columns:
            display_df["P/B"] = display_df["P/B"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── AI Report ──
    st.markdown('<div class="section-label">AI Equity Research Report</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="report-block">{report}</div>', unsafe_allow_html=True)

    # ── News ──
    if data["news"]:
        st.markdown('<div class="section-label">Recent News</div>', unsafe_allow_html=True)
        for n in data["news"][:6]:
            title = n.get("title", "")
            publisher = n.get("publisher", "")
            link = n.get("link", "#")
            st.markdown(f"**{title}**  \n<span style='color:#8a8578;font-size:0.8rem;'>{publisher}</span> · [Read →]({link})", unsafe_allow_html=True)
            st.markdown("")

    # ── Disclaimer ──
    st.markdown("""
    <div class="footer-text">
        ⚠️ <strong>Disclaimer:</strong> This is an AI-generated analysis for educational and informational purposes only. 
        It does not constitute financial advice, a recommendation, or an offer to buy or sell securities. 
        Always conduct your own research and consult a licensed financial advisor before making investment decisions.
        Data sourced from Yahoo Finance. AI analysis powered by Google Gemini.
    </div>
    """, unsafe_allow_html=True)

elif analyse_btn:
    st.warning("Please enter a ticker symbol.")
