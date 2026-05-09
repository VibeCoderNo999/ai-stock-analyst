import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests
import json
import re

# ─── Page Config ───
st.set_page_config(page_title="AI Stock Analyst", page_icon="🔍", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@700;900&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .hero-title { font-family:'Playfair Display',serif; font-size:2.8rem; font-weight:900; color:#F0E6D3; letter-spacing:-1px; margin-bottom:0; }
    .hero-sub { color:#8a8578; font-size:1.05rem; margin-top:-4px; margin-bottom:28px; }
    .ticker-badge { display:inline-block; background:linear-gradient(135deg,#C9A84C,#E8D48B); color:#1a1714; font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1.4rem; padding:6px 18px; border-radius:6px; letter-spacing:1px; }
    .section-label { font-family:'Playfair Display',serif; font-size:1.3rem; font-weight:700; color:#F0E6D3; margin:28px 0 12px; padding-bottom:6px; border-bottom:2px solid #C9A84C; display:inline-block; }
    .metric-card { background:linear-gradient(135deg,#1a1714,#22201b); border:1px solid #33302a; border-radius:10px; padding:16px 20px; margin-bottom:8px; }
    .metric-name { color:#8a8578; font-size:0.78rem; text-transform:uppercase; letter-spacing:1.2px; }
    .metric-val { font-family:'JetBrains Mono',monospace; font-size:1.2rem; font-weight:700; color:#F0E6D3; }
    .pos { color:#6BCB77; } .neg { color:#FF6B6B; }

    /* Report styling */
    .report-container { background:#1a1714; border:1px solid #33302a; border-radius:12px; padding:32px 36px; }
    .report-container h1 { font-family:'Playfair Display',serif; font-size:1.6rem; color:#F0E6D3; border-bottom:2px solid #C9A84C; padding-bottom:8px; margin-top:28px; }
    .report-container h2 { font-family:'Playfair Display',serif; font-size:1.35rem; color:#F0E6D3; border-bottom:1px solid #33302a; padding-bottom:6px; margin-top:24px; }
    .report-container h3 { font-family:'Playfair Display',serif; font-size:1.1rem; color:#E8D48B; margin-top:18px; }
    .report-container p { color:#d4cdc0; font-size:1rem; line-height:1.8; margin:10px 0; }
    .report-container li { color:#d4cdc0; font-size:1rem; line-height:1.8; }
    .report-container strong { color:#E8D48B; }
    .report-container table { width:100%; border-collapse:collapse; margin:16px 0; }
    .report-container th { background:#22201b; color:#C9A84C; padding:10px 14px; text-align:left; font-size:0.82rem; text-transform:uppercase; letter-spacing:1px; }
    .report-container td { padding:10px 14px; border-bottom:1px solid #2a2722; color:#d4cdc0; font-size:0.95rem; }

    .footer-text { text-align:center; color:#555; font-size:0.7rem; margin-top:40px; padding:12px; border-top:1px solid #22201b; }
    div[data-testid="stMetricValue"] { font-family:'JetBrains Mono',monospace; }
    .stTextInput > div > div > input { font-family:'JetBrains Mono',monospace; font-size:1.1rem; text-transform:uppercase; }
    div[data-testid="stProgress"] > div { background-color:#22201b !important; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ───
for key in ["done", "ticker", "data", "info", "hist_5y", "comp_df", "structured", "report_text", "price"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "done" not in st.session_state:
    st.session_state.done = False


# ─── Groq API ───
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

def call_groq(prompt: str, data_context: str) -> str:
    import time
    system_prompt = """You are a senior equity research analyst with CFA credentials and 15+ years of experience.

IMPORTANT: Start your response with a JSON block in EXACTLY this format:
```json
{
  "scorecard": {"profitability": 7, "growth": 6, "balance_sheet": 8, "cash_flow": 7, "valuation": 5, "overall": 6.6},
  "verdict": "BUY",
  "confidence": "MEDIUM",
  "target_bull": 0.0,
  "target_base": 0.0,
  "target_bear": 0.0,
  "risks": [
    {"text": "Describe risk here", "severity": "HIGH"},
    {"text": "Describe risk here", "severity": "MEDIUM"},
    {"text": "Describe risk here", "severity": "LOW"}
  ]
}
```
Replace all placeholder values with your actual assessment. Scores 1-10. Verdict: STRONG BUY / BUY / HOLD / SELL / STRONG SELL. Confidence: HIGH / MEDIUM / LOW. Targets are price numbers.

After the JSON, write the full report with these EXACT markdown headings:

## Company Overview
## Financial Health
## Valuation Analysis
## Growth Analysis
## Technical Signals
## Competitive Positioning
## Key Risks
## Recent News & Catalysts
## Target Price & Verdict

Rules: Use markdown tables for comparisons. Bold key numbers. Cite actual figures from the data. Keep paragraph text and table text the same size — do not use sub-headings excessively. Write for sophisticated investors."""

    models = ["llama-3.3-70b-versatile", "llama3-70b-8192"]
    for model in models:
        for attempt in range(3):
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"{prompt}\n\n--- FINANCIAL DATA ---\n{data_context}"},
                        ],
                        "temperature": 0.3, "max_tokens": 6000,
                    },
                    timeout=90,
                )
                if resp.status_code == 429:
                    time.sleep((attempt + 1) * 10); continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except requests.exceptions.Timeout:
                if attempt < 2: time.sleep(5); continue
                break
            except Exception:
                if attempt < 2: time.sleep(3); continue
                break
    return None


def parse_structured_data(raw: str) -> tuple:
    structured = {}
    report_text = raw
    match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        try:
            structured = json.loads(match.group(1))
            report_text = raw[match.end():].strip()
        except Exception:
            pass
    return structured, report_text


# ─── Technical Indicators ───
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9).mean()
    return macd_line, signal, macd_line - signal

def calc_bollinger(series, period=20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma, sma + 2 * std, sma - 2 * std


# ─── Data Fetching ───
@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    try: info = stock.info
    except: info = {}
    try:
        hist_5y = stock.history(period="5y")
        if isinstance(hist_5y.columns, pd.MultiIndex):
            hist_5y.columns = hist_5y.columns.get_level_values(0)
    except: hist_5y = pd.DataFrame()
    try: income = stock.financials
    except: income = pd.DataFrame()
    try: recs = stock.recommendations
    except: recs = pd.DataFrame()
    try: news = stock.news[:8] if stock.news else []
    except: news = []
    return {"info": info, "hist_5y": hist_5y, "income": income, "recommendations": recs, "news": news}


def build_data_context(data: dict, ticker: str) -> str:
    info = data["info"]
    hist = data["hist_5y"].tail(252) if not data["hist_5y"].empty else pd.DataFrame()
    sections = []

    basics = {"Ticker": ticker, "Name": info.get("longName","N/A"), "Sector": info.get("sector","N/A"),
               "Industry": info.get("industry","N/A"), "Country": info.get("country","N/A"),
               "Description": info.get("longBusinessSummary","N/A")[:500]}
    sections.append("=== COMPANY ===\n" + "\n".join(f"{k}: {v}" for k,v in basics.items()))

    valuation = {"Current Price": info.get("currentPrice", info.get("regularMarketPrice","N/A")),
                 "Market Cap": info.get("marketCap","N/A"), "EV": info.get("enterpriseValue","N/A"),
                 "Trailing P/E": info.get("trailingPE","N/A"), "Forward P/E": info.get("forwardPE","N/A"),
                 "PEG": info.get("pegRatio","N/A"), "P/B": info.get("priceToBook","N/A"),
                 "EV/EBITDA": info.get("enterpriseToEbitda","N/A"), "52W High": info.get("fiftyTwoWeekHigh","N/A"),
                 "52W Low": info.get("fiftyTwoWeekLow","N/A")}
    sections.append("=== VALUATION ===\n" + "\n".join(f"{k}: {v}" for k,v in valuation.items()))

    profit = {"Gross Margin": info.get("grossMargins","N/A"), "Op Margin": info.get("operatingMargins","N/A"),
              "Net Margin": info.get("profitMargins","N/A"), "ROE": info.get("returnOnEquity","N/A"),
              "ROA": info.get("returnOnAssets","N/A"), "Revenue": info.get("totalRevenue","N/A"),
              "Net Income": info.get("netIncomeToCommon","N/A"), "EBITDA": info.get("ebitda","N/A"),
              "EPS": info.get("trailingEps","N/A"), "Fwd EPS": info.get("forwardEps","N/A")}
    sections.append("=== PROFITABILITY ===\n" + "\n".join(f"{k}: {v}" for k,v in profit.items()))

    balance = {"Cash": info.get("totalCash","N/A"), "Debt": info.get("totalDebt","N/A"),
               "D/E": info.get("debtToEquity","N/A"), "Current Ratio": info.get("currentRatio","N/A"),
               "FCF": info.get("freeCashflow","N/A"), "Op CF": info.get("operatingCashflow","N/A")}
    sections.append("=== BALANCE SHEET ===\n" + "\n".join(f"{k}: {v}" for k,v in balance.items()))

    income_stmt = data["income"]
    if not income_stmt.empty and "Total Revenue" in income_stmt.index:
        try:
            rev = income_stmt.loc["Total Revenue"].dropna().sort_index()
            if len(rev) >= 2:
                g = [(rev.iloc[i]-rev.iloc[i-1])/abs(rev.iloc[i-1])*100 for i in range(1,len(rev))]
                sections.append(f"=== REVENUE GROWTH ===\n{', '.join(f'{x:.1f}%' for x in g)} (recent last)")
        except: pass

    if not hist.empty and "Close" in hist.columns:
        close = hist["Close"]
        rsi = calc_rsi(close)
        _, _, macd_hist = calc_macd(close)
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        cur = close.iloc[-1]
        tech = {"RSI(14)": f"{rsi.iloc[-1]:.1f}" if not pd.isna(rsi.iloc[-1]) else "N/A",
                "MACD Hist": f"{macd_hist.iloc[-1]:.4f}" if not pd.isna(macd_hist.iloc[-1]) else "N/A",
                "SMA50": f"{sma50.iloc[-1]:.2f}" if not pd.isna(sma50.iloc[-1]) else "N/A",
                "SMA200": f"{sma200.iloc[-1]:.2f}" if not pd.isna(sma200.iloc[-1]) else "N/A",
                "vs SMA50": "Above" if not pd.isna(sma50.iloc[-1]) and cur > sma50.iloc[-1] else "Below",
                "vs SMA200": "Above" if not pd.isna(sma200.iloc[-1]) and cur > sma200.iloc[-1] else "Below"}
        sections.append("=== TECHNICAL ===\n" + "\n".join(f"{k}: {v}" for k,v in tech.items()))
        perf = {}
        for label, days in [("1W",5),("1M",21),("3M",63),("6M",126),("1Y",252)]:
            if len(close) > days:
                perf[label] = f"{(cur-close.iloc[-days])/close.iloc[-days]*100:.1f}%"
        sections.append("=== PERFORMANCE ===\n" + "\n".join(f"{k}: {v}" for k,v in perf.items()))

    analyst = {"Target Mean": info.get("targetMeanPrice","N/A"), "Target High": info.get("targetHighPrice","N/A"),
               "Target Low": info.get("targetLowPrice","N/A"), "Num Analysts": info.get("numberOfAnalystOpinions","N/A"),
               "Consensus": info.get("recommendationKey","N/A")}
    sections.append("=== ANALYST ===\n" + "\n".join(f"{k}: {v}" for k,v in analyst.items()))

    recs = data["recommendations"]
    if recs is not None and not recs.empty:
        try: sections.append(f"=== RECS ===\n{recs.tail(4).to_string()}")
        except: pass

    news = data["news"]
    if news:
        lines = [f"- [{n.get('publisher','')}] {n.get('title','')}" for n in news[:6] if n.get("title")]
        sections.append("=== NEWS ===\n" + "\n".join(lines))

    return "\n\n".join(sections)


@st.cache_data(ttl=300)
def fetch_competitor_data(sector: str, ticker: str) -> pd.DataFrame:
    idx_peers = {
        "BBRI.JK": ["BBCA.JK","BMRI.JK","BBNI.JK","BRIS.JK"],
        "BBCA.JK": ["BBRI.JK","BMRI.JK","BBNI.JK","BRIS.JK"],
        "TLKM.JK": ["EXCL.JK","ISAT.JK","TBIG.JK"],
        "ASII.JK": ["UNTR.JK","SMSM.JK","AUTO.JK"],
        "UNVR.JK": ["ICBP.JK","INDF.JK","MYOR.JK"],
        "GOTO.JK": ["BUKA.JK","EMTK.JK"],
    }
    sector_peers = {
        "Technology": ["AAPL","MSFT","GOOGL","META","NVDA"],
        "Financial Services": ["JPM","BAC","GS","MS","C"],
        "Healthcare": ["JNJ","UNH","PFE","MRK","ABBV"],
        "Energy": ["XOM","CVX","SHEL","TTE","COP"],
        "Consumer Cyclical": ["AMZN","TSLA","HD","NKE","MCD"],
        "Communication Services": ["GOOGL","META","DIS","NFLX","CMCSA"],
        "Industrials": ["CAT","HON","UPS","BA","GE"],
        "Consumer Defensive": ["PG","KO","PEP","WMT","COST"],
        "Basic Materials": ["LIN","APD","NEM","FCX","DOW"],
        "Real Estate": ["PLD","AMT","CCI","SPG","EQIX"],
        "Utilities": ["NEE","DUK","SO","AEP","D"],
    }
    peers = idx_peers.get(ticker.upper(), [])
    if sector in sector_peers:
        for p in sector_peers[sector]:
            if p.upper() != ticker.upper() and p not in peers:
                peers.append(p)
            if len(peers) >= 5: break
    rows = []
    for p in peers[:5]:
        try:
            pi = yf.Ticker(p).info
            rows.append({"Ticker": p, "Name": pi.get("shortName", p),
                         "Market Cap": pi.get("marketCap"), "P/E": pi.get("trailingPE"),
                         "P/B": pi.get("priceToBook"), "ROE": pi.get("returnOnEquity"),
                         "Net Margin": pi.get("profitMargins"), "Div Yield": pi.get("dividendYield")})
        except: continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─── Native Visual Components ───
SCORE_COLORS = {(1,3): "#ef4444", (4,5): "#f97316", (6,7): "#C9A84C", (8,10): "#22c55e"}

def score_color(s):
    s = int(s) if s else 0
    for (lo, hi), c in SCORE_COLORS.items():
        if lo <= s <= hi: return c
    return "#8a8578"

def render_scorecard(scorecard: dict):
    st.markdown('<div class="section-label">Financial Health Scorecard</div>', unsafe_allow_html=True)
    overall = scorecard.get("overall", 0)
    col_overall = score_color(overall)

    st.markdown(f"""
    <div style="text-align:center;background:#1a1714;border:1px solid #33302a;border-radius:12px;padding:20px 20px 8px;">
        <div style="color:#8a8578;font-size:0.75rem;text-transform:uppercase;letter-spacing:2px;">Overall Score</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:2.8rem;font-weight:700;color:{col_overall};">{overall:.1f}<span style="font-size:1rem;color:#8a8578;">/10</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    labels = [("profitability","Profitability"), ("growth","Growth"),
              ("balance_sheet","Balance Sheet"), ("cash_flow","Cash Flow"), ("valuation","Valuation")]

    for key, label in labels:
        score = scorecard.get(key, 0) or 0
        color = score_color(score)
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f'<p style="color:#8a8578;font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;margin:2px 0;">{label}</p>', unsafe_allow_html=True)
            st.progress(int(score * 10))
        with col_b:
            st.markdown(f'<p style="font-family:JetBrains Mono,monospace;font-size:1.1rem;font-weight:700;color:{color};margin-top:18px;">{score}/10</p>', unsafe_allow_html=True)


def render_verdict(structured: dict, current_price: float):
    st.markdown('<div class="section-label">Verdict & Price Targets</div>', unsafe_allow_html=True)
    verdict = structured.get("verdict", "HOLD").upper()
    confidence = structured.get("confidence", "MEDIUM").upper()
    bull = structured.get("target_bull", 0) or 0
    base = structured.get("target_base", 0) or 0
    bear = structured.get("target_bear", 0) or 0

    verdict_colors = {"STRONG BUY": "#22c55e", "BUY": "#4ade80", "HOLD": "#C9A84C",
                      "SELL": "#f87171", "STRONG SELL": "#ef4444"}
    verdict_bg = {"STRONG BUY": "#0d4a1f", "BUY": "#0f3a1a", "HOLD": "#3a3000",
                  "SELL": "#3a0f0f", "STRONG SELL": "#4a0a0a"}
    color = verdict_colors.get(verdict, "#C9A84C")
    bg = verdict_bg.get(verdict, "#22201b")

    st.markdown(f"""
    <div style="background:{bg};border:2px solid {color};border-radius:12px;padding:20px;text-align:center;">
        <div style="color:#8a8578;font-size:0.75rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">AI Verdict</div>
        <div style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:900;color:{color};">{verdict}</div>
        <div style="margin-top:8px;display:inline-block;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;padding:3px 12px;border-radius:20px;background:#22201b;color:#8a8578;border:1px solid #33302a;">Confidence: {confidence}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if bull and base and bear and current_price:
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown(f'<div class="metric-card"><div class="metric-name">Bear Case</div><div class="metric-val" style="color:#ef4444;">{bear:,.2f}</div></div>', unsafe_allow_html=True)
        with t2:
            st.markdown(f'<div class="metric-card"><div class="metric-name">Base Case</div><div class="metric-val" style="color:#C9A84C;">{base:,.2f}</div></div>', unsafe_allow_html=True)
        with t3:
            st.markdown(f'<div class="metric-card"><div class="metric-name">Bull Case</div><div class="metric-val" style="color:#22c55e;">{bull:,.2f}</div></div>', unsafe_allow_html=True)

        # Target range chart
        all_vals = [v for v in [bear, current_price, base, bull] if v]
        if all_vals:
            min_v, max_v = min(all_vals)*0.93, max(all_vals)*1.07
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[bear, bull], y=[0.5, 0.5], mode="lines",
                line=dict(color="#33302a", width=16), showlegend=False, hoverinfo="skip"))
            for x, label, c in [(bull,"Bull","#22c55e"),(base,"Base","#C9A84C"),(bear,"Bear","#ef4444")]:
                if x:
                    fig.add_trace(go.Scatter(x=[x], y=[0.5], mode="markers+text",
                        marker=dict(color=c, size=16, symbol="diamond" if label=="Base" else "circle"),
                        text=[f"{label}\n{x:,.0f}"], textposition="top center",
                        textfont=dict(color=c, size=9), name=label,
                        hovertemplate=f"{label}: {x:,.2f}<extra></extra>"))
            fig.add_vline(x=current_price, line_dash="dash", line_color="#F0E6D3", line_width=1.5,
                          annotation_text=f"Now {current_price:,.2f}", annotation_font_color="#F0E6D3", annotation_font_size=9)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=110, margin=dict(l=0,r=0,t=30,b=0), showlegend=False,
                              xaxis=dict(range=[min_v,max_v], showgrid=False, zeroline=False),
                              yaxis=dict(visible=False, range=[0,1]))
            st.plotly_chart(fig, use_container_width=True)


def render_risks(risks: list):
    st.markdown('<div class="section-label">Key Risks</div>', unsafe_allow_html=True)
    for r in risks:
        severity = (r.get("severity") or "MEDIUM").upper()
        text = r.get("text", "")
        if severity == "HIGH":
            st.error(f"**HIGH** — {text}")
        elif severity == "MEDIUM":
            st.warning(f"**MEDIUM** — {text}")
        else:
            st.success(f"**LOW** — {text}")


def render_analyst_section(info: dict, recs_df):
    st.markdown('<div class="section-label">Analyst Estimates</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        target_mean = info.get("targetMeanPrice")
        target_high = info.get("targetHighPrice")
        target_low  = info.get("targetLowPrice")
        num_analysts = info.get("numberOfAnalystOpinions", 0)
        rec_key = (info.get("recommendationKey") or "N/A").upper()
        current = info.get("currentPrice", info.get("regularMarketPrice", 0)) or 0

        if target_mean and target_high and target_low and current:
            upside = (target_mean - current) / current * 100
            sign = "+" if upside >= 0 else ""
            color = "#22c55e" if upside >= 0 else "#ef4444"

            st.markdown(f"""
            <div style="background:#1a1714;border:1px solid #33302a;border-radius:12px;padding:20px;text-align:center;">
                <div style="color:#8a8578;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Consensus Price Target</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:2.2rem;font-weight:700;color:#F0E6D3;">{target_mean:,.2f}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;color:{color};margin-bottom:12px;">{sign}{upside:.1f}% from current price</div>
                <div style="display:flex;justify-content:space-around;padding-top:12px;border-top:1px solid #33302a;">
                    <div><div style="color:#8a8578;font-size:0.7rem;">LOW</div><div style="font-family:'JetBrains Mono',monospace;color:#ef4444;">{target_low:,.2f}</div></div>
                    <div><div style="color:#8a8578;font-size:0.7rem;">MEAN</div><div style="font-family:'JetBrains Mono',monospace;color:#C9A84C;">{target_mean:,.2f}</div></div>
                    <div><div style="color:#8a8578;font-size:0.7rem;">HIGH</div><div style="font-family:'JetBrains Mono',monospace;color:#22c55e;">{target_high:,.2f}</div></div>
                </div>
                <div style="margin-top:12px;color:#8a8578;font-size:0.8rem;">{num_analysts} analysts &nbsp;·&nbsp; Consensus: <strong style="color:#C9A84C;">{rec_key}</strong></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Analyst target data not available for this ticker.")

    with col2:
        if recs_df is not None and not recs_df.empty:
            try:
                latest = recs_df.iloc[-1]
                strong_buy  = int(latest.get("strongBuy", 0) or 0)
                buy         = int(latest.get("buy", 0) or 0)
                hold        = int(latest.get("hold", 0) or 0)
                sell        = int(latest.get("sell", 0) or 0)
                strong_sell = int(latest.get("strongSell", 0) or 0)
                total = strong_buy + buy + hold + sell + strong_sell

                if total > 0:
                    bullish_pct = (strong_buy + buy) / total * 100
                    bearish_pct = (sell + strong_sell) / total * 100
                    neutral_pct = hold / total * 100

                    fig = go.Figure(go.Bar(
                        x=[strong_buy, buy, hold, sell, strong_sell],
                        y=["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
                        orientation="h",
                        marker_color=["#22c55e","#4ade80","#C9A84C","#f87171","#ef4444"],
                        text=[str(v) if v > 0 else "" for v in [strong_buy, buy, hold, sell, strong_sell]],
                        textposition="inside",
                        textfont=dict(color="#1a1714", size=11, family="JetBrains Mono"),
                    ))
                    fig.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=220, margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(gridcolor="#22201b", tickfont=dict(size=11)),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    reliability = "Strong" if bullish_pct > 60 or bearish_pct > 60 else "Moderate" if bullish_pct > 40 or bearish_pct > 40 else "Mixed"
                    rel_color = "#22c55e" if reliability == "Strong" else "#C9A84C" if reliability == "Moderate" else "#f97316"
                    st.markdown(f'<p style="text-align:center;color:#8a8578;font-size:0.82rem;">Consensus Strength: <strong style="color:{rel_color};">{reliability}</strong> &nbsp;·&nbsp; {bullish_pct:.0f}% bullish · {neutral_pct:.0f}% neutral · {bearish_pct:.0f}% bearish</p>', unsafe_allow_html=True)
                else:
                    st.info("No breakdown available.")
            except:
                st.info("Recommendation data not available.")
        else:
            st.info("Recommendation data not available for this ticker.")


# ─── Main UI ───
st.markdown('<p class="hero-title">AI Stock Analyst</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Institutional-grade equity research · Enter any ticker worldwide</p>', unsafe_allow_html=True)

col_input, col_btn = st.columns([3, 1])
with col_input:
    ticker_input = st.text_input("Ticker", placeholder="e.g. AAPL, BBRI.JK, TSLA, 0700.HK", label_visibility="collapsed")
with col_btn:
    analyse_btn = st.button("Analyse", use_container_width=True, type="primary")

if analyse_btn and ticker_input.strip():
    ticker = ticker_input.strip().upper()
    with st.status("Gathering data...", expanded=True) as status:
        st.write("Fetching market data...")
        data = fetch_stock_data(ticker)
        info = data["info"]
        if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
            st.error(f"Could not find data for {ticker}. Check the ticker and try again.")
            st.stop()

        st.write("Computing technical indicators...")
        data_context = build_data_context(data, ticker)

        st.write("Fetching competitor data...")
        comp_df = fetch_competitor_data(info.get("sector",""), ticker)
        if not comp_df.empty:
            data_context += "\n\n=== COMPETITORS ===\n" + comp_df.to_string(index=False)

        st.write("Running AI analysis (~15 seconds)...")
        raw_output = call_groq(f"Full equity research report for {ticker} ({info.get('longName', ticker)}). Use ONLY the data provided.", data_context)
        if raw_output is None:
            st.error("AI analysis failed. Check your Groq API key in secrets.")
            st.stop()

        structured, report_text = parse_structured_data(raw_output)
        price = info.get("currentPrice", info.get("regularMarketPrice", 0)) or 0

        # ── Save everything to session state ──
        st.session_state.done = True
        st.session_state.ticker = ticker
        st.session_state.data = data
        st.session_state.info = info
        st.session_state.hist_5y = data["hist_5y"]
        st.session_state.comp_df = comp_df
        st.session_state.structured = structured
        st.session_state.report_text = report_text
        st.session_state.price = price

        status.update(label="Analysis complete", state="complete")

elif analyse_btn:
    st.warning("Please enter a ticker symbol.")


# ─── Display Results (from session state) ───
if st.session_state.done:
    ticker    = st.session_state.ticker
    info      = st.session_state.info
    hist_5y   = st.session_state.hist_5y
    comp_df   = st.session_state.comp_df
    structured = st.session_state.structured
    report_text = st.session_state.report_text
    price     = st.session_state.price
    data      = st.session_state.data

    # ── Header ──
    prev_close = info.get("previousClose", price) or price
    change = price - prev_close
    change_pct = (change / prev_close * 100) if prev_close else 0
    sign = "+" if change >= 0 else ""
    delta_color = "pos" if change >= 0 else "neg"

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:20px;margin-bottom:6px;flex-wrap:wrap;">
        <span class="ticker-badge">{ticker}</span>
        <span style="font-size:1.6rem;font-weight:700;color:#F0E6D3;font-family:'JetBrains Mono',monospace;">{price:,.2f}</span>
        <span class="{delta_color}" style="font-size:1.1rem;font-family:'JetBrains Mono',monospace;">{sign}{change:.2f} ({sign}{change_pct:.2f}%)</span>
    </div>
    <div style="color:#8a8578;font-size:0.85rem;margin-bottom:20px;">
        {info.get('longName', ticker)} &nbsp;·&nbsp; {info.get('sector','')} &nbsp;·&nbsp; {info.get('industry','')} &nbsp;·&nbsp; {info.get('country','')}
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics ──
    metric_cols = st.columns(6)
    metrics = [
        ("Market Cap",  f"${info.get('marketCap',0)/1e9:,.1f}B" if info.get("marketCap") else "N/A"),
        ("P/E Ratio",   f"{info.get('trailingPE',0):.1f}"       if info.get("trailingPE") else "N/A"),
        ("EPS",         f"{info.get('trailingEps',0):.2f}"      if info.get("trailingEps") else "N/A"),
        ("Div Yield",   f"{info.get('dividendYield',0)*100:.2f}%" if info.get("dividendYield") else "N/A"),
        ("52W Range",   f"{info.get('fiftyTwoWeekLow',0):,.0f} – {info.get('fiftyTwoWeekHigh',0):,.0f}"),
        ("Beta",        f"{info.get('beta',0):.2f}"             if info.get("beta") else "N/A"),
    ]
    for i, (label, val) in enumerate(metrics):
        with metric_cols[i]:
            st.markdown(f'<div class="metric-card"><div class="metric-name">{label}</div><div class="metric-val">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Interactive Price Chart ──
    st.markdown('<div class="section-label">Price Chart</div>', unsafe_allow_html=True)
    period_options = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252, "5Y": 1260}
    selected_period = st.radio("Period", list(period_options.keys()), index=4, horizontal=True, label_visibility="collapsed")

    chart_col, ta_col = st.columns([2, 1])
    with chart_col:
        if not hist_5y.empty and "Close" in hist_5y.columns:
            days = period_options[selected_period]
            hist = hist_5y.tail(days)
            close = hist["Close"]
            sma50 = close.rolling(50).mean()
            sma200 = close.rolling(200).mean()
            _, bb_upper, bb_lower = calc_bollinger(close)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"],
                low=hist["Low"], close=hist["Close"],
                increasing_line_color="#6BCB77", decreasing_line_color="#FF6B6B", name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=sma50, name="SMA 50", line=dict(color="#C9A84C", width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=sma200, name="SMA 200", line=dict(color="#8a8578", width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=bb_upper, line=dict(color="rgba(201,168,76,0.3)", width=1), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=bb_lower, line=dict(color="rgba(201,168,76,0.3)", width=1), fill="tonexty", fillcolor="rgba(201,168,76,0.05)", showlegend=False), row=1, col=1)
            bar_colors = ["#6BCB77" if c >= o else "#FF6B6B" for c, o in zip(hist["Close"], hist["Open"])]
            fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], marker_color=bar_colors, opacity=0.5, name="Volume"), row=2, col=1)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=450, margin=dict(l=0,r=0,t=10,b=0),
                legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=10)),
                xaxis_rangeslider_visible=False,
                yaxis=dict(gridcolor="#22201b"), yaxis2=dict(gridcolor="#22201b"),
                xaxis=dict(gridcolor="#22201b"), xaxis2=dict(gridcolor="#22201b"))
            st.plotly_chart(fig, use_container_width=True)

    with ta_col:
        if not hist_5y.empty and "Close" in hist_5y.columns:
            close_1y = hist_5y.tail(252)["Close"]
            rsi = calc_rsi(close_1y)
            _, _, macd_hist_vals = calc_macd(close_1y)

            rsi_val = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 0
            macd_val = macd_hist_vals.iloc[-1] if not pd.isna(macd_hist_vals.iloc[-1]) else 0
            sma50_val = close_1y.rolling(50).mean().iloc[-1]
            sma200_val = close_1y.rolling(200).mean().iloc[-1]
            trend = "Bullish" if not pd.isna(sma50_val) and not pd.isna(sma200_val) and sma50_val > sma200_val else "Bearish"
            price_pos = (close_1y.iloc[-1] - close_1y.min()) / (close_1y.max() - close_1y.min()) * 100 if close_1y.max() != close_1y.min() else 50

            st.markdown('<div class="section-label">Technical Signals</div>', unsafe_allow_html=True)
            rsi_label = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
            rsi_color = "neg" if rsi_val > 70 else "pos" if rsi_val < 30 else ""
            macd_color = "pos" if macd_val > 0 else "neg"
            trend_color = "pos" if trend == "Bullish" else "neg"

            for label, val, desc, color in [
                ("RSI (14)",    f"{rsi_val:.1f}", rsi_label, rsi_color),
                ("MACD",        f"{macd_val:.4f}", "Bullish" if macd_val > 0 else "Bearish", macd_color),
                ("Trend (SMA)", "", trend, trend_color),
                ("52W Position",f"{price_pos:.0f}%", "of range", ""),
            ]:
                st.markdown(f'<div class="metric-card"><div class="metric-name">{label}</div><div class="metric-val">{val} <span class="{color}" style="font-size:0.82rem;">{desc}</span></div></div>', unsafe_allow_html=True)

            fig_rsi = go.Figure(go.Indicator(
                mode="gauge+number", value=rsi_val,
                gauge=dict(axis=dict(range=[0,100], tickcolor="#8a8578"), bar=dict(color="#C9A84C"), bgcolor="#1a1714",
                           steps=[dict(range=[0,30],color="#1a3a1a"),dict(range=[30,70],color="#22201b"),dict(range=[70,100],color="#3a1a1a")],
                           threshold=dict(line=dict(color="#FF6B6B",width=2),thickness=0.8,value=rsi_val)),
                number=dict(font=dict(color="#F0E6D3", family="JetBrains Mono"))))
            fig_rsi.update_layout(height=180, margin=dict(l=20,r=20,t=30,b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#8a8578"))
            st.plotly_chart(fig_rsi, use_container_width=True)

    # ── Competitor Table ──
    if not comp_df.empty:
        st.markdown('<div class="section-label">Competitor Comparison</div>', unsafe_allow_html=True)
        disp = comp_df.copy()
        for col in ["ROE", "Net Margin", "Div Yield"]:
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
        if "Market Cap" in disp.columns:
            disp["Market Cap"] = disp["Market Cap"].apply(lambda x: f"${x/1e9:.1f}B" if pd.notna(x) and x else "N/A")
        if "P/E" in disp.columns:
            disp["P/E"] = disp["P/E"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        if "P/B" in disp.columns:
            disp["P/B"] = disp["P/B"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Analyst Section ──
    render_analyst_section(info, data["recommendations"])

    # ── Visual AI Summary ──
    st.markdown('<div class="section-label">AI Research Summary</div>', unsafe_allow_html=True)

    if structured:
        vis_col1, vis_col2 = st.columns(2)
        with vis_col1:
            render_scorecard(structured.get("scorecard", {}))
        with vis_col2:
            render_verdict(structured, price)

        if structured.get("risks"):
            render_risks(structured.get("risks", []))

    # ── Full Report ──
    st.markdown('<div class="section-label">Full Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown(report_text)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── News ──
    if data["news"]:
        st.markdown('<div class="section-label">Recent News</div>', unsafe_allow_html=True)
        for n in data["news"][:6]:
            title = n.get("title","")
            publisher = n.get("publisher","")
            link = n.get("link","#")
            if title:
                st.markdown(f"**{title}**  \n*{publisher}* · [Read]({link})")
                st.markdown("")

    st.markdown("""
    <div class="footer-text">
        Disclaimer: AI-generated analysis for educational purposes only. Not financial advice.
        Always do your own research. Data: Yahoo Finance · AI: Groq.
    </div>
    """, unsafe_allow_html=True)
