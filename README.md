# 🔍 AI Stock Analyst

Institutional-grade equity research powered by AI. Enter any ticker worldwide and get a comprehensive analyst report in seconds.

## Features

- **Real-time data** — prices, financials, ratios, balance sheet, cash flow from Yahoo Finance
- **Technical analysis** — RSI, MACD, Bollinger Bands, moving averages, trend detection
- **Competitor comparison** — auto-detects peers and compares key metrics
- **AI research report** — structured like a professional equity research note:
  - Financial health scorecard (1-10 ratings)
  - Valuation analysis (historical + peer comparison)
  - Bull case / Bear case / Base case target prices
  - Risk assessment with severity ratings
  - Final verdict: Strong Buy → Strong Sell
- **Recent news** — latest headlines for the stock
- **Interactive charts** — candlestick with overlays, RSI gauge, volume

## Works with any ticker

- US stocks: `AAPL`, `TSLA`, `NVDA`
- Indonesia (IDX): `BBRI.JK`, `TLKM.JK`, `ASII.JK`
- Hong Kong: `0700.HK`
- Any Yahoo Finance ticker

## Setup

### 1. Get a free Gemini API key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Click **Get API Key** → **Create API Key**
3. Copy the key

### 2. Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select repo → `app.py`
3. Before deploying, click **Advanced settings** → **Secrets**
4. Add your API key:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

5. Click **Deploy**

### 3. Run locally (optional)

```bash
pip install -r requirements.txt

# Create .streamlit/secrets.toml
echo 'GEMINI_API_KEY = "your-key"' > .streamlit/secrets.toml

streamlit run app.py
```

## Tech Stack

- **Streamlit** — web app framework
- **yfinance** — market data
- **Plotly** — interactive charts
- **Google Gemini 2.0 Flash** — AI analysis (free tier)

## Cost

**$0.** Gemini free tier covers ~1,500 analyses per day.

---

*⚠️ This tool is for educational and informational purposes only. Not financial advice.*
