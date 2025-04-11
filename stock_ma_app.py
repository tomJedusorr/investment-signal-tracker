import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
from trendline_equation import fit_trendline_from_lows
import plotly.express as px
from st_social_media_links import SocialMediaIcons

st.set_page_config(page_title="Investment Signal Tracker", page_icon="📈", layout="centered")

@st.cache_data(ttl=30, show_spinner=False)
def convert_comma_number(val):
    if isinstance(val, str):
        return float(val.replace(",", "."))
    return val

@st.cache_data(ttl=30, show_spinner=False)
def fetch_ticker_info(ticker):
    return yf.Ticker(ticker).info

@st.cache_data(ttl=30, show_spinner=False)
def calculate_mas(df, ticker, ma_periods, label_prefix):
    return {
        f"MA{p}{label_prefix}": df[ticker].rolling(window=p).mean().iloc[-1]
        for p in ma_periods
    }

@st.cache_data(ttl=30, show_spinner=False)
def get_latest_mas(tickers, values, horizon):
    summary = []
    ma_periods=[100, 200]

    for ticker, value in zip(tickers, values):
        try:
            df_full = (yf.download(ticker, start="2000-1-1")['Close'])
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            continue

        if df_full.empty or df_full.shape[0] < max(ma_periods):
            print(f"Not enough data for {ticker}")
            continue

        price_exp = fit_trendline_from_lows(ticker)
        mkt_cap = fetch_ticker_info(ticker).get("marketCap")
        latest_close = df_full[ticker].iloc[-1]
        latest_peak = df_full[ticker].max()
        spread = (latest_close / latest_peak) - 1

        # Calculate daily features
        ma_values = calculate_mas(df_full, ticker, ma_periods, "daily")
        daily_returns = df_full[ticker].pct_change()
        drawdown_d = daily_returns.min()
        std_dev_d = daily_returns.std()
        last_daily_return = daily_returns.iloc[-1]
        geo_return_d = (np.prod([1 + r for r in daily_returns[1:]]) ** (1 / len(daily_returns))) - 1
        adj_sharpe_d = geo_return_d / std_dev_d

        # Trendline positioning
        xma100d = (ma_values.get("MA100daily") - latest_close) / ma_values.get("MA100daily")
        xma200d = (ma_values.get("MA200daily") - latest_close) / ma_values.get("MA200daily")
        xtrenp2d = (price_exp.get("trend_p2") - latest_close) / price_exp.get("trend_p2")
        xmkt_cap = (np.log(mkt_cap)) / 100 if mkt_cap else 0
        xspread = -spread
        xworstrd = drawdown_d - last_daily_return
        xstd_devd = std_dev_d - last_daily_return

        features = np.array([xma100d, xma200d, xtrenp2d, adj_sharpe_d, xmkt_cap, xspread, xworstrd, xstd_devd])
        weights = np.array([0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0.05, 0.6])
        daily_y = np.sum(weights * features) * value

        # Similar pattern for weekly, monthly, yearly
        # Weekly
        weekly = df_full.resample('W', label='left').last()
        weekly_returns = weekly[ticker].pct_change()
        std_dev_w = weekly_returns.std()
        last_weekly_return = df_full[ticker].iloc[-1] / df_full[ticker].iloc[-6] - 1
        drawdown_w = weekly_returns.min()
        geo_return_w = (np.prod([1 + r for r in weekly_returns[1:]]) ** (1 / len(weekly_returns))) - 1 
        adj_sharpe_w = geo_return_w / std_dev_w
        xworstrw = drawdown_w - last_weekly_return
        xstd_devw = std_dev_w - last_weekly_return
        weekly_features = np.array([xma100d, xma200d, xtrenp2d, adj_sharpe_w, xmkt_cap, xspread, xworstrw, xstd_devw])
        weekly_y = np.sum(weights * weekly_features) * value

        # Monthly
        monthly = df_full.resample('ME', label='left').last()
        monthly_returns = monthly[ticker].pct_change()
        std_dev_m = monthly_returns.std()
        last_monthly_return = df_full[ticker].iloc[-1] / df_full[ticker].iloc[-22] - 1
        drawdown_m = monthly_returns.min()
        geo_return_m = (np.prod([1 + r for r in monthly_returns[1:]]) ** (1 / len(monthly_returns))) - 1 
        adj_sharpe_m = geo_return_m / std_dev_m
        xma100m = (calculate_mas(monthly, ticker, ma_periods, "weekly").get("MA100weekly") - latest_close) / calculate_mas(monthly, ticker, ma_periods, "weekly").get("MA100weekly")
        xma200m = (calculate_mas(monthly, ticker, ma_periods, "weekly").get("MA200weekly") - latest_close) / calculate_mas(monthly, ticker, ma_periods, "weekly").get("MA200weekly")
        xworstrm = drawdown_m - last_monthly_return
        xstd_devm = std_dev_m - last_monthly_return
        monthly_features = np.array([xma100m, xma200m, xtrenp2d, adj_sharpe_m, xmkt_cap, xspread, xworstrm, xstd_devm])
        monthly_y = np.sum(weights * monthly_features) * value

        # Yearly
        yearly = df_full.resample('YE', label='left').last()
        yearly_returns = yearly[ticker].pct_change()
        std_dev_y = std_dev_d * np.sqrt(252)
        last_yearly_return = df_full[ticker].iloc[-1] / df_full[ticker].iloc[-253] - 1
        drawdown_y = yearly_returns.min()
        geo_return_y = (np.prod([1 + r for r in yearly_returns[1:]]) ** (1 / len(yearly_returns))) - 1 
        adj_sharpe_y = geo_return_y / std_dev_y
        xworstry = drawdown_y - last_yearly_return
        xstd_devy = std_dev_y - last_yearly_return
        yearly_features = np.array([xma100m, xma200m, xtrenp2d, adj_sharpe_y, xmkt_cap, xspread, xworstry, xstd_devy])
        yearly_y = np.sum(weights * yearly_features) * value

        summary.append({
            "Ticker": ticker,
            "Daily Investment": daily_y,
            "Weekly Investment": weekly_y,
            "Monthly Investment": monthly_y,
            "Yearly Investment": yearly_y
        })

    df = pd.DataFrame(summary)
    return df[['Ticker', f"{horizon.title()} Investment"]]

# ------------------ UI ------------------

st.title("📈 Investment Sizing Dashboard")

tickers_input = st.text_input("Enter tickers (semicolon-separated)")
prices_input = st.text_input("Enter position values (semicolon-separated, comma for decimals)")
horizon = st.selectbox("Select horizon", ["daily", "weekly", "monthly", "yearly"])

tickers = [t.strip().upper() for t in tickers_input.split(";") if t.strip()]
values = [convert_comma_number(v.strip()) for v in prices_input.split(";") if v.strip()]

if st.button("Run Analysis"):
    if len(tickers) != len(values):
        st.error("Mismatch between number of tickers and values.")
        st.stop()

    with st.spinner("Running analysis..."):
        df_result = get_latest_mas(tickers, values, horizon)

    if not df_result.empty:
        invest_col = df_result.columns[1]
        df_result[invest_col] = df_result[invest_col].astype(float).round(2)

        st.markdown(f"### 📊 {invest_col} per Ticker")

        fig = px.bar(
            df_result,
            x="Ticker",
            y=invest_col,
            text=invest_col,
            title=f"Suggested {invest_col}",
            color=invest_col,
            color_continuous_scale="Blues"
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(yaxis_title="Amount", xaxis_title="Ticker", height=700)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔍 View Full Table"):
            st.dataframe(df_result)

        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "investment_suggestions.csv", "text/csv")

    else:
        st.warning("No data available.")

social_media_links = [
    "https://www.linkedin.com/in/rapha%C3%ABl-dahomay/",
    "https://github.com/tomJedusorr"
]

social_media_icons = SocialMediaIcons(social_media_links)

# Optional: Add a divider for visual separation
st.divider()

# Render the social media icons
st.subheader("Follow the creator")
social_media_icons.render()
