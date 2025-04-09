import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
from trendline_equation import fit_trendline_from_lows

def convert_comma_number(val):
    if isinstance(val, str):
        return float(val.replace(",", "."))
    return val

def calculate_mas(df, ma_periods, label_prefix):
    return {
        f"MA{p}{label_prefix}": df.rolling(window=p).mean().iloc[-1]
        for p in ma_periods
    }

def get_latest_mas(tickers, values, horizon):
    summary = []
    ma_periods=[100, 200]

    for ticker, value in zip(tickers, values):
        try:
            df_full = yf.download(ticker)['Close']
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            continue

        if df_full.empty or df_full.shape[0] < max(ma_periods):
            print(f"Not enough data for {ticker}")
            continue

        price_exp = fit_trendline_from_lows(ticker)

        dff = yf.Ticker(ticker)
        mkt_cap = dff.info.get("marketCap")

        ma_values = {}

        # Daily MAs
        ma_values.update(calculate_mas(df_full[ticker], ma_periods, "daily"))
        daily_returns = df_full[ticker].pct_change()
        drawdown_d = daily_returns.min()
        std_dev_d = daily_returns.std()
        last_daily_return = daily_returns.iloc[-1]
        geo_return_d = (np.prod([1 + r for r in daily_returns[1:]]) ** (1 / len(daily_returns))) - 1 
        adj_sharpe_d = geo_return_d / std_dev_d

        # Weekly
        weekly = df_full.resample('W', label='left').last()
        weekly_returns = weekly[ticker].pct_change()
        ma_values.update(calculate_mas(weekly[ticker], ma_periods, "weekly"))
        std_dev_w = weekly_returns.std()
        last_weekly_return = df_full[ticker].iloc[-1] / df_full[ticker].iloc[-6] - 1
        drawdown_w = weekly_returns.min()
        geo_return_w = (np.prod([1 + r for r in weekly_returns[1:]]) ** (1 / len(weekly_returns))) - 1 
        adj_sharpe_w = geo_return_w / std_dev_w

        # Monthly
        monthly = df_full.resample('M', label='left').last()
        monthly_returns = monthly[ticker].pct_change()
        ma_values.update(calculate_mas(monthly[ticker], ma_periods, "monthly"))
        std_dev_m = monthly_returns.std()
        last_monthly_return = df_full[ticker].iloc[-1] / df_full[ticker].iloc[-22] - 1
        drawdown_m = monthly_returns.min()
        geo_return_m = (np.prod([1 + r for r in monthly_returns[1:]]) ** (1 / len(monthly_returns))) - 1 
        adj_sharpe_m = geo_return_m / std_dev_m

        # Yearly
        yearly = df_full.resample('Y', label='left').last()
        yearly_returns = yearly[ticker].pct_change()
        std_dev_y = std_dev_d * np.sqrt(252)
        last_yearly_return = df_full[ticker].iloc[-1] / df_full[ticker].iloc[-253] - 1
        drawdown_y = yearly_returns.min()
        geo_return_y = (np.prod([1 + r for r in yearly_returns[1:]]) ** (1 / len(yearly_returns))) - 1 
        adj_sharpe_y = geo_return_y / std_dev_y

        latest_close = df_full[ticker].iloc[-1]
        latest_peak = df_full[ticker].max()
        spread = (latest_close / latest_peak) - 1

        # Sizing Formula
        wma100 = 0.05
        wma200 = 0.05
        wtrenp3 = 0.1
        wsharpe = 0.1
        wmkt_cap = 0.1
        wspread = 0.05
        wworstr = 0.1
        wstd_dev = 0.45

        weights = np.array([wma100, wma200, wtrenp3, wsharpe, wmkt_cap, wspread, wworstr, wstd_dev])

        # Daily parameters
        xma100d = ((ma_values.get("MA100daily")) - latest_close) / ma_values.get("MA100daily")
        xma200d = ((ma_values.get("MA200daily")) - latest_close) / ma_values.get("MA200daily")
        xtrenp3d = ((price_exp.get("trend_p2")) - latest_close) / (price_exp.get("trend_p2"))
        xsharped = adj_sharpe_d
        xmkt_cap = (np.log(mkt_cap)) / 100
        xspread = spread * (-1)
        xworstrd = drawdown_d - last_daily_return
        xstd_devd = std_dev_d - last_daily_return

        daily_p = np.array([xma100d, xma200d, xtrenp3d, xsharped, xmkt_cap, xspread, xworstrd, xstd_devd])

        daily_y = (np.sum(weights * daily_p)) * value

        # Weekly parameters
        xsharpew = adj_sharpe_w
        xworstrw = drawdown_w - last_weekly_return
        xstd_devw = std_dev_w - last_weekly_return

        weekly_p = np.array([xma100d, xma200d, xtrenp3d, xsharpew, xmkt_cap, xspread, xworstrw, xstd_devw])

        weekly_y = (np.sum(weights * weekly_p)) * value

        # Monthly parameters
        xma100m = ((ma_values.get("MA100weekly")) - latest_close) / ma_values.get("MA100weekly")
        xma200m = ((ma_values.get("MA200weekly")) - latest_close) / ma_values.get("MA200weekly")
        xsharpem = adj_sharpe_m
        xworstrm = drawdown_m - last_monthly_return
        xstd_devm = std_dev_m - last_monthly_return

        monthly_p = np.array([xma100d, xma200d, xtrenp3d, xsharpem, xmkt_cap, xspread, xworstrm, xstd_devm])

        monthly_y = (np.sum(weights * monthly_p)) * value

        # Yearly parameters
        xtrenp2y = ((price_exp.get("trend_p2")) - latest_close) / (price_exp.get("trend_p2"))
        xsharpey = adj_sharpe_y
        xworstry = drawdown_y - last_yearly_return
        xstd_devy = std_dev_y - last_yearly_return

        yearly_p = np.array([xma100m, xma200m, xtrenp3d, xsharpey, xmkt_cap, xspread, xworstry, xstd_devy])

        yearly_y = (np.sum(weights * yearly_p)) * value

        sharpes = {"Adj Sharpe Daily": adj_sharpe_d,
                   "Adj Sharpe Weekly": adj_sharpe_w,
                   "Adj Sharpe Monthly": adj_sharpe_m,
                   "Adj Sharpe Yearly": adj_sharpe_y}

        summary.append({
            "Ticker": ticker,
            "Latest Close": latest_close,
            "Daily Investment": daily_y,
            "Weekly Investment": weekly_y,
            "Monthly Investment": monthly_y,
            "Yearly Investment": yearly_y,
            **price_exp,
            **ma_values,
            **sharpes,
            "Spread from Peak": spread,
            "Std Dev Daily": std_dev_d,
            "Std Dev Weekly": std_dev_w,
            "Std Dev Monthly": std_dev_m,
            "Std Dev Yearly": std_dev_y,
            "Last Daily Return": last_daily_return,
            "Last Weekly Return": last_weekly_return,
            "Last Monthly Return": last_monthly_return,
            "Last Yearly Return": last_yearly_return,
            "Worst Daily Return": drawdown_d,
            "Worst Weekly Return": drawdown_w,
            "Worst Monthly Return": drawdown_m,
            "Worst yearly Return": drawdown_y,
            "Last Market Cap": mkt_cap
        })

        Table = pd.DataFrame(summary)

    if horizon == "daily":
        return Table[['Ticker', 'Daily Investment']]
    elif horizon == "weekly":
        return Table[['Ticker', 'Weekly Investment']]
    elif horizon == "monthly":
        return Table[['Ticker', 'Monthly Investment']]
    else:
        return Table[['Ticker', 'Yearly Investment']]

st.title("📈 technical Analysis Dashboard")

tickers_input = st.text_input("Enter stock tickers (semi-colon-separated)")
price_input = st.text_input("Enter your position value (semi-colon-separated)")

period = st.selectbox("Select Horizon", ["daily", "weekly", "monthly", "yearly"], index=0)

# Convert user input
tickers = [t.strip().upper() for t in tickers_input.split(";") if t.strip()]
values = [convert_comma_number(p.strip()) for p in price_input.split(";") if p.strip()]
horizon = period

if st.button("Run Analysis"):
    if tickers and values and horizon:
        with st.spinner("Fetching data..."):
            df_result = get_latest_mas(tickers, values, horizon)
        st.success("Done!")

        if not df_result.empty:
            st.dataframe(df_result)

            # Download as CSV
            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "moving_averages.csv", "text/csv")

        else:
            st.warning("No data to display.")
    else:
        st.error("Please enter valid tickers and MA periods.")
