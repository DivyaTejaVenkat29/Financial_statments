import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
# Define stock symbols
symbols = [
           'HINDUNILVR.NS', 'NESTLEIND.NS', 'VBL.NS', 'GODREJCP.NS', 'BRITANNIA.NS',
           'DABUR.NS', 'MARICO.NS', 'COLPAL.NS', 'ITC.NS', 'PGHH.NS', 'AWL.NS',
           'JUBLFOOD.NS', 'MARUTI.NS', 'M&M.NS', 'TATAMOTORS.NS', 'BAJAJ-AUTO.NS',
           'EICHERMOT.NS', 'TVSMOTOR.NS', 'HEROMOTOCO.NS', 'MOTHERSON.NS', 'CUMMINSIND.NS',
           'BOSCHLTD.NS', 'TIINDIA.NS', 'ASHOKLEY.NS', 'SCHAEFFLER.NS', 'BALKRISIND.NS',
           'ULTRACEMCO.NS', 'AMBUJACEM.NS', 'SHREECEM.NS', 'ACC.NS', 'DALBHARAT.NS',
           'JKCEMENT.NS', 'KAJARIACER.NS', 'RAMCOCEM.NS', 'LT.NS', 'ASIANPAINT.NS',
           'COALINDIA.NS', 'HINDZINC.NS', 'TATASTEEL.NS', 'JSWSTEEL.NS', 'VEDL.NS',
           'HINDALCO.NS', 'JINDALSTEL.NS', 'NMDC.NS', 'BHARATFORG.NS', 'JSL.NS',
           'SAIL.NS', 'NTPC.NS', 'SUZLON.NS', 'POWERGRID.NS', 'TATAPOWER.NS',
           'ADANIGREEN.NS', 'JSWENERGY.NS', 'NHPC.NS', 'TORNTPOWER.NS', 'SJVN.NS',
           'IREDA.NS', 'NLCINDIA.NS', 'CESC.NS', 'SWSOLAR.NS', 'IEX.NS', 'BHARTIARTL.NS',
           'IDEA.NS', 'INDUSTOWER.NS', 'TATACOMM.NS', 'ITI.NS', 'TEJASNET.NS', 'TTML.NS',
           'HFCL.NS', 'RAILTEL.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS',
           'KOTAKBANK.NS', 'BANKBARODA.NS', 'PNB.NS', 'IOB.NS', 'INDUSINDBK.NS', 'SUNPHARMA.NS',
           'CIPLA.NS', 'DIVISLAB.NS', 'ZYDUSLIFE.NS', 'DRREDDY.NS', 'TORNTPHARM.NS',
           'APOLLOHOSP.NS', 'MANKIND.NS', 'MAXHEALTH.NS', 'LUPIN.NS', 'AUROPHARMA.NS',
           'RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS', 'HINDPETRO.NS', 'OIL.NS',
           'PETRONET.NS', 'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'LTIM.NS',
           'TECHM.NS', 'OFSS.NS', 'PERSISTENT.NS', 'MPHASIS.NS', 'TATAELXSI.NS',
           'KPITTECH.NS', 'TITAN.NS', 'UPL.NS', 'TATACHEM.NS', 'ADANIPORTS.NS',
           'ZOMATO.NS', 'ADANIENT.NS', 'DEEPAKNTR.NS', 'IRCTC.NS', 'DMART.NS', 'BHEL.NS']
# Streamlit layout and styling
st.set_page_config(layout="wide")
st.title("Stock Financial Analysis - Research Tool")
st.write("Explore comprehensive financial insights and trends for selected stocks.")

# Sidebar for symbol selection and analysis period
st.sidebar.header("Filters")
selected_symbol = st.sidebar.selectbox("Choose a Stock Symbol", symbols)
selected_period = st.sidebar.selectbox("Select Period", ['1y', '3y', '5y', '10y'])

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch financial statements
def fetch_financials(ticker):
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    income_statement = stock.financials
    cash_flow = stock.cashflow
    return balance_sheet, income_statement, cash_flow

# Function to calculate quarter-over-quarter growth
def calculate_growth(data):
    return data.pct_change().dropna() * 100

# Function to calculate financial ratios
def calculate_ratios(balance_sheet, income_statement, cash_flow):
    ratios = {}
    # Profitability Ratios
    ratios['Gross Profit Margin'] = (income_statement.loc['Gross Profit'] / income_statement.loc['Total Revenue']) * 100
    ratios['Net Profit Margin'] = (income_statement.loc['Net Income'] / income_statement.loc['Total Revenue']) * 100
    ratios['Return on Equity'] = (income_statement.loc['Net Income'] / balance_sheet.loc['Stockholders Equity']) * 100
    
    # Liquidity Ratios
    ratios['Current Ratio'] = balance_sheet.loc['Current Assets'] / balance_sheet.loc['Current Liabilities']
    ratios['Quick Ratio'] = (balance_sheet.loc['Current Assets'] - balance_sheet.loc['Inventory']) / balance_sheet.loc['Current Liabilities']
    
    # Debt Ratios
    ratios['Debt to Equity'] = balance_sheet.loc['Total Debt'] / balance_sheet.loc['Stockholders Equity']
    
    return pd.Series(ratios)

# Streamlit UI
st.title('Interactive Financial Statement Analysis')

# User input for stock selection

# Fetch and display financials
balance_sheet, income_statement, cash_flow = fetch_financials(selected_symbol)

# Section: Balance Sheet
st.header('Balance Sheet')
st.dataframe(balance_sheet)
if st.button('View Total Assets'):
    total_assets = balance_sheet.loc['Total Assets'].sum()
    st.write(f"Total Assets: {total_assets}")


    # Visualization of Total Assets
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(balance_sheet.T.index, balance_sheet.loc['Total Assets'], color='blue',marker='o')
    ax.set_title('Total Assets Over Years')
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Assets')
    st.pyplot(fig)  # Pass the figure to Streamlit
# Section: Income Statement
st.header('Income Statement')
st.dataframe(income_statement)

# Section: Visualize Growth Rates
st.subheader("Quarter-over-Quarter Growth Rates")
if st.button('Visualize Growth Rates'):
    revenue_growth = calculate_growth(income_statement.loc['Total Revenue'])
    net_income_growth = calculate_growth(income_statement.loc['Net Income'])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(revenue_growth.index, revenue_growth.values, label='Revenue Growth (%)', marker='o')
    ax.plot(net_income_growth.index, net_income_growth.values, label='Net Income Growth (%)', marker='o')
    
    ax.set_title('Quarter-over-Quarter Growth Rates')
    ax.set_ylabel('Growth Rate (%)')
    ax.set_xlabel('Quarter')
    ax.legend()
    st.pyplot(fig)

# Section: Cash Flow Statement
st.header('Cash Flow Statement')
st.dataframe(cash_flow)
if st.button('Visualize Cash Flow Growth'):
    cash_flow_growth = calculate_growth(cash_flow.loc['Operating Cash Flow'])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cash_flow_growth.index, cash_flow_growth.values, label='Cash Flow Growth (%)', marker='o', color='green')
    
    ax.set_title('Quarter-over-Quarter Operating Cash Flow Growth')
    ax.set_ylabel('Growth Rate (%)')
    ax.set_xlabel('Quarter')
    ax.legend()
    st.pyplot(fig)

# Section: Additional Financial Insights
st.subheader("Key Financial Ratios")
financial_ratios = calculate_ratios(balance_sheet, income_statement, cash_flow)
st.write(financial_ratios)

# Section: Revenue and Profit Trends
st.subheader("Revenue and Profit Trends")
revenue_growth = calculate_growth(income_statement.loc['Total Revenue'])
net_income_growth = calculate_growth(income_statement.loc['Net Income'])
st.line_chart(revenue_growth, width=0, height=400)
st.line_chart(net_income_growth, width=0, height=400)

# Section: Discounted Cash Flow (DCF) Analysis
st.subheader("Discounted Cash Flow (DCF) Analysis")
expected_growth_rate = st.slider("Select Expected Growth Rate (%)", 5, 20, 10)
discount_rate = st.slider("Select Discount Rate (%)", 5, 15, 10)

# Assuming cash flow from operations for simplified DCF
cash_flow_forecast = (cash_flow.loc['Operating Cash Flow'].iloc[0] * (1 + expected_growth_rate / 100))
dcf_value = cash_flow_forecast / (discount_rate / 100)
st.write(f"**Estimated Stock Value (DCF):** ${dcf_value:.2f}")


# # Fetch financial data
@st.cache_data
def get_financial_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=selected_period)
    balance_sheet = stock.balance_sheet
    income_statement = stock.financials
    cashflow_statement = stock.cashflow
    return hist, balance_sheet, income_statement, cashflow_statement

# # Display data and insights
hist, balance_sheet, income_statement, cashflow_statement = get_financial_data(selected_symbol)

# # Show financial statements
# st.subheader("Financial Statements")
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.write("**Balance Sheet**")
#     st.write(balance_sheet)
# with col2:
#     st.write("**Income Statement**")
#     st.write(income_statement)
# with col3:
#     st.write("**Cashflow Statement**")
#     st.write(cashflow_statement)

# Key Metrics Section
st.subheader("Key Financial Metrics")
st.metric("Current Price", f"{hist['Close'][-1]:.2f}")
st.metric("52 Week High", f"\u20B9{hist['High'].max():.2f}")
st.metric("52 Week Low", f"\u20B9{hist['Low'].min():.2f}")

# Plot Stock Price Trends
st.subheader("Stock Price Trend")
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name="Close Price"))
fig.update_layout(title=f"Price Trend for {selected_symbol}", xaxis_title="Date", yaxis_title="Price (INR)")
st.plotly_chart(fig)


import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
import plotly.graph_objs as go

# Function to fetch historical stock data based on the selected period
def fetch_stock_data(ticker, period):
    stock_data = yf.download(ticker, period=period)
    stock_data.reset_index(inplace=True)
    return stock_data

# Function for sentiment analysis (Placeholder)
def fetch_sentiment_data(ticker):
    # Placeholder for sentiment data. You should integrate actual sentiment fetching logic here.
    dates = pd.date_range(start="2022-01-01", end="2024-01-01", freq='D')
    sentiment_scores = np.random.uniform(-1, 1, size=len(dates))  # Random sentiment scores
    sentiment_df = pd.DataFrame({'date': dates, 'sentiment': sentiment_scores})
    return sentiment_df

from prophet import Prophet

# Function to train the forecasting model
def train_prophet_model(data):
    # Ensure the 'Date' column is timezone naive
    data['Date'] = data['Date'].dt.tz_localize(None)
    
    # Rename columns for Prophet
    df = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Initialize and fit the model
    model = Prophet()
    model.fit(df)
    
    # Make future predictions
    future = model.make_future_dataframe(periods=30)  # Forecast for 30 days
    forecast = model.predict(future)
    
    return forecast

# Streamlit App
st.title("Stock Price Forecasting with Sentiment Analysis")

# Sidebar for stock ticker input and period selection
a_period = st.sidebar.selectbox("Select Period", [ '3mo', '6mo', '1y', '2y', '5y', '10y'])

if selected_symbol:
    # Fetch stock and sentiment data based on the selected period
    stock_data = fetch_stock_data(selected_symbol, a_period)
    sentiment_data = fetch_sentiment_data(selected_symbol)

    # Forecast stock prices
    forecast = train_prophet_model(stock_data)

    # Plotting Stock Prices and Forecast
    fig = go.Figure()
    
    # Actual stock price
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Actual Price'))
    
    # Forecasted prices
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Price'))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower CI', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper CI', line=dict(dash='dash')))

    # Overlay sentiment scores
    sentiment_data['sentiment'] = sentiment_data['sentiment'].apply(lambda x: (x + 1) / 2)  # Normalize sentiment scores to 0-1
    sentiment_data.set_index('date', inplace=True)
    sentiment_data = sentiment_data.reindex(stock_data['Date'], method='ffill')  # Align sentiment with stock data

    # Overlay sentiment scores
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=sentiment_data['sentiment'], mode='lines', name='Sentiment Score', yaxis='y2', line=dict(color='orange')))

    # Updating layout
    fig.update_layout(title=f"{selected_symbol} Stock Price Forecast", xaxis_title='Date', yaxis_title='Price', yaxis2=dict(title='Sentiment Score', overlaying='y', side='right'))
    st.plotly_chart(fig)

    # Display buy/sell signal zones based on forecasted prices
    buy_signals = forecast[(forecast['yhat'] > forecast['yhat'].shift(1)) & (forecast['yhat'].shift(1) < forecast['yhat'].shift(2))]
    sell_signals = forecast[(forecast['yhat'] < forecast['yhat'].shift(1)) & (forecast['yhat'].shift(1) > forecast['yhat'].shift(2))]

    st.subheader("Buy/Sell Signal Zones")
    st.write("Buy Signals:")
    st.write(buy_signals[['ds', 'yhat']])
    st.write("Sell Signals:")
    st.write(sell_signals[['ds', 'yhat']])



import pandas as pd
import numpy as np
import plotly.graph_objs as go


# Function to fetch historical stock data
def fetch_stock_data1(symbols, period):
    stock_data = {}
    for symbol in symbols:
        data = yf.download(symbol, period=period)
        stock_data[symbol] = data['Close']
    return pd.DataFrame(stock_data)

# Function to calculate Relative Strength Index (RSI)
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Streamlit App
st.title("Stock Peer Comparison")

# Select sector for peer comparison
symbols = {
        'FMCG': ['HINDUNILVR.NS', 'NESTLEIND.NS', 'VBL.NS', 'GODREJCP.NS', 'BRITANNIA.NS', 'DABUR.NS', 'MARICO.NS', 'COLPAL.NS', 'ITC.NS', 'PGHH.NS', 'AWL.NS', 'JUBLFOOD.NS'],
        'Auto': ['MARUTI.NS', 'M&M.NS', 'TATAMOTORS.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS', 'HEROMOTOCO.NS', 'MOTHERSON.NS', 'CUMMINSIND.NS', 'BOSCHLTD.NS', 'TIINDIA.NS', 'ASHOKLEY.NS', 'SCHAEFFLER.NS', 'BALKRISIND.NS'],
        'Cement': ['ULTRACEMCO.NS', 'AMBUJACEM.NS', 'SHREECEM.NS', 'ACC.NS', 'DALBHARAT.NS', 'JKCEMENT.NS', 'KAJARIACER.NS', 'RAMCOCEM.NS', 'LT.NS', 'ASIANPAINT.NS'],
        'Metal': ['HINDZINC.NS', 'TATASTEEL.NS', 'JSWSTEEL.NS', 'VEDL.NS', 'HINDALCO.NS', 'JINDALSTEL.NS', 'NMDC.NS', 'BHARATFORG.NS', 'JSL.NS', 'SAIL.NS'],
        'Energy': ['NTPC.NS', 'COALINDIA.NS', 'SUZLON.NS', 'POWERGRID.NS', 'TATAPOWER.NS', 'ADANIGREEN.NS', 'JSWENERGY.NS', 'NHPC.NS', 'TORNTPOWER.NS', 'SJVN.NS', 'IREDA.NS', 'NLCINDIA.NS', 'CESC.NS', 'SWSOLAR.NS', 'IEX.NS'],
        'Telecom': ['BHARTIARTL.NS', 'IDEA.NS', 'INDUSTOWER.NS', 'TATACOMM.NS', 'ITI.NS', 'TEJASNET.NS', 'TTML.NS', 'HFCL.NS', 'RAILTEL.NS'],
        'Bank': ['^NSEBANK', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'BANKBARODA.NS', 'PNB.NS', 'IOB.NS', 'INDUSINDBK.NS'],
        'Pharma': ['SUNPHARMA.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'ZYDUSLIFE.NS', 'DRREDDY.NS', 'TORNTPHARM.NS', 'APOLLOHOSP.NS', 'MANKIND.NS', 'MAXHEALTH.NS', 'LUPIN.NS', 'AUROPHARMA.NS'],
        'Oil': ['RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS', 'HINDPETRO.NS', 'OIL.NS', 'PETRONET.NS'],
        'IT': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'LTIM.NS', 'TECHM.NS', 'OFSS.NS', 'PERSISTENT.NS', 'MPHASIS.NS', 'TATAELXSI.NS', 'KPITTECH.NS', 'DIXON.NS'],
        'Miscellaneous': ['TITAN.NS', 'UPL.NS', 'TATACHEM.NS', 'ADANIPORTS.NS', 'ZOMATO.NS', 'ADANIENT.NS', 'DEEPAKNTR.NS', 'IRCTC.NS', 'DMART.NS', 'BHEL.NS'],
    }
# Sidebar for sector selection
selected_sector = st.sidebar.selectbox("Select Sector", list(symbols.keys()))
q_period = st.sidebar.selectbox("Select Period", ['1d', '5d', '1w', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'],key='unique_key2')

selected_symbols = symbols[selected_sector]

# Fetch stock data for the selected symbols
stock_data = fetch_stock_data1(selected_symbols,q_period)


# Calculate RSI for the selected symbols
rsi_data = calculate_rsi(stock_data)


st.subheader("Relative Strength Index (RSI) Comparison")
rsi_fig = go.Figure()
for symbol in selected_symbols:
    rsi_fig.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data[symbol], mode='lines', name=symbol))

rsi_fig.update_layout(title='RSI Comparison', xaxis_title='Date', yaxis_title='RSI', yaxis_range=[0, 100])
st.plotly_chart(rsi_fig)

def format_market_cap(value):
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"  # Convert to billions
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"  # Convert to millions
    else:
        return f"${value:.2f}"  # Return as is
# Display Market Capitalization
st.subheader("Market Capitalization Comparison")
market_cap_fig = go.Figure()
for symbol in selected_symbols:
    ticker = yf.Ticker(symbol)
    market_cap = ticker.info['marketCap']
    formatted_market_cap = format_market_cap(market_cap)  # Format the market cap
    market_cap_fig.add_trace(go.Bar(x=[symbol], y=[market_cap], name=symbol, text=[formatted_market_cap], textposition='auto'))
market_cap_fig.update_layout(title='Market Capitalization Comparison', xaxis_title='Stock', yaxis_title='Market Cap (in billions)', yaxis_tickformat=',')
st.plotly_chart(market_cap_fig)



# Function to fetch stock data (You can use yfinance for real stock data)
def fetch_data(symbols, start_date, end_date):
    # Placeholder for actual data fetching logic
    # Replace this with your data fetching code (e.g., using yfinance)
    return pd.DataFrame(np.random.randn(100, len(symbols)), columns=symbols)

# Risk Metrics Calculation
def calculate_metrics(portfolio_returns):
    mean_return = portfolio_returns.mean().mean()  # Get mean return across all stocks
    volatility = portfolio_returns.std().mean()  # Get volatility across all stocks
    sharpe_ratio = mean_return / volatility
    sortino_ratio = mean_return / portfolio_returns[portfolio_returns < 0].std().mean()
    
    return sharpe_ratio, sortino_ratio

# Value-at-Risk Calculation
def calculate_var(portfolio_returns, confidence_level=0.95):
    return np.percentile(portfolio_returns, (1 - confidence_level) * 100)

# Diversification Analysis
def diversification_analysis(portfolio_returns):
    st.subheader("Correlation Matrix")
    corr_matrix = portfolio_returns.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    ax.set_title('Diversification Analysis - Correlation Matrix')
    st.pyplot(fig)

    # Portfolio Distribution
    st.subheader("Portfolio Distribution")
    sector_weights = portfolio_returns.mean().to_frame(name='Mean Return').reset_index()
    sector_weights.columns = ['Stock', 'Mean Return']
    sector_weights['Weight'] = sector_weights['Mean Return'] / sector_weights['Mean Return'].sum()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=sector_weights, x='Weight', y='Stock', palette='viridis')
    ax.set_title('Portfolio Allocation by Stock')
    st.pyplot(fig)


# Streamlit UI
st.title('Portfolio Analysis and Risk Management')

# User inputs
selected_sectors = st.multiselect('Select Sectors', options=list(symbols.keys()))
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

# Fetch Data
if selected_sectors and start_date and end_date:
    selected_symbols = [symbol for sector in selected_sectors for symbol in symbols[sector]]
    data = fetch_data(selected_symbols, start_date, end_date)
    returns = data.pct_change().dropna()  # Calculate daily returns

    # Calculate Risk Metrics
    sharpe_ratio, sortino_ratio = calculate_metrics(returns)
    var_95 = calculate_var(returns)

    # Display Metrics
    st.write(f'Sharpe Ratio: {sharpe_ratio:.2f}')
    st.write(f'Sortino Ratio: {sortino_ratio:.2f}')
    st.write(f'Value at Risk (95% confidence): {var_95:.2f}')

    # Diversification Analysis
    diversification_analysis(returns)


# Adding options to save user preferences (to be implemented)
# Download financial data
combined_data = pd.concat([balance_sheet, income_statement, cashflow_statement], axis=1)

# Download Button for CSV
csv = combined_data.to_csv(index=True)
st.download_button(
    label="Download Financial Data as CSV",
    data=csv,
    file_name=f"{selected_symbol}_financials.csv",
    mime="text/csv"
)
