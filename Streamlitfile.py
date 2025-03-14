#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import streamlit as st

# Custom CSS for button
st.markdown(
    """
    <style>
    .top-right-button {
        position: relative;
        float: right;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'show_instructions' not in st.session_state:
    st.session_state.show_instructions = False

# Add the "How to Use the App" button
if st.button("Know About the App"):
    st.session_state.show_instructions = not st.session_state.show_instructions

# Toggle instructions visibility
if st.session_state.show_instructions:
    with st.expander("Know About the App", expanded=True):
        st.write("""
        This app helps you understand portfolio construction using stocks listed on **NSE India** by:
        1. Building **efficient frontiers** to visualize risk-return tradeoffs
        2. Identifying portfolios with the **highest Sharpe ratio**
        3. Showing ideal stock allocations for maximum risk-adjusted returns
        4. Comparing current vs. ideal allocations with **Portfolio Rebalancer**
        """)
        
        st.caption("""
        *Uses monthly stock returns. 10-year government bond yield is adjusted to monthly risk-free rate.  
        Results are illustrative - not investment advice.
        """)


# Set the app title
st.title("Portfolio Simulator")  # Changed title

# Import company and tickers
data = pd.read_csv("EQUITY_L.csv")
Company = data["NAME OF COMPANY"].tolist()
Symbol = data["SYMBOL"].tolist()
Company_to_symbol = dict(zip(Company, Symbol))

# Selecting the company name and dates
selected_option = st.sidebar.multiselect('Select the stocks in your portfolio:', Company)
start_date = st.sidebar.date_input('Enter start date')
end_date = st.sidebar.date_input('Enter end date')
RFR = st.sidebar.number_input('Enter the 10 year Government bond yield in %')
annual_risk_free_rate = (RFR / 100)

# Convert annual risk-free rate to monthly
monthly_risk_free_rate = (1 + annual_risk_free_rate) ** (1 / 12) - 1

# Check if selections are valid
if selected_option and start_date < end_date:
    tickers = [Company_to_symbol[stock] + ".NS" for stock in selected_option]
    try:
        data = yf.download(tickers, start=start_date, end=end_date, interval='1mo')['Close']
    except Exception as e:
        st.write(f"Error downloading data for {tickers}: {e}")

    if not data.empty:
        returns = data.pct_change().dropna()

        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        def portfolio_performance(weights, mean_returns, cov_matrix):
            returns = np.dot(weights, mean_returns)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return returns * 100, volatility * 100  # Convert to %

        def get_efficient_frontier(mean_returns, cov_matrix, monthly_risk_free_rate):
            num_assets = len(mean_returns)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((-1, 2) for asset in range(num_assets))

            def minimize_volatility(weights):
                return portfolio_performance(weights, mean_returns, cov_matrix)[1]

            initial_guess = num_assets * [1. / num_assets,]
            efficient_portfolios = []

            for target_return in np.linspace(mean_returns.min(), mean_returns.max(), 100):
                constraints = (
                    {'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0] - target_return * 100},  # Convert to %
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
                )
                result = minimize(minimize_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
                efficient_portfolios.append(result.x)

            return efficient_portfolios

        def calculate_efficient_frontier(mean_returns, cov_matrix, monthly_risk_free_rate):
            efficient_portfolios = get_efficient_frontier(mean_returns, cov_matrix, monthly_risk_free_rate)
            returns = []
            volatilities = []

            for portfolio in efficient_portfolios:
                p_returns, p_volatility = portfolio_performance(portfolio, mean_returns, cov_matrix)
                returns.append(p_returns)
                volatilities.append(p_volatility)

            return np.array(returns), np.array(volatilities), efficient_portfolios

        def plot_efficient_frontier(mean_returns, cov_matrix, monthly_risk_free_rate):
            returns, volatilities, efficient_portfolios = calculate_efficient_frontier(mean_returns, cov_matrix, monthly_risk_free_rate)
            sharpe_ratios = (returns - monthly_risk_free_rate * 100) / volatilities  # Convert risk-free rate to %

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Ideal Weight Allocation", "Minimum Variance Frontier"))

            weights = efficient_portfolios[np.argmax(sharpe_ratios)]

            bar_fig = go.Bar(x=selected_option, y=weights, text=[f'{weight:.2%}' for weight in weights], textposition='outside', showlegend=False)
            fig.add_trace(bar_fig, row=1, col=1)

            fig.update_yaxes(range=[min(weights) - 0.1, max(weights) + 0.1], row=1, col=1)
            fig.update_layout(yaxis=dict(title_text='Weights %'))

            fig.add_trace(go.Scatter(
                x=volatilities,
                y=returns,
                mode='markers',
                marker=dict(
                    color='white',
                    size=10
                ),
                text=[f'Sharpe Ratio: {sr:.2f}' for sr in sharpe_ratios],
                showlegend=False
            ), row=1, col=2)

            max_sharpe_idx = np.argmax(sharpe_ratios)
            fig.add_trace(go.Scatter(
                x=[volatilities[max_sharpe_idx]],
                y=[returns[max_sharpe_idx]],
                mode='markers',
                marker=dict(
                    color='blue',
                    size=15,
                    symbol='hexagram'
                ),
                name='Max Sharpe Ratio Portfolio'
            ), row=1, col=2)

            fig.update_layout(
                title='Ideal Portfolio Simulation',  # Added section heading
                xaxis2=dict(title='Volatility (Standard Deviation) %', tickformat=".2f", domain=[0.6, 1]),
                yaxis2=dict(title='Expected Return %', tickformat=".2f"),
                xaxis=dict(domain=[0, 0.4]),
                showlegend=True,
                legend=dict(font=dict(size=10))
            )

            st.plotly_chart(fig)

            return efficient_portfolios[max_sharpe_idx], max_sharpe_idx

        # Usage
        optimal_portfolio_weights, max_sharpe_idx = plot_efficient_frontier(mean_returns, cov_matrix, monthly_risk_free_rate)

    else:
        st.write("No data available for selected date range")
else:
    if not selected_option:
        st.write("Select stock in your Portfolio to Start the Analysis")
    if start_date >= end_date:
        st.write("End date must be after start date")


show_rebalancer = st.sidebar.checkbox('Show Portfolio Rebalancer')

if show_rebalancer:
    # Check if optimal_portfolio_weights is defined
    if 'optimal_portfolio_weights' not in globals():
        st.warning("Please select stocks and dates first to calculate optimal weights.")
    else:
        Investment_Amount = {}
        for stock in selected_option:
            amount = st.sidebar.number_input(f'Enter the amount invested in {stock}', min_value=0, value=0)
            Investment_Amount[stock] = amount

        if any(Investment_Amount.values()):
            stocks = list(Investment_Amount.keys())
            amounts = list(Investment_Amount.values())
            total_amount = sum(amounts)
            amounts_percent = [(amount / total_amount) * 100 for amount in amounts]

            # Map optimal weights to stocks
            weights = {stock: optimal_portfolio_weights[idx] * 100 for idx, stock in enumerate(selected_option)}

            # Create DataFrame
            df = pd.DataFrame({
                'Stock': stocks,
                'Current Allocation (%)': amounts_percent,
                'Target Allocation (%)': [weights[stock] for stock in stocks],
                'Rebalancing Needed (%)': [weights[stock] - amounts_percent[idx] for idx, stock in enumerate(stocks)]
            })

            # Melt DataFrame for Plotly
            df_melted = df.melt(id_vars=['Stock'], 
                                value_vars=['Current Allocation (%)', 'Target Allocation (%)', 'Rebalancing Needed (%)'], 
                                var_name='Metric', value_name='Value')

            # Create bar chart
            fig1 = px.bar(df_melted, x='Stock', y='Value', color='Metric', barmode='group', 
                          text=df_melted['Value'].map('{:.2f}%'.format),  
                          color_discrete_map={
                              'Rebalancing Needed (%)': 'darkblue',
                              'Current Allocation (%)': 'lightblue',
                              'Target Allocation (%)': 'lightgreen'
                          },
                          labels={'Value': 'Values', 'Stock': 'Stocks'}, 
                          title='Portfolio Weights: Actual vs. Recommended')

            fig1.update_traces(textposition='outside')  # Show text outside the bar
            st.plotly_chart(fig1)
        else:
            st.write("Please enter the investment amounts to view the chart.")

