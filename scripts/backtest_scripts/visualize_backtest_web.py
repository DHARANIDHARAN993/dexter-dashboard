import streamlit as st
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px

# Path to results CSV
data_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'data'))
results_csv = os.path.join(data_dir, 'BackTestHistoryResults.csv')

st.set_page_config(page_title='Backtest Results Dashboard', layout='wide')
st.title('Backtest Results Dashboard')

# Load data
def load_data():
    if not os.path.exists(results_csv):
        st.error(f"Results file not found: {results_csv}")
        st.stop()
    df = pd.read_csv(results_csv)
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header('Filters')

def get_symbol_options(df):
    syms = df['symbol'].dropna().unique().tolist()
    syms.sort()
    return syms

symbol_options = get_symbol_options(df)
symbol_filter = st.sidebar.multiselect('Symbol', symbol_options, default=symbol_options)

status_filter = st.sidebar.radio('Position Status', ['All', 'Open', 'Closed'], index=0)

if 'buy_date' in df.columns:
    df['buy_date'] = pd.to_datetime(df['buy_date'], errors='coerce')
    min_date = df['buy_date'].min()
    max_date = df['buy_date'].max()
    buy_date_range = st.sidebar.date_input('Buy Date Range', [min_date, max_date])
else:
    buy_date_range = None

filtered_df = df[df['symbol'].isin(symbol_filter)]

if status_filter == 'Open':
    filtered_df = filtered_df[filtered_df['hit_date'].isna()]
elif status_filter == 'Closed':
    filtered_df = filtered_df[filtered_df['hit_date'].notna()]

if buy_date_range and len(buy_date_range) == 2:
    start, end = pd.to_datetime(buy_date_range[0]), pd.to_datetime(buy_date_range[1])
    filtered_df = filtered_df[(filtered_df['buy_date'] >= start) & (filtered_df['buy_date'] <= end)]

def format_price(val):
    try:
        if pd.isna(val):
            return '-'
        return f"{float(val):.2f}"
    except Exception:
        return '-'

def build_table(df):
    qty = 10
    rows = []
    for _, row in df.iterrows():
        symbol = row.get('symbol', '-')
        current_price = format_price(row.get('current_price', '-'))
        buy_date = row.get('buy_date', '-') if pd.notna(row.get('buy_date', None)) else '-'
        buy_price = row.get('buy_price', float('nan'))
        buy_price_fmt = format_price(buy_price)
        ema_at_buy = format_price(row.get('ema_at_buy', '-'))
        buy_value = '-' if buy_price_fmt == '-' else format_price(float(buy_price_fmt) * qty)
        sell_date = row.get('hit_date', '-') if pd.notna(row.get('hit_date', None)) else '-'
        sell_price = row.get('target_price', '-') if pd.notna(row.get('hit_date', None)) and pd.notna(row.get('target_price', None)) else '-'
        sell_price_fmt = format_price(sell_price)
        sell_value = '-' if sell_price_fmt == '-' else format_price(float(sell_price_fmt) * qty)
        profit = '-' if sell_value == '-' or buy_value == '-' else format_price(float(sell_value) - float(buy_value))
        unrealized_pnl = format_price(row.get('unrealized_pnl', '-'))
        days_to_target = row.get('days_to_target', '-')
        if pd.isna(days_to_target):
            days_to_target = '-'
        rows.append([symbol, current_price, qty, buy_date, buy_price_fmt, ema_at_buy, buy_value, sell_date, sell_price_fmt, sell_value, profit, unrealized_pnl, days_to_target])
    columns = ['Symbol', 'Current Price', 'Qty', 'Buy Date', 'Buy Price', 'EMA at Buy', 'Buy Value', 'Sell Date', 'Sell Price', 'Sell Value', 'Profit', 'Unrealized PnL', 'No of Days to Hit Target']
    return pd.DataFrame(rows, columns=columns)

table_df = build_table(filtered_df)

# --- Metrics Row ---
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    total_profit = pd.to_numeric(table_df['Profit'], errors='coerce').dropna().sum()
    num_trades = len(table_df)
    num_open = table_df[table_df['Sell Date'] == '-'].shape[0]
    num_closed = table_df[table_df['Sell Date'] != '-'].shape[0]
    avg_days = pd.to_numeric(table_df['No of Days to Hit Target'], errors='coerce').dropna().mean()
    col1.metric("Total Profit", f"{total_profit:.2f}")
    col2.metric("Total Trades", num_trades)
    col3.metric("Open Positions", num_open)
    col4.metric("Avg Days to Target", f"{avg_days:.2f}" if not pd.isna(avg_days) else "-")

# --- Tabs for Visualizations ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Results Table",
    "Days to Target",
    "Open vs Closed",
    "Profit Distribution",
    "Buy Price vs Days to Target",
    "Top Profitable Trades",
    "Cumulative Profit"
])

with tab1:
    st.subheader('Filtered Results Table')
    st.dataframe(table_df, use_container_width=True)
    st.download_button('Download Filtered Data as CSV', table_df.to_csv(index=False), file_name='filtered_backtest_results.csv')

with tab2:
    st.subheader('Days to 2% Target (Closed Positions)')
    closed = table_df[table_df['No of Days to Hit Target'] != '-']
    if not closed.empty:
        chart_df = closed[['Symbol', 'No of Days to Hit Target']].copy()
        chart_df['No of Days to Hit Target'] = pd.to_numeric(chart_df['No of Days to Hit Target'])
        st.bar_chart(chart_df.set_index('Symbol'))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(chart_df['Symbol'], chart_df['No of Days to Hit Target'], color='skyblue')
        ax.set_xlabel('Stock Symbol')
        ax.set_ylabel('Days to 2% Gain')
        ax.set_title('Days Taken to Reach 2% Gain After Buy Signal (Matplotlib)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info('No closed positions to chart.')

with tab3:
    st.subheader('Open vs Closed Positions')
    open_count = table_df[table_df['Sell Date'] == '-'].shape[0]
    closed_count = table_df[table_df['Sell Date'] != '-'].shape[0]
    pie_df = pd.DataFrame({
        'Status': ['Open', 'Closed'],
        'Count': [open_count, closed_count]
    })
    fig_pie = px.pie(pie_df, names='Status', values='Count', title='Open vs Closed Positions')
    st.plotly_chart(fig_pie, use_container_width=True)

with tab4:
    st.subheader('Profit Distribution')
    profit_vals = pd.to_numeric(table_df['Profit'], errors='coerce').dropna()
    if not profit_vals.empty:
        fig_hist = px.histogram(profit_vals, nbins=20, title='Profit Distribution')
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info('No profit data to display.')

with tab5:
    st.subheader('Buy Price vs. Days to Target')
    scatter_df = table_df[(table_df['Buy Price'] != '-') & (table_df['No of Days to Hit Target'] != '-')]
    if not scatter_df.empty:
        fig_scatter = px.scatter(
            scatter_df,
            x='Buy Price',
            y='No of Days to Hit Target',
            color='Symbol',
            hover_data=['Buy Date', 'Sell Date', 'Profit'],
            title='Buy Price vs. Days to Target'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info('Not enough data for scatter plot.')

with tab6:
    st.subheader('Top 10 Profitable Trades')
    top_df = table_df[table_df['Profit'] != '-'].copy()
    top_df['Profit'] = pd.to_numeric(top_df['Profit'], errors='coerce')
    top_df = top_df.sort_values('Profit', ascending=False).head(10)
    if not top_df.empty:
        fig_top = px.bar(
            top_df,
            x='Symbol',
            y='Profit',
            color='Symbol',
            hover_data=['Buy Date', 'Sell Date', 'Buy Value', 'Sell Value'],
            title='Top 10 Profitable Trades'
        )
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info('No profit data to display.')

with tab7:
    st.subheader('Cumulative Profit Over Time')
    closed = table_df[(table_df['Sell Date'] != '-') & (table_df['Profit'] != '-')]
    if not closed.empty:
        closed['Sell Date'] = pd.to_datetime(closed['Sell Date'])
        closed = closed.sort_values('Sell Date')
        closed['Profit'] = pd.to_numeric(closed['Profit'], errors='coerce')
        closed['Cumulative Profit'] = closed['Profit'].cumsum()
        fig_cum = px.line(
            closed,
            x='Sell Date',
            y='Cumulative Profit',
            title='Cumulative Profit Over Time'
        )
        st.plotly_chart(fig_cum, use_container_width=True)
    else:
        st.info('No closed trades to show cumulative profit.')

st.caption('Powered by Streamlit | Data from BackTestResults.csv') 