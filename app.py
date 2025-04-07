# app.py
import gradio as gr
import matplotlib.pyplot as plt
from utils import make_predictions
import matplotlib.dates as mdates
import pandas as pd
import yfinance as yf
import tempfile
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Plotting function
def plot_predictions(result_df, symbol):
    fig, ax = plt.subplots(figsize=(10, 5))
    result_df['Date'] = pd.to_datetime(result_df['Date'])
    ax.plot(result_df['Date'], result_df['Actual'], label="Actual", color='blue')
    ax.plot(result_df['Date'], result_df['Predicted'], label="Predicted", color='orange')
    ax.set_title(f"{symbol} — Actual vs. Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45)
    return fig

# Get stock info from yfinance
def get_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        description = info.get("longBusinessSummary", "No description available.")
        current_price = info.get("currentPrice", "N/A")
        return f"💡 **{symbol} Info:**\n- 📌 Price: **${current_price}**\n- 📝 Description: {description}"
    except Exception:
        return f"⚠️ Could not retrieve info for {symbol}"

# Prediction + Metrics + Temp CSV Save
def predict_and_plot(symbol, start, end):
    result_df, _, error = make_predictions(symbol.upper(), start, end)
    if error:
        return f"❌ {error}", None, None, None, "", gr.update(visible=False)

    result_df['Date'] = pd.to_datetime(result_df['Date']).dt.strftime('%Y-%m-%d')
    fig = plot_predictions(result_df, symbol)
    status_msg = f"✅ Prediction Complete for {symbol.upper()}"
    display_df = result_df[['Date', 'Actual', 'Predicted']]

    # Metrics
    y_true = result_df['Actual'].values
    y_pred = result_df['Predicted'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics = f"📊 MAE: {mae:.4f} | RMSE: {rmse:.4f}"

    # Save to temp CSV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', newline='') as f:
        display_df.to_csv(f.name, index=False)
        csv_path = f.name

    return status_msg, fig, display_df, csv_path, metrics, gr.update(visible=False)

with gr.Blocks(title="Stock Analytics") as demo:
    # Header and Logo
    gr.HTML(
        """
        <div style='text-align: center; padding: 10px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/2721/2721290.png' width='60'/>
            <h1 style='margin-bottom: 0;'>Stock Price Predictor</h1>
            <p style='font-size: 18px; color: gray;'>Predict and visualize stock prices using deep learning 📈</p>
        </div>
        """
    )

    with gr.Row():
        symbol = gr.Dropdown(choices=['AAPL', 'TSLA', 'AMZN'], label="Choose Stock Symbol", value='AAPL')
        start = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2023-01-01")
        end = gr.Textbox(label="End Date (YYYY-MM-DD)", value="2024-12-31")
        get_info = gr.Button("ℹ️ Get Stock Info")

    stock_info = gr.Markdown("")
    get_info.click(fn=get_stock_info, inputs=symbol, outputs=stock_info)

    gr.Markdown("⚠️ **If error occurs during prediction, please refresh the website.**")

    
    btn = gr.Button("📊 Predict")
    status = gr.Textbox(label="Status")
    metrics = gr.Textbox(label="Performance Metrics")
    plot = gr.Plot(label="Actual vs Predicted")
    table = gr.Dataframe(label="Prediction Table", interactive=False)
    csv_output = gr.File(label="📥 Download CSV")
    loader = gr.Markdown("⏳ Running prediction, please wait...", visible=False)

    def start_loader():
        return gr.update(visible=True)

    def end_loader():
        return gr.update(visible=False)

    # Show loader first, then call prediction
    btn.click(fn=start_loader, outputs=loader)
    btn.click(fn=predict_and_plot, inputs=[symbol, start, end],
              outputs=[status, plot, table, csv_output, metrics, loader])

    # Style
    gr.HTML(
        """
        <style>
        body { background-color: #f7f9fc; }
        .gr-button { font-size: 16px !important; }
        .gr-textbox label, .gr-dropdown label, .gr-dataframe label, .gr-plot label {
            font-weight: bold;
            color: #333;
        }
        .gr-dataframe thead {
            background-color: #e8f0fe;
            color: #000;
        }
        .gr-dataframe tbody tr:nth-child(even) {
            background-color: #f1f3f4;
        }
        .gr-dataframe tbody tr:hover {
            background-color: #d7e3fc;
        }
        </style>
        """
    )
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000)),  # Use Render's default port
    inbrowser=False,                                 # Don't try to open a browser
    prevent_thread_lock=False                        # Required for Render to detect port
)