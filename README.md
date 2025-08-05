# PulseCoinRadar

**PulseCoinRadar** is a Python tool designed to forecast "coin pulses" — sudden, significant price movements in cryptocurrencies. It combines market data from Bybit with technical indicators (CCI, EMA) and social media signals (e.g., X activity) to predict pulses using a GRU neural network. The tool generates insightful visualizations with Seaborn, including correlation heatmaps and price plots, to help traders identify market opportunities.

## Features
- Fetches real-time OHLCV data from Bybit.
- Incorporates social media signals for enhanced predictions.
- Uses a GRU model to detect coin pulses.
- Generates correlation heatmaps and price plots with Seaborn.
- Configurable for various symbols and timeframes.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PulseCoinRadar.git
   cd PulseCoinRadar
