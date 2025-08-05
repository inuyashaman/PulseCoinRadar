import ccxt
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

class PulseCoinRadar:
    def __init__(self, symbol='SOL/USDT', timeframe='1h', lookback_days=10, api_key=None, api_secret=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.exchange = ccxt.bybit({'apiKey': api_key, 'secret': api_secret})
        self.scaler = MinMaxScaler()

    def fetch_ohlcv(self):
        """Получение исторических данных с Bybit."""
        since = int((datetime.now() - timedelta(days=self.lookback_days)).timestamp() * 1000)
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def fetch_social_signals(self):
        """Получение социальных сигналов с X (заглушка для API X)."""
        # Реальная версия требует API X для анализа сигналов
        return np.random.rand(len(self.fetch_ohlcv())) * 70

    def calculate_cci(self, df, period=20):
        """Расчет CCI (Commodity Channel Index)."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = (typical_price - sma).abs().rolling(window=period).mean()
        return (typical_price - sma) / (0.015 * mad)

    def calculate_ema(self, prices, period=12):
        """Расчет EMA (Exponential Moving Average)."""
        return prices.ewm(span=period, adjust=False).mean()

    def prepare_data(self, df):
        """Подготовка данных для модели."""
        df['returns'] = df['close'].pct_change()
        df['cci'] = self.calculate_cci(df)
        df['ema'] = self.calculate_ema(df['close'])
        df['social_signals'] = self.fetch_social_signals()
        features = df[['close', 'cci', 'ema', 'social_signals']].dropna()

        scaled_data = self.scaler.fit_transform(features)
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i])
            y.append(1 if scaled_data[i, 0] > np.percentile(scaled_data[:, 0], 90) else 0)  # Ценовой импульс
        return np.array(X), np.array(y)

    def build_model(self):
        """Создание GRU-модели."""
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(60, 4)),
            Dropout(0.3),
            GRU(32),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, X, y):
        """Обучение модели."""
        model = self.build_model()
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
        return model

    def predict_pulse(self, model, X):
        """Прогноз ценовых импульсов."""
        predictions = model.predict(X)
        return (predictions > 0.5).astype(int)

    def visualize_results(self, df, predictions):
        """Визуализация с Seaborn."""
        df = df.iloc[60:].copy()
        df['pulse_prediction'] = predictions

        # Тепловая карта корреляции
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[['close', 'cci', 'ema', 'social_signals', 'pulse_prediction']].corr(), annot=True, cmap='viridis')
        plt.title(f'Correlation Heatmap for {self.symbol}')
        plt.savefig('data/sample_output/pulse_heatmap.png')
        plt.close()

        # График цены с предсказаниями
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=df['timestamp'], y=df['close'], label='Price', color='blue')
        sns.scatterplot(x=df[df['pulse_prediction'] == 1]['timestamp'], 
                        y=df[df['pulse_prediction'] == 1]['close'], 
                        color='red', label='Predicted Pulse', s=100)
        plt.title(f'Price Pulses for {self.symbol}')
        plt.savefig('data/sample_output/pulse_plot.png')
        plt.show()

    def run(self):
        """Основной метод анализа."""
        df = self.fetch_ohlcv()
        X, y = self.prepare_data(df)
        model = self.train_model(X, y)
        predictions = self.predict_pulse(model, X)
        self.visualize_results(df, predictions)
        print(f"Coin pulses predicted: {np.sum(predictions)} out of {len(predictions)} periods.")

if __name__ == "__main__":
    radar = PulseCoinRadar(symbol='SOL/USDT', timeframe='1h', lookback_days=10)
    radar.run()
