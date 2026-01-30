
import pandas as pd
import numpy as np
from typing import List, Optional
from loguru import logger
import ta


class TechnicalIndicators:
    """Calculate technical indicators for stock data."""
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14, column: str = 'Close') -> pd.DataFrame:
        """Add Relative Strength Index."""
        df['RSI'] = ta.momentum.RSIIndicator(df[column], window=window).rsi()
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
        """Add MACD indicators."""
        macd = ta.trend.MACD(df[column])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20, column: str = 'Close') -> pd.DataFrame:
        """Add Bollinger Bands."""
        bb = ta.volatility.BollingerBands(df[column], window=window)
        df['BB_high'] = bb.bollinger_hband()
        df['BB_mid'] = bb.bollinger_mavg()
        df['BB_low'] = bb.bollinger_lband()
        df['BB_width'] = bb.bollinger_wband()
        df['BB_pct'] = bb.bollinger_pband()
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Average True Range."""
        df['ATR'] = ta.volatility.AverageTrueRange(
            df['High'], df['Low'], df['Close'], window=window
        ).average_true_range()
        return df
    
    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume."""
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            df['Close'], df['Volume']
        ).on_balance_volume()
        return df
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
        """Add various moving averages."""
        df['SMA_50'] = ta.trend.SMAIndicator(df[column], window=50).sma_indicator()
        df['SMA_200'] = ta.trend.SMAIndicator(df[column], window=200).sma_indicator()
        df['EMA_12'] = ta.trend.EMAIndicator(df[column], window=12).ema_indicator()
        df['EMA_26'] = ta.trend.EMAIndicator(df[column], window=26).ema_indicator()
        return df
    
    @staticmethod
    def add_stochastic(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        stoch = ta.momentum.StochasticOscillator(
            df['High'], df['Low'], df['Close'], window=window
        )
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        return df
    
    @staticmethod
    def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Average Directional Index."""
        df['ADX'] = ta.trend.ADXIndicator(
            df['High'], df['Low'], df['Close'], window=window
        ).adx()
        return df


class SentimentFeatures:
    """Create advanced sentiment-based features."""
    
    @staticmethod
    def add_lag_features(df: pd.DataFrame, column: str = 'sentiment_score', lags: List[int] = [1, 2, 3, 5, 7]) -> pd.DataFrame:
        """Add lagged sentiment features."""
        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)
        return df
    
    @staticmethod
    def add_rolling_features(df: pd.DataFrame, column: str = 'sentiment_score', windows: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
        """Add rolling window features for sentiment."""
        for window in windows:
            df[f'{column}_ma_{window}'] = df[column].rolling(window=window).mean()
            df[f'{column}_std_{window}'] = df[column].rolling(window=window).std()
            df[f'{column}_min_{window}'] = df[column].rolling(window=window).min()
            df[f'{column}_max_{window}'] = df[column].rolling(window=window).max()
        return df
    
    @staticmethod
    def add_momentum_features(df: pd.DataFrame, column: str = 'sentiment_score') -> pd.DataFrame:
        """Add sentiment momentum and acceleration."""
        df[f'{column}_momentum'] = df[column].diff()
        df[f'{column}_acceleration'] = df[f'{column}_momentum'].diff()
        return df
    
    @staticmethod
    def add_volatility_features(df: pd.DataFrame, column: str = 'sentiment_score', window: int = 14) -> pd.DataFrame:
        """Add sentiment volatility."""
        df[f'{column}_volatility'] = df[column].rolling(window=window).std()
        return df
    
    @staticmethod
    def add_news_count_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on news count."""
        if 'news_count' in df.columns:
            df['news_count_lag_1'] = df['news_count'].shift(1)
            df['news_count_ma_7'] = df['news_count'].rolling(window=7).mean()
            df['news_intensity'] = df['news_count'] / df['news_count'].rolling(window=30).mean()
        return df
    
    @staticmethod
    def add_sentiment_ema(df: pd.DataFrame, column: str = 'sentiment_score', span: int = 5) -> pd.DataFrame:
        """
        Adds Exponential Moving Average to sentiment to create a 'memory' of news 
        that decays over time rather than disappearing on no-news days.
        """
        if column in df.columns:
            # fillna(0) ensures the calculation starts even if the first row is empty
            df[f'{column}_ema'] = df[column].fillna(0).ewm(span=span, adjust=False).mean()
        return df


class MarketFeatures:
    """Create market-based features."""
    
    @staticmethod
    def add_return_features(df: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
        """Add various return calculations."""
        df['returns'] = df[column].pct_change()
        df['log_returns'] = np.log(df[column] / df[column].shift(1))
        df['returns_lag_1'] = df['returns'].shift(1)
        
        # Cumulative returns
        df['cum_returns'] = (1 + df['returns']).cumprod() - 1
        
        # Rolling returns
        for window in [5, 10, 20]:
            df[f'returns_{window}d'] = df[column].pct_change(window)
        
        return df
    
    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
        
        # Volume-price trend
        df['vpt'] = (df['Volume'] * df['returns']).cumsum()
        
        return df
    
    @staticmethod
    def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        df['high_low_spread'] = df['High'] - df['Low']
        df['open_close_spread'] = df['Close'] - df['Open']
        df['high_low_pct'] = (df['High'] - df['Low']) / df['Low']
        
        # Price position within the day
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        return df
    
    @staticmethod
    def add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add gap analysis features."""
        df['gap'] = df['Open'] - df['Close'].shift(1)
        df['gap_pct'] = df['gap'] / df['Close'].shift(1)
        df['is_gap_up'] = (df['gap'] > 0).astype(int)
        df['is_gap_down'] = (df['gap'] < 0).astype(int)
        
        return df


class TemporalFeatures:
    """Create time-based features."""

    
    @staticmethod
    def add_date_features(df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
        """Add calendar-based features."""
        df['Date'] = pd.to_datetime(df[date_column])
        
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_month'] = df['Date'].dt.day
        df['week_of_year'] = df['Date'].dt.isocalendar().week
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['year'] = df['Date'].dt.year
        
        # Binary flags
        df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype(int)
        df['is_year_start'] = df['Date'].dt.is_year_start.astype(int)
        df['is_year_end'] = df['Date'].dt.is_year_end.astype(int)
        
        # Convert back to date if needed
        df['Date'] = df['Date'].dt.date
        
        return df
    
    @staticmethod
    def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical encoding for periodic features."""
        if 'day_of_week' in df.columns:
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df


class FeatureEngineer:
    """Main feature engineering class that orchestrates all feature creation."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.technical = TechnicalIndicators()
        self.sentiment = SentimentFeatures()
        self.market = MarketFeatures()
        self.temporal = TemporalFeatures()
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features."""
        logger.info("Creating all features")
        initial_rows = len(df)
        
        try:
            # Technical indicators
            logger.info("Adding technical indicators")
            df = self.technical.add_rsi(df)
            df = self.technical.add_macd(df)
            df = self.technical.add_bollinger_bands(df)
            df = self.technical.add_atr(df)
            df = self.technical.add_obv(df)
            df = self.technical.add_moving_averages(df)
            df = self.technical.add_stochastic(df)
            df = self.technical.add_adx(df)
            
            # Sentiment features
            if 'sentiment_score' in df.columns:
                logger.info("Adding sentiment features")
                df = self.sentiment.add_lag_features(df)
                df = self.sentiment.add_rolling_features(df)
                df = self.sentiment.add_momentum_features(df)
                df = self.sentiment.add_volatility_features(df)
                df = self.sentiment.add_news_count_features(df)
                df = self.sentiment.add_sentiment_ema(df)
            
            # Market features
            logger.info("Adding market features")
            df = self.market.add_return_features(df)
            df = self.market.add_volume_features(df)
            df = self.market.add_price_features(df)
            df = self.market.add_gap_features(df)
            
            # Temporal features
            logger.info("Adding temporal features")
            df = self.temporal.add_date_features(df)
            df = self.temporal.add_cyclical_features(df)
            
            # Create target variable
            threshold = 0.002 
            df['Target'] = (df['returns'].shift(-1) > threshold).astype(int)
            
            # Drop rows with NaN values
            df_clean = df.dropna()
            rows_dropped = initial_rows - len(df_clean)
            
            logger.info(f"Feature engineering complete. Created {len(df.columns)} features")
            logger.info(f"Dropped {rows_dropped} rows with NaN values ({rows_dropped/initial_rows:.2%})")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise
    
    def get_feature_names(self, include_target: bool = False) -> List[str]:
        """Get list of all feature names."""
        # This would be populated after running create_all_features
        # For now, return a basic list
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
            'BB_high', 'BB_mid', 'BB_low', 'BB_width', 'BB_pct',
            'ATR', 'OBV', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
            'Stoch_K', 'Stoch_D', 'ADX',
            'sentiment_score', 'sentiment_score_lag_1', 'sentiment_score_lag_2', 'sentiment_score_lag_3',
            'sentiment_score_ma_7', 'sentiment_score_volatility',
            'returns', 'log_returns', 'volume_change', 'high_low_spread', 'open_close_spread',
            'day_of_week', 'month', 'quarter'
        ]
        
        if include_target:
            features.append('Target')
        
        return features


# Legacy function for backward compatibility
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy feature engineering function."""
    engineer = FeatureEngineer()
    return engineer.create_all_features(df)
