
import pandas as pd
import yfinance as yf
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from loguru import logger
import hashlib
import pickle
from datetime import datetime, timedelta


class DataValidator:
    """Validates data quality and integrity."""
    
    @staticmethod
    def validate_fnspid_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate FNSPID dataset."""
        issues = []
        
        # Check required columns
        required_cols = ['Date', 'Headline', 'Ticker']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for empty data
        if df.empty:
            issues.append("Dataset is empty")
        
        # Check date format
        if 'Date' in df.columns:
            try:
                pd.to_datetime(df['Date'], errors='coerce')
            except Exception as e:
                issues.append(f"Invalid date format: {str(e)}")
        
        # Check for null values
        null_counts = df.isnull().sum()
        critical_nulls = null_counts[null_counts > len(df) * 0.5]
        if not critical_nulls.empty:
            issues.append(f"High null percentage in columns: {critical_nulls.to_dict()}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_market_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate market data from yfinance."""
        issues = []
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for negative prices
        if 'Close' in df.columns and (df['Close'] < 0).any():
            issues.append("Negative prices detected")
        
        # Check for zero volume days
        if 'Volume' in df.columns:
            zero_vol_pct = (df['Volume'] == 0).sum() / len(df)
            if zero_vol_pct > 0.1:
                issues.append(f"High percentage of zero volume days: {zero_vol_pct:.2%}")
        
        # Check for price anomalies (daily change > 50%)
        if 'Close' in df.columns:
            returns = df['Close'].pct_change()
            if (abs(returns) > 0.5).any():
                issues.append("Extreme price movements detected (>50% daily change)")
        
        return len(issues) == 0, issues


class DataCache:
    """Simple file-based caching system."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_string = str(args)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data."""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def set(self, key: str, data: pd.DataFrame):
        """Save data to cache."""
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


class EnhancedDataLoader:
    """Enhanced data loader with validation and caching."""
    
    def __init__(self, config: dict):
        self.config = config
        self.cache = DataCache(config.get('data', {}).get('cache_dir', 'data/cache'))
        self.validator = DataValidator()
    
    def load_fnspid_data(
        self, 
        filepath: str, 
        nrows: Optional[int] = None,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load FNSPID dataset with validation and filtering.
        
        Args:
            filepath: Path to CSV file
            nrows: Number of rows to read (None = all)
            ticker: Filter by ticker symbol
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Validated and filtered DataFrame
        """
        logger.info(f"Loading FNSPID data from {filepath}")
        
        # Check cache
        cache_key = self.cache._get_cache_key(filepath, nrows, ticker, start_date, end_date)
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.info("Data loaded from cache")
            return cached_data
        
        # Load data
        try:
            df = pd.read_csv(filepath, nrows=nrows)
            logger.info(f"Loaded {len(df)} rows from file")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
        
        # Validate
        is_valid, issues = self.validator.validate_fnspid_data(df)
        if not is_valid:
            logger.warning(f"Data validation issues: {issues}")
        
        # Convert date
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Date'] = df['Date'].dt.date
        
        # Filter by ticker
        if ticker and 'Ticker' in df.columns:
            df = df[df['Ticker'] == ticker]
            logger.info(f"Filtered to ticker {ticker}: {len(df)} rows")
        
        # Filter by date
        if start_date:
            start_date = pd.to_datetime(start_date).date()
            df = df[df['Date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date).date()
            df = df[df['Date'] <= end_date]
        
        logger.info(f"Final dataset size: {len(df)} rows")
        
        # Cache the result
        self.cache.set(cache_key, df)
        
        return df
    
    def fetch_market_data(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch market data with validation and caching.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data
            
        Returns:
            Validated market data DataFrame
        """
        logger.info(f"Fetching market data for {ticker}")
        
        # Check cache
        cache_key = self.cache._get_cache_key('market', ticker, start_date, end_date)
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.info("Market data loaded from cache")
                return cached_data
        
        # Fetch from yfinance with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                if df.empty:
                    raise ValueError(f"No data returned for {ticker}")
                
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch market data after {max_retries} attempts")
                    raise
        
        # Process data
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # Validate
        is_valid, issues = self.validator.validate_market_data(df)
        if not is_valid:
            logger.warning(f"Market data validation issues: {issues}")
        
        logger.info(f"Fetched {len(df)} trading days")
        
        # Cache the result
        self.cache.set(cache_key, df)
        
        return df
    
    def merge_datasets(
        self, 
        news_df: pd.DataFrame, 
        market_df: pd.DataFrame,
        aggregation: str = 'mean'
    ) -> pd.DataFrame:
        """
        Merge sentiment and market data with improved aggregation.
        
        Args:
            news_df: News data with sentiment scores
            market_df: Market OHLCV data
            aggregation: Method to aggregate sentiment ('mean', 'weighted', 'max', 'last')
            
        Returns:
            Merged DataFrame with sentiment and market data
        """
        logger.info(f"Merging datasets with {aggregation} aggregation")
        
        # Handle different aggregation methods
        if aggregation == 'mean':
            daily_sentiment = news_df.groupby('Date')['sentiment_score'].mean()

        elif aggregation == 'weighted':
            # Weight by absolute sentiment (stronger signals matter more)
            news_df['weight'] = news_df['sentiment_score'].abs()

            daily_sentiment = (
                news_df
                .groupby('Date', group_keys=False)
                .apply(
                    lambda x: np.average(x['sentiment_score'], weights=x['weight']),
                    include_groups=False
                )
                .rename('sentiment_score')
            )

        elif aggregation == 'max':
            # Use the strongest sentiment of the day
            daily_sentiment = news_df.groupby('Date')['sentiment_score'].apply(
                lambda x: x.loc[x.abs().idxmax()] if len(x) > 0 else 0
            )

        elif aggregation == 'last':
            # Use the last sentiment of the day
            daily_sentiment = news_df.groupby('Date')['sentiment_score'].last()

        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        daily_sentiment = daily_sentiment.reset_index()

        # Add news count
        news_count = news_df.groupby('Date').size().reset_index(name='news_count')
        daily_sentiment = daily_sentiment.merge(news_count, on='Date', how='left')

        merged_df = pd.merge(market_df, daily_sentiment, on='Date', how='left')
        
        # Fill missing sentiment with 0 (neutral)
        merged_df['sentiment_score'] = merged_df['sentiment_score'].fillna(0)
        merged_df['news_count'] = merged_df['news_count'].fillna(0).astype(int)
        
        logger.info(f"Merged dataset size: {len(merged_df)} rows")
        logger.info(f"Days with news: {(merged_df['news_count'] > 0).sum()}")
        
        return merged_df
    
    def load_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        news_filepath: str
    ) -> dict:
        """
        Load data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            news_filepath: Path to FNSPID data
            
        Returns:
            Dictionary mapping tickers to merged DataFrames
        """
        results = {}
        
        for ticker in tickers:
            try:
                logger.info(f"Loading data for {ticker}")
                news_df = self.load_fnspid_data(
                    news_filepath,
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date
                )
                market_df = self.fetch_market_data(ticker, start_date, end_date)
                results[ticker] = (news_df, market_df)
            except Exception as e:
                logger.error(f"Failed to load data for {ticker}: {e}")
                
        return results


def load_fnspid_data(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Legacy function for backward compatibility.
    """
    config = {'data': {'cache_dir': 'data/cache'}}
    loader = EnhancedDataLoader(config)
    return loader.load_fnspid_data(filepath, nrows)


def fetch_market_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Legacy function for backward compatibility.
    """
    config = {'data': {'cache_dir': 'data/cache'}}
    loader = EnhancedDataLoader(config)
    return loader.fetch_market_data(ticker, start_date, end_date)


def merge_datasets(news_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy function for backward compatibility.
    """
    config = {'data': {'cache_dir': 'data/cache'}}
    loader = EnhancedDataLoader(config)
    return loader.merge_datasets(news_df, market_df)
