import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


from src.data_loader import EnhancedDataLoader
from src.sentiment import apply_sentiment_analysis
from src.preprocessing import run_preprocessing_pipeline
from src.feature_engineering import FeatureEngineer
from src.model import ModelTrainer

# Page config
st.set_page_config(
    page_title="Financial Sentiment Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Load config
@st.cache_resource
def load_config():
    config_path = Path('config/config.yaml')
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

# Title
st.markdown('<h1 class="main-header">📈 Enhanced NLP Stock Prediction System</h1>', unsafe_allow_html=True)
st.markdown("**Advanced Sentiment Analysis + Machine Learning for Stock Market Prediction**")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Ticker input
ticker = st.sidebar.text_input(
    "Stock Ticker Symbol", 
    value=config.get('defaults', {}).get('ticker', 'AA')
).upper()

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=pd.to_datetime(config.get('defaults', {}).get('start_date', '2010-06-10'))
    )
with col2:
    end_date = st.date_input(
        "End Date", 
        value=pd.to_datetime(config.get('defaults', {}).get('end_date', '2021-03-09'))
    )

# Advanced settings
with st.sidebar.expander("🔧 Advanced Settings"):
    aggregation_method = st.selectbox(
        "Sentiment Aggregation",
        ['mean', 'weighted', 'max', 'last'],
        help="Method to aggregate multiple news items per day"
    )
    
    model_selection = st.multiselect(
        "Models to Train",
        ['XGBoost', 'LightGBM', 'LSTM', 'Ensemble'],
        default=['XGBoost'],
        help="Select which models to train"
    )
    
    use_cache = st.checkbox("Use Data Cache", value=True)
    
    max_rows = st.number_input(
        "Max Rows to Load",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
        help="Limit data size for faster processing"
    )

# Data source
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Data Source")

LOCAL_DATA_PATH = "data/fnspid_sample.csv"
uploaded_file = st.sidebar.file_uploader("Upload FNSPID CSV", type="csv")

if uploaded_file is not None:
    data_source = uploaded_file
elif Path(LOCAL_DATA_PATH).exists():
    data_source = LOCAL_DATA_PATH
else:
    data_source = None
    st.sidebar.error("No data source available!")

# Run button
run_analysis = st.sidebar.button("🚀 Run Full Analysis", type="primary", width="stretch")

# Main content
if run_analysis and data_source:
    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize components
        status_text.text("Initializing data loader...")
        progress_bar.progress(10)
        
        loader = EnhancedDataLoader(config)
        engineer = FeatureEngineer(config)
        trainer = ModelTrainer(config)
        
        # Load news data
        status_text.text("Loading news data...")
        progress_bar.progress(20)
        
        news_df = loader.load_fnspid_data(
            data_source, 
            nrows=max_rows,
            ticker=ticker,
            start_date=str(start_date),
            end_date=str(end_date)
        )
        
        if news_df.empty:
            st.error(f"No news found for {ticker} in the selected date range.")
            st.stop()
        
        # Preprocess news
        status_text.text("Preprocessing text...")
        progress_bar.progress(30)
        
        news_df = run_preprocessing_pipeline(news_df, text_column='Headline')
        
        # Apply sentiment analysis
        status_text.text("Analyzing sentiment with FinBERT...")
        progress_bar.progress(40)
        
        news_df = apply_sentiment_analysis(news_df, column_name='Headline')
        
        # Fetch market data
        status_text.text("Fetching market data...")
        progress_bar.progress(50)
        
        market_df = loader.fetch_market_data(ticker, str(start_date), str(end_date), use_cache=use_cache)
        
        # Merge datasets
        status_text.text("Merging datasets...")
        progress_bar.progress(60)
        
        full_df = loader.merge_datasets(news_df, market_df, aggregation=aggregation_method)
        
        # Feature engineering
        status_text.text("Engineering features...")
        progress_bar.progress(70)
        
        full_df_features = engineer.create_all_features(full_df)
        
        # Train model
        status_text.text("Training models...")
        progress_bar.progress(80)
        
        feature_cols = [
                    'Close', 'Volume', 'RSI', 'MACD', 'sentiment_score', 
                    'sentiment_score_lag_1', 'returns', 'ATR', 'SMA_50', 'SMA_200'
                ]
        
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
            full_df_features, feature_cols, test_size=0.2, val_size=0.1
        )
        scaler = MinMaxScaler()

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Train selected models
        results = {}
        if 'XGBoost' in model_selection:
            xgb_model = trainer.train_xgboost(X_train, X_val, y_train, y_val)
            xgb_preds = xgb_model.predict(X_test)
            results['XGBoost'] = trainer.evaluate_model(y_test, xgb_preds, model_name="XGBoost")
        
        if 'LightGBM' in model_selection:
            lgbm_model = trainer.train_lightgbm(X_train, X_val, y_train, y_val)
            lgbm_preds = lgbm_model.predict(X_test)
            results['LightGBM'] = trainer.evaluate_model(y_test, lgbm_preds, model_name="LightGBM")

        # Train LSTM
        if 'LSTM' in model_selection:   
            lstm_model = trainer.train_lstm(X_train, X_val, y_train, y_val)
            
            # Create LSTM predictions for test set
            lstm_preds = []
            seq_len = 30
            lstm_model.eval()
            with torch.no_grad():
                for i in range(seq_len, len(X_test)):
                    seq = X_test[i-seq_len:i]
                    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(trainer.device)
                    pred = lstm_model(seq_tensor).cpu().numpy()[0][0]
                    lstm_preds.append(int(pred > 0.5))

            # Pad beginning
            lstm_preds = [0] * seq_len + lstm_preds
            results['LSTM'] = trainer.evaluate_model(y_test, lstm_preds, model_name="LSTM")

        # Ensemble model
        if 'Ensemble' in model_selection:
            ensemble_result = trainer.create_ensemble(X_test, y_test)
            ensemble_preds = ensemble_result['predictions']
            
            results['Ensemble'] = trainer.evaluate_model(
                y_test, ensemble_preds, 
                ensemble_result['probabilities'],
                model_name="Ensemble"
            )

        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # ============ RESULTS DISPLAY ============
        
        st.success("✅ Analysis completed successfully!")
        
        # Tab navigation
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🔍 Sentiment Analysis", 
            "📈 Price & Sentiment", 
            "🤖 Model Performance",
            "📉 Feature Analysis"
        ])
        
        # TAB 1: Overview
        with tab1:
            st.header("Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trading Days", len(full_df_features))
            with col2:
                st.metric("News Articles", len(news_df))
            with col3:
                st.metric("Days with News", (full_df['news_count'] > 0).sum())
            with col4:
                avg_sentiment = full_df['sentiment_score'].mean()
                st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
            
            # Data sample
            st.subheader("Sample Data")
            st.dataframe(
                full_df_features[['Date', 'Close', 'sentiment_score', 'RSI', 'MACD', 'Target']].head(10),
                width="stretch"
            )
        
        # TAB 2: Sentiment Analysis
        with tab2:
            st.header("Sentiment Analysis Results")
            
            # Sentiment distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Distribution")
                fig = px.histogram(
                    news_df, 
                    x='sentiment_score',
                    nbins=50,
                    title="Distribution of Sentiment Scores",
                    labels={'sentiment_score': 'Sentiment Score', 'count': 'Frequency'}
                )
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                st.subheader("News Volume Over Time")
                news_count_daily = news_df.groupby('Date').size().reset_index(name='count')
                fig = px.line(
                    news_count_daily, 
                    x='Date', 
                    y='count',
                    title="Daily News Article Count"
                )
                st.plotly_chart(fig, width="stretch")
            
            # Sample processed headlines
            st.subheader("Processed Headlines Sample")
            st.dataframe(
                news_df[['Headline', 'processed_headline', 'sentiment_score']].head(10),
                width="stretch"
            )
        
        # TAB 3: Price & Sentiment
        with tab3:
            st.header("Price vs Sentiment Visualization")
            
            # Create dual-axis plot
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=full_df['Date'],
                    y=full_df['Close'],
                    name="Stock Price",
                    line=dict(color='#1f77b4', width=2)
                ),
                secondary_y=False
            )
            
            # Add sentiment area
            fig.add_trace(
                go.Scatter(
                    x=full_df['Date'],
                    y=full_df['sentiment_score'],
                    name="Sentiment Score",
                    fill='tozeroy',
                    line=dict(color='#2ca02c', width=1),
                    opacity=0.3
                ),
                secondary_y=True
            )
            
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Stock Price ($)", secondary_y=False)
            fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
            
            fig.update_layout(
                title="Stock Price and Sentiment Over Time",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Correlation metrics
            st.subheader("Correlation Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                corr_val = full_df['sentiment_score'].corr(full_df['returns'])
                st.metric("Sentiment-Return Correlation", f"{corr_val:.4f}")
            
            with col2:
                full_df['Lagged_Sentiment'] = full_df['sentiment_score'].shift(1)
                lag_corr = full_df['Lagged_Sentiment'].corr(full_df['returns'])
                st.metric("Lagged Correlation (t-1)", f"{lag_corr:.4f}")
            
            with col3:
                lag_corr_3 = full_df['sentiment_score'].shift(3).corr(full_df['returns'])
                st.metric("Lagged Correlation (t-3)", f"{lag_corr_3:.4f}")
        
        # TAB 4: Model Performance
        with tab4:
            st.header("Machine Learning Model Performance")
            
            # Display metrics for each model
            for model_name, metrics in results.items():
                st.subheader(f"{model_name} Results")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                col2.metric("Precision", f"{metrics['precision']:.2%}")
                col3.metric("Recall", f"{metrics['recall']:.2%}")
                col4.metric("F1-Score", f"{metrics['f1_score']:.2%}")
            
            # Feature importance (if XGBoost was trained)
            if 'XGBoost' in model_selection and 'xgboost' in trainer.models:
                st.subheader("Top 20 Most Important Features")
                
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': trainer.models['xgboost'].feature_importances_
                }).sort_values('importance', ascending=False).head(20)
                
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Feature Importance"
                )
                st.plotly_chart(fig, width="stretch")
        
        # TAB 5: Feature Analysis
        with tab5:
            st.header("Feature Correlation Analysis")
            
            # Select key features for correlation
            key_features = ['Close', 'sentiment_score', 'RSI', 'MACD', 'BB_width', 
                          'ATR', 'OBV', 'SMA_50', 'returns', 'volume_change']
            
            available_features = [f for f in key_features if f in full_df_features.columns]
            
            if len(available_features) > 1:
                corr_matrix = full_df_features[available_features].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Feature Correlation Heatmap"
                )
                st.plotly_chart(fig, width="stretch")
            
            # Time series of key features
            st.subheader("Key Features Over Time")
            
            features_to_plot = st.multiselect(
                "Select features to visualize",
                available_features,
                default=available_features[:3]
            )
            
            if features_to_plot:
                fig = go.Figure()
                for feature in features_to_plot:
                    fig.add_trace(go.Scatter(
                        x=full_df_features['Date'],
                        y=full_df_features[feature],
                        name=feature,
                        mode='lines'
                    ))
                
                fig.update_layout(
                    title="Features Over Time",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig, width="stretch")
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)
        
else:
    # Welcome screen
    st.info("👈 Configure your analysis parameters in the sidebar and click 'Run Full Analysis' to begin.")
    
    st.markdown("""
    ## 🎯 System Features
    
    ### Data Processing
    - ✅ Intelligent data validation and quality checks
    - ✅ Smart caching for faster reloads
    - ✅ Multiple sentiment aggregation strategies
    
    ### NLP & Sentiment Analysis
    - ✅ FinBERT domain-specific sentiment analysis
    - ✅ Financial "numberness" preservation
    - ✅ Batch processing for efficiency
    
    ### Feature Engineering
    - ✅ 80+ advanced features
    - ✅ Technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
    - ✅ Sentiment features (lags, rolling stats, momentum)
    - ✅ Temporal features (cyclical encoding)
    
    ### Machine Learning
    - ✅ XGBoost classifier
    - ✅ LightGBM classifier
    - ✅ LSTM neural network
    - ✅ Ensemble methods
    
    ### Visualization
    - ✅ Interactive Plotly charts
    - ✅ Correlation analysis
    - ✅ Feature importance
    - ✅ Model performance metrics
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>NLP Stock Prediction Sentiment Anlysis System</strong></p>
    <p>Addis Ababa University | School of Information Technology and Engineering</p>
</div>
""", unsafe_allow_html=True)
