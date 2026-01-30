# 📈 NLP-Based Stock Prediction and Sentiment Analysis System

## 🎯 Project Overview

This is NLP-based financial news sentiment analysis system for predicting stock market trends. The system combines state-of-the-art NLP (FinBERT), advanced machine learning algorithms (XGBoost, LightGBM, LSTM), and comprehensive feature engineering to predict stock price movements.

### Key features
- ✅ **Multiple ML Algorithms**: XGBoost, LightGBM, LSTM, and Ensemble methods
- ✅ **80+ Advanced Features**: Technical indicators, sentiment features, temporal features
- ✅ **Production-Ready Architecture**: Proper logging, error handling, model persistence
- ✅ **Data Validation**: Comprehensive data quality checks and validation
- ✅ **Time Series Cross-Validation**: Proper temporal validation strategies
- ✅ **Caching System**: Faster data loading with intelligent caching
- ✅ **Configuration Management**: YAML-based configuration for easy customization

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  FNSPID News │  │  yFinance    │  │  Custom Data │     │
│  │   Dataset    │  │   Market     │  │    Sources   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Data Processing Pipeline                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Validation  │→ │   Cleaning   │→ │    Caching   │     │
│  │   & Quality  │  │  & Merging   │  │   & Storage  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    NLP Processing Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Tokenization │→ │   FinBERT    │→ │  Sentiment   │     │
│  │   & Lemma    │  │  Sentiment   │  │ Aggregation  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Feature Engineering Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Technical   │  │  Sentiment   │  │   Temporal   │     │
│  │  Indicators  │  │   Features   │  │   Features   │     │
│  │  (RSI, MACD) │  │ (Lags, MAs)  │  │ (Day, Month) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    ML Training Pipeline                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   XGBoost    │  │   LightGBM   │  │     LSTM     │     │
│  │  Classifier  │  │  Classifier  │  │   Network    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                           ↓                                 │
│                  ┌──────────────┐                          │
│                  │   Ensemble   │                          │
│                  │    Model     │                          │
│                  └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│               Visualization & Serving Layer                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Streamlit   │  │   FastAPI    │  │   MLflow     │     │
│  │  Dashboard   │  │  REST API    │  │  Tracking    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd improved_nlp_stock_prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Configuration

Edit `config/config.yaml` to customize:
- Data paths and sources
- Model hyperparameters
- Feature engineering options
- Logging and tracking settings

### Running the Dashboard

```bash
streamlit run app.py
```


## 📁 Project Structure

```
improved_nlp_stock_prediction/
│
├── config/
│   └── config.yaml              # Configuration file
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Enhanced data loading with validation
│   ├── preprocessing.py         # NLP preprocessing
│   ├── sentiment.py             # FinBERT sentiment analysis
│   ├── feature_engineering.py   # 80+ features creation
│   ├── model.py                 # ML model training (XGB, LGBM, LSTM)
│
├── data/
│
│
├── app.py                       # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🎨 Features

### 1. Data Processing
- **Validation Layer**: Comprehensive data quality checks
- **Smart Caching**: Hash-based caching for faster reloads
- **Multi-ticker Support**: Analyze multiple stocks simultaneously
- **Robust Error Handling**: Graceful degradation and informative errors

### 2. NLP Pipeline
- **FinBERT Sentiment**: Domain-specific financial sentiment
- **Numberness Preservation**: Keeps important financial figures
- **Batch Processing**: Efficient processing of large datasets
- **Sentiment Aggregation**: Multiple aggregation strategies (mean, weighted, max, last)

### 3. Feature Engineering (80+ Features)

#### Technical Indicators
- RSI (Relative Strength Index)
- MACD with signal and divergence
- Bollinger Bands (high, mid, low, width, percentage)
- ATR (Average True Range)
- OBV (On-Balance Volume)
- Moving Averages (SMA 50/200, EMA 12/26)
- Stochastic Oscillator
- ADX (Average Directional Index)

#### Sentiment Features
- Lag features (t-1, t-2, t-3, t-5, t-7)
- Rolling statistics (mean, std, min, max for 3, 7, 14, 30 days)
- Sentiment momentum and acceleration
- Sentiment volatility
- News count and intensity features

#### Market Features
- Multiple return calculations (simple, log, rolling)
- Volume analysis (change, ratio, volume-price trend)
- Price spreads (high-low, open-close)
- Gap analysis (gap, gap percentage, gap direction)

#### Temporal Features
- Calendar features (day, week, month, quarter, year)
- Cyclical encoding (sin/cos transformations)
- Binary flags (month/quarter/year start/end)

### 4. Machine Learning Models

#### XGBoost Classifier
- Gradient boosting with tree-based models
- Feature importance analysis
- Hyperparameter tuning with cross-validation

#### LightGBM Classifier
- Fast gradient boosting framework
- Efficient memory usage
- Better performance on large datasets

#### LSTM Neural Network
- Captures temporal dependencies
- Sequence-based prediction
- Deep learning approach for pattern recognition

#### Ensemble Model
- Combines predictions from all models
- Weighted or simple averaging
- Improved generalization and robustness

### 5. Model Evaluation

Comprehensive metrics:
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Trading**: Sharpe Ratio, Maximum Drawdown, Win Rate
- **Visualization**: Confusion matrix, ROC curves, feature importance
- **Backtesting**: Historical performance simulation

### 6. Production Features

- **Model Persistence**: Save/load trained models
- **MLflow Integration**: Experiment tracking and model registry
- **Structured Logging**: Comprehensive logging with Loguru
- **Configuration Management**: YAML-based config
- **API Endpoints**: FastAPI for model serving
- **Docker Support**: Containerized deployment
- **CI/CD Ready**: Automated testing and deployment

---


## 📚 References

- **FinBERT**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- **FNSPID Dataset**: Financial News and Stock Price Integration Dataset

---

## 👥 Team

- **Aderajew Yeshiwendm** (UGR/3662/14)
- **Aregawi Fikre** (UGR/6531/14)
- **Yohannes Alemayehu** (UGR/249714/)

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Addis Ababa University, College of Technology and Built Environment
- School of Information Technology and Engineering
- ProsusAI for the FinBERT model
---
