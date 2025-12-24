# Emotion–Fundamental Trend Amplifier with Cross-Market Signals

## Overview
This project develops a **deep learning framework** for predicting **weekly stock trading signals (BUY, SELL, HOLD)** by integrating **company fundamentals**, **macroeconomic indicators**, **investor sentiment**, and **cross-market signals**.

The core idea is **signal coherence**: reliable trading signals emerge when multiple data sources align.  
The framework combines **Transformer-based architectures**, **temporal convolution**, and **ensemble learning** to capture both short-term dynamics and long-horizon dependencies in financial markets.

---

## Problem
Short-term stock prediction is difficult because signals are:
- noisy and highly non-linear  
- distributed across heterogeneous data sources  
- updated at different frequencies (weekly prices vs quarterly fundamentals)  

Traditional statistical models and single-source ML approaches fail to capture how **fundamentals, sentiment, and broader market dynamics interact**. This project addresses that gap using **multimodal deep learning**.

---

## Data
The dataset spans **2009–2024** and covers **502 S&P 500 stocks**, with approximately **420,000 weekly observations**.

**Data sources**
- **Fundamentals:** profitability, leverage, liquidity, valuation ratios  
- **Macroeconomic indicators:** inflation, interest rates, growth signals  
- **Investor sentiment:** sentiment scores from Reddit posts and earnings calls  
- **Cross-market signals:** Bitcoin, oil, VIX and related market indicators  

📎 **Public dataset (Kaggle):**  
https://www.kaggle.com/datasets/chitamvuong/st456-final-dataset-of-deep-visionaries/data

---

## Target Variable
Each stock-week is classified into:
- **BUY** (weekly return > +2%)
- **SELL** (weekly return < −2%)
- **HOLD** (within ±2%)

This setup reflects realistic short-term trading decisions and naturally produces an imbalanced class distribution (~50% HOLD, 30% BUY, 20% SELL).

---

## Methodology

### Preprocessing
- Feature standardisation across all inputs  
- Temporal alignment across mixed data frequencies  
- Reshaping into **52-week rolling sequences (52 × 47 features)**  
- Stratified train–test split (80/20)  

### Models
The framework evaluates and compares multiple architectures:
- **LSTM**
- **TimesNet**
- **Cross-Modal Transformer**
- **TFT-GAT (Temporal Fusion Transformer + Graph Attention)**
- **Informer**
- **Informer-TCN (hybrid attention + temporal convolution)**

A **stacked ensemble using XGBoost** aggregates predictions and produces a **coherence score** reflecting agreement across data modalities.

---

## Evaluation
Models are assessed using:
- Accuracy, Precision, Recall, **F1-score**
- **ROC-AUC**
- Coherence-based filtering for high-confidence signals
- Portfolio-level backtesting

---

## Key Results
- **Informer-TCN** achieved the strongest overall performance  
  - F1-score ≈ **0.39**
  - ROC-AUC ≈ **0.59**
- High-coherence signals (~11% of predictions) delivered a  
  **+17.97% portfolio return**, outperforming the S&P 500 (+9%)  
- SHAP analysis shows meaningful contributions from  
  **macroeconomic and cross-market variables**, validating the multimodal design  

---

## Takeaways
- Multimodal deep learning improves short-term trend detection when signals align  
- Sentiment is valuable but noisy; coherence filtering is essential  
- Hybrid architectures (attention + convolution) outperform single-model approaches  

---

## Limitations
- Sentiment signals remain noisy despite attention mechanisms  
- Computational cost is high for long sequences and large feature sets  
- Results focus on large-cap US equities and may not generalise directly  

---

## Future Work
- Adaptive temporal alignment across data sources  
- Financial-domain LLMs for richer sentiment extraction  
- Online / incremental learning for regime shifts  
- Lighter-weight architectures for faster inference  

---

## Notes
This repository contains a **cleaned and independent version** of work originally developed in a collaborative academic setting.

All **course identifiers, group labels, and submission artifacts** have been removed.  
The project is presented for **professional, reproducible, and portfolio use**.
