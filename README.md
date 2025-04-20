# Stock Trend Prediction Using Machine Learning
## Project Overview
This project focuses on predicting the directional movement of stock prices (up or down) for the next trading day using machine learning models. Instead of forecasting exact prices, which are highly volatile and influenced by many unpredictable factors, we reframe the problem as a binary classification task. This approach makes the problem more tractable and reduces noise from outlier price movements.

## Overview of Dataset
Dataset contains historical daily prices and volumes of U.S. stocks and ETFs over the period of 2011 to 2017. We selected the top 50 stocks by market capitalization, ensuring high liquidity and diversity across industries. Each observation contains **date, opening price, high, low, closing price, and trading volume**. Our final cleaned dataset is **filtered_stocks_combined.csv**.

## Evaluation Metrics
For this classification problem, we used three key metrics to evaluate model performance:
Accuracy: Percentage of correct predictions.
F1 Score: Harmonic mean of precision and recall, balancing both.
AUC-ROC: Measures the model’s ability to distinguish between classes (1.0 = perfect, 0.5 = random guessing).

## Models
We implemented the following models:
1) Logistic Regression
2) Multilayer Perceptron (MLP)
3) Recurrent Neural Network (RNN)
4) Long Short Term Memory (LSTM)
5) XGBoost

The results of our models are as follows:
|         | Accuracy | F1 Score | Area under ROC Curve |
|--------------|--------|--------|--------|
| Logistic Regression | 0.7647 | 0.7777 | 0.8280 |
| MLP | 0.5130 | 0.5072 | 0.5201 | 
| RNN | 0.5165 | 0.6706 | 0.5098 | 
| LSTM | 0.5119 | 0.3550 | 0.5119 | 
| XGBoost | 0.6419 | 0.6734 | 0.7047 | 

Logistic Regression performs the best on all 3 metrics. With just a few features used like “Open”, “High”, “Low” and “Volume” and about 75,000 observations, the model is less likely to overfit and can effectively separate the classes without being overwhelmed by noise or complex interactions. In such settings, simpler models like logistic regression often perform well and can even rival more complex algorithms, as there is less risk of the model being misled by spurious patterns or high-dimensional noise. Other more complicated models were simply not able to match the performance of the logistic model because of the characteristic of the dataset. 

## Future Improvements
1) Improved Feature Engineering
Additional features such as momentum indicators, volatility measures, or cross‑asset signals can be included. We can also incorporate external macroeconomic or sentiment data to further augment the dataset. This would likely play into the strengths of the more complex models given their ability to model non-linear interactions and high-dimensional settings.
2) Refining the rolling‑window scheme
To further improve the project, we can consider optimising window length, overlap, or retraining frequency via cross‑validation

## Guide to the Github Repository 

To run the code, please either clone the repository using Git: 
``` 
git clone https://github.com/brandono7/stock_price_prediction.git
```
or download the repository as a zip file and extract it to a location of your choice.

**Prerequisites**

Before running the project, make sure to install the following python libraries on your virtual environment via terminal / command prompt:
```
pip install -r requirements.txt
```

**Repository Structure**
1. **`Stocks` folder**: Contains the raw data of all the stocks that we obtained from [Kaggle](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/data).
2. nasdaq_screener_1742913519579.csv: Used as a reference to filter out the top 50 stocks in market capitalization 
3. filtered_stocks_combined.csv: Final cleaned dataset
4. logit.ipynb: Codes that implement the logit model
5. data_preprocessing.ipynb: Codes that clean the data and perform EDA
6. xgboost.ipynb: Codes that implement the XGBoost model
7. RNN.ipynb: Codes that implement the RNN model

This project is a group assignment for a NUS course - CS3244 - Machine Learning.

Authors: Brandon Ong Cae Jun, Chew Yun Yi, Gao Yening, Leong Deng Jun, Zhang Yijian, Wang Chengmao (Group 24)

Last Updated: 20 Apr 2025