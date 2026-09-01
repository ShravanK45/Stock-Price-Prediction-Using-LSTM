# Stock Price Prediction using LSTM

A Deep Learning project that uses a **Long Short-Term Memory (LSTM)** network to predict future stock closing prices based on the previous **60 days of historical price data**.

The project demonstrates how sequential data can be transformed into sliding windows and processed using LSTM networks for time-series forecasting.

---

## 📌 Project Overview

Unlike traditional machine learning datasets where rows are treated independently, stock prices are sequential in nature.

The price at a particular time is influenced by historical patterns and previous observations.

This project uses:

> **Previous 60 days of stock closing prices → Predict the next day's closing price**

The model learns temporal patterns from historical stock prices using an LSTM-based neural network.

---

## 🧠 Workflow

```text
Historical Stock Data
        ↓
Select Closing Price
        ↓
Chronological Train-Test Split
        ↓
Feature Scaling
        ↓
Create 60-Day Sliding Windows
        ↓
Reshape Data for LSTM
        ↓
Train LSTM Model
        ↓
Predict Future Prices
        ↓
Inverse Scaling
        ↓
Compare Actual vs Predicted Prices
```

---

## 📊 Sliding Window Approach

A window of the previous **60 days** is used as input to predict the next day's stock price.

Example:

```text
Day 1 → Day 60
        ↓
Predict Day 61


Day 2 → Day 61
        ↓
Predict Day 62
```

This sliding-window technique converts a single time series into multiple supervised learning samples.

---

## 🔧 Data Preprocessing

### 1. Chronological Train-Test Split

The dataset is split chronologically:

* **95% Training Data**
* **5% Testing Data**

Random splitting is avoided because time-series data must preserve temporal order.

```text
Past Data → Training
Future Data → Testing
```

---

### 2. Feature Scaling

`StandardScaler` is used to normalize the stock prices.

The scaler is fitted only on the training data to prevent data leakage.

```python
scaler.fit_transform(training_data)
```

The test data is transformed using the same scaler:

```python
scaler.transform(test_data)
```

---

### 3. Sequence Generation

The previous 60 days are used as input features.

The data is reshaped into:

```text
(samples, timesteps, features)
```

For this project:

```text
(samples, 60, 1)
```

Where:

* `samples` → Number of training sequences
* `60` → Previous 60 trading days
* `1` → Closing price feature

---

## 🤖 Model Architecture

The LSTM model consists of:

```text
Input
  ↓
LSTM (64 units)
  ↓
LSTM (64 units)
  ↓
Dense Layer (128, ReLU)
  ↓
Dropout (0.5)
  ↓
Dense Layer (1)
  ↓
Predicted Stock Price
```

---

## ⚙️ Model Configuration

* Optimizer: Adam
* Loss Function: Mean Absolute Error (MAE)
* Evaluation Metric: Root Mean Squared Error (RMSE)
* Batch Size: 32
* Epochs: 20
* Sequence Length: 60 Days

---

## 📈 Prediction Results

The model predictions are inverse-transformed back to the original stock price scale and compared against actual test data.

### Actual vs Predicted Stock Prices

![Stock Predictions](images/stock_predictions.png)

The model captures the overall trend of the stock price reasonably well.

As expected in time-series forecasting, the predictions may lag behind sudden spikes or sharp market movements since the model relies only on historical closing-price information.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow / Keras
* LSTM

---

## 📂 Project Structure

```text
Stock-Price-Prediction-Using-LSTM/
│
├── Stock_Price_Prediction_LSTM.ipynb
├── README.md
├── requirements.txt
├── MicrosoftStock.csv
│
└── images/
    └── stock_predictions.png
```

---

## 🚀 Key Learnings

* Time-series forecasting fundamentals
* Chronological train-test splitting
* Avoiding data leakage
* Feature scaling for sequential data
* Sliding-window sequence generation
* LSTM input structure
* `(samples, timesteps, features)` format
* Training deep learning models for regression
* Inverse transforming predictions
* Comparing actual vs predicted time-series values

---

## 🔮 Future Improvements

* Use additional features such as Open, High, Low, and Volume
* Experiment with GRU and Bidirectional LSTM
* Hyperparameter tuning
* Add technical indicators
* Compare LSTM with Transformer-based forecasting models
* Evaluate using MAE, RMSE, and MAPE
* Multi-step stock price forecasting

---

## ⚠️ Disclaimer

This project is created for educational purposes to understand time-series forecasting using LSTM networks.

Stock markets are influenced by numerous external factors, and historical price data alone cannot guarantee accurate future predictions.

---

## 👨‍💻 Author

**Shravan Kundap**

Aspiring AI/ML Engineer | Deep Learning | Machine Learning
