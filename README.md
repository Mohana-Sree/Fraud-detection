# 🔒 Credit Card Fraud Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-black?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-blueviolet?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A real-time machine learning system for detecting fraudulent credit card transactions**

[Demo](#-demo) • [Features](#-key-features) • [Installation](#-installation) • [Usage](#-usage) • [Model Performance](#-model-performance)

</div>

---

## 🎯 About The Project

This project implements a **machine learning-based fraud detection system** for credit card transactions. The system uses advanced data science techniques to identify potentially fraudulent transactions in real-time, helping financial institutions minimize losses and protect customers.

### 🚀 **Live Demo**
Experience the fraud detection system: **[Try Live Demo](https://web-production-f8cfb.up.railway.app/)**
 
---

## ✨ Key Features

- 🔍 **Real-time Fraud Detection** - Instant classification of transactions
- 📊 **High Accuracy Model** - 94% overall accuracy with 88.8% fraud detection rate
- 🌐 **Web Interface** - User-friendly Flask web application
- 📈 **Confidence Scoring** - Probability-based predictions with confidence levels
- 🎲 **Random Sample Generator** - Test the system with realistic transaction data
- ⚡ **Fast Processing** - Sub-second prediction response times
- 📱 **Responsive Design** - Works on desktop and mobile devices

---

## 🛠️ Technology Stack

### **Backend**
- **Python 3.12** - Core programming language
- **Flask** - Web application framework
- **scikit-learn** - Machine learning library
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### **Machine Learning**
- **Logistic Regression** - Classification algorithm
- **StandardScaler** - Feature preprocessing
- **Class Balancing** - Handling imbalanced datasets
- **Cross-validation** - Model evaluation

### **Data Source**
- **Kaggle API** - Automated dataset download
- **Credit Card Fraud Dataset** - Real-world transaction data

---

## 🚀 Installation

### **Prerequisites**
- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### **Quick Setup**

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mohana-Sree/Fraud-detection.git
   cd Fraud-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**
   ```bash
   python model_training.py
   ```
   *This will automatically download the dataset from Kaggle and train the model*

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://127.0.0.1:5000`

---

## 💻 Usage

### **Web Interface**

1. **Manual Input**: Enter transaction details (V1-V5 features and Amount)
2. **Random Sample**: Generate realistic test data with one click
3. **Analyze**: Click "Analyze Transaction" to get fraud prediction
4. **Results**: View prediction with confidence percentage

### **Example Transaction**
```
V1: 1.234     V2: -0.567    V3: 2.891
V4: -1.456    V5: 0.789     Amount: 150.50
Result: "This transaction is likely LEGITIMATE with 95.67% confidence"
```

### **API Usage**
```python
import requests

# Example API call
data = {
    'v1': 1.234,
    'v2': -0.567,
    'v3': 2.891,
    'v4': -1.456,
    'v5': 0.789,
    'amount': 150.50
}

response = requests.post('http://127.0.0.1:5000/predict', data=data)
result = response.json()
print(result['result'])
```

---

## 📊 Model Performance

### **Dataset Statistics**
- **Total Transactions**: 284,807
- **Legitimate**: 227,451 (99.83%)
- **Fraudulent**: 394 (0.17%)
- **Class Imbalance Ratio**: 577:1

### **Model Metrics**
| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 94.01% |
| **Fraud Detection Rate (Recall)** | 88.8% |
| **Precision** | 2.5% |
| **False Alarm Rate** | 5.98% |
| **True Positives** | 87 fraud cases detected |
| **False Negatives** | 11 fraud cases missed |

### **Confusion Matrix**
```
                    Predicted
                 Legit    Fraud
Actual  Legit   53,465   3,399
        Fraud      11      87
```

### **Business Impact**
- ✅ **High Fraud Detection**: Catches 9 out of 10 fraudulent transactions
- ✅ **Manageable False Alarms**: ~6% require manual review
- ✅ **Cost-Effective**: Prevents significant financial losses

---

## 📊 Dataset Information

### **Source**
- **Kaggle**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Origin**: European cardholders transactions (2013)

### **Features**
- **V1-V28**: PCA-transformed features (anonymized for privacy)
- **Time**: Seconds elapsed between transactions
- **Amount**: Transaction amount
- **Class**: 0 = Legitimate, 1 = Fraud

### **Privacy & Security**
- All sensitive features are anonymized using PCA transformation
- No personally identifiable information (PII) included
- Compliant with financial data privacy standards

---

## 🔌 API Documentation

### **Endpoints**

#### `GET /`
- **Description**: Render the main web interface
- **Response**: HTML page

#### `POST /predict`
- **Description**: Predict fraud for a transaction
- **Parameters**:
  - `v1` (float): PCA feature 1
  - `v2` (float): PCA feature 2
  - `v3` (float): PCA feature 3
  - `v4` (float): PCA feature 4
  - `v5` (float): PCA feature 5
  - `amount` (float): Transaction amount
- **Response**:
  ```json
  {
    "result": "This transaction is likely LEGITIMATE with 95.67% confidence",
    "is_fraud": false
  }
  ```

#### `GET /health`
- **Description**: Health check endpoint
- **Response**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true
  }
  ```

---

## 🧠 Machine Learning Approach

### **Algorithm Choice: Logistic Regression**
- **Interpretability**: Essential for financial compliance
- **Probability Output**: Provides confidence scores
- **Speed**: Fast inference for real-time predictions
- **Proven**: Industry standard for fraud detection

### **Handling Class Imbalance**
```python
# Balanced class weights
model = LogisticRegression(class_weight='balanced')
# Fraud class gets 289x more weight than legitimate
```

### **Feature Engineering**
- **Standardization**: Critical for logistic regression
- **PCA Features**: V1-V5 capture most important patterns
- **Amount Scaling**: Normalized transaction amounts

### **Evaluation Strategy**
- **Business-focused metrics**: Prioritize fraud detection over precision
- **Cost-sensitive approach**: Missing fraud costs more than false alarms
- **Stratified sampling**: Maintains class distribution in train/test split

---

**Project Link**: [https://github.com/Mohana-Sree/Fraud-detection](https://github.com/Mohana-Sree/Fraud-detection)

---
