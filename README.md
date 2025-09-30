# Credit Card Fraud Detection System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-blueviolet?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-cyan?logo=numpy&logoColor=white)](https://numpy.org/)

</div>

A real-time fraud detection application using a machine learning model built with **Python**, **Flask**, and **scikit-learn**.

---

## 📜 Table of Contents
- [About The Project](#about-the-project)  
- [Key Features](#key-features)  
- [Getting Started](#getting-started)  

---

## 📖 About The Project
This project provides an interactive web interface to demonstrate a real-world machine learning application.  
It uses a **Logistic Regression** model to classify credit card transactions as either legitimate or fraudulent based on anonymized features.  
The system is designed as a practical example of deploying an ML model with Flask.

---

## ✨ Key Features
- 🤖 **Machine Learning Model**: Logistic Regression trained to detect fraudulent transactions.  
- ⚙️ **Data Scaling**: Implements `StandardScaler` to normalize input features.  
- 🌐 **Interactive Web Interface**: Flask app with HTML/CSS/JavaScript front end.  
- 🎲 **Dynamic Data Generation**: *Generate Random Sample* button for quick testing with randomized realistic data.  
- 📈 **Real-time Predictions**: Displays prediction and confidence score instantly.  

---

## 🚀 Getting Started
Follow these instructions to set up and run the project locally.

### 1. Prerequisites
- Python **3.7+**
- `pip` (Python package installer)
- `git`

### 2. Installation & Setup
Clone the repository:
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

Create and activate a virtual environment:
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the dataset:  
The model is trained on the **Credit Card Fraud Detection** dataset from Kaggle.  

👉 [Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  

Place the downloaded `creditcard.csv` file in the **root directory** of the project.

---

## ▶️ How to Run

### 1. Train the model
Run the training script:
```bash
python model_training.py
```
This creates a `model/` folder containing the trained model and scaler.

### 2. Start the web application
```bash
python app.py
```

### 3. Access the application
Open your browser and navigate to:  
```
http://127.0.0.1:5000
```

---

## 📁 Project Structure
```
credit-card-fraud-detection/
├─ app.py                  # Flask web server
├─ model_training.py       # Model training script
├─ requirements.txt        # Dependencies
├─ README.md               # Project documentation
├─ creditcard.csv          # Dataset (downloaded from Kaggle, not included in repo)
├─ model/
│  ├─ logreg_model.joblib
│  └─ scaler.joblib
├─ templates/
│  └─ index.html
├─ static/
│  ├─ css/
│  └─ js/
└─ LICENSE
```

---

```

---

## 🤝 Contributing
Contributions are welcome and appreciated!  

Steps:
1. Fork the Project  
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)  
4. Push to the Branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request  

---

