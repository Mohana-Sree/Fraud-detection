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

## 📋 Table of Contents
- [About The Project](#-about-the-project)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Dataset Information](#-dataset-information)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 About The Project

This project implements a **machine learning-based fraud detection system** for credit card transactions. The system uses advanced data science techniques to identify potentially fraudulent transactions in real-time, helping financial institutions minimize losses and protect customers.

### 🚀 **Live Demo**
Experience the fraud detection system: **[Try Live Demo](https://your-deployed-app.herokuapp.com)** *(Deploy and update this link)*

### 🎥 **Demo Video**
![Fraud Detection Demo](https://via.placeholder.com/800x400/2196F3/FFFFFF?text=Fraud+Detection+System+Demo)

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

## 📁 Project Structure

```
Fraud-detection/
│
├── 📊 model/                    # Trained model files
│   └── fraud_detection_model.pkl
│
├── 🎨 templates/               # HTML templates
│   └── index.html
│
├── 🎨 styles/                  # CSS styling
│   └── styles.css
│
├── 🐍 app.py                   # Flask web application
├── 🤖 model_training.py        # ML model training script
├── 📋 requirements.txt         # Python dependencies
├── 📖 README.md               # Project documentation
└── 🚫 .gitignore              # Git ignore rules
```

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

## � Future Enhancements

### **Technical Improvements**
- [ ] **Ensemble Methods**: Combine multiple algorithms
- [ ] **Deep Learning**: Neural networks for complex patterns
- [ ] **Feature Engineering**: Time-based and velocity features
- [ ] **Online Learning**: Continuous model updates

### **Business Features**
- [ ] **Risk Scoring**: Multi-level risk categories
- [ ] **Explanation AI**: Feature importance for decisions
- [ ] **A/B Testing**: Model comparison framework
- [ ] **Real-time Monitoring**: Performance dashboards

### **Deployment**
- [ ] **Cloud Deployment**: AWS/GCP production deployment
- [ ] **API Authentication**: Secure API access
- [ ] **Database Integration**: Transaction logging
- [ ] **Monitoring**: Performance and drift detection

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### **How to Contribute**
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### **Areas for Contribution**
- Model improvements and new algorithms
- Frontend enhancements and UI/UX
- Additional features and functionality
- Documentation and examples
- Testing and quality assurance

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Mohana Sree**
- 📧 Email: [your.email@example.com](mailto:your.email@example.com)
- 💼 LinkedIn: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)
- 🌐 Portfolio: [your-portfolio.com](https://your-portfolio.com)
- 💻 GitHub: [@Mohana-Sree](https://github.com/Mohana-Sree)

**Project Link**: [https://github.com/Mohana-Sree/Fraud-detection](https://github.com/Mohana-Sree/Fraud-detection)

---

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the credit card fraud dataset
- [scikit-learn](https://scikit-learn.org/) for the machine learning framework
- [Flask](https://flask.palletsprojects.com/) for the web application framework
- The open-source community for inspiration and resources

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Made with ❤️ for fraud prevention and financial security**

</div>  

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

