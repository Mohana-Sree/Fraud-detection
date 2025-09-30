import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
import pickle
import os
import kagglehub
import numpy as np

# --- Step 1: Download and Load The Data ---
print("Downloading dataset from Kaggle...")
try:
    # Download latest version of the credit card fraud dataset
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print("Path to dataset files:", path)
    
    # Find the creditcard.csv file in the downloaded path
    dataset_file = os.path.join(path, 'creditcard.csv')
    if not os.path.exists(dataset_file):
        # Check if it's in a subdirectory
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('creditcard.csv') or file.endswith('.csv'):
                    dataset_file = os.path.join(root, file)
                    break
    
    print(f"Loading dataset from: {dataset_file}")
    data = pd.read_csv(dataset_file)
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {data.shape}")
    
except Exception as e:
    print(f"Error downloading or loading dataset: {e}")
    print("Trying to load from local 'creditcard.csv' file...")
    try:
        data = pd.read_csv('creditcard.csv')
        print("Local dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: No dataset found. Please ensure you have access to Kaggle or place 'creditcard.csv' in the project directory.")
        exit()


# --- Step 2: Prepare The Data ---
print("Preparing data for training...")
features_to_keep = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount']
target = 'Class'
print(f"Using a simplified model with {len(features_to_keep)} features.")
X = data[features_to_keep]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data preparation complete.")


# --- Step 3: Scale The Features ---
print("Scaling numerical features...")
# Initialize the StandardScaler
scaler = StandardScaler()
# Fit the scaler ONLY on the training data to avoid data leakage
X_train_scaled = scaler.fit_transform(X_train)
# Transform the test data using the scaler fitted on the training data
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.")


# --- Step 4: Train The Model ---
print("Training the Logistic Regression model...")
# Using balanced class weights to handle imbalanced data
# This automatically adjusts weights inversely proportional to class frequencies
model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
# Train the model on the SCALED training data
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# Print class weights used
class_weights = model.class_weight
if hasattr(model, 'classes_'):
    for i, class_label in enumerate(model.classes_):
        weight = len(y_train) / (len(model.classes_) * np.bincount(y_train)[i])
        print(f"Effective weight for class {class_label}: {weight:.2f}")
print()


# --- Step 5: Evaluate The Model ---
print("\n--- Model Evaluation ---")

# Check class distribution
print("Class Distribution in Training Set:")
print(f"Legitimate (0): {sum(y_train == 0):,} ({sum(y_train == 0)/len(y_train)*100:.2f}%)")
print(f"Fraudulent (1): {sum(y_train == 1):,} ({sum(y_train == 1)/len(y_train)*100:.2f}%)")
print(f"Imbalance Ratio: {sum(y_train == 0)/sum(y_train == 1):.1f}:1\n")

# Evaluate on the SCALED test data
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate more meaningful metrics for imbalanced data
tn, fp, fn, tp = cm.ravel()
print(f"\nDetailed Metrics:")
print(f"True Negatives (Legitimate correctly identified): {tn:,}")
print(f"False Positives (Legitimate flagged as fraud): {fp:,}")
print(f"False Negatives (Fraud missed): {fn:,}")
print(f"True Positives (Fraud correctly detected): {tp:,}")

print(f"\nFraud Detection Rate (Recall): {tp/(tp+fn)*100:.1f}%")
print(f"False Alarm Rate: {fp/(fp+tn)*100:.2f}%")
print(f"Precision (When we predict fraud, how often correct): {tp/(tp+fp)*100:.1f}%")

print("-------------------------\n")


# --- Step 6: Save The Model and Scaler ---
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")

# We need to save both the model and the scaler for the web app to use
# We'll save them together in a dictionary
model_and_scaler = {
    'model': model,
    'scaler': scaler
}

model_path = os.path.join(model_dir, 'fraud_detection_model.pkl')
print(f"Saving the trained model and scaler to {model_path}...")
with open(model_path, 'wb') as file:
    pickle.dump(model_and_scaler, file)

print("Model and scaler saved successfully!")
print("You can now run 'app.py' to start the web application.")

