#!/usr/bin/env python3
"""
Deployment script - simplified for Railway
"""
import os

def check_model_exists():
    """Check if model exists"""
    model_path = os.path.join('model', 'fraud_detection_model.pkl')
    
    if os.path.exists(model_path):
        print("✅ Model found! Ready for deployment.")
        return True
    else:
        print("❌ Model not found. Please ensure model is included in repository.")
        print("💡 Run 'python model_training.py' locally and commit the model.")
        return False

if __name__ == "__main__":
    success = check_model_exists()
    if success:
        print("🚀 Deployment setup complete!")
    else:
        print("⚠️  Warning: Model missing, but continuing deployment...")