#!/usr/bin/env python3
"""
Deployment script to ensure model is trained before starting the app
"""
import os
import subprocess
import sys

def ensure_model_exists():
    """Check if model exists, if not train it"""
    model_path = os.path.join('model', 'fraud_detection_model.pkl')
    
    if not os.path.exists(model_path):
        print("🤖 Model not found. Training model for deployment...")
        print("This may take a few minutes...")
        
        try:
            # Run model training
            result = subprocess.run([sys.executable, 'model_training.py'], 
                                  capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ Model training completed successfully!")
                return True
            else:
                print(f"❌ Error training model: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Model training timeout (>10 minutes)")
            return False
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False
    else:
        print("✅ Model already exists!")
        return True

if __name__ == "__main__":
    success = ensure_model_exists()
    if not success:
        print("❌ Deployment failed - could not train model")
        sys.exit(1)
    else:
        print("🚀 Ready for deployment!")