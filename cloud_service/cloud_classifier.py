import json
import time
import sys
import numpy as np
from datetime import datetime, timezone
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import os
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# ==============================
# Configuration
# ==============================
with open("config.json") as f:
    cfg = json.load(f)

AWS_ENDPOINT     = cfg["aws_endpoint"]
ROOT_CA_PATH     = cfg["root_ca_path"]
CERT_PATH        = cfg["cert_path"]
PRIVATE_KEY_PATH = cfg["private_key_path"]
CLIENT_ID        = cfg["client_id"]
TOPIC_RAW_DATA   = cfg["topic_raw_data"]   # Topic for receiving raw IMU data
TOPIC_RESULTS    = cfg["topic_results"]    # Topic for sending classification results

# ==============================
# Cloud Classification Service
# ==============================

class CloudDoorClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load pre-trained SVM model"""
        try:
            if os.path.exists("door_classifier_model.pkl"):
                self.model = joblib.load("door_classifier_model.pkl")
                self.scaler = joblib.load("door_scaler.pkl")
                print("Cloud model loaded successfully")
            else:
                print("No trained model found. Please train the model first.")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def classify_door_state(self, features):
        """Classify door state using trained SVM model"""
        if self.model is None or self.scaler is None:
            return "unknown", 0.0
        
        try:
            # Normalize features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            
            # Get prediction probability (if available)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 0.85  # Default confidence
            
            # Map prediction to door state
            if prediction == 0:
                return "closed", confidence
            elif prediction == 1:
                return "open", confidence
            else:
                return "unknown", 0.0
                
        except Exception as e:
            print(f"Classification error: {e}")
            return "unknown", 0.0

# ==============================
# AWS IoT MQTT setup
# ==============================
mqtt = AWSIoTMQTTClient(CLIENT_ID)
mqtt.configureEndpoint(AWS_ENDPOINT, 8883)
mqtt.configureCredentials(ROOT_CA_PATH, PRIVATE_KEY_PATH, CERT_PATH)

# ==============================
# Helpers
# ==============================
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def on_raw_data_received(client, userdata, message):
    """Handle incoming raw IMU data for classification"""
    try:
        payload = json.loads(message.payload.decode('utf-8'))
        
        device_id = payload.get('device_id')
        features = np.array(payload.get('features', []))
        request_id = payload.get('request_id')
        timestamp = payload.get('timestamp')
        
        print(f"Received classification request from {device_id}")
        
        # Classify door state
        door_state, confidence = classifier.classify_door_state(features)
        
        # Send result back
        result_payload = {
            "device_id": device_id,
            "timestamp": now_iso(),
            "door_state": door_state,
            "confidence": confidence,
            "request_id": request_id,
            "processed_at": now_iso()
        }
        
        mqtt.publish(TOPIC_RESULTS, json.dumps(result_payload), 0)
        print(f"Sent classification result: {door_state} (confidence: {confidence:.2f})")
        
    except Exception as e:
        print(f"Error processing raw data: {e}")

def cleanup_and_exit():
    try:
        mqtt.disconnect()
    except Exception:
        pass
    print("Clean exit.")
    sys.exit(0)

# ==============================
# Main
# ==============================
def main():
    global classifier
    
    print("Starting Cloud Door Classification Service...")
    print("Connecting to AWS IoT...")
    
    try:
        mqtt.connect()
        print("Connected to AWS IoT.")
        
        # Initialize classifier
        classifier = CloudDoorClassifier()
        
        # Subscribe to raw data topic
        mqtt.subscribe(TOPIC_RAW_DATA, 1, on_raw_data_received)
        print(f"Subscribed to {TOPIC_RAW_DATA}")
        print("Waiting for classification requests...")
        
        # Keep the service running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Exit requested by user.")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        cleanup_and_exit()

if __name__ == "__main__":
    main()
