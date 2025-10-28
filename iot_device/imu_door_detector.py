import json
import time
import sys
import numpy as np
from datetime import datetime, timezone
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import os

import RPi.GPIO as GPIO
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
TOPIC            = cfg["topic"]

# IMU Configuration
IMU_SAMPLE_RATE = 50  # Hz
WINDOW_SIZE = 100  # samples for feature extraction
MOVEMENT_THRESHOLD = 0.5  # threshold for detecting movement
ANGLE_THRESHOLD_OPEN = 45  # degrees
ANGLE_THRESHOLD_CLOSE = 20  # degrees

# GPIO Configuration
GPIO.setmode(GPIO.BOARD)
LED_PIN = 3  # Status LED
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

# ==============================
# IMU Data Collection
# ==============================
class IMUCollector:
    def __init__(self):
        self.data_buffer = []
        self.is_moving = False
        self.movement_start_time = None
        self.current_state = "unknown"
        self.last_classification_time = 0
        self.min_time_between_classifications = 2.0  # seconds
        
        # Load trained model if it exists
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load pre-trained SVM model"""
        try:
            if os.path.exists("door_classifier_model.pkl"):
                self.model = joblib.load("door_classifier_model.pkl")
                self.scaler = joblib.load("door_scaler.pkl")
                print("Model loaded successfully")
            else:
                print("No trained model found. Please train the model first.")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def simulate_imu_data(self):
        """Simulate IMU data for testing (replace with actual IMU reading)"""
        # In real implementation, replace this with actual IMU sensor reading
        # For now, simulate accelerometer and gyroscope data
        import random
        
        # Simulate door movement patterns
        if self.is_moving:
            # During movement, add more variation
            accel_x = random.uniform(-2.0, 2.0)
            accel_y = random.uniform(-2.0, 2.0)
            accel_z = random.uniform(8.0, 12.0)  # gravity + movement
            gyro_x = random.uniform(-50.0, 50.0)
            gyro_y = random.uniform(-50.0, 50.0)
            gyro_z = random.uniform(-50.0, 50.0)
        else:
            # At rest, minimal variation
            accel_x = random.uniform(-0.5, 0.5)
            accel_y = random.uniform(-0.5, 0.5)
            accel_z = random.uniform(9.5, 10.5)  # mostly gravity
            gyro_x = random.uniform(-5.0, 5.0)
            gyro_y = random.uniform(-5.0, 5.0)
            gyro_z = random.uniform(-5.0, 5.0)
        
        return {
            'timestamp': time.time(),
            'accel_x': accel_x,
            'accel_y': accel_y,
            'accel_z': accel_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z
        }
    
    def detect_movement(self, data):
        """Detect if door is moving based on IMU data"""
        # Calculate magnitude of acceleration and angular velocity
        accel_mag = np.sqrt(data['accel_x']**2 + data['accel_y']**2 + data['accel_z']**2)
        gyro_mag = np.sqrt(data['gyro_x']**2 + data['gyro_y']**2 + data['gyro_z']**2)
        
        # Simple threshold-based movement detection
        movement_detected = (abs(accel_mag - 9.81) > MOVEMENT_THRESHOLD or 
                           gyro_mag > MOVEMENT_THRESHOLD)
        
        return movement_detected
    
    def extract_features(self, data_window):
        """Extract features from IMU data window"""
        if len(data_window) < WINDOW_SIZE:
            return None
        
        # Convert to numpy arrays
        accel_x = np.array([d['accel_x'] for d in data_window])
        accel_y = np.array([d['accel_y'] for d in data_window])
        accel_z = np.array([d['accel_z'] for d in data_window])
        gyro_x = np.array([d['gyro_x'] for d in data_window])
        gyro_y = np.array([d['gyro_y'] for d in data_window])
        gyro_z = np.array([d['gyro_z'] for d in data_window])
        
        features = []
        
        # Statistical features for each axis
        for axis_data in [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]:
            features.extend([
                np.mean(axis_data),           # Mean
                np.std(axis_data),            # Standard deviation
                np.max(axis_data),            # Maximum
                np.min(axis_data),            # Minimum
                np.var(axis_data),            # Variance
                np.median(axis_data),         # Median
            ])
        
        # Additional features
        # Magnitude of acceleration
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        features.extend([
            np.mean(accel_mag),
            np.std(accel_mag),
            np.max(accel_mag) - np.min(accel_mag)  # Range
        ])
        
        # Magnitude of angular velocity
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        features.extend([
            np.mean(gyro_mag),
            np.std(gyro_mag),
            np.max(gyro_mag) - np.min(gyro_mag)  # Range
        ])
        
        # Energy features
        features.extend([
            np.sum(accel_x**2),  # Energy in X-axis acceleration
            np.sum(accel_y**2),  # Energy in Y-axis acceleration
            np.sum(accel_z**2),  # Energy in Z-axis acceleration
            np.sum(gyro_x**2),   # Energy in X-axis gyro
            np.sum(gyro_y**2),   # Energy in Y-axis gyro
            np.sum(gyro_z**2),   # Energy in Z-axis gyro
        ])
        
        return np.array(features)
    
    def classify_door_state(self, data_window):
        """Classify door state using trained SVM model"""
        if self.model is None or self.scaler is None:
            return "unknown"
        
        features = self.extract_features(data_window)
        if features is None:
            return "unknown"
        
        try:
            # Normalize features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            
            # Map prediction to door state
            if prediction == 0:
                return "closed"
            elif prediction == 1:
                return "open"
            else:
                return "unknown"
        except Exception as e:
            print(f"Classification error: {e}")
            return "unknown"
    
    def process_data(self):
        """Main data processing loop"""
        while True:
            # Get IMU data
            imu_data = self.simulate_imu_data()
            
            # Add to buffer
            self.data_buffer.append(imu_data)
            if len(self.data_buffer) > WINDOW_SIZE * 2:
                self.data_buffer = self.data_buffer[-WINDOW_SIZE * 2:]
            
            # Detect movement
            movement_detected = self.detect_movement(imu_data)
            
            if movement_detected and not self.is_moving:
                # Movement started
                self.is_moving = True
                self.movement_start_time = time.time()
                print("Movement detected - starting classification window")
                
            elif not movement_detected and self.is_moving:
                # Movement ended
                self.is_moving = False
                movement_duration = time.time() - self.movement_start_time
                
                print(f"Movement ended after {movement_duration:.2f} seconds")
                
                # Classify door state if we have enough data
                if len(self.data_buffer) >= WINDOW_SIZE:
                    current_time = time.time()
                    if current_time - self.last_classification_time > self.min_time_between_classifications:
                        door_state = self.classify_door_state(self.data_buffer[-WINDOW_SIZE:])
                        self.current_state = door_state
                        self.last_classification_time = current_time
                        
                        print(f"Door state classified as: {door_state}")
                        
                        # Publish result
                        return door_state
            
            time.sleep(1.0 / IMU_SAMPLE_RATE)
        
        return None

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

def publish_door_state(door_state):
    """Publish door state to AWS IoT"""
    payload = {
        "device_id": CLIENT_ID,
        "timestamp": now_iso(),
        "door_state": door_state,
        "confidence": 0.85  # You can implement confidence scoring
    }
    
    try:
        mqtt.publish(TOPIC, json.dumps(payload), 0)
        print(f"Published door state: {door_state}")
        
        # Blink LED to indicate successful publish
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(LED_PIN, GPIO.LOW)
        
    except Exception as e:
        print(f"Error publishing: {e}")

def cleanup_and_exit():
    try:
        GPIO.cleanup()
    except Exception:
        pass
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
    print("Starting Door State Detection System...")
    print("Connecting to AWS IoT...")
    
    try:
        mqtt.connect()
        print("Connected to AWS IoT.")
        
        # Initialize IMU collector
        imu_collector = IMUCollector()
        
        print("Starting IMU data collection and classification...")
        
        while True:
            door_state = imu_collector.process_data()
            if door_state and door_state != "unknown":
                publish_door_state(door_state)
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            
    except KeyboardInterrupt:
        print("Exit requested by user.")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        cleanup_and_exit()

if __name__ == "__main__":
    main()
