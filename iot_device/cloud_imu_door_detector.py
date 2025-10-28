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
TOPIC_RAW_DATA   = cfg["topic_raw_data"]  # Topic for sending raw IMU data
TOPIC_RESULTS    = cfg["topic_results"]   # Topic for receiving classification results

# IMU Configuration
IMU_SAMPLE_RATE = 50  # Hz
WINDOW_SIZE = 100  # samples for feature extraction
MOVEMENT_THRESHOLD = 0.5  # threshold for detecting movement

# GPIO Configuration
GPIO.setmode(GPIO.BOARD)
LED_PIN = 3  # Status LED
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

# ==============================
# IMU Data Collection (Cloud-based)
# ==============================
class CloudIMUCollector:
    def __init__(self):
        self.data_buffer = []
        self.is_moving = False
        self.movement_start_time = None
        self.current_state = "unknown"
        self.last_classification_time = 0
        self.min_time_between_classifications = 2.0  # seconds
        self.pending_classification = False
    
    def simulate_imu_data(self):
        """Simulate IMU data for testing (replace with actual IMU reading)"""
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
                
                # Send data for cloud classification if we have enough data
                if len(self.data_buffer) >= WINDOW_SIZE:
                    current_time = time.time()
                    if current_time - self.last_classification_time > self.min_time_between_classifications:
                        features = self.extract_features(self.data_buffer[-WINDOW_SIZE:])
                        if features is not None:
                            self.pending_classification = True
                            self.last_classification_time = current_time
                            
                            print("Sending data to cloud for classification...")
                            
                            # Return features for cloud processing
                            return features
            
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

def publish_raw_data(features):
    """Publish raw IMU features to cloud for classification"""
    payload = {
        "device_id": CLIENT_ID,
        "timestamp": now_iso(),
        "features": features.tolist(),
        "request_id": f"{CLIENT_ID}_{int(time.time())}"
    }
    
    try:
        mqtt.publish(TOPIC_RAW_DATA, json.dumps(payload), 0)
        print(f"Published raw data for classification")
        
        # Blink LED to indicate successful publish
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(LED_PIN, GPIO.LOW)
        
    except Exception as e:
        print(f"Error publishing raw data: {e}")

def on_classification_result(client, userdata, message):
    """Handle classification results from cloud"""
    try:
        payload = json.loads(message.payload.decode('utf-8'))
        
        if payload.get('device_id') == CLIENT_ID:
            door_state = payload.get('door_state', 'unknown')
            confidence = payload.get('confidence', 0.0)
            request_id = payload.get('request_id')
            
            print(f"Received classification result: {door_state} (confidence: {confidence:.2f})")
            
            # Publish final result to door state topic
            publish_door_state(door_state, confidence)
            
    except Exception as e:
        print(f"Error processing classification result: {e}")

def publish_door_state(door_state, confidence):
    """Publish final door state result"""
    payload = {
        "device_id": CLIENT_ID,
        "timestamp": now_iso(),
        "door_state": door_state,
        "confidence": confidence,
        "classification_method": "cloud"
    }
    
    try:
        mqtt.publish("door/state", json.dumps(payload), 0)
        print(f"Published door state: {door_state}")
        
    except Exception as e:
        print(f"Error publishing door state: {e}")

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
    print("Starting Cloud-based Door State Detection System...")
    print("Connecting to AWS IoT...")
    
    try:
        mqtt.connect()
        print("Connected to AWS IoT.")
        
        # Subscribe to classification results
        mqtt.subscribe(TOPIC_RESULTS, 1, on_classification_result)
        print(f"Subscribed to {TOPIC_RESULTS}")
        
        # Initialize IMU collector
        imu_collector = CloudIMUCollector()
        
        print("Starting IMU data collection and cloud classification...")
        
        while True:
            features = imu_collector.process_data()
            if features is not None:
                publish_raw_data(features)
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            
    except KeyboardInterrupt:
        print("Exit requested by user.")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        cleanup_and_exit()

if __name__ == "__main__":
    main()
