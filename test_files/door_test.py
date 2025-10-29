#!/usr/bin/env python3
# door_detector_live.py
"""
Real-time door state detection using MPU6050 IMU and trained SVM model
Reads sensor data and classifies as: Open or Close
"""

import os
import sys
import time
import numpy as np
import joblib
from math import sqrt
from datetime import datetime
from smbus2 import SMBus
from collections import deque

# ========= Configuration =========
SAMPLE_HZ = 100         # Must match training data
WINDOW_SIZE = 100       # Number of samples for classification
MOVEMENT_THRESHOLD = 15.0  # deg/s - LOWERED threshold to detect gentler movements

# I2C / MPU6050 constants
I2C_BUS      = 1
MPU_ADDR     = 0x68
PWR_MGMT_1   = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H  = 0x43
ACCEL_SCALE  = 16384.0   # LSB/g, ±2g
GYRO_SCALE   = 131.0     # LSB/(deg/s), ±250 dps

# Model paths
MODEL_PATH = "30_entries_model.pkl"
SCALER_PATH = "30_entries_scaler.pkl"

# Error handling
MAX_I2C_RETRIES = 3
I2C_RETRY_DELAY = 0.1

# ========= MPU6050 Functions =========
def _read_word(bus, reg, retries=MAX_I2C_RETRIES):
    """Read 16-bit signed value from MPU6050 with retry logic"""
    for attempt in range(retries):
        try:
            hi = bus.read_byte_data(MPU_ADDR, reg)
            lo = bus.read_byte_data(MPU_ADDR, reg+1)
            v  = (hi << 8) | lo
            return v - 65536 if v >= 32768 else v
        except OSError as e:
            if attempt < retries - 1:
                print(f"⚠️  I2C read error (attempt {attempt+1}/{retries}): {e}")
                time.sleep(I2C_RETRY_DELAY)
                # Try to reinitialize
                try:
                    bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)
                    time.sleep(0.05)
                except:
                    pass
            else:
                raise

def init_imu(bus):
    """Initialize MPU6050 - wake it up"""
    try:
        bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)
        # Verify it's working by reading WHO_AM_I register
        who_am_i = bus.read_byte_data(MPU_ADDR, 0x75)
        if who_am_i == 0x68:
            print(f"✓ MPU6050 initialized (WHO_AM_I: 0x{who_am_i:02X})")
        else:
            print(f"⚠️  Unexpected WHO_AM_I: 0x{who_am_i:02X} (expected 0x68)")
    except OSError as e:
        print(f"✗ Failed to initialize MPU6050: {e}")
        print("  Check connections:")
        print("    - VCC → 3.3V or 5V")
        print("    - GND → GND")
        print("    - SDA → GPIO 2 (Pin 3)")
        print("    - SCL → GPIO 3 (Pin 5)")
        raise

def read_imu_data(bus):
    """
    Read IMU data from MPU6050 with error handling
    Returns dict with ax, ay, az, gx, gy, gz, a_mag, w_mag or None on error
    """
    try:
        # Read accelerometer (in g's)
        ax = _read_word(bus, ACCEL_XOUT_H)   / ACCEL_SCALE
        ay = _read_word(bus, ACCEL_XOUT_H+2) / ACCEL_SCALE
        az = _read_word(bus, ACCEL_XOUT_H+4) / ACCEL_SCALE
        
        # Read gyroscope (in deg/s)
        gx = _read_word(bus, GYRO_XOUT_H)    / GYRO_SCALE
        gy = _read_word(bus, GYRO_XOUT_H+2)  / GYRO_SCALE
        gz = _read_word(bus, GYRO_XOUT_H+4)  / GYRO_SCALE
        
        # Calculate magnitudes
        a_mag = sqrt(ax*ax + ay*ay + az*az)
        w_mag = sqrt(gx*gx + gy*gy + gz*gz)
        
        return {
            'ax': ax,
            'ay': ay,
            'az': az,
            'gx': gx,
            'gy': gy,
            'gz': gz,
            'a_mag': a_mag,
            'w_mag': w_mag,
            'timestamp': time.time()
        }
    except OSError as e:
        print(f"✗ I2C Error reading sensor: {e}")
        return None

# ========= Door State Detector =========
class DoorStateDetector:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.data_buffer = deque(maxlen=WINDOW_SIZE * 2)
        self.is_moving = False
        self.movement_start_time = None
        self.current_state = "open"
        self.last_classification_time = 0
        self.min_time_between_classifications = 0.5  # seconds
        
        # Class names (must match training)
        self.class_names = ['Open', 'Close']
        
        # Load trained model
        self.model = None
        self.scaler = None
        self.load_model(model_path, scaler_path)
        
        # Statistics
        self.total_classifications = 0
        self.state_counts = {'Open': 0, 'Close': 0}
        self.i2c_errors = 0
    
    def load_model(self, model_path, scaler_path):
        """Load pre-trained SVM model and scaler"""
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Model loaded: {model_path}")
                print(f"✓ Scaler loaded: {scaler_path}")
            else:
                print(f"✗ Model files not found!")
                print(f"  Looking for: {model_path} and {scaler_path}")
                print(f"  Train the model first: python train_svm_2class.py")
                sys.exit(1)
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
    
    def detect_movement(self, data):
        """Detect if door is moving based on gyroscope magnitude"""
        return data['w_mag'] > MOVEMENT_THRESHOLD
    
    def extract_features(self, data_window):
        """
        Extract features from IMU data window
        Features match training data: ax, ay, az, gx, gy, gz, a_mag, w_mag
        """
        if len(data_window) < WINDOW_SIZE:
            return None
        
        # Convert to numpy arrays
        features = []
        for sample in data_window:
            features.append([
                sample['ax'],
                sample['ay'],
                sample['az'],
                sample['gx'],
                sample['gy'],
                sample['gz'],
                sample['a_mag'],
                sample['w_mag']
            ])
        
        # Calculate mean of features across the window
        features_array = np.array(features)
        mean_features = np.mean(features_array, axis=0)
        
        return mean_features
    
    def classify_door_state(self, data_window):
        """Classify door state using trained SVM model"""
        if self.model is None or self.scaler is None:
            return None, None
        
        features = self.extract_features(data_window)
        if features is None:
            return None, None
        
        try:
            # Normalize features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            class_name = self.class_names[prediction]
            
            return prediction, class_name
            
        except Exception as e:
            print(f"✗ Classification error: {e}")
            return None, None
    
    def print_status(self, state):
        """Print formatted status to terminal with colors"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Terminal color codes
        RESET = '\033[0m'
        BOLD = '\033[1m'
        
        if state == "Open":
            COLOR = '\033[92m'  # Green
            SYMBOL = "↑"
        elif state == "Close":
            COLOR = '\033[91m'  # Red
            SYMBOL = "↓"
        else:
            COLOR = '\033[93m'  # Yellow
            SYMBOL = "?"
        
        # Update statistics
        self.total_classifications += 1
        if state in self.state_counts:
            self.state_counts[state] += 1
        
        status_str = f"[{timestamp}] {COLOR}{BOLD}{SYMBOL} {state.upper()}{RESET}"
        status_str += f"  (Total: {self.total_classifications}, "
        status_str += f"Open: {self.state_counts['Open']}, "
        status_str += f"Close: {self.state_counts['Close']})"
        
        print(status_str)
    
    def process_sample(self, imu_data):
        """
        Process a single IMU sample
        Returns: (classified, state) tuple
          classified: True if a new classification was made
          state: The classified state (Open/Close) or None
        """
        if imu_data is None:
            self.i2c_errors += 1
            return False, None
            
        # Add to buffer
        self.data_buffer.append(imu_data)
        
        # Detect movement
        movement_detected = self.detect_movement(imu_data)
        
        if movement_detected and not self.is_moving:
            # Movement started
            self.is_moving = True
            self.movement_start_time = time.time()
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] 🔄 Movement detected (w_mag={imu_data['w_mag']:.2f} deg/s)")
            return False, None
            
        elif not movement_detected and self.is_moving:
            # Movement ended
            self.is_moving = False
            movement_duration = time.time() - self.movement_start_time
            
            # Classify door state if we have enough data
            if len(self.data_buffer) >= WINDOW_SIZE:
                current_time = time.time()
                if current_time - self.last_classification_time > self.min_time_between_classifications:
                    # Get the data window for classification
                    data_window = list(self.data_buffer)[-WINDOW_SIZE:]
                    prediction, class_name = self.classify_door_state(data_window)
                    
                    if class_name:
                        self.current_state = class_name
                        self.last_classification_time = current_time
                        self.print_status(class_name)
                        return True, class_name
        
        return False, None

def main():
    print("\n" + "="*70)
    print("Door State Detection System - Live MPU6050 Sensor")
    print("="*70)
    print(f"Sampling rate: {SAMPLE_HZ} Hz")
    print(f"Window size: {WINDOW_SIZE} samples")
    print(f"Movement threshold: {MOVEMENT_THRESHOLD} deg/s")
    print("\nLegend:")
    print("  🟢 ↑ Open  - Door is opening")
    print("  🔴 ↓ Close - Door is closing")
    print("\nPress Ctrl+C to stop")
    print("="*70 + "\n")
    
    # Initialize detector
    detector = DoorStateDetector()
    
    # Initialize I2C and IMU
    bus = SMBus(I2C_BUS)
    
    try:
        init_imu(bus)
    except Exception as e:
        print(f"\n✗ Cannot initialize sensor. Exiting.")
        bus.close()
        sys.exit(1)
    
    print("✓ Starting real-time detection...\n")
    
    # Timing for precise sampling
    period = 1.0 / SAMPLE_HZ
    next_sample_time = time.monotonic()
    
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 10
    
    try:
        while True:
            # Read IMU data
            imu_data = read_imu_data(bus)
            
            if imu_data is None:
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"\n✗ Too many consecutive I2C errors ({consecutive_errors})")
                    print("  Possible causes:")
                    print("    1. Loose wire connection")
                    print("    2. Power supply issue")
                    print("    3. Sensor failure")
                    print("  Please check hardware and restart")
                    break
                time.sleep(0.01)
                continue
            else:
                consecutive_errors = 0
            
            # Process the sample
            detector.process_sample(imu_data)
            
            # Precise timing to maintain sample rate
            next_sample_time += period
            sleep_time = next_sample_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Fell behind, reset timing
                next_sample_time = time.monotonic()
    
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("Detection stopped by user")
        print("="*70)
        print(f"Total classifications: {detector.total_classifications}")
        print(f"  Open:  {detector.state_counts['Open']}")
        print(f"  Close: {detector.state_counts['Close']}")
        print(f"I2C Errors: {detector.i2c_errors}")
        print("="*70 + "\n")
    
    finally:
        bus.close()
        print("✓ I2C bus closed")

if __name__ == "__main__":
    # Check if running on Raspberry Pi
    try:
        from smbus2 import SMBus
    except ImportError:
        print("✗ Error: smbus2 not installed")
        print("  Install with: pip install smbus2")
        sys.exit(1)
    
    main()