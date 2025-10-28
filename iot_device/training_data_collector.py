import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
import os
import random

# ==============================
# Training Data Collection Script
# ==============================

class TrainingDataCollector:
    def __init__(self):
        self.training_data = []
        self.current_session = []
        self.is_collecting = False
        self.current_label = None
        self.session_count = 0
        
    def simulate_imu_data(self, door_state="closed"):
        """Simulate IMU data based on door state"""
        # Simulate different patterns for open vs closed states
        
        if door_state == "open":
            # Door is open - more variation due to air movement, vibrations
            accel_x = random.uniform(-1.5, 1.5)
            accel_y = random.uniform(-1.5, 1.5)
            accel_z = random.uniform(8.5, 11.5)  # gravity + movement
            gyro_x = random.uniform(-30.0, 30.0)
            gyro_y = random.uniform(-30.0, 30.0)
            gyro_z = random.uniform(-30.0, 30.0)
        else:  # closed
            # Door is closed - more stable readings
            accel_x = random.uniform(-0.3, 0.3)
            accel_y = random.uniform(-0.3, 0.3)
            accel_z = random.uniform(9.7, 10.3)  # mostly gravity
            gyro_x = random.uniform(-2.0, 2.0)
            gyro_y = random.uniform(-2.0, 2.0)
            gyro_z = random.uniform(-2.0, 2.0)
        
        return {
            'timestamp': time.time(),
            'accel_x': accel_x,
            'accel_y': accel_y,
            'accel_z': accel_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z
        }
    
    def simulate_door_movement(self, start_state, end_state, duration=2.0):
        """Simulate door movement from one state to another"""
        movement_data = []
        samples = int(duration * 50)  # 50 Hz sampling rate
        
        for i in range(samples):
            # Interpolate between states during movement
            progress = i / samples
            
            if start_state == "closed" and end_state == "open":
                # Opening movement - increasing variation
                accel_x = random.uniform(-0.3 + progress * 1.2, 0.3 + progress * 1.2)
                accel_y = random.uniform(-0.3 + progress * 1.2, 0.3 + progress * 1.2)
                accel_z = random.uniform(9.7 + progress * 1.8, 10.3 + progress * 1.2)
                gyro_x = random.uniform(-2.0 + progress * 28.0, 2.0 + progress * 28.0)
                gyro_y = random.uniform(-2.0 + progress * 28.0, 2.0 + progress * 28.0)
                gyro_z = random.uniform(-2.0 + progress * 28.0, 2.0 + progress * 28.0)
            else:  # closing movement
                # Closing movement - decreasing variation
                accel_x = random.uniform(-1.5 - progress * 1.2, 1.5 - progress * 1.2)
                accel_y = random.uniform(-1.5 - progress * 1.2, 1.5 - progress * 1.2)
                accel_z = random.uniform(8.5 + progress * 1.8, 11.5 - progress * 1.2)
                gyro_x = random.uniform(-30.0 + progress * 28.0, 30.0 - progress * 28.0)
                gyro_y = random.uniform(-30.0 + progress * 28.0, 30.0 - progress * 28.0)
                gyro_z = random.uniform(-30.0 + progress * 28.0, 30.0 - progress * 28.0)
            
            movement_data.append({
                'timestamp': time.time() + i * 0.02,
                'accel_x': accel_x,
                'accel_y': accel_y,
                'accel_z': accel_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z
            })
        
        return movement_data
    
    def extract_features(self, data_window):
        """Extract features from IMU data window"""
        if len(data_window) < 50:  # Minimum window size
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
                np.percentile(axis_data, 25), # 25th percentile
                np.percentile(axis_data, 75), # 75th percentile
            ])
        
        # Additional features
        # Magnitude of acceleration
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        features.extend([
            np.mean(accel_mag),
            np.std(accel_mag),
            np.max(accel_mag) - np.min(accel_mag),  # Range
            np.var(accel_mag)
        ])
        
        # Magnitude of angular velocity
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        features.extend([
            np.mean(gyro_mag),
            np.std(gyro_mag),
            np.max(gyro_mag) - np.min(gyro_mag),  # Range
            np.var(gyro_mag)
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
        
        # Frequency domain features (simplified)
        for axis_data in [accel_x, accel_y, accel_z]:
            fft = np.fft.fft(axis_data)
            features.extend([
                np.mean(np.abs(fft)),  # Mean frequency magnitude
                np.std(np.abs(fft)),   # Std frequency magnitude
            ])
        
        return np.array(features)
    
    def collect_training_session(self, door_state, num_samples=100):
        """Collect training data for a specific door state"""
        print(f"Collecting {num_samples} samples for door state: {door_state}")
        
        session_data = []
        
        # Collect static state data
        for i in range(num_samples // 2):
            imu_data = self.simulate_imu_data(door_state)
            session_data.append(imu_data)
            time.sleep(0.02)  # 50 Hz
        
        # Collect movement data
        if door_state == "open":
            movement_data = self.simulate_door_movement("closed", "open")
        else:
            movement_data = self.simulate_door_movement("open", "closed")
        
        session_data.extend(movement_data)
        
        # Extract features and label
        features = self.extract_features(session_data)
        if features is not None:
            label = 1 if door_state == "open" else 0
            self.training_data.append({
                'features': features,
                'label': label,
                'door_state': door_state,
                'timestamp': datetime.now().isoformat()
            })
            print(f"Added training sample with {len(features)} features, label: {label}")
        
        return len(session_data)
    
    def save_training_data(self, filename="training_data.json"):
        """Save training data to file"""
        # Convert to serializable format
        serializable_data = []
        for sample in self.training_data:
            serializable_data.append({
                'features': sample['features'].tolist(),
                'label': sample['label'],
                'door_state': sample['door_state'],
                'timestamp': sample['timestamp']
            })
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Saved {len(self.training_data)} training samples to {filename}")
    
    def load_training_data(self, filename="training_data.json"):
        """Load training data from file"""
        if not os.path.exists(filename):
            print(f"Training data file {filename} not found")
            return
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.training_data = []
        for sample in data:
            self.training_data.append({
                'features': np.array(sample['features']),
                'label': sample['label'],
                'door_state': sample['door_state'],
                'timestamp': sample['timestamp']
            })
        
        print(f"Loaded {len(self.training_data)} training samples from {filename}")

def main():
    print("Door State Training Data Collection")
    print("===================================")
    
    collector = TrainingDataCollector()
    
    # Check if we have existing training data
    if os.path.exists("training_data.json"):
        response = input("Existing training data found. Load it? (y/n): ")
        if response.lower() == 'y':
            collector.load_training_data()
    
    print(f"Current training samples: {len(collector.training_data)}")
    
    while True:
        print("\nOptions:")
        print("1. Collect 'closed' door samples")
        print("2. Collect 'open' door samples")
        print("3. Save training data")
        print("4. Load training data")
        print("5. Show statistics")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            num_samples = int(input("Number of samples to collect (default 100): ") or "100")
            collector.collect_training_session("closed", num_samples)
        
        elif choice == '2':
            num_samples = int(input("Number of samples to collect (default 100): ") or "100")
            collector.collect_training_session("open", num_samples)
        
        elif choice == '3':
            collector.save_training_data()
        
        elif choice == '4':
            collector.load_training_data()
        
        elif choice == '5':
            if collector.training_data:
                closed_count = sum(1 for sample in collector.training_data if sample['label'] == 0)
                open_count = sum(1 for sample in collector.training_data if sample['label'] == 1)
                print(f"Total samples: {len(collector.training_data)}")
                print(f"Closed door samples: {closed_count}")
                print(f"Open door samples: {open_count}")
            else:
                print("No training data available")
        
        elif choice == '6':
            if collector.training_data:
                save = input("Save training data before exiting? (y/n): ")
                if save.lower() == 'y':
                    collector.save_training_data()
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
