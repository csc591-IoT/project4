# IoT Device - Door State Detection System

## Overview
This is the **MQTT Publisher** application that runs on a Raspberry Pi with an MPU6050 IMU sensor. It detects door open/close events using machine learning (SVM classifier) and publishes the classification results to AWS IoT Core.

## Quick Start - How to Execute

**Step 1:** Complete installation and configuration (see below)  
**Step 2:** Ensure model files exist (`30_entries_model.pkl`, `30_entries_scaler.pkl`)  
**Step 3:** Run the application:
```bash
cd iot_device
python publisher_node.py
```

The application will:
- Initialize the MPU6050 sensor
- Connect to AWS IoT Core
- Begin monitoring door movement
- Publish door state events when detected

**To stop:** Press `Ctrl+C`

## Prerequisites
- Raspberry Pi with GPIO enabled
- Python 3.8+
- MPU6050 IMU sensor connected via I2C
- An AWS IoT Thing, with:
  - Root CA certificate
  - Device certificate
  - Private key
  - Correct IoT policy attached (must allow publish to `door/state` topic)
- Trained SVM model files (`30_entries_model.pkl` and `30_entries_scaler.pkl`)

## Installation

Before installing dependencies, it is recommended to create and activate a virtual environment to keep your setup isolated.

### 1. Create a Virtual Environment
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Upgrade pip and Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Additional dependency for Raspberry Pi:**
```bash
pip install smbus2
```

## Project Structure
```
iot_device/
├── publisher_node.py          # Main door detection and AWS IoT publisher
├── train_svm_model.py          # Train SVM model from CSV data
├── iot_data_logger.py          # Data collection tool for training
├── config.json                 # AWS IoT configuration
├── requirements.txt            # Python dependencies
├── 30_entries_model.pkl        # Trained SVM model (must exist)
├── 30_entries_scaler.pkl       # Feature scaler (must exist)
├── README.md                   # This file
└── certs/                      # AWS IoT certificates (not in repo)
    ├── AmazonRootCA1.pem
    ├── certificate.pem.crt
    └── private.pem.key
```

## Configure

### 1. Set up AWS IoT Certificates

In the project root, create a directory named `certs` and place your AWS IoT device certificates in it:
- `AmazonRootCA1.pem` - AWS Root CA certificate
- `certificate.pem.crt` - Device certificate
- `private.pem.key` - Device private key

### 2. Update config.json

Update the `config.json` file with correct paths:

```json
{
  "aws_endpoint": "a2k13a18qjxz7o-ats.iot.us-east-2.amazonaws.com",
  "root_ca_path": "certs/AmazonRootCA1.pem",
  "cert_path": "certs/certificate.pem.crt",
  "private_key_path": "certs/private.pem.key",
  "client_id": "door-sensor-pi-001",
  "topic": "door/state",
  "topic_raw_data": "door/raw_data",
  "topic_results": "door/classification_results"
}
```

**To find the aws_endpoint:**
1. On the AWS console, search for "IoT Core" in the search bar
2. Select IoT Core from the services
3. In the left navbar, click on "Settings"
4. Copy the "Device data endpoint" and paste it in `aws_endpoint` here

**Tip:** Add `certs/` and `config.json` to your `.gitignore` file to keep sensitive data private.

### 3. Hardware Connection

Connect the MPU6050 to your Raspberry Pi:
- **VCC** → 3.3V or 5V
- **GND** → GND
- **SDA** → GPIO 2 (Pin 3)
- **SCL** → GPIO 3 (Pin 5)

Enable I2C on Raspberry Pi:
```bash
sudo raspi-config
# Navigate to: Interfacing Options → I2C → Enable
```

### 4. Train the Model (If needed)

If you don't have the model files, you'll need to train the SVM model:

1. **Collect training data:**
   ```bash
   python iot_data_logger.py
   ```
   Follow the prompts to collect door open/close events.

2. **Train the model:**
   ```bash
   python train_svm_model.py sessions/data_with_30_entries.csv
   ```
   This will generate `30_entries_model.pkl` and `30_entries_scaler.pkl`.

## Run

Start the door detection system:

```bash
python publisher_node.py
```

You should see:
- MPU6050 initialization confirmation
- AWS IoT connection status
- Real-time door state classifications (Open/Close)
- Messages published to AWS IoT when door events are detected

## Expected Output

```
======================================================================
Door State Detection System - Live MPU6050 Sensor + AWS IoT
======================================================================
Sampling rate: 100 Hz
Window size: 100 samples
Movement threshold: 15.0 deg/s

Press Ctrl+C to stop
======================================================================

✓ AWS IoT config loaded from config.json
✓ MPU6050 initialized (WHO_AM_I: 0x68)
✓ Connected to AWS IoT Core
✓ Starting real-time detection...

[14:30:45.123] Movement detected (w_mag=25.43 deg/s)
[14:30:45.523] ↑ OPEN  (Total: 1, Open: 1, Close: 0)
  Published to AWS IoT: door/state - Open
```

## Features

- **Real-time IMU sampling** at 100 Hz
- **Movement detection** using gyroscope magnitude threshold
- **Machine learning classification** using trained SVM model
- **AWS IoT publishing** of door state events
- **Error handling** for I2C communication failures
- **Statistics tracking** of classifications and I2C errors

## Troubleshooting

### I2C Connection Issues
- Check physical connections
- Verify I2C is enabled: `sudo i2cdetect -y 1`
- MPU6050 should appear at address `0x68`

### AWS IoT Connection Failed
- Verify certificate paths in `config.json`
- Check AWS IoT policy allows publish to `door/state`
- Ensure device certificate is active in AWS IoT

### Model Files Not Found
- Run `train_svm_model.py` to generate model files
- Ensure training data CSV exists in `sessions/` directory

### Sensor Reading Errors
- Check power supply (3.3V or 5V)
- Verify I2C pull-up resistors are present
- Try restarting the application

