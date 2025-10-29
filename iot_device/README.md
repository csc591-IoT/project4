# IoT Device

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
  "aws_endpoint": "endpoint.iot.us-east-2.amazonaws.com",
  "root_ca_path": "certs/AmazonRootCA1.pem",
  "cert_path": "certs/certificate.pem.crt",
  "private_key_path": "certs/private.pem.key",
  "client_id": "door-sensor-pi-001",
  "topic": "door/state",
  "topic_raw_data": "door/raw_data",
  "topic_results": "door/classification_results"
}
```
## Run

Start the door detection system:

```bash
python publisher_node.py
```
