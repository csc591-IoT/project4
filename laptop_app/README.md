# Door Monitor - Laptop/Smartphone Application

## Overview
This is the **MQTT Subscriber** application that runs on a laptop or smartphone to monitor door status. It receives door open/close events from the IoT device (publisher) via AWS IoT Core and displays the current status to the user.

## Architecture
- **IoT Device**: MQTT Publisher (detects door events and publishes to AWS IoT)
- **Cloud Service**: AWS IoT Core (MQTT Broker)
- **Laptop/Smartphone**: MQTT Subscriber (this application - receives and displays door status)

## Features
- ✅ Connects to AWS IoT Core as MQTT subscriber
- ✅ Receives door state events in real-time
- ✅ Displays current door status (OPEN/CLOSED)
- ✅ Shows timestamp of latest decision
- ✅ Shows confidence score (if available)
- ✅ Minimal output - only prints when state changes

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS IoT credentials in `config.json`:
   - Ensure the certificate paths point to the correct location
   - Update endpoint if different

## Usage

Run the door monitor:
```bash
python door_monitor.py
```

## Output Example

When started:
```
======================================================================
Door Monitor - Laptop/Smartphone Subscriber
======================================================================
✓ Connected to AWS IoT
✓ Monitoring topic: door/state
======================================================================

Waiting for door events... (Press Ctrl+C to stop)
```

When a door event is received:
```
======================================================================
  Door Status: 🟢 OPEN
  Last Updated: 2025-10-28T14:30:45.123456
  Confidence: 0.95
======================================================================
```

## Requirements

- Python 3.x
- AWSIoTPythonSDK
- Valid AWS IoT certificates
- Network connection to AWS IoT Core

## Notes

- The application runs continuously until stopped with Ctrl+C
- Only displays updates when new door events are received
- Uses the same AWS IoT certificates as the IoT device
- Client ID is different from the IoT device to avoid conflicts
