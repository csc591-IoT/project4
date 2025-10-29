# System Architecture - Door Monitoring System

## Architecture Overview

This system follows the MQTT publisher-subscriber pattern with AWS IoT Core as the broker:

```
┌─────────────────┐         ┌──────────────┐         ┌────────────────────┐
│   IoT Device    │         │   AWS IoT    │         │ Laptop/Smartphone  │
│   (Publisher)   │────────▶│     Core     │────────▶│   (Subscriber)     │
│                 │         │   (Broker)   │         │                    │
│ - MPU6050       │         │              │         │ - door_monitor.py  │
│ - SVM Model     │         │              │         │ - Displays status  │
│ - Publishes to  │         │              │         │ - Shows timestamp  │
│   door/state    │         │              │         │                    │
└─────────────────┘         └──────────────┘         └────────────────────┘
```

## Component Roles

### 1. IoT Device (Publisher)
**File**: `iot_device/publisher_node.py`
- Reads IMU sensor data from MPU6050
- Classifies door state using trained SVM model
- **Publishes** door open/close decisions to AWS IoT Core
- Topic: `door/state`
- Message format:
  ```json
  {
    "state": "open",
    "timestamp": "2025-10-28T14:30:45.123456",
    "confidence": 0.95
  }
  ```

### 2. AWS IoT Core (Broker)
- MQTT message broker in the cloud
- Routes messages from publisher to subscribers
- Handles authentication and authorization
- Topic: `door/state`

### 3. Laptop/Smartphone (Subscriber)
**File**: `laptop_app/door_monitor.py`
- **Subscribes** to `door/state` topic
- Receives door open/close events
- **Displays current status** to the user
- **Shows timestamp** of latest decision
- Updates display **only when new events arrive**

## Message Flow

1. **Detection**: IoT device detects door movement via IMU sensor
2. **Classification**: SVM model classifies the event as "open" or "close"
3. **Publish**: IoT device publishes decision to AWS IoT Core
4. **Route**: AWS IoT Core routes message to all subscribers
5. **Receive**: Laptop/smartphone receives the message
6. **Display**: Current status and timestamp are shown to user

## Key Features (Requirement Compliance)

✅ **IoT device is MQTT publisher**
- Publishes door state decisions to AWS IoT Core

✅ **Laptop/smartphone is MQTT subscriber**
- Subscribes to door state topic

✅ **Shows current door status**
- Displays 🟢 OPEN or 🔴 CLOSED

✅ **Shows timestamp of latest decision**
- Displays when the last event was detected

✅ **Updates only on new events**
- Display updates only when new door open/close events are received
- No continuous polling or unnecessary output

✅ **Minimal print statements**
- Only prints when:
  - Application starts
  - New door event is received
  - Error occurs
  - Application stops

## Running the System

### Start the Subscriber (Laptop/Smartphone)
```bash
cd laptop_app
python door_monitor.py
```

### Start the Publisher (IoT Device)
```bash
cd iot_device
python publisher_node.py
```

## Expected Behavior

1. Subscriber starts and waits for events
2. Publisher detects door movement and classifies it
3. Publisher sends decision to AWS IoT Core
4. Subscriber receives message and updates display
5. User sees current door status and timestamp

**Output is minimal** - only shows important information when state changes occur.
