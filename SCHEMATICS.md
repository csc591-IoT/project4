# Door State Detection System - Schematics and Connections

## System Architecture Diagram

```
┌─────────────────┐    MQTT     ┌─────────────────┐    MQTT     ┌─────────────────┐
│   IMU Sensor    │─────────────▶│  Raspberry Pi   │─────────────▶│  AWS IoT Core   │
│                 │              │                 │              │                 │
│ • Accelerometer │              │ • Data Collection│              │ • MQTT Broker   │
│ • Gyroscope     │              │ • Feature Extract│              │ • Message Route │
│ • I2C/SPI       │              │ • Classification │              │ • Security      │
└─────────────────┘              └─────────────────┘              └─────────────────┘
                                           │                                │
                                           │                                │
                                           ▼                                ▼
                                  ┌─────────────────┐              ┌─────────────────┐
                                  │  Local SVM      │              │  Cloud Service  │
                                  │  Classification │              │                 │
                                  └─────────────────┘              │ • SVM Model     │
                                                                   │ • Web Dashboard │
                                                                   │ • Statistics     │
                                                                   └─────────────────┘
                                                                           │
                                                                           │ MQTT
                                                                           ▼
                                                                  ┌─────────────────┐
                                                                  │ Laptop/Smartphone│
                                                                  │                 │
                                                                  │ • Real-time UI  │
                                                                  │ • Status Display│
                                                                  │ • Message Log   │
                                                                  └─────────────────┘
```

## Hardware Connection Diagram

### Raspberry Pi Pin Connections

```
Raspberry Pi 4B Pinout:

┌─────────────────────────────────────────────────────────────┐
│ 3.3V  │ 5V  │ GPIO2 │ 5V  │ GPIO3 │ GND │ GPIO4 │ 14 │ GND │
│ GPIO2 │ 5V  │ GPIO3 │ GND │ GPIO4 │ 14  │ GND   │ 15 │ 17  │
│ 3.3V  │ 18  │ GPIO5 │ GND │ GPIO6 │ 27  │ GND   │ 22 │ 3.3V│
│ GPIO10│ 9   │ GND   │ 11  │ GPIO8 │ GND │ GPIO7 │ 25 │ 8   │
│ GND   │ 23  │ GPIO24│ 3.3V│ GPIO25│ 28  │ GND   │ 1  │ 7   │
│ GPIO18│ GND │ GPIO15│ 3   │ GND   │ 5   │ GPIO16│ 6  │ 12  │
│ GND   │ 19  │ GPIO20│ GND │ GPIO21│ 26  │ GND   │ 16 │ 20  │
└─────────────────────────────────────────────────────────────┘

IMU Sensor Connections:
┌─────────────────┐
│   MPU6050 IMU   │
│                 │
│ VCC  ──────────▶│ 3.3V (Pin 1)
│ GND  ──────────▶│ GND (Pin 6)
│ SCL  ──────────▶│ GPIO3 (Pin 5) - I2C Clock
│ SDA  ──────────▶│ GPIO2 (Pin 3) - I2C Data
│ INT  ──────────▶│ GPIO4 (Pin 7) - Interrupt (Optional)
└─────────────────┘

LED Indicator:
┌─────────────────┐
│   Status LED    │
│                 │
│ Anode ─────────▶│ GPIO3 (Pin 5)
│ Cathode ───────▶│ GND (Pin 6)
│                 │
│ Resistor: 220Ω  │
└─────────────────┘
```

## Software Architecture

### Data Flow Diagram

```
IMU Sensor Data
       │
       ▼
┌─────────────────┐
│ Data Collection │
│ • 50Hz Sampling │
│ • Buffer Mgmt   │
│ • Movement Det. │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Feature Extract │
│ • Statistical   │
│ • Magnitude     │
│ • Energy        │
│ • Frequency     │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Classification  │
│ • SVM Model     │
│ • Normalization │
│ • Prediction    │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ MQTT Publisher  │
│ • JSON Payload  │
│ • QoS Level 0   │
│ • Error Handle  │
└─────────────────┘
```

### Classification Pipeline

```
Raw IMU Data (6-axis)
       │
       ▼
┌─────────────────┐
│ Data Preprocess │
│ • Noise Filter  │
│ • Outlier Det.  │
│ • Window Mgmt   │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Feature Extract │
│ • Time Domain   │
│ • Frequency     │
│ • Statistical   │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Feature Select  │
│ • Normalization │
│ • Scaling       │
│ • Dimensionality│
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ SVM Classifier  │
│ • Kernel Opt.   │
│ • Hyperparams   │
│ • Cross-Val     │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Post-Process    │
│ • Confidence    │
│ • Threshold     │
│ • State Mgmt    │
└─────────────────┘
```

## Network Topology

### MQTT Topic Structure

```
door/
├── state/                    # Final door state decisions
│   ├── device-001/
│   ├── device-002/
│   └── ...
├── raw_data/                 # Raw IMU features
│   ├── device-001/
│   ├── device-002/
│   └── ...
├── classification_results/  # Cloud classification results
│   ├── device-001/
│   ├── device-002/
│   └── ...
└── status/                   # Device status messages
    ├── device-001/
    ├── device-002/
    └── ...
```

### Message Flow

```
Device A (Publisher)          AWS IoT Core          Device B (Subscriber)
      │                           │                        │
      │─── MQTT Publish ─────────▶│                        │
      │   Topic: door/state       │                        │
      │   QoS: 0                  │                        │
      │                           │─── MQTT Message ──────▶│
      │                           │   Topic: door/state   │
      │                           │   QoS: 0              │
      │                           │                        │
      │                           │                        │─── Process Message
      │                           │                        │    Update UI
      │                           │                        │    Log Event
```

## Power and Signal Specifications

### Power Requirements

| Component | Voltage | Current | Power |
|-----------|---------|---------|-------|
| Raspberry Pi 4B | 5V | 600mA | 3W |
| MPU6050 IMU | 3.3V | 3.9mA | 13mW |
| Status LED | 3.3V | 15mA | 50mW |
| **Total** | | | **~3.1W** |

### Signal Levels

| Signal | Voltage | Logic | Notes |
|--------|---------|-------|-------|
| I2C SCL | 3.3V | High/Low | Pull-up required |
| I2C SDA | 3.3V | High/Low | Pull-up required |
| GPIO | 3.3V | High/Low | 5V tolerant |
| Interrupt | 3.3V | High/Low | Optional |

### Timing Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| I2C Clock | 400kHz | Standard mode |
| Sampling Rate | 50Hz | Configurable |
| Classification Latency | <2s | Movement end to result |
| MQTT Publish Rate | 1/min | During movement |
| Heartbeat Interval | 30s | Device status |

## Mechanical Mounting

### Door Attachment Points

```
Door Frame
    │
    ├── Hinge Side (Recommended)
    │   ├── Distance from hinge: 10-20cm
    │   ├── Height: Any convenient level
    │   └── Orientation: Vertical mounting
    │
    ├── Handle Side (Avoid)
    │   └── Too far from pivot point
    │
    └── Top/Bottom (Alternative)
        ├── Distance from edge: 5-10cm
        └── Orientation: Horizontal mounting
```

### Sensor Orientation

```
IMU Sensor Mounting:
┌─────────────────┐
│   MPU6050      │
│                 │
│ X ─────────────▶│ Door opening direction
│ Y ─────────────▶│ Vertical (up/down)
│ Z ─────────────▶│ Into/out of door plane
│                 │
└─────────────────┘

Coordinate System:
- X-axis: Door movement direction
- Y-axis: Vertical axis
- Z-axis: Perpendicular to door surface
```

## Troubleshooting Connections

### Common Connection Issues

1. **I2C Communication Failures**
   - Check pull-up resistors (4.7kΩ)
   - Verify voltage levels (3.3V)
   - Confirm device address

2. **Power Issues**
   - Ensure stable 5V supply
   - Check current capacity
   - Monitor voltage drops

3. **MQTT Connection Problems**
   - Verify certificates
   - Check network connectivity
   - Confirm endpoint URL

4. **Classification Accuracy**
   - Validate training data
   - Check sensor mounting
   - Adjust movement thresholds

### Debug Connections

```
Debug Header (Optional):
┌─────────────────┐
│   Debug Pins    │
│                 │
│ GPIO14 ────────▶│ Serial TX
│ GPIO15 ────────▶│ Serial RX
│ GPIO18 ────────▶│ Status LED
│ GPIO23 ────────▶│ Error LED
│ GPIO24 ────────▶│ Activity LED
└─────────────────┘
```

This comprehensive schematic documentation provides all necessary information for hardware assembly, software configuration, and system troubleshooting.
