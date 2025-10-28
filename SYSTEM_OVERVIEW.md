# Door State Detection System - Technical Overview

## System Architecture

This system implements a complete IoT-based door state detection solution using IMU sensors, machine learning classification, and cloud services. The system consists of several components working together to detect and communicate door open/close events.

### Components Overview

1. **IoT Device (Raspberry Pi)** - Collects IMU data and performs classification
2. **Laptop/Smartphone App** - Displays door state in real-time
3. **Cloud Service** - Alternative classification service and web dashboard
4. **MQTT Communication** - AWS IoT Core for message passing

## Technical Implementation

### 1. IMU Data Collection and Processing

#### Data Collection
- **Sampling Rate**: 50 Hz
- **Window Size**: 100 samples for feature extraction
- **Movement Detection**: Threshold-based detection using acceleration and gyroscope magnitude

#### Feature Extraction
The system extracts comprehensive features from IMU data:

**Statistical Features (per axis)**:
- Mean, Standard Deviation, Variance
- Maximum, Minimum, Median
- 25th and 75th percentiles

**Magnitude Features**:
- Acceleration magnitude statistics
- Angular velocity magnitude statistics
- Range calculations

**Energy Features**:
- Sum of squared values for each axis
- Frequency domain features (simplified FFT)

**Total Features**: ~60 features per classification window

### 2. Machine Learning Classification

#### Support Vector Machine (SVM)
- **Algorithm**: Support Vector Classification
- **Kernel Options**: RBF, Linear, Polynomial
- **Hyperparameter Tuning**: Automated cross-validation
- **Feature Normalization**: StandardScaler preprocessing

#### Training Process
1. **Data Collection**: Use `training_data_collector.py` to gather labeled data
2. **Feature Extraction**: Automatic feature extraction from raw IMU data
3. **Model Training**: Use `train_svm_model.py` for model training and evaluation
4. **Model Persistence**: Trained models saved as pickle files

#### Classification Output
- **Classes**: 0 (Closed), 1 (Open)
- **Confidence**: Probability scores when available
- **Decision Threshold**: Configurable confidence thresholds

### 3. MQTT Communication Protocol

#### Message Structure
```json
{
  "device_id": "door-sensor-pi-001",
  "timestamp": "2025-01-XX T XX:XX:XX.XXXZ",
  "door_state": "open|closed",
  "confidence": 0.85
}
```

#### Topics
- `door/state` - Final door state decisions
- `door/raw_data` - Raw IMU features for cloud classification
- `door/classification_results` - Cloud classification results

### 4. System Modes

#### Local Classification Mode
- **File**: `imu_door_detector.py`
- **Process**: IMU → Feature Extraction → Local SVM → MQTT
- **Advantages**: Low latency, works offline
- **Requirements**: Trained model on device

#### Cloud Classification Mode
- **Files**: `cloud_imu_door_detector.py` + `cloud_classifier.py`
- **Process**: IMU → Feature Extraction → MQTT → Cloud SVM → MQTT
- **Advantages**: Centralized processing, easier model updates
- **Requirements**: Internet connection, cloud service running

### 5. User Interface

#### Desktop Application
- **Framework**: Tkinter (Python)
- **Features**: Real-time door state display, message logging, connection status
- **Updates**: Automatic refresh every 5 seconds

#### Web Dashboard
- **Framework**: Flask (Python) + HTML/CSS/JavaScript
- **Features**: Statistics, device monitoring, message history
- **Access**: http://localhost:5000

## Hardware Requirements

### IoT Device (Raspberry Pi)
- **Processor**: Raspberry Pi 3B+ or newer
- **Memory**: 1GB RAM minimum
- **Storage**: 8GB SD card minimum
- **Sensors**: 6-axis IMU (accelerometer + gyroscope)
- **Connectivity**: WiFi or Ethernet for MQTT

### IMU Sensor Specifications
- **Accelerometer**: 3-axis, ±2g to ±16g range
- **Gyroscope**: 3-axis, ±250 to ±2000 dps range
- **Interface**: I2C or SPI
- **Power**: 3.3V or 5V operation

## Software Dependencies

### IoT Device
```
AWSIoTPythonSDK==1.5.4
RPi.GPIO
numpy
scikit-learn
joblib
pandas
matplotlib
seaborn
```

### Laptop Application
```
AWSIoTPythonSDK==1.5.4
tkinter
numpy
scikit-learn
joblib
pandas
matplotlib
seaborn
```

### Cloud Service
```
Flask==2.3.3
AWSIoTPythonSDK==1.5.4
numpy
scikit-learn
joblib
```

## Configuration

### AWS IoT Core Setup
1. **Thing Creation**: Create IoT thing in AWS IoT Console
2. **Certificate Generation**: Generate device certificates
3. **Policy Attachment**: Attach policies for MQTT publish/subscribe
4. **Endpoint Configuration**: Update endpoint in config files

### Certificate Structure
```
certs/
├── AmazonRootCA1.pem
├── device-certificate.pem.crt
└── private.pem.key
```

## Performance Characteristics

### Classification Accuracy
- **Target**: >90% accuracy on test data
- **Latency**: <2 seconds from movement end to classification
- **False Positive Rate**: <5% for door state changes

### System Performance
- **CPU Usage**: <20% on Raspberry Pi 3B+
- **Memory Usage**: <100MB RAM
- **Network**: <1KB per classification message
- **Power**: <2W average consumption

## Deployment Considerations

### Training Data Requirements
- **Minimum Samples**: 200 per class (open/closed)
- **Data Collection**: 5-10 minutes per door state
- **Validation**: Cross-validation during training
- **Retraining**: Recommended when door characteristics change

### Environmental Factors
- **Door Type**: Works with various door types and sizes
- **Mounting Position**: Avoid door edges, prefer near hinge
- **Movement Threshold**: Configurable angle thresholds
- **Interference**: Minimal impact from vibrations

### Scalability
- **Multiple Devices**: Support for multiple door sensors
- **Cloud Processing**: Horizontal scaling possible
- **Message Volume**: Handles 100+ messages per minute
- **Storage**: Minimal local storage requirements

## Troubleshooting

### Common Issues
1. **MQTT Connection Failures**: Check certificates and endpoint
2. **Classification Errors**: Verify training data quality
3. **Movement Detection**: Adjust thresholds in configuration
4. **Performance Issues**: Monitor CPU and memory usage

### Debug Features
- **Logging**: Comprehensive logging throughout system
- **Status Indicators**: LED indicators for device status
- **Web Dashboard**: Real-time monitoring and statistics
- **Message History**: Complete message log for debugging

## Security Considerations

### Data Privacy
- **Local Processing**: Option for local-only classification
- **Encrypted Communication**: TLS 1.2 for MQTT
- **Certificate Authentication**: Device-level authentication
- **No Personal Data**: Only sensor readings transmitted

### Network Security
- **AWS IoT Core**: Managed security service
- **Certificate Rotation**: Support for certificate updates
- **Access Control**: Policy-based access control
- **Monitoring**: CloudWatch integration for monitoring

This system provides a robust, scalable solution for door state detection with both local and cloud processing options, comprehensive monitoring capabilities, and professional-grade security features.
