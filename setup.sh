#!/bin/bash

# ==============================
# Door State Detection System Setup Script
# ==============================

echo "🚪 Door State Detection System Setup"
echo "===================================="

# Check if running on Raspberry Pi
if grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "✅ Raspberry Pi detected"
    IS_PI=true
else
    echo "ℹ️  Not running on Raspberry Pi (simulation mode)"
    IS_PI=false
fi

# Function to install Python dependencies
install_dependencies() {
    local dir=$1
    echo "📦 Installing dependencies for $dir..."
    
    if [ -f "$dir/requirements.txt" ]; then
        pip3 install -r "$dir/requirements.txt"
        echo "✅ Dependencies installed for $dir"
    else
        echo "⚠️  No requirements.txt found in $dir"
    fi
}

# Function to setup AWS IoT certificates
setup_certificates() {
    echo "🔐 Setting up AWS IoT certificates..."
    
    # Create certs directory if it doesn't exist
    mkdir -p certs
    
    echo "Please place your AWS IoT certificates in the certs/ directory:"
    echo "- AmazonRootCA1.pem"
    echo "- device-certificate.pem.crt"
    echo "- private.pem.key"
    echo ""
    echo "You can download these from AWS IoT Console > Security > Certificates"
    echo ""
    
    # Check if certificates exist
    if [ -f "certs/AmazonRootCA1.pem" ] && [ -f "certs/device-certificate.pem.crt" ] && [ -f "certs/private.pem.key" ]; then
        echo "✅ All required certificates found"
    else
        echo "⚠️  Please add the required certificates to the certs/ directory"
        echo "   The system will not work without proper certificates"
    fi
}

# Function to update configuration files
update_config() {
    echo "⚙️  Updating configuration files..."
    
    # Get AWS endpoint from user
    read -p "Enter your AWS IoT endpoint: " aws_endpoint
    
    if [ -z "$aws_endpoint" ]; then
        echo "⚠️  AWS endpoint not provided. Please update config.json manually."
        return
    fi
    
    # Update IoT device config
    if [ -f "iot_device/config.json" ]; then
        sed -i "s/your-aws-iot-endpoint.amazonaws.com/$aws_endpoint/g" iot_device/config.json
        echo "✅ Updated IoT device config"
    fi
    
    # Update laptop app config
    if [ -f "laptop_app/config.json" ]; then
        sed -i "s/your-aws-iot-endpoint.amazonaws.com/$aws_endpoint/g" laptop_app/config.json
        echo "✅ Updated laptop app config"
    fi
    
    # Update cloud service config
    if [ -f "cloud_service/config.json" ]; then
        sed -i "s/your-aws-iot-endpoint.amazonaws.com/$aws_endpoint/g" cloud_service/config.json
        echo "✅ Updated cloud service config"
    fi
}

# Function to create systemd service for IoT device
create_service() {
    if [ "$IS_PI" = true ]; then
        echo "🔧 Creating systemd service for IoT device..."
        
        cat > /tmp/door-detector.service << EOF
[Unit]
Description=Door State Detection Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$(pwd)/iot_device
ExecStart=/usr/bin/python3 $(pwd)/iot_device/imu_door_detector.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        sudo mv /tmp/door-detector.service /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable door-detector.service
        
        echo "✅ Systemd service created"
        echo "   Use 'sudo systemctl start door-detector' to start the service"
        echo "   Use 'sudo systemctl status door-detector' to check status"
    fi
}

# Main setup process
main() {
    echo "Starting setup process..."
    echo ""
    
    # Install dependencies for each component
    install_dependencies "iot_device"
    install_dependencies "laptop_app"
    install_dependencies "cloud_service"
    
    echo ""
    
    # Setup certificates
    setup_certificates
    
    echo ""
    
    # Update configuration
    update_config
    
    echo ""
    
    # Create systemd service (Raspberry Pi only)
    if [ "$IS_PI" = true ]; then
        create_service
        echo ""
    fi
    
    echo "🎉 Setup completed!"
    echo ""
    echo "Next steps:"
    echo "1. Collect training data: python3 iot_device/training_data_collector.py"
    echo "2. Train the model: python3 iot_device/train_svm_model.py"
    echo "3. Run the IoT device: python3 iot_device/imu_door_detector.py"
    echo "4. Run the laptop app: python3 laptop_app/door_monitor.py"
    echo "5. Run cloud service: python3 cloud_service/cloud_classifier.py"
    echo "6. Run web dashboard: python3 cloud_service/web_dashboard.py"
    echo ""
    echo "For cloud-based classification, use:"
    echo "- IoT device: python3 iot_device/cloud_imu_door_detector.py"
    echo "- Cloud classifier: python3 cloud_service/cloud_classifier.py"
}

# Run main function
main
