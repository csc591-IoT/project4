import json
import time
import threading
from datetime import datetime, timezone
from flask import Flask, render_template, jsonify
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
TOPIC_RAW_DATA   = cfg["topic_raw_data"]
TOPIC_RESULTS    = cfg["topic_results"]

# ==============================
# Flask Web Dashboard
# ==============================
app = Flask(__name__)

# Global data storage
dashboard_data = {
    'raw_data_messages': [],
    'classification_results': [],
    'device_status': {},
    'statistics': {
        'total_messages': 0,
        'open_events': 0,
        'closed_events': 0,
        'last_update': None
    }
}

# MQTT client
mqtt_client = None

class DashboardMQTTClient:
    def __init__(self):
        self.client = AWSIoTMQTTClient(f"{CLIENT_ID}_dashboard")
        self.client.configureEndpoint(AWS_ENDPOINT, 8883)
        self.client.configureCredentials(ROOT_CA_PATH, PRIVATE_KEY_PATH, CERT_PATH)
        
    def connect(self):
        try:
            self.client.connect()
            print("Dashboard MQTT client connected")
            
            # Subscribe to all door-related topics
            self.client.subscribe("door/#", 1, self.on_message)
            print("Subscribed to door topics")
            
        except Exception as e:
            print(f"Dashboard MQTT connection failed: {e}")
    
    def on_message(self, client, userdata, message):
        """Handle incoming MQTT messages"""
        try:
            payload = json.loads(message.payload.decode('utf-8'))
            topic = message.topic
            
            # Store message based on topic
            if topic == TOPIC_RAW_DATA:
                dashboard_data['raw_data_messages'].append({
                    'timestamp': payload.get('timestamp'),
                    'device_id': payload.get('device_id'),
                    'request_id': payload.get('request_id'),
                    'received_at': datetime.now(timezone.utc).isoformat()
                })
                
                # Update device status
                device_id = payload.get('device_id')
                dashboard_data['device_status'][device_id] = {
                    'last_seen': datetime.now(timezone.utc).isoformat(),
                    'status': 'active'
                }
                
            elif topic == TOPIC_RESULTS:
                dashboard_data['classification_results'].append({
                    'timestamp': payload.get('timestamp'),
                    'device_id': payload.get('device_id'),
                    'door_state': payload.get('door_state'),
                    'confidence': payload.get('confidence'),
                    'request_id': payload.get('request_id'),
                    'processed_at': payload.get('processed_at')
                })
                
                # Update statistics
                dashboard_data['statistics']['total_messages'] += 1
                door_state = payload.get('door_state')
                if door_state == 'open':
                    dashboard_data['statistics']['open_events'] += 1
                elif door_state == 'closed':
                    dashboard_data['statistics']['closed_events'] += 1
                
                dashboard_data['statistics']['last_update'] = datetime.now(timezone.utc).isoformat()
            
            elif topic == "door/state":
                # Final door state updates
                dashboard_data['classification_results'].append({
                    'timestamp': payload.get('timestamp'),
                    'device_id': payload.get('device_id'),
                    'door_state': payload.get('door_state'),
                    'confidence': payload.get('confidence'),
                    'classification_method': payload.get('classification_method', 'local')
                })
            
            # Keep only last 100 messages to prevent memory issues
            if len(dashboard_data['raw_data_messages']) > 100:
                dashboard_data['raw_data_messages'] = dashboard_data['raw_data_messages'][-100:]
            
            if len(dashboard_data['classification_results']) > 100:
                dashboard_data['classification_results'] = dashboard_data['classification_results'][-100:]
                
        except Exception as e:
            print(f"Error processing dashboard message: {e}")

# Initialize MQTT client
mqtt_dashboard = DashboardMQTTClient()

# ==============================
# Flask Routes
# ==============================
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """API endpoint to get dashboard data"""
    return jsonify(dashboard_data)

@app.route('/api/statistics')
def get_statistics():
    """API endpoint to get statistics"""
    return jsonify(dashboard_data['statistics'])

@app.route('/api/devices')
def get_devices():
    """API endpoint to get device status"""
    return jsonify(dashboard_data['device_status'])

@app.route('/api/messages')
def get_messages():
    """API endpoint to get recent messages"""
    return jsonify({
        'raw_data': dashboard_data['raw_data_messages'][-20:],
        'results': dashboard_data['classification_results'][-20:]
    })

# ==============================
# Main
# ==============================
def start_mqtt_client():
    """Start MQTT client in separate thread"""
    try:
        mqtt_dashboard.connect()
        
        # Keep MQTT client running
        while True:
            time.sleep(1)
            
    except Exception as e:
        print(f"MQTT client error: {e}")

def main():
    print("Starting Door State Dashboard...")
    
    # Start MQTT client in background thread
    mqtt_thread = threading.Thread(target=start_mqtt_client, daemon=True)
    mqtt_thread.start()
    
    # Start Flask web server
    print("Starting web server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()
