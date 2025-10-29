import json
import time
from datetime import datetime
import AWSIoTPythonSDK.MQTTLib as AWSIoTPyMQTT

CONFIG_PATH = "config.json"

current_state = {
    'status': 'Unknown',
    'timestamp': None,
    'confidence': None
}

def update_display():
    emoji = "OPEN" if current_state['status'].lower() == 'open' else "CLOSED"
    if current_state['status'].lower() == 'unknown':
        emoji = "UNKNOWN"
    
    print(f"\n{'='*70}")
    print(f"  Door Status: {emoji}")
    if current_state['timestamp']:
        print(f"  Last Updated: {current_state['timestamp']}")
    if current_state['confidence']:
        print(f"  Confidence: {current_state['confidence']:.2f}")
    print(f"{'='*70}\n")


def on_message_callback(client, userdata, message):
    """
    Callback when door state message is received from IoT device
    Updates current status and displays to user
    """
    try:
        # Parse incoming message
        payload = json.loads(message.payload.decode('utf-8'))
        
        # Update current state
        if 'state' in payload:
            current_state['status'] = payload['state'].capitalize()
        
        if 'timestamp' in payload:
            current_state['timestamp'] = payload['timestamp']
        
        if 'confidence' in payload:
            current_state['confidence'] = payload['confidence']
        
        # Display updated status
        update_display()
        
    except Exception as e:
        print(f"Error processing message: {e}")


def main():
    print("\n" + "="*70)
    print("Door Monitor - Laptop/Smartphone Subscriber")
    print("="*70)
    
    # Load configuration
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        
        AWS_ENDPOINT     = cfg["aws_endpoint"]
        ROOT_CA_PATH     = cfg["root_ca_path"]
        CERT_PATH        = cfg["cert_path"]
        PRIVATE_KEY_PATH = cfg["private_key_path"]
        CLIENT_ID        = cfg["client_id"] + "-subscriber"
        TOPIC            = cfg["topic"]
        
        print(f"Connected to AWS IoT")
        print(f"Monitoring topic: {TOPIC}")
        
    except Exception as e:
        print(f"Configuration error: {e}")
        return
    
    # 
    try:
        myAWSIoTMQTTClient = AWSIoTPyMQTT.AWSIoTMQTTClient(CLIENT_ID)
        myAWSIoTMQTTClient.configureEndpoint(AWS_ENDPOINT, 8883)
        myAWSIoTMQTTClient.configureCredentials(ROOT_CA_PATH, PRIVATE_KEY_PATH, CERT_PATH)
        
        myAWSIoTMQTTClient.connect()
        myAWSIoTMQTTClient.subscribe(TOPIC, 1, on_message_callback)
        
        print(f"{'='*70}\n")
        print("Waiting for door events...\n")
        
        # Keep running to receive messages
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n{'='*70}")
        print("Monitor stopped")
        print(f"{'='*70}\n")
    finally:
        try:
            myAWSIoTMQTTClient.disconnect()
        except:
            pass


if __name__ == "__main__":
    main()