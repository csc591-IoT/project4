import json
import time
import sys
from datetime import datetime, timezone
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
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
TOPIC            = cfg["topic"]

# ==============================
# Door Monitor GUI Application
# ==============================

class DoorMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Door State Monitor")
        self.root.geometry("600x500")
        self.root.configure(bg='#f0f0f0')
        
        # Current door state
        self.current_state = "Unknown"
        self.last_update_time = None
        self.message_count = 0
        
        # MQTT client
        self.mqtt_client = None
        self.is_connected = False
        
        # Setup GUI
        self.setup_gui()
        
        # Setup MQTT
        self.setup_mqtt()
        
        # Start MQTT connection in separate thread
        self.mqtt_thread = threading.Thread(target=self.connect_mqtt, daemon=True)
        self.mqtt_thread.start()
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Title
        title_label = tk.Label(
            self.root, 
            text="🚪 Door State Monitor", 
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=20)
        
        # Status frame
        status_frame = tk.Frame(self.root, bg='#f0f0f0')
        status_frame.pack(pady=10)
        
        # Connection status
        self.connection_label = tk.Label(
            status_frame,
            text="🔴 Disconnected",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#ff0000'
        )
        self.connection_label.pack(side=tk.LEFT, padx=10)
        
        # Message count
        self.count_label = tk.Label(
            status_frame,
            text="Messages: 0",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#666666'
        )
        self.count_label.pack(side=tk.LEFT, padx=10)
        
        # Door state display
        state_frame = tk.Frame(self.root, bg='#f0f0f0')
        state_frame.pack(pady=20)
        
        tk.Label(
            state_frame,
            text="Current Door State:",
            font=("Arial", 14),
            bg='#f0f0f0',
            fg='#333333'
        ).pack()
        
        self.state_label = tk.Label(
            state_frame,
            text="Unknown",
            font=("Arial", 24, "bold"),
            bg='#f0f0f0',
            fg='#666666'
        )
        self.state_label.pack(pady=10)
        
        # Last update time
        self.time_label = tk.Label(
            state_frame,
            text="Last Update: Never",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#888888'
        )
        self.time_label.pack()
        
        # Log area
        log_frame = tk.Frame(self.root, bg='#f0f0f0')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        tk.Label(
            log_frame,
            text="Message Log:",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#333333'
        ).pack(anchor=tk.W)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            font=("Consolas", 9),
            bg='#ffffff',
            fg='#333333',
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        self.connect_button = tk.Button(
            button_frame,
            text="Connect",
            command=self.toggle_connection,
            font=("Arial", 10),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=5
        )
        self.connect_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = tk.Button(
            button_frame,
            text="Clear Log",
            command=self.clear_log,
            font=("Arial", 10),
            bg='#f44336',
            fg='white',
            padx=20,
            pady=5
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.exit_button = tk.Button(
            button_frame,
            text="Exit",
            command=self.exit_app,
            font=("Arial", 10),
            bg='#666666',
            fg='white',
            padx=20,
            pady=5
        )
        self.exit_button.pack(side=tk.LEFT, padx=5)
    
    def setup_mqtt(self):
        """Setup MQTT client"""
        self.mqtt_client = AWSIoTMQTTClient(CLIENT_ID)
        self.mqtt_client.configureEndpoint(AWS_ENDPOINT, 8883)
        self.mqtt_client.configureCredentials(ROOT_CA_PATH, PRIVATE_KEY_PATH, CERT_PATH)
        
        # Configure connection settings
        self.mqtt_client.configureAutoReconnectBackoffTime(1, 32, 20)
        self.mqtt_client.configureConnectDisconnectTimeout(10)
        self.mqtt_client.configureMQTTOperationTimeout(5)
        
        # Set up callback
        self.mqtt_client.configureLastWillMessage(TOPIC, json.dumps({
            "device_id": CLIENT_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "offline"
        }), 0)
    
    def connect_mqtt(self):
        """Connect to MQTT broker"""
        try:
            self.log_message("Connecting to AWS IoT...")
            self.mqtt_client.connect()
            self.mqtt_client.subscribe(TOPIC, 1, self.on_message)
            
            self.is_connected = True
            self.root.after(0, self.update_connection_status)
            self.log_message("Connected to AWS IoT successfully!")
            
        except Exception as e:
            self.log_message(f"Connection failed: {str(e)}")
            self.is_connected = False
            self.root.after(0, self.update_connection_status)
    
    def on_message(self, client, userdata, message):
        """Handle incoming MQTT messages"""
        try:
            payload = json.loads(message.payload.decode('utf-8'))
            
            # Update door state
            if 'door_state' in payload:
                self.current_state = payload['door_state']
                self.last_update_time = payload.get('timestamp', datetime.now(timezone.utc).isoformat())
                self.message_count += 1
                
                # Update GUI in main thread
                self.root.after(0, self.update_display)
                
                # Log message
                self.log_message(f"Door state updated: {self.current_state}")
            
        except Exception as e:
            self.log_message(f"Error processing message: {str(e)}")
    
    def update_display(self):
        """Update the GUI display"""
        # Update door state
        self.state_label.config(text=self.current_state.title())
        
        # Update colors based on state
        if self.current_state.lower() == "open":
            self.state_label.config(fg='#ff4444')  # Red for open
        elif self.current_state.lower() == "closed":
            self.state_label.config(fg='#44ff44')  # Green for closed
        else:
            self.state_label.config(fg='#666666')  # Gray for unknown
        
        # Update time
        if self.last_update_time:
            try:
                dt = datetime.fromisoformat(self.last_update_time.replace('Z', '+00:00'))
                local_time = dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
                self.time_label.config(text=f"Last Update: {local_time}")
            except:
                self.time_label.config(text=f"Last Update: {self.last_update_time}")
        
        # Update message count
        self.count_label.config(text=f"Messages: {self.message_count}")
    
    def update_connection_status(self):
        """Update connection status display"""
        if self.is_connected:
            self.connection_label.config(text="🟢 Connected", fg='#00aa00')
            self.connect_button.config(text="Disconnect", bg='#f44336')
        else:
            self.connection_label.config(text="🔴 Disconnected", fg='#ff0000')
            self.connect_button.config(text="Connect", bg='#4CAF50')
    
    def toggle_connection(self):
        """Toggle MQTT connection"""
        if self.is_connected:
            try:
                self.mqtt_client.disconnect()
                self.is_connected = False
                self.update_connection_status()
                self.log_message("Disconnected from AWS IoT")
            except Exception as e:
                self.log_message(f"Error disconnecting: {str(e)}")
        else:
            # Start connection in separate thread
            self.mqtt_thread = threading.Thread(target=self.connect_mqtt, daemon=True)
            self.mqtt_thread.start()
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
    
    def clear_log(self):
        """Clear the log area"""
        self.log_text.delete(1.0, tk.END)
    
    def exit_app(self):
        """Exit the application"""
        try:
            if self.mqtt_client and self.is_connected:
                self.mqtt_client.disconnect()
        except:
            pass
        self.root.quit()
        sys.exit(0)

# ==============================
# Main
# ==============================
def main():
    root = tk.Tk()
    app = DoorMonitorApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    finally:
        app.exit_app()

if __name__ == "__main__":
    main()
