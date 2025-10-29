# Door Monitor

## Overview
This is the **MQTT Subscriber** application that runs on a laptop or smartphone to monitor door status. It receives door open/close events from the IoT device (publisher) via AWS IoT Core and displays the current status to the user.

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
python subscriber_node.py
```
