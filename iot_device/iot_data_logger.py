#!/usr/bin/env python3
# imu_event_logger.py - Captures complete door motion EVENTS
import os, csv, time, sys, threading, pathlib
from math import sqrt
from datetime import datetime
from smbus2 import SMBus

# ========= Session config =========
DOOR_ID    = "doorA"
MOUNT_POS  = "pos1"      # UPDATE THIS: pos1 = 1/3 from hinge, pos2 = 1/2 from hinge
SAMPLE_HZ  = 100
OUT_DIR    = "sessions"

# Motion detection thresholds
MOTION_THRESHOLD = 10.0   # deg/s - gyro magnitude to detect motion start
MOTION_TIMEOUT   = 0.5    # seconds of stillness to end event
PRE_MOTION_BUFFER = 0.3   # seconds to capture before motion detected
# ==================================

# I2C / MPU6050 constants
I2C_BUS      = 1
MPU_ADDR     = 0x68
PWR_MGMT_1   = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H  = 0x43
ACCEL_SCALE  = 16384.0
GYRO_SCALE   = 131.0

# State variables
_current_label = None
_is_armed = False       # Ready to capture next event
_running = True
_event_count = {"opening": 0, "closing": 0}

def _kb_listener():
    """
    ARM the system to capture next event:
    o = Arm for OPENING event
    c = Arm for CLOSING event
    q = Quit and save file
    """
    global _current_label, _is_armed, _running, _event_count
    
    print("\n" + "="*70)
    print("EVENT CAPTURE CONTROLS:")
    print("="*70)
    print("  'o' = ARM for next OPENING event")
    print("       → Press 'o', then open door when ready")
    print("       → System auto-detects motion and captures complete event")
    print()
    print("  'c' = ARM for next CLOSING event")
    print("       → Press 'c', then close door when ready")
    print("       → System auto-detects motion and captures complete event")
    print()
    print("  'q' = QUIT and save file")
    print("="*70)
    print("\nRECOMMENDED WORKFLOW:")
    print("  1. Start with door in any position")
    print("  2. Press 'o' → open door smoothly → wait for 'captured' message")
    print("  3. Press 'c' → close door smoothly → wait for 'captured' message")
    print("  4. Repeat steps 2-3 about 20-30 times")
    print("  5. Vary speeds: slow, medium, fast, partial opens/closes")
    print("  6. Press 'q' when done")
    print("="*70)
    print("\nWaiting for command...\n")
    
    while _running:
        ch = sys.stdin.read(1)
        
        if ch == 'o':
            _current_label = "opening"
            _is_armed = True
            print("\n" + "="*70)
            print(f">>> ARMED FOR OPENING EVENT #{_event_count['opening']+1} <<<")
            print(">>> OPEN THE DOOR NOW <<<")
            print("="*70)
            
        elif ch == 'c':
            _current_label = "closing"
            _is_armed = True
            print("\n" + "="*70)
            print(f">>> ARMED FOR CLOSING EVENT #{_event_count['closing']+1} <<<")
            print(">>> CLOSE THE DOOR NOW <<<")
            print("="*70)
            
        elif ch == 'q':
            _running = False
            print("\n[!] Quitting...")

def _read_word(bus, reg):
    hi = bus.read_byte_data(MPU_ADDR, reg)
    lo = bus.read_byte_data(MPU_ADDR, reg+1)
    v  = (hi << 8) | lo
    return v - 65536 if v >= 32768 else v

def main():
    global _running, _is_armed, _event_count, _current_label
    
    pathlib.Path(OUT_DIR).mkdir(exist_ok=True)
    fname    = f"{DOOR_ID}_{MOUNT_POS}_events_{int(time.time())}.csv"
    csv_path = os.path.join(OUT_DIR, fname)

    # Init I2C + wake IMU
    bus = SMBus(I2C_BUS)
    bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)
    time.sleep(0.1)

    # Prepare terminal for single-key reads
    import termios, tty
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    period = 1.0 / SAMPLE_HZ
    next_t = time.monotonic()
    
    # Circular buffer for pre-motion data
    buffer_size = int(PRE_MOTION_BUFFER * SAMPLE_HZ)
    ring_buffer = []
    
    # Event capture state
    capturing_event = False
    event_data = []
    stillness_counter = 0
    stillness_samples = int(MOTION_TIMEOUT * SAMPLE_HZ)

    print(f"\n[i] Data will be saved to: {csv_path}")
    print(f"[i] Sampling at {SAMPLE_HZ} Hz")
    print(f"[i] Motion threshold: {MOTION_THRESHOLD} deg/s")
    print(f"[i] Event timeout: {MOTION_TIMEOUT}s of stillness")

    # Start keyboard thread
    t = threading.Thread(target=_kb_listener, daemon=True)
    t.start()
    
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "event_id","sample_num","ts","door_id","mount_pos","sample_hz",
            "ax","ay","az","gx","gy","gz","a_mag","w_mag","label"
        ])

        try:
            while _running and t.is_alive():
                # Always READ from sensor
                ax = _read_word(bus, ACCEL_XOUT_H)   / ACCEL_SCALE
                ay = _read_word(bus, ACCEL_XOUT_H+2) / ACCEL_SCALE
                az = _read_word(bus, ACCEL_XOUT_H+4) / ACCEL_SCALE
                gx = _read_word(bus, GYRO_XOUT_H)    / GYRO_SCALE
                gy = _read_word(bus, GYRO_XOUT_H+2)  / GYRO_SCALE
                gz = _read_word(bus, GYRO_XOUT_H+4)  / GYRO_SCALE

                a_mag = sqrt(ax*ax + ay*ay + az*az)
                w_mag = sqrt(gx*gx + gy*gy + gz*gz)
                
                ts = time.time()
                sample = [ts, ax, ay, az, gx, gy, gz, a_mag, w_mag]

                # Always maintain ring buffer (for pre-motion capture)
                ring_buffer.append(sample)
                if len(ring_buffer) > buffer_size:
                    ring_buffer.pop(0)

                # State machine for event capture
                if _is_armed and not capturing_event:
                    # Waiting for motion to start
                    if w_mag > MOTION_THRESHOLD:
                        # Motion detected! Start capturing
                        capturing_event = True
                        _is_armed = False
                        
                        # Include pre-motion buffer
                        event_data = ring_buffer.copy()
                        
                        print(f"\n[🔴 RECORDING] Motion detected! Gyro={w_mag:.1f} °/s")
                        print("              Keep moving door until motion stops...")
                        stillness_counter = 0
                
                elif capturing_event:
                    # Currently capturing an event
                    event_data.append(sample)
                    
                    if w_mag < MOTION_THRESHOLD:
                        stillness_counter += 1
                    else:
                        stillness_counter = 0
                    
                    # Check if motion has ended
                    if stillness_counter >= stillness_samples:
                        # Event complete! Save to file
                        event_id = f"{_current_label}_{_event_count[_current_label]+1:03d}"
                        
                        for i, (ts, ax, ay, az, gx, gy, gz, a_mag, w_mag) in enumerate(event_data):
                            w.writerow([
                                event_id, i, f"{ts:.6f}", 
                                DOOR_ID, MOUNT_POS, SAMPLE_HZ,
                                f"{ax:.6f}", f"{ay:.6f}", f"{az:.6f}",
                                f"{gx:.3f}", f"{gy:.3f}", f"{gz:.3f}",
                                f"{a_mag:.6f}", f"{w_mag:.6f}", _current_label
                            ])
                        
                        _event_count[_current_label] += 1
                        
                        # Calculate total angular displacement (integrate gyro)
                        # Using dominant axis (you'll need to identify which one)
                        gyro_x_sum = sum(s[4] for s in event_data)  # gx
                        total_angle = abs(gyro_x_sum) / SAMPLE_HZ
                        
                        print("\n" + "="*70)
                        print(f"✅ EVENT CAPTURED: {_current_label.upper()} #{_event_count[_current_label]}")
                        print("="*70)
                        print(f"   Samples: {len(event_data)} | Duration: {len(event_data)/SAMPLE_HZ:.2f}s | Est. angle: ~{total_angle:.1f}°")
                        print(f"   Total events: opening={_event_count['opening']}, closing={_event_count['closing']}")
                        print("="*70)
                        print("\n>>> READY FOR NEXT EVENT - Press 'o' or 'c' <<<\n")
                        
                        # Reset for next event
                        capturing_event = False
                        event_data = []
                        stillness_counter = 0
                        _current_label = None

                # Precise pacing
                next_t += period
                sleep = next_t - time.monotonic()
                if sleep > 0: 
                    time.sleep(sleep)
                else: 
                    next_t = time.monotonic()

        except KeyboardInterrupt:
            print("\n[!] Interrupted by user")
        finally:
            bus.close()
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print("\n" + "="*70)
    print("RECORDING COMPLETE")
    print("="*70)
    print(f"File saved: {csv_path}")
    print(f"Total events collected:")
    print(f"  - Opening events: {_event_count['opening']}")
    print(f"  - Closing events: {_event_count['closing']}")
    print(f"  - Total: {_event_count['opening'] + _event_count['closing']}")
    print(f"\nRecommended: Collect at least 20-30 events of each type")
    print("="*70)

if __name__ == "__main__":
    main()