#!/usr/bin/env python3
# imu_data_logger.py
import os, csv, time, sys, threading, pathlib
from math import sqrt
from datetime import datetime
from smbus2 import SMBus

# ========= Session config (edit these for each recording) =========
DOOR_ID    = "doorA"     # e.g., doorA, doorB
MOUNT_POS  = "pos1"      # e.g., pos1 (3–5" from hinge, mid height)
SAMPLE_HZ  = 100         # 50–100 Hz is good
OUT_DIR    = "sessions"  # CSV output directory
# ================================================================

# I2C / MPU6050 constants
I2C_BUS      = 1
MPU_ADDR     = 0x68
PWR_MGMT_1   = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H  = 0x43
ACCEL_SCALE  = 16384.0   # LSB/g, ±2g
GYRO_SCALE   = 131.0     # LSB/(deg/s), ±250 dps

# Label state (keyboard controlled)
_current_label = "idle"
_running = True

def _kb_listener():
    """Single-key label toggles: i=idle, o=open, c=close, q=quit"""
    global _current_label, _running
    print("[keys] i=idle, o=open, c=close, q=quit")
    while _running:
        ch = sys.stdin.read(1)
        if ch == 'i':
            _current_label = "idle";  print("[label] idle")
        elif ch == 'o':
            _current_label = "open";  print("[label] open")
        elif ch == 'c':
            _current_label = "close"; print("[label] close")
        elif ch == 'q':
            _running = False
            print("[i] quitting…")

def _read_word(bus, reg):
    hi = bus.read_byte_data(MPU_ADDR, reg)
    lo = bus.read_byte_data(MPU_ADDR, reg+1)
    v  = (hi << 8) | lo
    return v - 65536 if v >= 32768 else v

def main():
    global _running
    pathlib.Path(OUT_DIR).mkdir(exist_ok=True)
    fname    = f"{DOOR_ID}_{MOUNT_POS}_{int(time.time())}.csv"
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

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ts","door_id","mount_pos","sample_hz",
            "ax","ay","az","gx","gy","gz","a_mag","w_mag","label"
        ])
        print(f"[i] logging → {csv_path} at {SAMPLE_HZ} Hz")
        print("[i] keep door closed & still ~5–10s to capture baseline…")

        # Start keyboard thread
        t = threading.Thread(target=_kb_listener, daemon=True)
        t.start()

        try:
            while _running and t.is_alive():
                ax = _read_word(bus, ACCEL_XOUT_H)   / ACCEL_SCALE
                ay = _read_word(bus, ACCEL_XOUT_H+2) / ACCEL_SCALE
                az = _read_word(bus, ACCEL_XOUT_H+4) / ACCEL_SCALE
                gx = _read_word(bus, GYRO_XOUT_H)    / GYRO_SCALE
                gy = _read_word(bus, GYRO_XOUT_H+2)  / GYRO_SCALE
                gz = _read_word(bus, GYRO_XOUT_H+4)  / GYRO_SCALE

                a_mag = sqrt(ax*ax + ay*ay + az*az)
                w_mag = sqrt(gx*gx + gy*gy + gz*gz)

                w.writerow([
                    f"{time.time():.6f}", DOOR_ID, MOUNT_POS, SAMPLE_HZ,
                    f"{ax:.6f}", f"{ay:.6f}", f"{az:.6f}",
                    f"{gx:.3f}", f"{gy:.3f}", f"{gz:.3f}",
                    f"{a_mag:.6f}", f"{w_mag:.6f}", _current_label
                ])

                # precise pacing
                next_t += period
                sleep = next_t - time.monotonic()
                if sleep > 0: time.sleep(sleep)
                else: next_t = time.monotonic()

        except KeyboardInterrupt:
            pass
        finally:
            bus.close()
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print("[i] done.")

if __name__ == "__main__":
    main()
