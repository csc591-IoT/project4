#!/usr/bin/env python3
# imu_test.py - Simple IMU placement verification tool
import time
from math import sqrt
from smbus2 import SMBus

# I2C / MPU6050 constants
I2C_BUS      = 1
MPU_ADDR     = 0x68
PWR_MGMT_1   = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H  = 0x43
ACCEL_SCALE  = 16384.0   # LSB/g, ±2g
GYRO_SCALE   = 131.0     # LSB/(deg/s), ±250 dps

def read_word(bus, reg):
    """Read 16-bit signed value from IMU register"""
    hi = bus.read_byte_data(MPU_ADDR, reg)
    lo = bus.read_byte_data(MPU_ADDR, reg+1)
    v  = (hi << 8) | lo
    return v - 65536 if v >= 32768 else v

def clear_screen():
    """Clear terminal screen"""
    print("\033[2J\033[H", end="")

def main():
    print("Initializing IMU...")
    
    # Init I2C and wake up MPU6050
    bus = SMBus(I2C_BUS)
    bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)
    time.sleep(0.1)
    
    print("IMU ready! Starting live display...\n")
    time.sleep(1)
    
    try:
        while True:
            # Read accelerometer (in g's)
            ax = read_word(bus, ACCEL_XOUT_H)   / ACCEL_SCALE
            ay = read_word(bus, ACCEL_XOUT_H+2) / ACCEL_SCALE
            az = read_word(bus, ACCEL_XOUT_H+4) / ACCEL_SCALE
            
            # Read gyroscope (in deg/s)
            gx = read_word(bus, GYRO_XOUT_H)    / GYRO_SCALE
            gy = read_word(bus, GYRO_XOUT_H+2)  / GYRO_SCALE
            gz = read_word(bus, GYRO_XOUT_H+4)  / GYRO_SCALE
            
            # Calculate magnitudes
            a_mag = sqrt(ax*ax + ay*ay + az*az)
            w_mag = sqrt(gx*gx + gy*gy + gz*gz)
            
            # Clear screen and display
            clear_screen()
            
            print("="*70)
            print("IMU SENSOR TEST - Real-time Display".center(70))
            print("="*70)
            print("\nACCELEROMETER (linear acceleration in g):")
            print(f"  X: {ax:7.3f} g  {'█' * int(abs(ax)*20)}")
            print(f"  Y: {ay:7.3f} g  {'█' * int(abs(ay)*20)}")
            print(f"  Z: {az:7.3f} g  {'█' * int(abs(az)*20)}")
            print(f"  Magnitude: {a_mag:.3f} g")
            
            print("\nGYROSCOPE (angular velocity in deg/s):")
            print(f"  X: {gx:7.1f} °/s  {'█' * int(abs(gx)/10)}")
            print(f"  Y: {gy:7.1f} °/s  {'█' * int(abs(gy)/10)}")
            print(f"  Z: {gz:7.1f} °/s  {'█' * int(abs(gz)/10)}")
            print(f"  Magnitude: {w_mag:.1f} °/s")
            
            print("\n" + "="*70)
            print("WHAT TO LOOK FOR:")
            print("="*70)
            print("1. STATIC (door not moving):")
            print("   - Accel should read ~1.0g on ONE axis (gravity)")
            print("   - Gyro should be near 0 on all axes")
            print("\n2. OPENING/CLOSING DOOR:")
            print("   - ONE gyro axis should spike significantly (rotation)")
            print("   - Accel should change on 1-2 axes (centripetal force)")
            print("\n3. GOOD PLACEMENT:")
            print("   - Strong, clear signals when door moves")
            print("   - Gyro spikes > 50 °/s during normal door movement")
            print("   - If signals are weak, move sensor closer to door edge")
            print("="*70)
            print("Press Ctrl+C to exit")
            
            time.sleep(0.1)  # Update 10 times per second
            
    except KeyboardInterrupt:
        print("\n\nTest stopped by user.")
        print("\n✓ IMU is working if you saw changing values!")
        print("✓ Good placement if you saw strong gyro spikes when moving door")
    finally:
        bus.close()

if __name__ == "__main__":
    main()