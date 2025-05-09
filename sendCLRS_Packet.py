#send a CRSF packet over serial to the expressLRS

import sys
import math
from serial.serialutil import SerialException
from serial import Serial as PySerial
import time
import struct
import serial.tools.list_ports
import threading

from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server


UDP_IP = "127.0.0.1"
UDP_PORT = 5632

POLY = 0xD5 

# Global variables for RC values
PITCH = 1500
ROLL = 1500
YAW = 1500
THROTTLE = 1000
ARM = 2000

# Configure the serial port
port = 'COM5'
baudrate = 921600

def list_available_ports():
    """List all available COM ports"""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No COM ports found!")
    else:
        print("\nAvailable COM ports:")
        for port in ports:
            print(f"- {port.device}: {port.description}")

def us_to_crsf(us_val):
    us_val = max(988, min(2012, us_val))
    return int((us_val - 988) * (1811 - 172) / (2012 - 988) + 172)

def pack_channels(channels):
    buf = bytearray(22)
    bitbuf = 0
    bits = 0
    idx = 0
    for ch in channels:
        bitbuf |= (ch & 0x7FF) << bits
        bits += 11
        while bits >= 8:
            buf[idx] = bitbuf & 0xFF
            bitbuf >>= 8
            bits -= 8
            idx += 1
    return buf

def crc8(data, poly=POLY):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc

def build_crsf_packet(payload_type, payload):
    frame = bytearray()
    frame.append(0xC8)             # Address: Flight Controller
    frame.append(len(payload) + 2) # Payload length + type
    frame.append(payload_type)     # Type: 0x16 = RC_CHANNELS_PACKED
    frame.extend(payload)
    crc = crc8(frame[2:])
    frame.append(crc)
    return frame

def send_rc_packet(ser, pitch=1500, roll=1500, yaw=1500, throttle=1000):
    """
    Create and send a CRSF RC packet
    """
    # print(f"Pitch: {pitch}, Roll: {roll}, Yaw: {yaw}, Throttle: {throttle}")
    
    # Define channel values in microseconds
    channels_us = [
        roll,    # Roll
        pitch,   # Pitch
        yaw,     # Yaw
        throttle, # Throttle
        ARM,    # AUX1 (e.g., arm switch) # setting this to 2000 will arm the drone
        *[1000]*11  # AUX2–AUX12
    ]
    
    # Convert to CRSF values
    channels_crsf = [us_to_crsf(v) for v in channels_us]

    # Build packet
    payload = pack_channels(channels_crsf)
    packet = build_crsf_packet(0x16, payload)
    
    # Print debug info
    print(f"Sending packet (len={len(packet)}): {[hex(x) for x in packet]}")
    # print("CRSF RC_CHANNELS_PACKED Packet:")
    # print(" ".join(f"{byte:02X}" for byte in packet))
    
    ser.write(packet)

def arm_handler(unused_addr, arm):
   ARM = arm
#    print(f"Received OSC message: ARM={ARM}")

def RC_UDP_handler(unused_addr, Roll, Pitch, Yaw, Throttle, Arm):
    """Handle incoming OSC messages for RC control"""
    global PITCH, ROLL, YAW, THROTTLE, ARM
    print(f"Received OSC message: Roll={Roll}, Pitch={Pitch}, Yaw={Yaw}, Throttle={Throttle}, Arm={Arm}")
    PITCH = Pitch
    ROLL = Roll
    YAW = Yaw
    THROTTLE = Throttle
    ARM = Arm

def start_osc_server():
    """Start the OSC server in a separate thread"""
    dispatcher = Dispatcher()
    dispatcher.map("/RC", RC_UDP_handler)
    #dispatcher.map("/arm", arm_handler)
    
    server = osc_server.ThreadingOSCUDPServer((UDP_IP, UDP_PORT), dispatcher)
    print(f"OSC Server listening on {UDP_IP}:{UDP_PORT}")
    
    # Start the server in a thread
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True  # Thread will close when main program exits
    server_thread.start()
    return server

def main():
    # First list available ports
    list_available_ports()
    print(f"\nAttempting to open {port} at {baudrate} baud...")

    try:
        # Open the serial port
        ser = PySerial(port, baudrate, timeout=1)
        print(f"Successfully opened {port}")

        # Start OSC server
        osc_server = start_osc_server()

        # Main loop for sending CRSF packets
        try:
            while True:
                # print(f"ARM: {ARM}")
                send_rc_packet(ser, PITCH, ROLL, YAW, THROTTLE)
                time.sleep(0.02)  # 50Hz update rate
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            if ser.is_open:
                ser.close()
                print(f"Closed {port}")
                
    except SerialException as e:
        print(f"Error opening {port}: {str(e)}")
        print("Please check if the port is correct and available.")
        sys.exit(1)

if __name__ == "__main__":
    main()

