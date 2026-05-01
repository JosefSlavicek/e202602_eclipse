import serial
import time

PORT = '/dev/ttyUSB0' # or your COM port

try:
    # 1. Open the port. 
    # With the new wiring, the LED should stay OFF.
    ser = serial.Serial(PORT, 9600)
    
    # 2. Set DTR to the state that keeps the LED OFF
    # Try True first; if it stays off, you're good.
    ser.setDTR(True) 
    print("System Stable - LED should be OFF")
    time.sleep(1)

    # 3. Fire the Shutter
    print("CLICK!")
    ser.setDTR(False) # This creates the voltage difference to light the LED
    
    ser.close()
except Exception as e:
    print(f"Error: {e}")
