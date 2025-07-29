#!/usr/bin/env python3
"""
FT232H ULN2003 Driver Controller with Socket Communication

This program connects to an FT232H via USB and controls a ULN2003 Darlington 
transistor array driver. Now includes socket server for inter-process communication.

Requirements:
- pyftdi library: pip install pyftdi
- FT232H board connected via USB
- ULN2003 driver IC connected to FT232H GPIO pins

Commands supported over socket:
- open: Move 30 degrees clockwise
- close: Move 30 degrees counter-clockwise
- move <degrees> [cw/ccw]: Move specific degrees
- stop: Stop all outputs
- status: Get current status
- quit: Shutdown server
"""

import time
import sys
import socket
import threading
import json
from pyftdi.ftdi import Ftdi
from pyftdi.gpio import GpioAsyncController

class FT232H_ULN2003_Controller:
    def __init__(self, url='ftdi://ftdi:232h/1'):
        """
        Initialize the FT232H controller
        
        Args:
            url: FTDI device URL (default works for most single FT232H setups)
        """
        self.url = url
        self.gpio = None
        self.pins_mask = 0xFF  # Use all 8 pins (D0-D7) as outputs
        self.connected = False
        self.current_position = 0.0  # Track current position in degrees
        self.motor_busy = False
        self.lock = threading.Lock()  # Thread safety for motor operations
        
    def connect(self):
        """Connect to the FT232H device"""
        try:
            self.gpio = GpioAsyncController()
            self.gpio.configure(self.url, direction=self.pins_mask)
            self.connected = True
            print(f"Successfully connected to FT232H at {self.url}")
            return True
        except Exception as e:
            print(f"Failed to connect to FT232H: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the FT232H device"""
        if self.gpio:
            self.gpio.close()
            self.connected = False
            print("Disconnected from FT232H")
    
    def set_outputs(self, value):
        """
        Set the output state of all ULN2003 inputs
        
        Args:
            value: 8-bit value (0-255) representing the state of outputs
                   Bit 0 = ULN2003 IN1, Bit 1 = IN2, etc.
        """
        if not self.gpio or not self.connected:
            print("Error: Not connected to FT232H")
            return False
            
        try:
            self.gpio.write(value)
            return True
        except Exception as e:
            print(f"Error setting outputs: {e}")
            return False
    
    def stepper_motor_degrees(self, degrees, direction='cw', delay=0.00075, motor_type='28BYJ-48'):
        """
        Move stepper motor by a specific number of degrees
        
        Args:
            degrees: Number of degrees to move (float)
            direction: 'cw' for clockwise, 'ccw' for counter-clockwise
            delay: Delay between steps in seconds
            motor_type: Type of stepper motor ('28BYJ-48', 'NEMA17', 'custom')
        
        Returns:
            bool: True if successful, False otherwise
        """
        with self.lock:
            if self.motor_busy:
                print("Motor is busy, please wait...")
                return False
            
            self.motor_busy = True
        
        try:
            # Motor specifications (steps per revolution)
            motor_specs = {
                '28BYJ-48': {
                    'steps_per_rev': 4096,  # With 64:1 gear ratio in half-step mode
                    'step_angle': 360.0 / 4096,
                    'description': '28BYJ-48 with ULN2003 (half-step mode)'
                },
                'NEMA17': {
                    'steps_per_rev': 3200,  # 1.8° step angle, 1/16 microstepping
                    'step_angle': 360.0 / 3200,
                    'description': 'NEMA17 (1.8° step, 1/16 microstep)'
                }
            }
            
            if motor_type not in motor_specs:
                print(f"Unknown motor type: {motor_type}")
                return False
            
            motor = motor_specs[motor_type]
            
            # Calculate number of steps needed
            steps_needed = int(abs(degrees) / motor['step_angle'])
            actual_degrees = steps_needed * motor['step_angle']
            
            print(f"Moving {actual_degrees}° {direction} ({steps_needed} steps)")
            
            # Determine direction (negative degrees reverse the direction)
            if degrees < 0:
                direction = 'ccw' if direction == 'cw' else 'cw'
            
            # Execute the movement
            result = self.stepper_motor_step(steps_needed, direction, delay)
            
            # Update position tracking
            if result:
                if direction == 'cw':
                    self.current_position += actual_degrees
                else:
                    self.current_position -= actual_degrees
                
                # Keep position within 0-360 range
                self.current_position = self.current_position % 360
            
            return result
            
        finally:
            self.motor_busy = False
    
    def stepper_motor_step(self, steps=1, direction='cw', delay=0.00075):
        """
        Control a stepper motor connected to ULN2003
        Assumes 4-pin stepper motor connected to pins 0-3
        
        Args:
            steps: Number of steps to move
            direction: 'cw' for clockwise, 'ccw' for counter-clockwise
            delay: Delay between steps in seconds
        """
        if not self.connected:
            print("Error: Not connected to FT232H")
            return False
            
        # 28BYJ-48 stepper motor sequence (half-step mode)
        step_sequence = [
            0b0001,  # 1000
            0b0011,  # 1100
            0b0010,  # 0100
            0b0110,  # 0110
            0b0100,  # 0010
            0b1100,  # 1010
            0b1000,  # 1000
            0b1001   # 1001
        ]
        
        if direction == 'ccw':
            step_sequence.reverse()
        
        for step in range(steps):
            for pattern in step_sequence:
                if not self.set_outputs(pattern):
                    return False
                time.sleep(delay)
        
        # Turn off all outputs after stepping
        self.set_outputs(0)
        return True
    
    def get_status(self):
        """Get current motor status"""
        return {
            'connected': self.connected,
            'position': self.current_position,
            'busy': self.motor_busy
        }
    
    def stop_motor(self):
        """Emergency stop - turn off all outputs"""
        self.set_outputs(0)
        return True

class MotorServer:
    def __init__(self, host='localhost', port=8888):
        """
        Initialize the motor server
        
        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.controller = FT232H_ULN2003_Controller()
        self.server_socket = None
        self.running = False
        
    def start_server(self):
        """Start the motor control server"""
        if not self.controller.connect():
            print("Failed to connect to motor controller")
            return False
        
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            print(f"Motor server started on {self.host}:{self.port}")
            print("Waiting for connections...")
            
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"Connection from {address}")
                    
                    self.handle_client(client_socket)
                    # Handle client in a separate thread
                    # client_thread = threading.Thread(
                    #     target=self.handle_client,
                    #     args=(client_socket,)
                    # )
                    # client_thread.daemon = True
                    # client_thread.start()
                    
                except socket.error:
                    if self.running:
                        print("Socket error occurred")
                    break
                    
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            self.cleanup()
    
    def handle_client(self, client_socket):
        """Handle individual client connections"""
        try:
            while self.running:
                # Receive command from client
                data = client_socket.recv(1024).decode('utf-8').strip()
                if not data:
                    break
                
                print(f"Received command: {data}")
                
                # Process command
                response = self.process_command(data)
                
                # Send response back to client
                client_socket.send(json.dumps(response).encode('utf-8'))
                
        except Exception as e:
            print(f"Client handling error: {e}")
        finally:
            client_socket.close()
    
    def process_command(self, command):
        """
        Process incoming commands
        
        Args:
            command: Command string from client
            
        Returns:
            dict: Response dictionary
        """
        try:
            parts = command.lower().split()
            cmd = parts[0]
            
            if cmd == 'open':
                # Move 30 degrees clockwise
                success = self.controller.stepper_motor_degrees(30, 'cw')
                return {
                    'success': success,
                    'message': 'Moved 30 degrees clockwise' if success else 'Failed to move',
                    'position': self.controller.current_position
                }
            
            elif cmd == 'close':
                # Move 30 degrees counter-clockwise
                success = self.controller.stepper_motor_degrees(30, 'ccw')
                return {
                    'success': success,
                    'message': 'Moved 30 degrees counter-clockwise' if success else 'Failed to move',
                    'position': self.controller.current_position
                }
            
            elif cmd == 'move' and len(parts) >= 2:
                # Move specific degrees: move <degrees> [cw/ccw]
                degrees = float(parts[1])
                direction = parts[2] if len(parts) > 2 else 'cw'
                
                if direction not in ['cw', 'ccw']:
                    return {'success': False, 'message': 'Invalid direction. Use cw or ccw'}
                
                success = self.controller.stepper_motor_degrees(degrees, direction)
                return {
                    'success': success,
                    'message': f'Moved {degrees} degrees {direction}' if success else 'Failed to move',
                    'position': self.controller.current_position
                }
            
            elif cmd == 'stop':
                # Stop motor and turn off outputs
                self.controller.stop_motor()
                return {
                    'success': True,
                    'message': 'Motor stopped',
                    'position': self.controller.current_position
                }
            
            elif cmd == 'status':
                # Get current status
                status = self.controller.get_status()
                return {
                    'success': True,
                    'message': 'Status retrieved',
                    'status': status
                }
            
            elif cmd == 'quit':
                # Shutdown server
                self.running = False
                return {
                    'success': True,
                    'message': 'Server shutting down'
                }
            
            else:
                return {
                    'success': False,
                    'message': f'Unknown command: {command}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Command processing error: {str(e)}'
            }
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if self.controller:
            self.controller.stop_motor()  # Turn off all outputs
            self.controller.disconnect()
        print("Server cleanup complete")

def main():
    """Main program"""
    print("FT232H ULN2003 Motor Control Server")
    print("=" * 40)
    
    # Check for available FTDI devices
    try:
        devices = Ftdi.list_devices()
        if not devices:
            print("No FTDI devices found. Please connect an FT232H device.")
            return
        print(f"Found {len(devices)} FTDI device(s)")
    except Exception as e:
        print(f"Error checking FTDI devices: {e}")
        return
    
    # Create and start server
    server = MotorServer()
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\nServer interrupted by user")
    finally:
        server.cleanup()

if __name__ == "__main__":
    main()