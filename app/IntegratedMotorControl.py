#!/usr/bin/env python3
"""
Motor Control Client

This script demonstrates how to send commands to the motor control server.
It can be used as a standalone script or imported as a module.

Commands:
- open: Move 30 degrees clockwise
- close: Move 30 degrees counter-clockwise
- move <degrees> [cw/ccw]: Move specific degrees
- stop: Stop all outputs
- status: Get current status
- quit: Shutdown server
"""

import socket
import json
import time
import sys

class MotorClient:
    def __init__(self, host='localhost', port=8888):
        """
        Initialize the motor client
        
        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
    
    def connect(self):
        """Connect to the motor server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to motor server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to motor server: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the motor server"""
        if self.socket:
            self.socket.close()
            self.connected = False
            print("Disconnected from motor server")
    
    def send_command(self, command):
        """
        Send a command to the motor server
        
        Args:
            command: Command string to send
            
        Returns:
            dict: Response from server, or None if error
        """
        if not self.connected:
            print("Not connected to server")
            return None
        
        try:
            # Send command
            self.socket.send(command.encode('utf-8'))
            
            # Receive response
            response = self.socket.recv(1024).decode('utf-8')
            return json.loads(response)
            
        except Exception as e:
            print(f"Communication error: {e}")
            return None
    
    def open_valve(self):
        """Open valve (30 degrees clockwise)"""
        return self.send_command('open')
    
    def close_valve(self):
        """Close valve (30 degrees counter-clockwise)"""
        return self.send_command('close')
    
    def move_degrees(self, degrees, direction='cw'):
        """
        Move motor by specific degrees
        
        Args:
            degrees: Number of degrees to move
            direction: 'cw' for clockwise, 'ccw' for counter-clockwise
        """
        return self.send_command(f'move {degrees} {direction}')
    
    def stop_motor(self):
        """Stop motor and turn off outputs"""
        return self.send_command('stop')
    
    def get_status(self):
        """Get current motor status"""
        return self.send_command('status')
    
    def shutdown_server(self):
        """Shutdown the motor server"""
        return self.send_command('quit')

def demo_sequence():
    """Demonstrate basic motor control commands"""
    print("Motor Control Client Demo")
    print("=" * 30)
    
    # Create client and connect
    client = MotorClient()
    if not client.connect():
        return
    
    try:
        # Get initial status
        print("\n1. Getting initial status...")
        response = client.get_status()
        if response and response['success']:
            print(f"Status: {response['status']}")
        else:
            print("Failed to get status")
        
        # Open valve
        print("\n2. Opening valve (30° CW)...")
        response = client.open_valve()
        if response:
            print(f"Response: {response['message']}")
            if response['success']:
                print(f"Current position: {response['position']:.2f}°")
        
        time.sleep(2)
        
        # Close valve
        print("\n3. Closing valve (30° CCW)...")
        response = client.close_valve()
        if response:
            print(f"Response: {response['message']}")
            if response['success']:
                print(f"Current position: {response['position']:.2f}°")
        
        time.sleep(2)
        
        # Custom movement
        print("\n4. Custom movement (45° CW)...")
        response = client.move_degrees(45, 'cw')
        if response:
            print(f"Response: {response['message']}")
            if response['success']:
                print(f"Current position: {response['position']:.2f}°")
        
        time.sleep(2)
        
        # Return to start
        print("\n5. Returning to start (45° CCW)...")
        response = client.move_degrees(45, 'ccw')
        if response:
            print(f"Response: {response['message']}")
            if response['success']:
                print(f"Current position: {response['position']:.2f}°")
        
        # Final status
        print("\n6. Getting final status...")
        response = client.get_status()
        if response and response['success']:
            print(f"Final status: {response['status']}")
        
        # Stop motor
        print("\n7. Stopping motor...")
        response = client.stop_motor()
        if response:
            print(f"Response: {response['message']}")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        client.disconnect()

def interactive_mode():
    """Interactive command mode"""
    print("Motor Control Client - Interactive Mode")
    print("=" * 40)
    
    client = MotorClient()
    if not client.connect():
        return
    
    print("\nAvailable commands:")
    print("  open          - Move 30° clockwise")
    print("  close         - Move 30° counter-clockwise")
    print("  move <deg>    - Move specific degrees clockwise")
    print("  move <deg> ccw - Move specific degrees counter-clockwise")
    print("  stop          - Stop motor")
    print("  status        - Get current status")
    print("  quit          - Exit client")
    print("  shutdown      - Shutdown server")
    
    try:
        while True:
            command = input("\nEnter command: ").strip()
            
            if command.lower() == 'quit':
                break
            elif command.lower() == 'shutdown':
                response = client.shutdown_server()
                if response:
                    print(f"Response: {response['message']}")
                break
            elif command == '':
                continue
            else:
                response = client.send_command(command)
                if response:
                    print(f"Success: {response['success']}")
                    print(f"Message: {response['message']}")
                    if 'position' in response:
                        print(f"Position: {response['position']:.2f}°")
                    if 'status' in response:
                        print(f"Status: {response['status']}")
                else:
                    print("No response received")
    
    except KeyboardInterrupt:
        print("\nInteractive mode interrupted")
    finally:
        client.disconnect()

def main():
    """Main program"""
    if len(sys.argv) > 1:
        if sys.argv[1] == 'demo':
            demo_sequence()
        elif sys.argv[1] == 'interactive':
            interactive_mode()
        else:
            print("Usage: python motor_client.py [demo|interactive]")
    else:
        # Default to interactive mode
        interactive_mode()

# Example usage as a module
def example_usage():
    """Example of how to use this as a module in your own code"""
    
    # Create and connect client
    client = MotorClient()
    if not client.connect():
        return
    
    try:
        # Your motor control logic here
        print("Opening valve...")
        response = client.open_valve()
        if response and response['success']:
            print("Valve opened successfully")
        
        # Wait for some process
        time.sleep(5)
        
        print("Closing valve...")
        response = client.close_valve()
        if response and response['success']:
            print("Valve closed successfully")
        
        # Check status
        status = client.get_status()
        if status and status['success']:
            print(f"Motor status: {status['status']}")
        
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()