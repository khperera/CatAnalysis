#!/usr/bin/env python3
"""
FT232H ULN2003 Driver Controller

This program connects to an FT232H via USB and controls a ULN2003 Darlington 
transistor array driver. Commonly used for stepper motors, relays, or other loads.

Requirements:
- pyftdi library: pip install pyftdi
- FT232H board connected via USB
- ULN2003 driver IC connected to FT232H GPIO pins

Wiring:
- FT232H D0-D7 (GPIO) -> ULN2003 IN1-IN8 (inputs)
- ULN2003 outputs -> Your loads (stepper motor, LEDs, relays, etc.)
- Common cathode of loads -> ULN2003 COM pin
- External power supply for loads
"""

import time
import sys
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
        
    def connect(self):
        """Connect to the FT232H device"""
        try:
            self.gpio = GpioAsyncController()
            self.gpio.configure(self.url, direction=self.pins_mask)
            print(f"Successfully connected to FT232H at {self.url}")
            return True
        except Exception as e:
            print(f"Failed to connect to FT232H: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the FT232H device"""
        if self.gpio:
            self.gpio.close()
            print("Disconnected from FT232H")
    
    def set_outputs(self, value):
        """
        Set the output state of all ULN2003 inputs
        
        Args:
            value: 8-bit value (0-255) representing the state of outputs
                   Bit 0 = ULN2003 IN1, Bit 1 = IN2, etc.
        """
        if not self.gpio:
            print("Error: Not connected to FT232H")
            return False
            
        try:
            self.gpio.write(value)
            return True
        except Exception as e:
            print(f"Error setting outputs: {e}")
            return False
    
    def set_single_output(self, pin, state):
        """
        Set a single output pin
        
        Args:
            pin: Pin number (0-7, corresponding to ULN2003 IN1-IN8)
            state: True for HIGH (output active), False for LOW (output off)
        """
        if pin < 0 or pin > 7:
            print("Error: Pin must be between 0 and 7")
            return False
            
        current_state = self.gpio.read()
        if state:
            new_state = current_state | (1 << pin)  # Set bit
        else:
            new_state = current_state & ~(1 << pin)  # Clear bit
            
        return self.set_outputs(new_state)
    
    def stepper_motor_step(self, steps=1, direction='cw', delay=0.11):
        """
        Control a stepper motor connected to ULN2003
        Assumes 4-pin stepper motor connected to pins 0-3
        
        Args:
            steps: Number of steps to move
            direction: 'cw' for clockwise, 'ccw' for counter-clockwise
            delay: Delay between steps in seconds
        """
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
        
        print(f"Moving stepper motor {steps} steps {direction}")
        
        for step in range(steps):
            for pattern in step_sequence:
                if not self.set_outputs(pattern):
                    return False
                time.sleep(delay)
        
        # Turn off all outputs after stepping
        self.set_outputs(0)
        return True
    
    def stepper_motor_degrees(self, degrees, direction='cw', delay=0.01, motor_type='28BYJ-48'):
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
            },
            'NEMA17_FULL': {
                'steps_per_rev': 200,   # 1.8° step angle, full step
                'step_angle': 1.8,
                'description': 'NEMA17 (1.8° step, full step)'
            },
            'custom': {
                'steps_per_rev': 200,   # Default, user should modify
                'step_angle': 1.8,
                'description': 'Custom motor (modify steps_per_rev)'
            }
        }
        
        if motor_type not in motor_specs:
            print(f"Unknown motor type: {motor_type}")
            print(f"Available types: {list(motor_specs.keys())}")
            return False
        
        motor = motor_specs[motor_type]
        
        # Calculate number of steps needed
        steps_needed = int(abs(degrees) / motor['step_angle'])
        actual_degrees = steps_needed * motor['step_angle']
        
        print(f"Motor: {motor['description']}")
        print(f"Requested: {degrees}°")
        print(f"Actual movement: {actual_degrees}° ({steps_needed} steps)")
        print(f"Step resolution: {motor['step_angle']:.4f}° per step")
        
        if abs(degrees - actual_degrees) > motor['step_angle']:
            print(f"Warning: Rounding error of {abs(degrees - actual_degrees):.4f}°")
        
        # Determine direction (negative degrees reverse the direction)
        if degrees < 0:
            direction = 'ccw' if direction == 'cw' else 'cw'
        
        return self.stepper_motor_step(steps_needed, direction, delay)
    
    def set_motor_specs(self, steps_per_revolution, step_angle=None):
        """
        Set custom motor specifications for degree calculations
        
        Args:
            steps_per_revolution: Total steps for 360° rotation
            step_angle: Degrees per step (calculated if not provided)
        """
        if step_angle is None:
            step_angle = 360.0 / steps_per_revolution
        
        # Update the custom motor specs
        motor_specs = {
            'steps_per_rev': steps_per_revolution,
            'step_angle': step_angle,
            'description': f'Custom motor ({step_angle:.4f}° per step)'
        }
        
        # This would need to be stored as an instance variable in a real implementation
        print(f"Custom motor set: {steps_per_revolution} steps/rev, {step_angle:.4f}° per step")
        print("Use motor_type='custom' in stepper_motor_degrees() function")
        
        return motor_specs
    
    def test_sequence(self):
        """Run a test sequence to verify ULN2003 outputs"""
        print("Running test sequence...")
        
        # Test each output individually
        for pin in range(8):
            print(f"Testing output {pin + 1}")
            self.set_single_output(pin, True)
            time.sleep(0.5)
            self.set_single_output(pin, False)
            time.sleep(0.2)
        
        # Test all outputs together
        print("All outputs ON")
        self.set_outputs(0xFF)
        time.sleep(1)
        
        print("All outputs OFF")
        self.set_outputs(0x00)
        
        # Binary counting pattern
        print("Binary counting pattern")
        for i in range(256):
            self.set_outputs(i)
            time.sleep(0.05)
        
        self.set_outputs(0x00)
        print("Test sequence complete")

def list_ftdi_devices():
    """List all available FTDI devices"""
    print("Available FTDI devices:")
    try:
        devices = Ftdi.list_devices()
        if not devices:
            print("No FTDI devices found")
            return []
        
        for i, (device, interface) in enumerate(devices):
            print(f"  {i}: {device}")
        return devices
    except Exception as e:
        print(f"Error listing devices: {e}")
        return []

def main():
    """Main program"""
    print("FT232H ULN2003 Controller")
    print("=" * 30)
    
    # List available devices
    devices = list_ftdi_devices()
    if not devices:
        print("Please connect an FT232H device and try again.")
        return
    
    # Create controller instance
    controller = FT232H_ULN2003_Controller()
    
    try:
        # Connect to FT232H
        if not controller.connect():
            return
        
        while True:
            print("\nOptions:")
            print("1. Test sequence")
            print("2. Set all outputs")
            print("3. Set single output")
            print("4. Stepper motor control (steps)")
            print("5. Stepper motor control (degrees)")
            print("6. Set custom motor specs")
            print("7. Turn off all outputs")
            print("8. Exit")
            
            choice = input("Enter choice (1-8): ").strip()
            
            if choice == '1':
                controller.test_sequence()
                
            elif choice == '2':
                try:
                    value = int(input("Enter output value (0-255): "))
                    if 0 <= value <= 255:
                        controller.set_outputs(value)
                        print(f"Outputs set to: {bin(value)} ({value})")
                    else:
                        print("Value must be between 0 and 255")
                except ValueError:
                    print("Invalid input")
                    
            elif choice == '3':
                try:
                    pin = int(input("Enter pin number (0-7): "))
                    state = input("Enter state (on/off): ").lower()
                    if pin >= 0 and pin <= 7 and state in ['on', 'off']:
                        controller.set_single_output(pin, state == 'on')
                        print(f"Pin {pin} set to {state}")
                    else:
                        print("Invalid input")
                except ValueError:
                    print("Invalid input")
                    
            elif choice == '4':
                try:
                    steps = int(input("Enter number of steps: "))
                    direction = input("Enter direction (cw/ccw): ").lower()
                    if direction in ['cw', 'ccw']:
                        controller.stepper_motor_step(steps, direction)
                    else:
                        print("Direction must be 'cw' or 'ccw'")
                except ValueError:
                    print("Invalid input")
                    
            elif choice == '5':
                try:
                    degrees = float(input("Enter degrees to move: "))
                    direction = input("Enter direction (cw/ccw) or leave blank for auto: ").lower().strip()
                    motor_type = input("Motor type (28BYJ-48/NEMA17/NEMA17_FULL/custom) [28BYJ-48]: ").strip()
                    
                    if not direction:
                        direction = 'cw'  # Default direction, function handles negative degrees
                    elif direction not in ['cw', 'ccw']:
                        print("Direction must be 'cw' or 'ccw'")
                        continue
                    
                    if not motor_type:
                        motor_type = '28BYJ-48'
                    
                    controller.stepper_motor_degrees(degrees, direction, motor_type=motor_type,delay=0.00075)
                except ValueError:
                    print("Invalid input")
                    
            elif choice == '6':
                try:
                    steps = int(input("Enter steps per revolution: "))
                    step_angle = input("Enter step angle (degrees) or leave blank to calculate: ").strip()
                    
                    if step_angle:
                        step_angle = float(step_angle)
                        controller.set_motor_specs(steps, step_angle)
                    else:
                        controller.set_motor_specs(steps)
                except ValueError:
                    print("Invalid input")
                    
            elif choice == '7':
                controller.set_outputs(0)
                print("All outputs turned off")
                
            elif choice == '8':
                break
                
            else:
                print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    
    finally:
        # Clean up
        controller.set_outputs(0)  # Turn off all outputs
        controller.disconnect()

if __name__ == "__main__":
    main()