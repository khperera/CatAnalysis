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
import os
import sys

import threading
import app.IntegratedMotorControl as MotorControl

class StateController:
    def __init__(self):
        """
        Initialize the motor client
        
        Args:
            host: Server host address
            port: Server port number
        """
          
        self.shitlist = {"bunt": 0, "tot":0}

        #Can go up to 500, will close the door if above 50
        self.BadGuyTimer = 0
        self.BadGuyTimerMax = 500
        self.BadGuyTimerThreshold = 50

        self.MachineOn = False

        self.GateOpen = False

        self.ForceGateOpen = False
        
        self.motorControl = MotorControl.MotorClient()

        

        Disconnected = True
        

        while(Disconnected):
            Disconnected = not self.motorControl.connect()
            if(not Disconnected):
                print("Connected!")
                continue
            print("Not connected. Trying again in 5 seconds.")

            time.sleep(5)


    def AddCat(self, list_name):
        #Can go up to 500, will close the door if above 50
        if("bunt" in list_name or "tot" in list_name):
            self.BadGuyTimer += 1

    def OpenValve(self):
        if(self.GateOpen):
            return
        self.motorControl.open_valve()
        

        pass

    def CloseValve(self):
        if(not self.GateOpen):
            return

        self.motorControl.close_valve()



        
    def timer_for_machine(self):
        if(self.MachineOn):

            return
        
        self.MachineOn = True

        while(self.MachineOn):

             


            time.sleep(0.200)



            










    

    
        
