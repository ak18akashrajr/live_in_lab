import dronekit as dk
from dronekit import Vehicle
import time


class Controller:
    def __init__(self, drone: Vehicle):
        self.drone = drone
        self.drone.mode = dk.VehicleMode("GUIDED")

    def arm(self):
        if not self.drone.is_armable:
            print("The can't be armed")
            return 404

        self.drone.arm(wait=True, timeout=10)
        return 200

    def takeoff(self, altitude):
        '''
        Args :
            altitude : float (meters)
        '''
        altitude = float(altitude)
        self.drone.simple_takeoff(altitude)

    def arm_and_takeoff(self, altitude):
        altitude = float(altitude)
        self.arm()
        time.sleep(0.1)
        self.takeoff(altitude)

    def goto(self, lat, long, alt=10):
        location = dk.LocationGlobalRelative(lat, long, alt)
        self.drone.simple_goto(location)

    def return_to_land(self):
        self.drone.mode = dk.VehicleMode("RTL")
        time.sleep(0.1)
        print("Returning to Launch")

    def land(self):
        self.drone.mode = dk.VehicleMode("LAND")
        time.sleep(0.1)
        print("Returning to Land")
