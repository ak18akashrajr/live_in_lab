from drone import Drone, Controller
from dronekit import Vehicle
import time

drone: Vehicle = Drone.get_vechile(sim=True)
controller: Controller = Controller(drone)

commands = ["arm", "takeoff", "armandtakeoff", "goto", "rtl", "land", "exit"]

while True:
    command = input("Enter command: ")
    if command == "exit":
        break
    elif command == "arm":
        controller.arm()

    elif command == "takeoff":
        alt = float(input("Enter the altitude(meters): "))
        controller.takeoff(alt)

    elif command == "armandtakeoff":
        alt = float(input("Enter the altitude(meters): "))
        controller.arm_and_takeoff(alt)

    elif command == "goto":
        lat = float(input("Enter latitude  : "))
        lon = float(input("Enter longitude : "))

    elif command == "rtl":
        controller.return_to_land()

    elif command == "land":
        controller.land()

    else:
        print('Invalid command')

    time.sleep(1)
