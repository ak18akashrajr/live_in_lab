from drone.utils import Drone
from dronekit import Vehicle

drone: Vehicle = Drone.get_vechile(sim=True)
