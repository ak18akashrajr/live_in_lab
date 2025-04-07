import dronekit as dk
from threading import Lock

"""
Create a singleton **instance** of Dronekit for Drone Controller and Flask Server.
"""


class Drone:
    _vechicle = None
    _lock = Lock()

    @classmethod
    def get_vechile(self, sim=False, connection_string=""):
        with Drone._lock:
            if Drone._vechicle is None:
                if sim:
                    con_string = '127.0.0.1:14551'
                else:
                    con_string = connection_string

                vehicle = dk.connect(
                    con_string, wait_ready=True, vehicle_class=dk.Vehicle)
                Drone._vechicle = vehicle

            return Drone._vechicle
