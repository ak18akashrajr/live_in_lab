import dronekit as dk
from dronekit import Vehicle


class DroneData:
    def __init__(self, drone: Vehicle):
        self.drone = drone

    def get_drone_data(self):
        return get_drone_data(self.drone)


def get_drone_data(vehicle):
    return {
        "battery": {
            "current": vehicle.battery.current or 0,
            "percentage": vehicle.battery.level or 0,
            "voltage": vehicle.battery.voltage or 0
        },
        "is_armable": vehicle.is_armable,
        "last_heartbeat": vehicle.last_heartbeat,
        "location": {
            "altitude": vehicle.location.global_frame.alt,
            "latitude": vehicle.location.global_frame.lat,
            "longitude": vehicle.location.global_frame.lon
        },
        "mode": vehicle.mode.name,
        "sat": {
            "fix_type": getattr(vehicle.gps_0, "fix_type", 0),
            "num_of_sat": getattr(vehicle.gps_0, "satellites_visible", 0)
        },
        "system_status": vehicle.system_status.state
    }
