from threading import Thread
from drone import Drone
from disaster_management import DisasterManagement
from forest_survey import ForestSurvey
from port_survey import PortSurvey
from server import start_server
import time

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Drone Setup @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
drone = Drone.get_vechile(sim=True)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Disaster Management @@@@@@@@@@@@@@@@@@@@@@
disaster_management = DisasterManagement("yolov8n.pt")

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ START Server @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
start_server(disaster_management)
