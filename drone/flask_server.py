from flask import Flask, request, jsonify
from drone.utils import Drone
from dronekit import Vehicle

drone: Vehicle = Drone.get_vechile(sim=True)
app = Flask(__name__)


@app.route("/get_data", method=["GET", "POST"])
def get_data():
    pass


def start_server():
    app.run()
