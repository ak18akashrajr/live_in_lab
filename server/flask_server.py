from flask import Flask, request, jsonify
from drone.drone_singleton import Drone
from drone.drone_data import DroneData
from dronekit import Vehicle
import requests

drone: Vehicle = Drone.get_vechile(sim=True)
app = Flask(__name__)
Model = None
drone_data_obj = None


@app.route("/get_data")
def get_data():
    ml_predict = Model.get_json_data()
    data = drone_data_obj.get_drone_data()
    data["ml_prediction"] = ml_predict
    return jsonify(data)


def start_server(model):
    global Model
    Model = model
    global drone_data_obj
    drone_data_obj = DroneData(drone)
    app.run(port=5000, use_reloader=False)
