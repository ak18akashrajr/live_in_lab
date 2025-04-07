# Simulation setup for Drone

## Install Required Dependencies

Run the below command to install all the required dependencies

```bash
pip install mavproxy
pip install dronekit
pip install dronekit-sitl
```

## Start the COPTER

Run the following command to create a copter simulation. It takes slightly longer time for the first simulation to be created, much faster after the first simulation

```bash
dronekit-sitl copter
```

Also note the TCP port where the simulation runs. example port: **127.0.0.1:5760**

## Convert the TCP port to UDP port

Generally the copter is simulated using TCP, but most of the Drone implementations are expect it in UDP. Run the following command to convert the TCP port to UDP port. Replace the **TCP port** from the above command.

```bash
mavproxy.py --master tcp:127.0.0.1:5760 --out 127.0.0.1:14550
```

The simulator will created successfully. Now we can use dronekit to run the simulation.
