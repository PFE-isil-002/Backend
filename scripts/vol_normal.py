import airsim
import time
import csv
import math
import random
import os
import sys
import json
from datetime import datetime

# Ensure real-time output
sys.stdout.reconfigure(line_buffering=True)

def distance(p1, p2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))

waypoints = [
    (0, 0, -5),
    (5, 0, -5),
    (10, 5, -5),
    (14, 5, -5),
    (18, 5, -5),
    (22, 5, -5),
    (25, 5, -5),
    (28, 5, -5),
    (31, 5, -5),
    (35, 5, -5),
    (38, 5, -5),
    (41, 5, -5),
    (43, 5, -5),
    (46, 5, -5),
    (48, 5, -5),
    (50, 5, -5),
    (55, 5, -5),
    (50, 5, -5),
    (40, 5, -5),
    (35, 5, -5),
    (30, 5, -5),
    
]

SEUIL_ERREUR = 1.5
NB_MISSIONS = 4
BATTERY_THRESHOLD = 20
BATTERY_DRAIN_PER_MISSION = 50
battery_level = 100

client = airsim.MultirotorClient(ip="172.17.112.1")
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

start_state = client.getMultirotorState().kinematics_estimated.position
start_position = (start_state.x_val, start_state.y_val, start_state.z_val)

client.simEnableWeather(True)
base_wind = airsim.Vector3r(3, 2, 0.5)

file_exists = os.path.isfile("donnees_vol_multiple.csv")
with open("donnees_vol_multiple.csv", "a", newline='') as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "mission_id", "time",
            "target_x", "target_y", "target_z",
            "real_x", "real_y", "real_z",
            "vx", "vy", "vz",
            "ax", "ay", "az",
            "roll", "pitch", "yaw",
            "wind_x", "wind_y", "wind_z",
            "erreur_m", "erreur_ok",
            "velocity", "erreur_cumulee_m",
            "battery_level", "rain", "fog", "snow"
        ])

    for mission_id in range(1, NB_MISSIONS + 1):
        print(f"\n Batterie virtuelle avant Mission {mission_id}: {battery_level}%")

        if battery_level < BATTERY_THRESHOLD:
            print(f" Batterie faible ({battery_level}%). Recharge en cours...")
            battery_level = 100
            print(f" Batterie rechargée à {battery_level}%.")

            client.reset()
            time.sleep(2)
            client.enableApiControl(True)
            client.armDisarm(True)
            client.takeoffAsync().join()

            start_state = client.getMultirotorState().kinematics_estimated.position
            start_position = (start_state.x_val, start_state.y_val, start_state.z_val)

        rain = round(random.uniform(0, 0.2), 2)
        fog = round(random.uniform(0, 0.2), 3)
        snow = round(random.uniform(0, 0.2), 3)

        client.simSetWeatherParameter(airsim.WeatherParameter.Rain, rain)
        client.simSetWeatherParameter(airsim.WeatherParameter.Fog, fog)
        client.simSetWeatherParameter(airsim.WeatherParameter.Snow, snow)

        velocity = round(random.uniform(3, 10), 2)
        print(f" Météo : Rain={rain}, Fog={fog}, Snow={snow}")
        print(f"\n MISSION {mission_id}/{NB_MISSIONS} (vitesse = {velocity} m/s)")

        erreur_cumulee = 0.0
        erreurs_liste = []

        for target in waypoints:
            noise = airsim.Vector3r(
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(-0.1, 0.1)
            )
            wind = airsim.Vector3r(
                base_wind.x_val + noise.x_val,
                base_wind.y_val + noise.y_val,
                base_wind.z_val + noise.z_val
            )
            client.simSetWind(wind)

            x, y, z = target
            client.moveToPositionAsync(x, y, z, velocity).join()
            time.sleep(0.5)

            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            acc = client.getImuData().linear_acceleration
            orient = airsim.to_eularian_angles(state.kinematics_estimated.orientation)

            real_pos = (pos.x_val, pos.y_val, pos.z_val)
            err = distance(real_pos, (x, y, z))
            erreur_ok = err < SEUIL_ERREUR

            erreur_cumulee += err
            erreurs_liste.append(err)

            writer.writerow([
                mission_id, time.time(),
                x, y, z,
                real_pos[0], real_pos[1], real_pos[2],
                vel.x_val, vel.y_val, vel.z_val,
                acc.x_val, acc.y_val, acc.z_val,
                orient[0], orient[1], orient[2],
                wind.x_val, wind.y_val, wind.z_val,
                err, erreur_ok,
                velocity, erreur_cumulee,
                battery_level, rain, fog, snow
            ])

            # ✅ Print real-time telemetry as JSON for WebSocket streaming
            telemetry = {
                "position": {"x": real_pos[0], "y": real_pos[1], "z": real_pos[2]},
                "timestamp": datetime.now().isoformat(),
                "velocity": {"x": vel.x_val, "y": vel.y_val, "z": vel.z_val},
                "orientation": {"roll": orient[0], "pitch": orient[1], "yaw": orient[2]},
                "battery": battery_level,
                "signal_strength": 100.0,
                "packet_loss": 0.0,
                "latency": 0.0
            }
            print(json.dumps(telemetry))  # 👈 sent to WebSocket in real-time

            print(f"[Mission {mission_id}]  Target: {target}, Erreur: {err:.2f} m,  Erreur cumulée: {erreur_cumulee:.2f} m")

        moyenne = erreur_cumulee / len(erreurs_liste)
        print(f"📉 Moyenne d’erreur de la mission {mission_id}: {moyenne:.2f} m")

        battery_level -= BATTERY_DRAIN_PER_MISSION
        if battery_level < 0:
            battery_level = 0

        print(f" Batterie restante après Mission {mission_id}: {battery_level}%")
        print(" Retour au point de départ...")
        client.moveToPositionAsync(*start_position, 3).join()
        time.sleep(1)

print("\n Atterrissage...")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print(" Toutes les missions sont terminées.")
