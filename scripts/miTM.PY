import airsim
import time
import csv
import math
import random
import os
import sys
import json
from datetime import datetime

# Force stdout to flush immediately for real-time streaming
sys.stdout.reconfigure(line_buffering=True)

def distance(p1, p2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))

def generate_trajectory_with_mitm(n_points=21, step_size=2):
    trajectory = []
    x, y, z = 0, 0, -5
    for i in range(n_points):
        # MITM attack simulation: random GPS spoof every few waypoints
        if i % random.randint(5, 10) == 0 and i != 0:
            spoof = random.uniform(-5, 5)
            trajectory.append((x + spoof, y + spoof, z + spoof))
        else:
            trajectory.append((x, y, z))
        x += step_size
    return trajectory

client = airsim.MultirotorClient(ip="172.17.112.1")
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

start_state = client.getMultirotorState().kinematics_estimated.position
start_position = (start_state.x_val, start_state.y_val, start_state.z_val)

file_exists = os.path.isfile("donnees_mitm.csv")
with open("donnees_mitm.csv", "a", newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "mission_id", "time",
            "target_x", "target_y", "target_z",
            "real_x", "real_y", "real_z",
            "vx", "vy", "vz",
            "roll", "pitch", "yaw",
            "battery", "spoofed"
        ])

    for mission_id in range(1, 6):
        print(f"🚨 MITM Mission {mission_id} starting")
        trajectory = generate_trajectory_with_mitm()

        for idx, target in enumerate(trajectory):
            x, y, z = target
            spoofed = False

            if idx % 7 == 0 and idx != 0:
                # simulate delay/spoof effect
                x += random.uniform(-3, 3)
                y += random.uniform(-3, 3)
                z += random.uniform(-1, 1)
                spoofed = True

            client.moveToPositionAsync(x, y, z, 5).join()
            time.sleep(0.3)

            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            orient = airsim.to_eularian_angles(state.kinematics_estimated.orientation)

            telemetry = {
                "position": {"x": pos.x_val, "y": pos.y_val, "z": pos.z_val},
                "timestamp": datetime.now().isoformat(),
                "velocity": {"x": vel.x_val, "y": vel.y_val, "z": vel.z_val},
                "orientation": {"roll": orient[0], "pitch": orient[1], "yaw": orient[2]},
                "battery": 100.0,
                "signal_strength": 95.0,
                "packet_loss": random.uniform(0, 1) if spoofed else 0.0,
                "latency": random.uniform(10, 200) if spoofed else 30.0
            }

            # JSON output for FastAPI WebSocket parsing
            print(json.dumps(telemetry))

            writer.writerow([
                mission_id, time.time(),
                x, y, z,
                pos.x_val, pos.y_val, pos.z_val,
                vel.x_val, vel.y_val, vel.z_val,
                orient[0], orient[1], orient[2],
                100.0, spoofed
            ])

            print(f"[{mission_id}] Target: {x:.2f}, {y:.2f}, {z:.2f} | Spoofed: {spoofed}")

        # Return to home
        client.moveToPositionAsync(*start_position, 3).join()
        time.sleep(1)

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print("✅ MITM simulation completed.")
