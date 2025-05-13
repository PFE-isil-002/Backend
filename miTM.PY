import airsim
import time
import csv
import math
import random
import os

def distance(p1, p2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))
def generate_fixed_trajectory( n_points=21, step_size=1):
    trajectory = []
    x, y, z = 0, 0, -5
    style = random.choice(["ligne_droite", "spirale", "zigzag", "montante", "descendante"])
    if style == "ligne_droite":
        for _ in range(n_points):
            trajectory.append((x, y, z))
            x += step_size

    elif style == "spirale":
        for i in range(n_points):
            angle = i * 2 * math.pi / n_points
            radius = 2 + 0.3 * i
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = -5 + 0.05 * i  # monte doucement
            trajectory.append((x, y, z))

    elif style == "descendante":
        for _ in range(n_points):
            trajectory.append((x, y, z))
            x += step_size
            z += 0.1  # descend

    elif style == "montante":
        for _ in range(n_points):
            trajectory.append((x, y, z))
            x += step_size
            z -= 0.1  # monte

    elif style == "zigzag":
        for i in range(n_points):
            y = step_size if i % 2 == 0 else -step_size
            trajectory.append((x, y, z))
            x += step_size

    else:
        raise ValueError("Style inconnu. Choisissez entre : ligne_droite, spirale, descendante, montante, zigzag.")

    return trajectory
SEUIL_ERREUR = 1.5
NB_MISSIONS = 100
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
csv_file = "donnees_vol_mitm_variable.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, "a", newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "mission_id", "step", "time",
            "target_x", "target_y", "target_z",
            "real_x", "real_y", "real_z",
            "vx", "vy", "vz",
            "ax", "ay", "az",
            "roll", "pitch", "yaw",
            "wind_x", "wind_y", "wind_z",
            "erreur_m", "is_attack",
            "velocity", "erreur_cumulee_m",
            "battery_level", "rain", "fog", "snow"
        ])

    for mission_id in range(1, NB_MISSIONS + 1):
        print(f"\n🔋 Batterie virtuelle avant Mission {mission_id}: {battery_level}%")

        if battery_level < BATTERY_THRESHOLD:
            print(f"⚠️ Batterie faible ({battery_level}%). Recharge en cours...")
            battery_level = 100
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
        velocity = round(random.uniform(3, 7), 2)

        client.simSetWeatherParameter(airsim.WeatherParameter.Rain, rain)
        client.simSetWeatherParameter(airsim.WeatherParameter.Fog, fog)
        client.simSetWeatherParameter(airsim.WeatherParameter.Snow, snow)

        erreur_cumulee = 0.0
        is_attack = 0  # ✅ Toutes les missions ici sont MITM
        erreurs_liste = []
        step=0
        waypoints= generate_fixed_trajectory()
        # 🔀 Attaque MITM : position et durée aléatoire
        mitm_start = random.randint(5, len(waypoints) - 10)
        mitm_duration = random.randint(1,5)
        mitm_end = mitm_start + mitm_duration
        print(f"\n🚀 MISSION {mission_id} | Vitesse = {velocity} m/s | MITM de step {mitm_start} à {mitm_end}")
        print(f"🌦️ Météo : Rain={rain}, Fog={fog}, Snow={snow}")
        for target in waypoints:
            # 💥 Injection MITM sur certains steps
            if mitm_start <= step < mitm_end:
                print(f"⚠️ MITM active à l'étape {step}")
                target = (
                    target[0] + random.uniform(5, 12),
                    target[1] + random.uniform(5, 10),
                    target[2] + random.uniform(1,4)
                )
                print(f"🎯 Target modifié : {target}")

            # Vent bruité
            noise = airsim.Vector3r(
                random.uniform(-0.2, 0.2),
                random.uniform(-0.2, 0.2),
                random.uniform(-0.1, 0.1)
            )
            wind = airsim.Vector3r(
                base_wind.x_val + noise.x_val,
                base_wind.y_val + noise.y_val,
                base_wind.z_val + noise.z_val
            )
            client.simSetWind(wind)

            x, y, z = target
            client.moveToPositionAsync(x, y, z, velocity)

            # Attente réelle d’arrivée (tolérance)
            start_time = time.time()
            while True:
                pos = client.getMultirotorState().kinematics_estimated.position
                if distance((pos.x_val, pos.y_val, pos.z_val), (x, y, z)) < 20:
                    break
                time.sleep(1)

            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            acc = client.getImuData().linear_acceleration
            orient = airsim.to_eularian_angles(state.kinematics_estimated.orientation)

            real_pos = (pos.x_val, pos.y_val, pos.z_val)
            err = distance(real_pos, (x, y, z))
            erreur_cumulee += err
            erreurs_liste.append(err)

            writer.writerow([
                mission_id, step, time.time(),
                x, y, z,
                real_pos[0], real_pos[1], real_pos[2],
                vel.x_val, vel.y_val, vel.z_val,
                acc.x_val, acc.y_val, acc.z_val,
                orient[0], orient[1], orient[2],
                wind.x_val, wind.y_val, wind.z_val,
                err, is_attack,
                velocity, erreur_cumulee,
                battery_level, rain, fog, snow
            ])

            print(f"[Mission {mission_id}] Step {step+1}/{len(waypoints)} 🎯 Target={target} | Erreur={err:.2f} m")
            step=step+1
        moyenne = erreur_cumulee / len(erreurs_liste)
        print(f"📉 Moyenne d’erreur : {moyenne:.2f} m")

        battery_level -= BATTERY_DRAIN_PER_MISSION
        battery_level = max(0, battery_level)
        print(f"🔋 Batterie restante : {battery_level}%")

        print("🔁 Retour au point de départ...")
        client.moveToPositionAsync(*start_position, 3, timeout_sec=10).join()
        time.sleep(1)
print("\n🛬 Atterrissage...")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print("✅ Toutes les missions MITM sont terminées.")