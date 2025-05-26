import subprocess
import time
import signal
CSV_PATH = "false_mission.csv"
CHECK_INTERVAL = 120
MISSION_DURATION = 600
def get_line_count(filepath):
    try:
        with open(filepath, "r") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return -1
while True :
    process = subprocess.Popen(["python3", "false_flight.py"])
    start_time = time.time()
    last_line_count = get_line_count(CSV_PATH)
    while time.time() - start_time < MISSION_DURATION:
        time.sleep(CHECK_INTERVAL)

        current_line_count = get_line_count(CSV_PATH)
        if current_line_count == last_line_count:
            print("Aucune nouvelle ligne détectée dans le CSV depuis 1m30. Interruption.")
            process.send_signal(signal.SIGINT)
            break
        last_line_count = get_line_count(CSV_PATH)
    process.send_signal(signal.SIGINT)