from datetime import datetime
from time import sleep

UPDATE_DELAY_S = 1

start_t = float(open("./logs/start_time.txt").read())

TOTAL_ROWS = 188876840852

def count_rows_done():
    n = 0
    with open("./logs/progress.txt") as f:
        lines = f.readlines()
        for line in lines:
            apparitions = int(line.strip().split("_")[2])
            n += apparitions
    return n

def current_partitions_in_progress():
    inprog = []
    with open("./logs/progress.txt") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split("_")
            pid = int(parts[0])
            status = int(parts[1])
            if status == 1:
                inprog.append(pid)
    return inprog

while True:
    rows_done = count_rows_done()
    now_t = datetime.now().timestamp()
    elapsed = now_t - start_t
    rate = rows_done / elapsed

    left_to_do = TOTAL_ROWS - rows_done
    time_left = left_to_do / rate
    time_left_hrs = time_left / 3600

    est_end_datetime = datetime.fromtimestamp(now_t + time_left)
    pct_done = rows_done / TOTAL_ROWS * 100

    inprog = current_partitions_in_progress()

    # clear the output
    print("\033[H\033[J")
    print(f"{pct_done:.2f}% done. {rows_done} rows done. Estimated time left: {time_left_hrs:.2f} hours. Estimated end time: {est_end_datetime}.")
    print(f"Partitions in progress: {inprog}")
    sleep(UPDATE_DELAY_S)

