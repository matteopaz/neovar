from datetime import datetime
from time import sleep
import os
import pandas as pd

UPDATE_DELAY_S = 2.5

start_t = float(open("./logs/start_time.txt").read())

TOTAL_ROWS = 188876840852

def count_rows_done():
    n = 0
    with open("./logs/progress.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if len(line.strip().split("_")) < 4:
                continue
            try:
                apparitions = int(line.strip().split("_")[2])
            except:
                continue
            n += apparitions
    return n

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

    # clear the output
    print("\033[H\033[J")
    print(f"{pct_done:.2f}% done. {rows_done} rows done. Estimated time left: {time_left_hrs:.2f} hours. Estimated end time: {est_end_datetime}.")
    sleep(UPDATE_DELAY_S)

