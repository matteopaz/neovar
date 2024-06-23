from datetime import datetime

TOTAL_N_ROWS = 188876840852

with open("./logs/progress.txt", "w") as f:
    f.write("")

with open("./logs/errors.txt", "w") as f:
    f.write("")

with open("./logs/masterlog.txt", "w") as f:
    f.write("")

with open("./logs/start_time.txt", "w") as f:
    f.write(str(int(datetime.now().timestamp())))