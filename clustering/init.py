from datetime import datetime
import json
import os

TOTAL_N_ROWS = 188876840852

N_JOBS = 10

def unfinished_partitions():
    directory = os.fsencode("/home/mpaz/neowise-clustering/clustering/out/")

    partition_ids = set(list(range(12288)))
        
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            parts = filename.split("_")
            pid = int(parts[1])
            partition_ids.remove(pid)
    return partition_ids

parts_to_do = unfinished_partitions()

assignments = [[] for _ in range(N_JOBS)]
for i, part in enumerate(parts_to_do):
    assignments[i % N_JOBS].append(part)

with open("./assignments.json", "w") as f:
    json.dump(assignments, f)

with open("./logs/progress.txt", "w") as f:
    f.write("")

with open("./logs/errors.txt", "w") as f:
    f.write("")

with open("./logs/masterlog.txt", "w") as f:
    f.write("")

with open("./logs/start_time.txt", "w") as f:
    f.write(str(int(datetime.now().timestamp())))