from datetime import datetime
import json
import os
import re

TOTAL_N_ROWS = 188876840852

N_JOBS = 11

polars = [2559, 8450, 9730, 10041, 2901, 9376, 9738, 10060, 2903, 9378, 9779, 10096, 3839, 9379, 9818, 4010, 9384, 9856, 4011, 9385, 9884, 8276, 9386, 9920, 8277, 9387, 9941, 8448, 9643, 9977, 8449, 9728, 10004]

def unfinished_partitions():
    directory = os.fsencode("/home/mpaz/neowise-clustering/clustering/out/")
    partition_ids = set(list(range(12288)))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        # check if matches regex
        match = re.match(r"partition_._cntr_to_cluster_id_map.csv", filename)

        if match:
            parts = filename.split("_")
            pid = int(parts[1])
            partition_ids.remove(pid)
    return partition_ids

parts_to_do = unfinished_partitions()

assignments = [[] for _ in range(N_JOBS)]
for i, part in enumerate(parts_to_do):
    assignments[i % N_JOBS].append(part)

for assignmentlist in assignments: # place polar partitions to be done at the very end
    for polarid in polars:
        if polarid in assignmentlist:
            assignmentlist.remove(polarid)
            assignmentlist.append(polarid)

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