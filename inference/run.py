from run_partition import run_partition

partitions_to_do = range(12888)
partitions_to_do = [0]

for partition_id in partitions_to_do:
    subcatalog = run_partition(partition_id)
    subcatalog.to_csv(f"out/partition_{partition_id}_subcatalog.csv", index=False)
