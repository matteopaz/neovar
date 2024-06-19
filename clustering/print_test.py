import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--partition_id", type=int)
args = parser.parse_args()



print("Hello. Partition ID is: ", args.partition_id)