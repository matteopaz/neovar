import os
ids = set([int(s.split("_")[1][:-4]) for s in os.listdir("out") if s.endswith(".csv")])
print(set(range(12288)) - ids)

print(len(ids))
print(len(ids) / 122.88, ' %')