import os
import matplotlib.pyplot as plt
import numpy as np

parts = set()

sizes = []
for f in os.listdir("cached"):
    s = f[:-8]
    # remove all letters
    s = "".join([x for x in s if not x.isalpha()])
    parts = parts.union(set([int(x) for x in s.split("_")]))
    sizes.append(os.path.getsize(f"cached/{f}") / (1024))

print(set(range(12288)) - parts)
print(len(os.listdir("cached")))

print(np.max(sizes))
print(np.mean(sizes))
print(np.std(sizes))
print(np.min(sizes))

plt.hist(sizes, bins=25)
plt.savefig("sizes.png")