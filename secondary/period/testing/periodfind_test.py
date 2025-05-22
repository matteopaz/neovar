import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from periodfind import ls, ce, aov
from time import time
import matplotlib.pyplot as plt

# VARIABLES
N_PERIODS = 100000
DATAFILE = "./TEST_OBJECT.csv"
N_OBJECTS_SIMULATED = 100

# Defining constants and acquiring object data
trialpds = np.linspace(0.5, 50, N_PERIODS, dtype=np.float32)**2

object_data = pd.read_csv(DATAFILE)
object_t = np.array(object_data["mjd"].values, dtype=np.float32)
object_m = np.array(object_data["w1mpro"].values, dtype=np.float32)

batch_t = [object_t for _ in range(N_OBJECTS_SIMULATED)]
batch_m = [object_m for _ in range(N_OBJECTS_SIMULATED)]

args = {
    "times": batch_t,
    "mags": batch_m,
    "periods": trialpds,
    "period_dts": np.array([0.0], dtype=np.float32),
    "n_stats": N_PERIODS,
}


# Running period analysis

print("Start")
pgram = aov.AOV(n_phase=10)
t1 = time()
out = pgram.calc(**args)
t2 = time()

pgram_periods = np.array([a.params[0] for a in out[0]])
pgram_values = np.array([a.significance for a in out[0]])
sorter = np.argsort(pgram_periods)
pgram_periods = pgram_periods[sorter]
pgram_values = pgram_values[sorter]

print("Total time: ", t2-t1)
print("Rate: ", (t2-t1)/N_OBJECTS_SIMULATED, " seconds per object")
print("Op. Rate: ", (10e9*(t2-t1))/(N_OBJECTS_SIMULATED*N_PERIODS), " ns / operation")

# side by side
fig, axs = plt.subplots(2, 1)
axs[0].scatter(object_t, object_m, s=1)
axs[1].plot(pgram_periods[pgram_periods < 1], pgram_values[pgram_periods < 1])
fig.savefig("test.png")
plt.close(fig)