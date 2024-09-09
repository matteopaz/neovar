import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

# Galactic Center
# partitiontbl = pd.read_parquet("/home/mpaz/neowise-clustering/clustering/out/partition_7207_cluster_id_to_data.parquet")
# Orion
# partitiontbl = pd.read_parquet("/home/mpaz/neowise-clustering/clustering/out/partition_5359_cluster_id_to_data.parquet")
# Bootes
# partitiontbl = pd.read_parquet("/home/mpaz/neowise-clustering/clustering/out/partition_2186_cluster_id_to_data.parquet")
# partitiontbl = pd.read_parquet("/home/mpaz/neowise-clustering/clustering/out/partition_0_cluster_id_to_data.parquet")

# partitions = [0, 5359, 2186, 10, 12, 11512, 9090, 4354, 29, 18]
partitions = [20]

partitiontbl = pd.concat([pd.read_parquet(f"/home/mpaz/neowise-clustering/clustering/out/partition_{p}_cluster_id_to_data.parquet") for p in partitions])

tomag = lambda x: -2.5 * np.log10(0.00000154851985514 * x / 309.54)



partitiontbl["medianw1flux"] = partitiontbl["w1flux"].apply(np.mean)
partitiontbl["w1mpro"] = partitiontbl["medianw1flux"].apply(tomag)

partitiontbl = partitiontbl[partitiontbl["w1mpro"] < 16.5]
partitiontbl.dropna(subset=["w1flux", "w1sigflux"], inplace=True)

partitiontbl["medianw1sigflux"] = partitiontbl["w1sigflux"].apply(np.median)

temp = partitiontbl["medianw1flux"] + partitiontbl["medianw1sigflux"]
partitiontbl["w1sigmpro"] = -temp.apply(tomag) + partitiontbl["medianw1flux"].apply(tomag)

partitiontbl = partitiontbl[partitiontbl["w1sigmpro"] < 0.6]

# partitiontbl["w1sigflux"] = partitiontbl["w1sigflux"].apply(lambda x: np.where(x < 0, 0, x))
# partitiontbl["w1score"] = (partitiontbl["w1sigflux"] - partitiontbl["medianw1sigflux"]) / partitiontbl["medianw1sigflux"]

y=np.array(partitiontbl["w1mpro"].values)
x=np.array(partitiontbl["w1sigmpro"].values)

plt.plot(x, y, 'r.')
# 80% opacity and make the points smaller
plt.plot(x, y, 'r.', alpha=0.8, markersize=0.5)
plt.xlabel("W1 Uncertainty")
plt.ylabel("W1 Magnitude")
plt.title("W1 Uncertainty vs. W1 Magnitude Universally")
plt.savefig("w1tosig.png")


# scores = partitiontbl["w1score"].values
# scores = [item for sublist in scores for item in sublist]
# scores = np.array(scores)
# pickle.dump(scores, open("w1scores.pkl", "wb"))
# scores = pickle.load(open("w1scores.pkl", "rb"))
# get rid of outliers
# scores = np.log(scores)
# scores = scores[~np.isnan(scores)]
# scores = scores[~np.isinf(scores)]
# scores = scores[(scores > -1.0) & (scores < 2.0)]
# scores = scores - np.min(scores) + 1e-6

# plot the histogram, with y as a probability density
# plt.hist(scores, bins=100, density=True)
# plt.title("W1 Score Distribution")
# plt.xlabel("W1 Score")
# plt.ylabel("Probability Density")



# from scipy.stats import gamma 

# fit a gamma distribution to the data with MLE
# mu, var = np.mean(scores), np.var(scores)
# alpha = mu**2 / var
# beta = mu / var
# loc = 1
# alpha, loc, beta = gamma.fit(scores, loc=0.85, scale=0.115, f0=1.7)
# print(alpha, loc, beta)
# x = np.linspace(0, 5, 500)
# pdf_fitted = gamma.pdf(x, 1.7, loc=self.s([0.785, 0.9]), scale=self.s([0.1, 0.2]))

# pdf_fitted = gamma.pdf(x, alpha, loc, beta)
# plt.plot(x, pdf_fitted, 'r-')


# plt.savefig("w1score.png")
# fit a lognormal distribution to the data with MLE
# print(mu, sigma)