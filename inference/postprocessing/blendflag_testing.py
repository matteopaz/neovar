import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture as GaussianMixture
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg
import itertools
import time
import pickle as pkl


color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)

# TESTING
blend = pd.read_csv('blend1.csv')

radec_blend = np.array(blend[['ra', 'dec']])
# radec_blend = pkl.load(open('blends.pkl', 'rb'))


# radec_blend[:, 0] = radec_blend[:, 0] * np.cos(np.radians(radec_blend[:, 1]))

radec_blend = radec_blend - np.mean(radec_blend, axis=0)
radec_blend = radec_blend * 3600

# rescale ra, dec to [0, 1]

def separate_models(twocomp):
    means = twocomp.means_
    covariance = twocomp.covariances_
    precision = twocomp.precisions_cholesky_
    weights = twocomp.weights_

    m1 = GaussianMixture(n_components=1, covariance_type='tied')
    m2 = GaussianMixture(n_components=1, covariance_type='tied')
    m1.means_ = np.array([means[0]])
    m2.means_ = np.array([means[1]])
    m1.covariances_ = covariance
    m2.covariances_ = covariance
    m1.precisions_cholesky_ = precision
    m2.precisions_cholesky_ = precision
    m1.weights_ = np.array([weights[0]])
    m2.weights_ = np.array([weights[1]])
    return m1, m2


# Fit a Gaussian Mixture Model to the data
t1 = time.time()
gmm_blend_one = GaussianMixture(n_components=1).fit(radec_blend)
gmm_blend_two = GaussianMixture(n_components=2, covariance_type='tied').fit(radec_blend)
t2 = time.time()
print('Time to fit GMM: ', t2 - t1)

m1, m2 = separate_models(gmm_blend_two)


fitblendone = np.median(gmm_blend_one.score_samples(radec_blend))
fitblendtwo = np.median(gmm_blend_two.score_samples(radec_blend))

fitblend_comp1 = np.median(m1.score_samples(radec_blend))
fitblend_comp2 = np.median(m2.score_samples(radec_blend))
print(fitblend_comp1, fitblend_comp2)


r1 = (fitblendtwo - fitblendone) / np.abs(fitblendone)
print(r1)
print(fitblendone, fitblendtwo)

plot_results(radec_blend, gmm_blend_one.predict(radec_blend), gmm_blend_one.means_, gmm_blend_one.covariances_, 0, 'Blended Sources')
plot_results(radec_blend, gmm_blend_two.predict(radec_blend), gmm_blend_two.means_, [gmm_blend_two.covariances_]*2, 1, 'Blended Sources')
plt.savefig('gmm.png')