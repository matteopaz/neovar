

# Epsilon Function

After inspecting clustering on many different regions of the sky, it is obvious that the general background noise level of the region is inverse to the quality of the database, as the validity of the PSF degrades in these regions. Using existing columns we already had queried, we define this background noise to be

$\eta = snr \times w1flux = snr \times \left(309.54 \times 10^{\frac{w1mpro}{2.5}}\right)$

We find that reducing the epsilon parameter of the DBSCAN algorithm is the best way to ensure our extracted clusters are valid - at the necessary cost of some loss in cluster count and size. Heuristically, going up an order of magnitude in $\eta$ was best adressed by halving $\varepsilon$. Starting at $\eta = 0.06, \varepsilon=2$...

$\varepsilon = 2^{-\left(1+\log(\frac{\eta}{6})\right)}$
