# Author: Troy Raen (raen@ipac.caltech.edu) and Matthew Paz (mpaz@ipac.caltech.edu)
# Created: 2024-05-31
import numpy as np
import hpgeom
import pyarrow.compute
import pyarrow.dataset
from sklearn.cluster import DBSCAN
import pandas as pd

PARTITION_K = 5
PARTITION_NSIDE = hpgeom.order_to_nside(order=PARTITION_K)

# [TODO]
# Set MAX_K to the max order stored in the new files.
MAX_K = 13
MAX_K_NSIDE = hpgeom.order_to_nside(order=MAX_K)
# Choose ITER_K such that one pixel plus the margin fits comfortably in memory (all rows, all years).
ITER_K = 6
ITER_K_NSIDE = hpgeom.order_to_nside(order=ITER_K)
# Choose MARGIN_K such that the pixels are as small as possible while still including everything
# that needs to be considered when clustering an ITER_K pixel.
# To see how big a pixel is, you may want to use an hpgeom method like max_pixel_radius or nside_to_pixel_area.
MARGIN_K = 12
MARGIN_K_NSIDE = hpgeom.order_to_nside(order=MARGIN_K)


# def run():
#     base_path = "/stage/irsa-data-parquetlinks/links-tmp/neowiser/healpix_k5/"
#     year_path = "year<N>_skinny/neowiser-healpix_k5-year<N>_skinny.parquet/_metadata"
#     neowise_path = lambda year: base_path + year_path.replace("<N>", str(year))
#     year_datasets = [
#     pyarrow.dataset.parquet_dataset(neowise_path(year), partitioning="hive") for year in range(1, 11)
#     ]
#     neowise_ds = pyarrow.dataset.dataset(year_datasets)
#     # for partition_k_pixel_id in range(hpgeom.nside_to_npixel(PARTITION_NSIDE)):  # [TODO] check
#     partition_k_pixel_id = 10936
#     find_clusters_one_partition(partition_k_pixel_id, neowise_ds)
    

def find_clusters_one_partition(partition_k_pixel_id: int, neowise_ds: pyarrow.dataset.Dataset):
    partition_k_cluster_map_tbl = pd.DataFrame(columns=["source_id", "cluster_id"])
    # Get the list of iter_k pixels to iterate over.
    iter_k_pixel_ids = change_k(pix=partition_k_pixel_id, pix_k=PARTITION_K, new_k=ITER_K)

    # Iterate over pixels, load the data, and cluster.
    for iter_k_pixel_id in iter_k_pixel_ids:
        filters = construct_filters(partition_k_pixel_id, iter_k_pixel_id)
        pixel_tbl = neowise_ds.to_table(
            columns=["source_id", "ra", "dec", "w1snr", f"healpix_k{PARTITION_K}", f"healpix_k{MAX_K}"], 
            filter=filters)
        positional_tbl = pixel_tbl.select(["ra", "dec"])
        iter_k_source_to_cluster_map_tbl = cluster(positional_tbl, iter_k_pixel_id)
        partition_k_cluster_map_tbl = partition_k_cluster_map_tbl.append(iter_k_source_to_cluster_map_tbl)
    
    return partition_k_cluster_map_tbl


def change_k(*, pix: int, pix_k: int, new_k: int, return_field=False) -> int | list[int] | pyarrow.Field:
    """Convert pix to pixel ID(s) at order new_k.

    Returns
    -------
        If return_field is True, this returns a pyarrow field to be used in dataset filters.
        Else,
            If knew <= kpix, there is a unique pixel (parent or self), and this returns an int.
            If knew > kpix, there are multiple (child) pixels, and this returns a list of ints.
    """
    if return_field:
        # Assume kpix > knew. Use formula from next section, but with pyarrow.compute functions.
        new_k_field = pyarrow.compute.floor(
            pyarrow.compute.divide(pix, pyarrow.compute.power(4, pyarrow.compute.subtract(pix_k, new_k)))
        )
        return new_k_field

    koffset = np.abs(pix_k - new_k)
    if pix_k > new_k:
        return int(np.floor(pix / 4**koffset))
    if pix_k < new_k:
        return [pix * 4**koffset + ki for ki in range(4**koffset)]
    return pix


def construct_filters(partition_k_pixel_id: int, iter_k_pixel_id: int):
    """Return a pyarrow filter that selects rows if they are in either the ITER_K pixel or
    the MARGIN_K pixels along its border (outside).

    It's important to include filters on the partitioning column for speed even though it doesn't
    change the results.
    """
    # Construct pyarrow fields for each column. We need to use pyarrow.compute for all math functions.
    partition_k_field = pyarrow.compute.field(f"healpix_k{PARTITION_K}")
    max_k_field = pyarrow.compute.field(f"healpix_k{MAX_K}")
    snr_field = pyarrow.compute.field("w1snr")
    w1cc_map = pyarrow.compute.field("w1cc_map")

    # Pixel IDs at ITER_K and MARGIN_K are not stored in the catalog, but the values can be
    # calculated from the max_k_field on the fly and used in the filters.
    iter_k_field = change_k(pix=max_k_field, pix_k=MAX_K, new_k=ITER_K, return_field=True)
    margin_k_field = change_k(pix=max_k_field, pix_k=MAX_K, new_k=MARGIN_K, return_field=True)

    # Construct a filter for the ITER_K pixel.
    pixel_filter = (partition_k_field == partition_k_pixel_id) & (iter_k_field == iter_k_pixel_id)

    # Construct a filter for the margin around the ITER_K pixel.
    margin_filter = _construct_margin_filter(
        partition_k_pixel_id, iter_k_pixel_id, partition_k_field, margin_k_field
    )

    ## QUALITY FILTERS ##

    snr_cutoff = snr_field > 4.0 
    artifact_filter = pyarrow.compute.bit_wise_and(w1cc_map, 0b100000000) == 0 # According to database flags, checks that the detection is not spurious

    quality_filter = snr_cutoff & artifact_filter


    # Construct the full filter for rows that are in the pixel or the margin.
    filters = (pixel_filter | margin_filter) & quality_filter
    return filters


def _construct_margin_filter(
    partition_k_pixel_id: int,
    iter_k_pixel_id: int,
    partition_k_field: pyarrow.Field,
    margin_k_field: pyarrow.Field,
):
    # Find the MARGIN_K pixels that border the ITER_K pixel. This uses hpgeom, but
    # there's probably a better way to do it with pure math (healpix nested ordering scheme).
    margin_k_pixels_within_iter_k_pixel = change_k(pix=iter_k_pixel_id, pix_k=ITER_K, new_k=MARGIN_K)
    all_neighbors = hpgeom.neighbors(
        nside=hpgeom.order_to_nside(order=MARGIN_K), pix=margin_k_pixels_within_iter_k_pixel, nest=True
    )
    all_neighbors_clean = [n for n in all_neighbors.flatten() if n != -1]  # -1 means no neighbor at that position
    border_margin_ids = sorted(set(all_neighbors_clean) - set(margin_k_pixels_within_iter_k_pixel))

    # Find neighboring partition_k pixels so we can include a filter on the partition column.
    partition_neighbor_ids = sorted(
        set([change_k(pix=pix, pix_k=MARGIN_K, new_k=PARTITION_K) for pix in border_margin_ids])
    )

    # Construct the margin filter.
    margin_filter = partition_k_field.isin([partition_k_pixel_id] + partition_neighbor_ids) & margin_k_field.isin(
        border_margin_ids
    )
    return margin_filter


def cluster(pixel_tbl: pyarrow.Table, iter_k_pixel_id: int) -> pyarrow.Table:
    positional_tbl = pixel_tbl.select(["ra", "dec"]).to_pandas()
    # Converting to lon/lat in radians
    lon_lat = positional_tbl * np.pi / 180.0
    lon_lat["ra"] = lon_lat["ra"] - np.pi

    source_id_to_cluster_id = pixel_tbl.select(["source_id"]).to_pandas()
    source_id_to_cluster_id.insert(1, "cluster_id", pd.NA)

    EPS = 1.25 / 3600

    dbscan = DBSCAN(
        eps=EPS, 
        min_samples=12, 
        n_jobs=-1, 
        algorithm="kd_tree",
        metric="haversine",
        )
    dbscan.fit(lon_lat)
    labels = dbscan.labels_

    all_clusters = {} # All clusters, whether in margin or not. Labels are integers here, not cluster_ids

    for i, label in enumerate(labels): # Place cluster members into a dictionary
        if label not in all_clusters:
            all_clusters[label] = []
        all_clusters[label].append(i)

    del all_clusters[-1]  # Remove noise label

    for label, indices in all_clusters.items():
        if len(indices) < 16: # Delete tiny clusters
            del all_clusters[label]
            continue
        
        
    for label in all_clusters:
        indices = all_clusters[label]
        lon_lat_in_cluster = lon_lat.iloc[indices].to_numpy() # Acquire the positions of cluster members

        longitude_in_cluster = lon_lat_in_cluster[:, 0] # Extract the longitude and latitude separately
        latitude_in_cluster = lon_lat_in_cluster[:, 1]

        cosine_of_latitude = np.cos(latitude_in_cluster) # Precompute this value
        cartesian_in_cluster = np.array([
            cosine_of_latitude * np.cos(longitude_in_cluster), 
            cosine_of_latitude * np.sin(longitude_in_cluster), 
            np.sin(latitude_in_cluster)]).T
        cartesian_centroid = np.mean(cartesian_in_cluster, axis=0) # Convert to cartesian and take the mean. Accounts for spherical geometry and wrapping around 0/360
        centroid_lon, centroid_lat = hpgeom.hpgeom.vector_to_angle(cartesian_centroid, lonlat=True, degrees=False) # Convert back to spherical coordinates of RA, DEC
        centroid_RA, centroid_Dec = (centroid_lon * 180 / np.pi + 180.0, centroid_lat * 180 / np.pi) # Convert to degrees and shift long

        # Check if the centroid of the cluster lies within the ITER_K pixel. If it does not, we do not include it in this write. It will be included in the adjacent pixel's write.
        if hpgeom.hpgeom.angle_to_pixel(ITER_K_NSIDE, centroid_lon, centroid_lat, degrees=False, lonlat=True) == iter_k_pixel_id:
            cluster_id = "{:.4f}".format(centroid_RA).zfill(8) + ("+" if centroid_Dec > 0 else "-") + "{:.4f}".format(np.abs(centroid_Dec)).zfill(7) # Create a unique cluster_id
            source_id_to_cluster_id.iloc[indices]["cluster_id"] = cluster_id

    
    clean_source_id_to_cluster_id = source_id_to_cluster_id.dropna() # Remove rows with no cluster_id
    clean_source_id_to_cluster_id = clean_source_id_to_cluster_id.astype("str")

    return clean_source_id_to_cluster_id