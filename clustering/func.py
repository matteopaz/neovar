# Author: Troy Raen (raen@ipac.caltech.edu) and Matthew Paz (mpaz@ipac.caltech.edu)
# Created: 2024-05-31
import numpy as np
import hpgeom
import pyarrow.compute
import pyarrow.dataset
import pyarrow.parquet as pq
from sklearn.cluster import DBSCAN
import pandas as pd
import line_profiler
# import plotly.graph_objects as go

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
    # Get the list of iter_k pixels to iterate over.
    iter_k_pixel_ids = change_k(pix=partition_k_pixel_id, pix_k=PARTITION_K, new_k=ITER_K)

    # Iterate over pixels, load the data, and cluster.
    iter_k_source_to_cluster_map_tbls = []
    iter_k_cluster_to_source_map_tbls = []

    for iter_k_pixel_id in iter_k_pixel_ids:
        filters = construct_filters(partition_k_pixel_id, iter_k_pixel_id)
        pixel_tbl = neowise_ds.to_table(
            columns=["source_id", "ra", "dec", "w1snr", f"healpix_k{PARTITION_K}", f"healpix_k{MAX_K}"], 
            filter=filters)
        
        print("{} apparitions in pixel {}".format(len(pixel_tbl), iter_k_pixel_id))

        iter_k_source_to_cluster_map_tbl, iter_k_cluster_to_source_map_tbl = cluster(pixel_tbl, iter_k_pixel_id)
        iter_k_source_to_cluster_map_tbls.append(iter_k_source_to_cluster_map_tbl)
        iter_k_cluster_to_source_map_tbls.append(iter_k_cluster_to_source_map_tbl)
    
    partition_k_source_to_cluster_map_tbl = pd.concat(iter_k_source_to_cluster_map_tbls, axis=0)
    partition_k_cluster_to_source_map_tbl = pyarrow.concat_tables(iter_k_cluster_to_source_map_tbls)

    
    return partition_k_source_to_cluster_map_tbl, partition_k_cluster_to_source_map_tbl 


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
    cc_flags_field = pyarrow.compute.field("cc_flags")
    # w1cc_map = pyarrow.compute.field("w1cc_map")

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

    snr_cutoff = pyarrow.compute.greater(snr_field, 4)  # SNR > 4
    # artifact_filter = pyarrow.compute.bit_wise_and(w1cc_map, 0b100000000) == 0 # According to database flags, checks that the detection is not spurious
    first_char_cc_flags = pyarrow.compute.utf8_slice_codeunits(cc_flags_field, start=0, stop=1)
    artifact_filter = pyarrow.compute.match_substring_regex(first_char_cc_flags, pattern=r"/D|d|P|O/gm")
    # Should not be on a diffraction spike at all, should not be a persistence or ghost artifact

    # quality_filter = snr_cutoff & artifact_filter
    quality_filter = snr_cutoff


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

@profile
def cluster(pixel_tbl: pyarrow.Table, iter_k_pixel_id: int) -> pyarrow.Table:
    positional_tbl = pixel_tbl.select(["ra", "dec"]).to_pandas()
    # Converting to lon/lat in radians
    lon_lat = positional_tbl * np.pi / 180.0
    lon_lat["ra"] = lon_lat["ra"] - np.pi

    source_id_to_cluster_id = pixel_tbl.select(["source_id"]).to_pandas()
    source_id_to_cluster_id.insert(1, "cluster_id", pd.NA)

    EPS = (0.85 / 3600) * np.pi / 180.0 # 1.25 arcseconds in radians

    dbscan = DBSCAN(
        eps=EPS, 
        min_samples=12, 
        n_jobs=-1, 
        algorithm="ball_tree",
        metric="haversine",
        )
    dbscan.fit(lon_lat)
    labels = dbscan.labels_

    all_clusters = {} # All clusters, whether in margin or not. Labels are integers here, not cluster_ids

    for i, label in enumerate(labels): # Place cluster members into a dictionary
        if label not in all_clusters:
            all_clusters[label] = []
        all_clusters[label].append(i)

    if -1 not in all_clusters:
        # print warning
        raise Warning("No noise label found in DBSCAN clustering in iter_k_pixel_id={}. Highly unusual".format(iter_k_pixel_id))
    else:
        del all_clusters[-1]  # Remove noise label

    labels_to_delete = []
    for label, indices in all_clusters.items():
        if len(indices) < 16: # Delete tiny clusters
            labels_to_delete.append(label) # Save to list to delete later, cannot delete while iterating
    
    for label in labels_to_delete:
        del all_clusters[label]
        
    filtered_cluster_dict = {}

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
        partition_k_pixel_id = change_k(pix=iter_k_pixel_id, pix_k=ITER_K, new_k=PARTITION_K)

        belongs_to_iter_k_pixel = hpgeom.hpgeom.angle_to_pixel(ITER_K_NSIDE, centroid_lon + np.pi, centroid_lat, degrees=False, lonlat=True)[0]
        if belongs_to_iter_k_pixel == iter_k_pixel_id:


            cluster_designation = "{:.4f}".format(float(centroid_RA)).zfill(8) + ("+" if centroid_Dec > 0 else "-") + "{:.4f}".format(float(np.abs(centroid_Dec))).zfill(7) # Create a unique cluster_id

            # Constructing binary cluster_id
            trunc_RA = np.round(centroid_RA, 4)
            trunc_Dec = np.round(centroid_Dec, 4)
            partition_k_id_bin = np.binary_repr(partition_k_pixel_id, width=16)
            trunc_RA_bin = np.binary_repr(int(trunc_RA * 10000), width=24)
            trunc_Dec_bin = np.binary_repr(int(trunc_Dec * 10000), width=24)
            cluster_id = partition_k_id_bin + trunc_RA_bin + trunc_Dec_bin # 64 bit integer with leading 16 bits for partition_k_pixel_id, 24 bits for RA, 24 bits for Dec
            cluster_id = int(cluster_id, 2) # to integer
            
            filtered_cluster_dict[cluster_id] = source_id_to_cluster_id.loc[indices, "source_id"].to_list() # Add to dictionary
            
            source_id_to_cluster_id.loc[indices, "cluster_id"] = cluster_id

    clean_source_id_to_cluster_id = source_id_to_cluster_id.dropna() # Remove rows with no cluster_id
    cluster_id_to_source_ids = dict_to_parquet(filtered_cluster_dict) # Convert to dictionary

    return clean_source_id_to_cluster_id, cluster_id_to_source_ids


def dict_to_parquet(d: dict):
    # Schema should be an ID int64, a list of ID Strings
    schema = pyarrow.schema([
        pyarrow.field("cluster_id", pyarrow.int64()),
        pyarrow.field("source_ids", pyarrow.list_(pyarrow.string()))
    ])
    # each entry of the dictionary is a row of this table
    data = {"cluster_id": [], "source_ids": []}
    for cluster_id, source_ids in d.items():
        data["cluster_id"].append(cluster_id)
        data["source_ids"].append(source_ids)
    # construct pyarrow table from this
    table = pyarrow.table(data, schema)
    return table



