# Author: Troy Raen (raen@ipac.caltech.edu) and Matthew Paz (mpaz@ipac.caltech.edu)
# Created: 2024-05-31
import numpy as np
import hpgeom
import pyarrow
import pyarrow.compute
import pyarrow.dataset
import pyarrow.parquet as pq
from sklearn.cluster import DBSCAN
import pandas as pd
import psutil
from line_profiler import profile
from time import perf_counter



PARTITION_K = 5
PARTITION_NSIDE = hpgeom.order_to_nside(order=PARTITION_K)


# Set MAX_K to the max order stored in the new files.
MAX_K = 13
# Choose ITER_K such that one pixel plus the margin fits comfortably in memory (all rows, all years).
ITER_K = 5
# Choose MARGIN_K such that the pixels are as small as possible while still including everything
# that needs to be considered when clustering an ITER_K pixel.
# To see how big a pixel is, you may want to use an hpgeom method like max_pixel_radius or nside_to_pixel_area.
MARGIN_K = 13

POLAR_REGION = False


def get_memory_usage_pct():
    return psutil.virtual_memory().percent

class ChunkedDataset:
    def __init__(self, neowise_ds: pyarrow.Table, partition_k: int, iter_ks: list[int], chunk_size: int):
        self.ds = neowise_ds
        self.partition_k = partition_k
        self.iter_ks = iter_ks
        self.chunk_size = chunk_size
        self.columns = [
            "cntr", "ra", "dec",
            "mjd", "w1flux", "w1sigflux", "w2flux", "w2sigflux",
            "qual_frame", "w1rchi2", "w2rchi2", "healpix_k5", "healpix_k13"
        ]
        self.chunk = []
        self.current_tbl = None

    def quality_filters(self):
        w1flux_field = pyarrow.compute.field("w1flux")
        w1sigflux_field = pyarrow.compute.field("w1sigflux")
        w2flux_field = pyarrow.compute.field("w2flux")
        w2sigflux_field = pyarrow.compute.field("w2sigflux")
        w1cc_map_field = pyarrow.compute.field("w1cc_map")
        w2cc_map_field = pyarrow.compute.field("w2cc_map")

        w1_real_detection = pyarrow.compute.invert(pyarrow.compute.is_null(w1sigflux_field))
        w2_real_detection = pyarrow.compute.invert(pyarrow.compute.is_null(w2sigflux_field))
        real_detection_filter = w1_real_detection | w2_real_detection

        w1snr = pyarrow.compute.if_else(w1_real_detection, pyarrow.compute.divide(w1flux_field, w1sigflux_field), 0) # If else to prevent error
        w2snr = pyarrow.compute.if_else(w2_real_detection, pyarrow.compute.divide(w2flux_field, w2sigflux_field), 0)

        w1_snr_cutoff = pyarrow.compute.greater(w1snr, 4) 
        w2_snr_cutoff = pyarrow.compute.greater(w2snr, 4)
        snr_cutoff = w1_snr_cutoff | w2_snr_cutoff
        w1_artifact_filter = pyarrow.compute.equal(pyarrow.compute.bit_wise_and(w1cc_map_field, 0b111111111), 0)
        w2_artifact_filter = pyarrow.compute.equal(pyarrow.compute.bit_wise_and(w2cc_map_field, 0b111111111), 0)
        artifact_filter = w1_artifact_filter & w2_artifact_filter # According to database flags, checks that the detection is not spurious

        # Should not be on a diffraction spike at all, should not be a persistence or ghost artifact

        # quality_filter = snr_cutoff & artifact_filter
        quality_filter = real_detection_filter & snr_cutoff & artifact_filter

        return quality_filter
    
    def query_next_chunk(self):
        if len(self.iter_ks) <= self.index:
            raise StopIteration("No more chunks to query")

        iter_ks_chunk = self.iter_ks[self.index:self.index+self.chunk_size]
        filters = self.construct_filters_multiple(iter_ks_chunk)
 
        pixel_tbl = self.ds.to_table(columns=self.columns, filter=filters)
        self.current_tbl = pixel_tbl
        self.chunk = iter_ks_chunk
        return

    def construct_filters_multiple(self, iter_ks: list[int]):
        partition_k_field = pyarrow.compute.field("healpix_k5")
        max_k_field = pyarrow.compute.field("healpix_k13")

        margin_k_field = change_k(pix=max_k_field, pix_k=MAX_K, new_k=MARGIN_K, return_field=True)

        margin_k_pixels_within_iter_k_pixel = [change_k(pix=iter_k_pixel_id, pix_k=ITER_K, new_k=MARGIN_K) for iter_k_pixel_id in iter_ks]
        margin_k_pixels_within_iter_k_pixel = [pix for sublist in margin_k_pixels_within_iter_k_pixel for pix in sublist]

        all_neighbors = hpgeom.neighbors(
            nside=hpgeom.order_to_nside(MARGIN_K), pix=margin_k_pixels_within_iter_k_pixel, nest=True
        )
        all_neighbors_clean = [n for n in all_neighbors.flatten() if n != -1]  # -1 means no neighbor at that position
        border_margin_ids = sorted(set(all_neighbors_clean) - set(margin_k_pixels_within_iter_k_pixel))

        # Find neighboring partition_k pixels so we can include a filter on the partition column.
        partition_neighbor_ids = sorted(
            set([change_k(pix=pix, pix_k=MARGIN_K, new_k=PARTITION_K) for pix in border_margin_ids])
        )

        # Construct the margin filter.
        pixel_filter_with_margin = partition_k_field.isin([self.partition_k] + partition_neighbor_ids) & margin_k_field.isin(
            border_margin_ids + margin_k_pixels_within_iter_k_pixel
        )

        quality_filter = self.quality_filters()

        return pixel_filter_with_margin & quality_filter

    def _merge_filters(self, list_of_filters: list): # recursively merge the filters pairwise
        if len(list_of_filters) == 1:
            return list_of_filters[0]
        else:
            half = len(list_of_filters) // 2

            merged_1 = self._merge_filters(list_of_filters[:half])
            merged_2 = self._merge_filters(list_of_filters[half:])
            return merged_1 | merged_2 
        
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= len(self.iter_ks):
            raise StopIteration
        
        iter_k_id = self.iter_ks[self.index]

        if iter_k_id not in self.chunk:
            self.query_next_chunk()
        
        if self.chunk_size == 1:
            self.index += 1 
            return self.current_tbl, iter_k_id
        
        posfilter = pixel_filters(self.partition_k, iter_k_id)
        filtered_tbl = self.current_tbl.filter(posfilter)
        
        self.index += 1
        return filtered_tbl, iter_k_id
    
def pixel_filters(partition_k_pixel_id: int, iter_k_pixel_id: int, include_partition_k=True):
    partition_k_field = pyarrow.compute.field(f"healpix_k5").cast(pyarrow.int32())
    max_k_field = pyarrow.compute.field(f"healpix_k13").cast(pyarrow.int32())
    iter_k_field = change_k(pix=max_k_field, pix_k=MAX_K, new_k=ITER_K, return_field=True)
    margin_k_field = change_k(pix=max_k_field, pix_k=MAX_K, new_k=MARGIN_K, return_field=True)

    if include_partition_k:
        pixel_filter = pyarrow.compute.equal(partition_k_field, partition_k_pixel_id) & pyarrow.compute.equal(iter_k_field, iter_k_pixel_id)
    else:
        pixel_filter = pyarrow.compute.equal(iter_k_field, iter_k_pixel_id)

    margin_filter = _construct_margin_filter(
        partition_k_pixel_id, iter_k_pixel_id, partition_k_field, margin_k_field
    )

    filter = (pixel_filter | margin_filter)

    return filter

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
        nside=hpgeom.order_to_nside(MARGIN_K), pix=margin_k_pixels_within_iter_k_pixel, nest=True
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

def find_clusters_one_partition(partition_k_pixel_id: int, neowise_ds: pyarrow.dataset.Dataset, iter_k_ord: int, iter_k_list: list[int], log_outfile=None):
    global ITER_K
    ITER_K = iter_k_ord

    if iter_k_ord > 6:
        global POLAR_REGION
        POLAR_REGION = True

    # Get the list of iter_k pixels to iterate over.
    iter_k_pixel_ids = iter_k_list
    if ITER_K == PARTITION_K:
        iter_k_pixel_ids = [iter_k_pixel_ids]
    
    # Iterate over pixels, load the data, and cluster.

    tbl_list_entry_ids = []
    iter_k_cntr_to_cluster_map_tbls = []
    iter_k_cluster_to_data_map_tbls = []

    chunksize = max(1, 2**(iter_k_ord - 5))


    for pixel_tbl, iter_k_pixel_id in ChunkedDataset(neowise_ds, partition_k_pixel_id, iter_k_pixel_ids, chunksize):
        t1 = perf_counter()

        with open(log_outfile, "a") as f:
            f.write(f"Pixel table for iterkid {iter_k_pixel_id} has {pixel_tbl.num_rows} rows")

        iter_k_cntr_to_cluster_map_tbl, iter_k_cluster_to_data_map_tbl = cluster(pixel_tbl, iter_k_pixel_id)

        tbl_list_entry_ids.append(iter_k_pixel_id)
        iter_k_cntr_to_cluster_map_tbls.append(iter_k_cntr_to_cluster_map_tbl)
        iter_k_cluster_to_data_map_tbls.append(iter_k_cluster_to_data_map_tbl)

        if get_memory_usage_pct() > 90:
            # Emergency, raise error. Hopefully should not reach this given the memory checks
            raise MemoryError("Memory usage is too high. Exiting to prevent crash.")

        if get_memory_usage_pct() > 75:
            start_iterk_PIDS = tbl_list_entry_ids[0]
            end_iterk_PIDS = tbl_list_entry_ids[-1]
            partition_k_cntr_to_cluster_map_tbl = pd.concat(iter_k_cntr_to_cluster_map_tbls, axis=0)
            partition_k_cluster_to_data_map_tbl = pyarrow.concat_tables(iter_k_cluster_to_data_map_tbls)
            # Save to disk
            PATH_TO_OUTPUT_DIRECTORY = "/home/mpaz/neowise-clustering/clustering/out"
            partition_k_cntr_to_cluster_map_tbl.to_csv(
                PATH_TO_OUTPUT_DIRECTORY +
                f"/partition_{partition_k_pixel_id}_subpartitions_{start_iterk_PIDS}to{end_iterk_PIDS}_cntr_to_id.csv"
                )
            pq.write_table(partition_k_cluster_to_data_map_tbl, 
                PATH_TO_OUTPUT_DIRECTORY + f"/partition_{partition_k_pixel_id}_subpartitions_{start_iterk_PIDS}to{end_iterk_PIDS}_cntr_to_data.parquet")
            # Clear memory
            del partition_k_cntr_to_cluster_map_tbl
            del partition_k_cluster_to_data_map_tbl

            iter_k_cntr_to_cluster_map_tbls = []
            iter_k_cluster_to_data_map_tbls = []
            tbl_list_entry_ids = []
        
    
    partition_k_cntr_to_cluster_map_tbl = pd.concat(iter_k_cntr_to_cluster_map_tbls, axis=0)
    partition_k_cluster_to_cntr_map_tbl = pyarrow.concat_tables(iter_k_cluster_to_data_map_tbls)

    
    return partition_k_cntr_to_cluster_map_tbl, partition_k_cluster_to_cntr_map_tbl 


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

@profile
def cluster(pixel_tbl: pyarrow.Table, iter_k_pixel_id: int) -> pyarrow.Table:
    positional_tbl = pixel_tbl.select(["ra", "dec"]).to_pandas().to_numpy()
    positional_tbl = np.radians(positional_tbl)
    latlon = positional_tbl[:, [1,0]]

    # cntr_to_cluster_id = pixel_tbl.select(["cntr"]).to_pandas()
    # cntr_to_cluster_id.insert(1, "cluster_id", pd.NA)
    cntrs = pixel_tbl.column("cntr").to_pandas()
    cluster_ids_np = np.zeros(len(cntrs), dtype=np.int64)

    EPS = (0.85 / 3600) * np.pi / 180.0
    min_samples = 12
    leaf_size = 5

    if POLAR_REGION:
        min_samples = 96
        leaf_size = 15

    dbscan = DBSCAN(
        eps=EPS, 
        min_samples=min_samples, 
        n_jobs=-1, 
        algorithm="ball_tree",
        metric="haversine",
        leaf_size=leaf_size
        )
    dbscan.fit(latlon)
    labels = dbscan.labels_

    all_clusters = {} # All clusters, whether in margin or not. Labels are integers here, not cluster_ids
    
    for idx, label in enumerate(labels):
        if label not in all_clusters:
            all_clusters[label] = []
        all_clusters[label].append(idx)

    if -1 not in all_clusters:
        raise Warning("No noise label found in DBSCAN clustering in iter_k_pixel_id={}. Highly unusual".format(iter_k_pixel_id))
    else:
        del all_clusters[-1]  # Remove noise label


    for label, indices in list(all_clusters.items()):
        if len(indices) < 16: # Delete tiny clusters
            del all_clusters[label] # Save to list to delete later, cannot delete while iterating
            continue

        ra_dec_in_cluster = positional_tbl[indices] # Acquire the positions of cluster members

        ra_in_cluster = ra_dec_in_cluster[:, 0] # Extract the longitude and latitude separately
        dec_in_cluster = ra_dec_in_cluster[:, 1]

        cosine_of_latitude = np.cos(dec_in_cluster) # Precompute this value
        cartesian_in_cluster = np.array([
            cosine_of_latitude * np.cos(ra_in_cluster), 
            cosine_of_latitude * np.sin(ra_in_cluster), 
            np.sin(dec_in_cluster)]).T
        cartesian_centroid = np.mean(cartesian_in_cluster, axis=0) # Convert to cartesian and take the mean. Accounts for spherical geometry and wrapping around 0/360
        centroid_RA, centroid_Dec = hpgeom.hpgeom.vector_to_angle(cartesian_centroid, lonlat=True, degrees=True) # Convert back to spherical coordinates of RA, DEC

        # Check if the centroid of the cluster lies within the ITER_K pixel. If it does not, we do not include it in this write. It will be included in the adjacent pixel's write.
        partition_k_pixel_id = change_k(pix=iter_k_pixel_id, pix_k=ITER_K, new_k=PARTITION_K)

        belongs_to_iter_k_pixel = hpgeom.hpgeom.angle_to_pixel(hpgeom.order_to_nside(ITER_K), centroid_RA, centroid_Dec, degrees=True, lonlat=True)[0]
        if belongs_to_iter_k_pixel == iter_k_pixel_id:
            cluster_id = get_cluster_id(partition_k_pixel_id, centroid_RA, centroid_Dec)
            all_clusters[cluster_id] = all_clusters.pop(label) # Rename the label to cluster_id
            cluster_ids_np[indices] = cluster_id # set the correct cntrs to the cluster_id
        else:
            del all_clusters[label] # Remove the cluster from the dictionary if it does not belong to the current pixel

    cntr_to_cluster_id = pd.DataFrame({"cntr": cntrs, "cluster_id": cluster_ids_np})

    clean_cntr_to_cluster_id = cntr_to_cluster_id.dropna() # Remove rows with no cluster_id
    cluster_id_to_data = dict_to_data_tbl(all_clusters, pixel_tbl) # Convert to dictionary

    return clean_cntr_to_cluster_id, cluster_id_to_data

def get_cluster_id(partition_k_id, center_ra, center_dec):
    # Truncate to required specificity
    trunc_RA = np.round(center_ra, 4)
    colat = 90.0 - center_dec
    trunc_colat = np.round(colat, 4)
    partition_k_id_bin = np.binary_repr(int(partition_k_id), width=16)

    bin_ra = np.binary_repr(int(trunc_RA * 10000), width=24)
    bin_colat = np.binary_repr(int(trunc_colat * 10000), width=24)
    cluster_id = partition_k_id_bin + bin_ra + bin_colat
    cluster_id = int(cluster_id, 2)
    return cluster_id


def dict_to_data_tbl(d: dict, data_tbl: pyarrow.Table):
    # Schema should be an ID int64, a list of ID Strings
    schema = pyarrow.schema([
        pyarrow.field("cluster_id", pyarrow.int64()),
        pyarrow.field("cntr", pyarrow.list_(pyarrow.int64())),
        pyarrow.field("ra", pyarrow.list_(pyarrow.float64())),
        pyarrow.field("dec", pyarrow.list_(pyarrow.float64())),
        pyarrow.field("mjd", pyarrow.list_(pyarrow.float64())),
        pyarrow.field("w1flux", pyarrow.list_(pyarrow.float64())),
        pyarrow.field("w1sigflux", pyarrow.list_(pyarrow.float64())),
        pyarrow.field("w2flux", pyarrow.list_(pyarrow.float64())),
        pyarrow.field("w2sigflux", pyarrow.list_(pyarrow.float64())),
        pyarrow.field("qual_frame", pyarrow.list_(pyarrow.int64())),
        pyarrow.field("w1rchi2", pyarrow.list_(pyarrow.float64())),
        pyarrow.field("w2rchi2", pyarrow.list_(pyarrow.float64()))
    ])
    # each entry of the dictionary is a row of this table
    data = {"cntr": [], "ra": [], "dec": [], "mjd": [], "w1flux": [], "w1sigflux": [], "w2flux": [], "w2sigflux": [], "qual_frame": [], "w1rchi2": [], "w2rchi2": []}

    pd_tbl = data_tbl.select(list(data.keys())).to_pandas()
    np_tbl = pd_tbl.to_numpy()
    colnames = list(pd_tbl.columns)
    index_dict = {col: colnames.index(col) for col in data}

    data["cluster_id"] = list(d.keys())
    for indices in d.values():
        cut_tbl = np_tbl[indices]
        for col in data:
            if col == "cluster_id":
                continue
            data[col].append(cut_tbl[:, index_dict[col]].tolist())
    # construct pyarrow table from this
    table = pyarrow.table(data, schema)
    return table