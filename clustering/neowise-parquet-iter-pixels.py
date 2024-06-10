# Author: Troy Raen (raen@ipac.caltech.edu)
# Date: 2024-05-31
import numpy as np
import hpgeom
import pyarrow.compute
import pyarrow.dataset


PARTITION_K = 5
PARTITION_NSIDE = hpgeom.order_to_nside(order=PARTITION_K)

# [TODO]
# Set MAX_K to the max order stored in the new files.
MAX_K = 10
# Choose ITER_K such that one pixel plus the margin fits comfortably in memory (all rows, all years).
ITER_K = 8
# Choose MARGIN_K such that the pixels are as small as possible while still including everything
# that needs to be considered when clustering an ITER_K pixel.
# To see how big a pixel is, you may want to use an hpgeom method like max_pixel_radius or nside_to_pixel_area.
MARGIN_K = 10


def run():
    # I would parallelize over partitions, but just to show the basics with a for-loop ...
    year_datasets = []  # [TODO] fill in
    neowise_ds = pyarrow.dataset.dataset(year_datasets)
    for partition_k_pixel_id in range(hpgeom.nside_to_npixel(PARTITION_NSIDE)):  # [TODO] check
        find_clusters_one_partition(partition_k_pixel_id, neowise_ds)


def find_clusters_one_partition(partition_k_pixel_id: int, neowise_ds: pyarrow.dataset.Dataset):
    # Get the list of iter_k pixels to iterate over.
    iter_k_pixel_ids = change_k(pix=partition_k_pixel_id, pix_k=PARTITION_K, new_k=ITER_K)

    # Iterate over pixels, load the data, and cluster.
    for iter_k_pixel_id in iter_k_pixel_ids:
        filters = construct_filters(partition_k_pixel_id, iter_k_pixel_id)
        pixel_tbl = neowise_ds.to_table(filter=filters)  # pyarrow Table. could convert to pandas, astropy, etc.
        your_clustering_code(pixel_tbl)


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

    # Construct the full filter for rows that are in the pixel or the margin.
    filters = pixel_filter | margin_filter
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


def your_clustering_code(pixel_tbl):
    # run your clustering code
    # you'll probably want to write the results to disk rather than returning them
    pass
