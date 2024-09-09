import pandas as pd
import torch
import numpy as np
from time import perf_counter as pc
from lib import get_centroid
# from line_profiler import profile
from math import ceil

DATA_DIRECTORY = "./in"

class PartitionDataLoader:
    def __init__(self, parquet_db: pd.DataFrame, chunk_size: int):
        self.tbl = parquet_db
        self.expected_flux_conversion_factor_w1 = 0.00000154851985514
        self.expected_flux_conversion_factor_w2 = 0.00000249224248693
        self.chunksize = chunk_size
        self.n_sources = 0
    
    def load_parquet(self):
        tbl = self.tbl

        self.tbl = tbl.apply(self.process_row, axis=1)
        self.tbl = self.tbl[self.tbl["designation"].notna()]
        # add nrows column
        self.tbl["npts"] = self.tbl["mjd"].apply(len)
        self.tbl = self.tbl.sort_values("npts", ascending=False)

        self.n_sources = len(self.tbl)        
        return tbl
    
    def __iter__(self):
        self.load_parquet()
        self.i = 0
        return self
        
    def designation(self, centroid):
        ra = np.round(centroid[0], 6)
        dec = np.abs(np.round(centroid[1], 6))
        if dec > 0:
            sgn = "+"
        else:
            sgn = "-"
        return "NEOVAR {:.6f}{}{:.6f}".format(ra, sgn, dec)
    
    def process_row(self, row):
        # Choose if we will be using w1 or w2
        ra  = np.array(row["ra"])
        dec = np.array(row["dec"])
        w1flux = np.array(self.expected_flux_conversion_factor_w1 * row["w1flux"])
        w1sigflux = np.array(self.expected_flux_conversion_factor_w1 * row["w1sigflux"])
        w2flux = np.array(self.expected_flux_conversion_factor_w2 * row["w2flux"])
        w2sigflux = np.array(self.expected_flux_conversion_factor_w2 * row["w2sigflux"])
        mjd = np.array(row["mjd"])
        w1rchi2 = np.array(row["w1rchi2"])
        w2rchi2 = np.array(row["w2rchi2"])
        qual_frame = np.array(row["qual_frame"])

        flux = w1flux
        sig = w1sigflux
        chi = w1rchi2
        
        valid_bright = flux > 0
        valid_sigma = sig > 0
        valid_snr = (flux / sig) > 4
        good_qual_frame = qual_frame > 0
        good_chi_squared = chi < 10

        first_order_filter = valid_bright & valid_sigma & good_qual_frame & valid_snr & good_chi_squared
        removed_count = np.where(first_order_filter)[0]

        empty = len(removed_count) == 0

        # Remove 5% of highest chi-squared values
        # chi = chi[first_order_filter]
        # chi_filter_short = chi < np.quantile(chi, 0.95)
        # chi_filter = np.array([False] * len(first_order_filter))
        # chi_filter[removed_count[chi_filter_short]] = True # Map up to the original indices

        # Outlier Rejection
        outlier_filter = np.array([True] * len(first_order_filter))
        if not empty:
            fluxes = flux[first_order_filter]
            outlier_rejection = np.abs(fluxes - np.mean(fluxes)) < 4 * np.std(fluxes)

            outlier_filter = np.array([False] * len(first_order_filter))
            outlier_filter[removed_count[outlier_rejection]] = True

        final_filter = first_order_filter & outlier_filter
        
        w1flux = w1flux[final_filter]
        w1sigflux = w1sigflux[final_filter]
        w2flux = w2flux[final_filter]
        w2sigflux = w2sigflux[final_filter]
        ra = ra[final_filter]
        dec = dec[final_filter]
        mjd = mjd[final_filter]
        w1rchi2 = w1rchi2[final_filter]
        w2rchi2 = w2rchi2[final_filter]
        qual_frame = qual_frame[final_filter]

        if len(w1flux) < 20:
            designation = pd.NA
        else:
            designation = self.designation(get_centroid(ra, dec))

        final_row = pd.Series({
            "designation": designation,
            "w1flux": w1flux,
            "w1sigflux": w1sigflux,
            "w2flux": w2flux,
            "w2sigflux": w2sigflux,
            "mjd": mjd,
            "ra": ra,
            "dec": dec,
            "w1rchi2": w1rchi2,
            "w2rchi2": w2rchi2,
            "qual_frame": qual_frame
        })

        for key in row.keys():
            if key not in ["w1flux", "w1sigflux", "w2flux", "w2sigflux", "mjd", "ra", "dec", "w1rchi2", "w2rchi2", "qual_frame"]:
                final_row[key] = row[key] # Copy the rest of the columns
        return final_row
    
    def to_tensor(self, slice):
        # tensors = Parallel(n_jobs=-4)(delayed(self._to_tensor_row)(row[1]) for row in slice.iterrows())
        tensors = [self._to_tensor_row(row[1]) for row in slice.iterrows()]
        batched = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        return batched
    
    def _to_tensor_row(self, row):

        values = self.normalize(row["w1flux"], row["w1sigflux"], row["mjd"])

        sorter = np.argsort(values[2])
        arr = np.stack(values, axis=1)
        arr = arr[sorter]
        return torch.tensor(arr, dtype=torch.float32).cuda()

    def normalize(self, flux, sigflux, mjd):    
        day = (mjd - np.min(mjd)) / 4000

        w1flux_norm = flux - np.median(flux)
        iqr = np.quantile(flux, 0.75) - np.quantile(flux, 0.25)
        w1flux_norm = w1flux_norm / iqr
        w1flux_norm = np.arcsinh(w1flux_norm)

        sigflux = np.arcsinh(sigflux / iqr)

        return w1flux_norm, sigflux, day

    def __next__(self):
        # get n rows
        if self.i >= self.n_sources:
            raise StopIteration
        
        slice = self.tbl.iloc[self.i: self.i + self.chunksize].copy()
        self.i += self.chunksize
        tens = self.to_tensor(slice)

        return slice, tens
    
    def __len__(self):
        return ceil(len(self.tbl) / self.chunksize)