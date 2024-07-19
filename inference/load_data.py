import pandas as pd
import torch
import numpy as np
from time import perf_counter as pc
from lib import get_centroid
from line_profiler import profile
from math import ceil

DATA_DIRECTORY = "./in"

@profile
class PartitionDataLoader:
    def __init__(self, parquet_db: pd.DataFrame, chunk_size: int, prefilter: bool = True):
        self.tbl = parquet_db
        self.expected_flux_conversion_factor_w1 = 0.00000154851985514
        self.expected_flux_conversion_factor_w2 = 0.00000249224248693
        self.chunksize = chunk_size
        self.prefilter = prefilter
        self.n_sources = 0
    
    def load_parquet(self):
        tbl = self.tbl
        self.n_sources = len(tbl)
        if self.prefilter:
            self.tbl = tbl.apply(self.process_row, axis=1)

        # add nrows column
        self.tbl["nrows"] = self.tbl["mjd"].apply(len)
        self.tbl = self.tbl.sort_values("nrows", ascending=False)
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
        return f"NEOVAR {ra}{sgn}{dec}"
    
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

        if False:
            flux = w2flux
            sig = w2sigflux
            chi = w2rchi2
        else:
            flux = w1flux
            sig = w1sigflux
            chi = w1rchi2

        
        valid_bright = flux > 0
        valid_sigma = sig > 0
        good_qual_frame = qual_frame == 10

        first_order_filter = valid_bright & valid_sigma & good_qual_frame
        removed_count = np.where(first_order_filter)[0]

        # Remove 5% of highest chi-squared values
        chi = np.array(chi)
        chi = chi[first_order_filter]
        chi_filter_short = chi < np.quantile(chi, 0.95)
        chi_filter = np.array([False] * len(first_order_filter))
        chi_filter[removed_count[chi_filter_short]] = True # Map up to the original indices

        # Remove 4.5 sigma outliers
        flux = flux[first_order_filter]
        outlier_filter_short = np.abs(flux - np.mean(flux)) < 4 * np.std(flux)
        outlier_filter = np.array([False] * len(first_order_filter))
        outlier_filter[removed_count[outlier_filter_short]] = True

        final_filter = first_order_filter & chi_filter & outlier_filter

        w1flux = w1flux[final_filter]
        w1sigflux = w1sigflux[final_filter]
        w2flux = w2flux[final_filter]
        w2sigflux = w2sigflux[final_filter]
        ra = ra[final_filter]
        dec = dec[final_filter]
        mjd = mjd[final_filter]
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
            "cluster_id": row["cluster_id"],
        })

        for key in row.keys():
            if key not in ["w1flux", "w1sigflux", "w2flux", "w2sigflux", "mjd", "ra", "dec", "cluster_id"]:
                final_row[key] = row[key] # Copy the rest of the columns
        return final_row
    
    @profile
    def to_tensor(self, slice):
        # tensors = Parallel(n_jobs=-4)(delayed(self._to_tensor_row)(row[1]) for row in slice.iterrows())
        tensors = [self._to_tensor_row(row[1]) for row in slice.iterrows()]
        batched = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        return batched
    
    def _to_tensor_row(self, row):
        w1 = torch.Tensor(row["w1flux"])
        w1sig = torch.Tensor(row["w1sigflux"])
        w2 = torch.Tensor(row["w2flux"])
        w2sig = torch.Tensor(row["w2sigflux"])
        mjd = torch.Tensor(row["mjd"])

        if torch.isnan(w1).any():
            values = self.normalize(w2, w2sig, mjd)
        else:
            values = self.normalize(w1, w1sig, mjd)

        sorter = torch.argsort(values[2])
        tensor = torch.stack(values, dim=1)
        tensor = tensor[sorter]
        return tensor.to(torch.float32).cuda()

    def normalize(self, flux, sigflux, mjd):    
        day = (mjd - torch.min(mjd)) / 4000
        w1flux_norm = (flux - torch.mean(flux)) / torch.std(flux)
        w1flux_norm = torch.arcsinh(w1flux_norm)
        w1snr_est = -torch.log10(flux) / 3

        return w1flux_norm, w1snr_est, day

    def __next__(self):
        # get n rows
        if self.i >= self.n_sources:
            raise StopIteration
        
        slice = self.tbl.iloc[self.i: self.i + self.chunksize]
        if not self.prefilter:
            slice = slice.apply(self.process_row, axis=1)
        self.i += self.chunksize
        tens = self.to_tensor(slice)

        return slice, tens
    
    def __len__(self):
        return ceil(self.n_sources / self.chunksize)