import torch
from line_profiler import profile
import pandas as pd
from torch.utils.data import IterableDataset
import numpy as np
import time
from joblib import Parallel, delayed
from math import ceil
import scipy.stats

RANDOM_SEED = 42
# device = "cpu" if not torch.cuda.is_available() else "cuda"
device = "cpu"
torch.set_default_dtype(torch.float32)
default_dtype = torch.get_default_dtype()

keeptypes = ["ea", "ew", "lpv", "rot", "rr", "cep", "yso"]

def data_tbl_apply_mask(data_tbl):
    if "filter_mask" not in data_tbl.columns:
        raise KeyError("data_tbl must have a 'filter_mask' column")
    def applymask(row):
        mask = row["filter_mask"]
        if not isinstance(mask, np.ndarray):
            raise ValueError("Filter mask is invalid - possibly missing?")
        for key in row.keys():
            if key == "filter_mask":
                continue
            itm = row[key]
            if isinstance(itm, np.ndarray):
                row[key] = itm[mask]
        return row
    return data_tbl.apply(applymask, axis=1)


def data_tbl_to_nn_input(data_tbl, bins, bin_overlap_frac=0, thread=False):
    def row_to_tensor(row):
        w1 = row["w1flux"]
        w1s = row["w1sigflux"]
        w2 = row["w2flux"]
        w2s = row["w2sigflux"]
        t = row["mjd"]

        overall_var = np.nanvar(w1)
        if np.isnan(w1).any():
            raise ValueError("NaNs in W1 flux")
        if np.isnan(t).any():
            raise ValueError("NaNs in MJD")
        
        if bins:
            if t.min() < 0 or t.max() > 1:
                # print(row["mjd"])
                # print(t.min(), t.max())
                # print(row)
                raise ValueError("MJD must be normalized to [0, 1] for binning")
            features = []
            for i in range(bins):
                binmask = (t >= (i - bin_overlap_frac)/bins) & (t < (i+1+bin_overlap_frac)/bins)
                if np.sum(binmask) < 2:
                    features.append([0, 0])
                    continue
                y1bin = np.median(w1[binmask])
                y1sbin = np.median(w1s[binmask])
                y2bin = np.median(w1[binmask])
                y2sbin = np.median(w1s[binmask])
                var = np.var(w1[binmask]) / overall_var
                features.append([y1bin, var])
            
            
            tens = torch.tensor(features, device="cpu")
            nonzero = ~(tens == 0)
            
            # minmax scaling over dimension 1
            tens[:, 0] = (tens[:, 0] - tens[:, 0].min()) / (tens[:, 0].max() - tens[:, 0].min())
            tens[:, 1] = (tens[:, 1] - tens[:, 1].min()) / (tens[:, 1].max() - tens[:, 1].min())
            return tens
        else:
            y1 = torch.tensor(w1, device="cpu")
            y1 = (y1 - torch.median(y1)) / (torch.quantile(y1, 0.75) - torch.quantile(y1, 0.25))
            t = torch.tensor(t, device="cpu")
            t = (t - t.min()) / (t.max() - t.min())
            return torch.stack([y1, t], dim=1)

    if not thread:
        rows = data_tbl.apply(row_to_tensor, axis=1).values.tolist()
    else:
        # print("using multithread")
        chunksize = (len(data_tbl) // 9)
        slices = [data_tbl.iloc[i*chunksize:(i+1)*chunksize] for i in range(ceil(len(data_tbl) / chunksize))]
        chunks = Parallel(n_jobs=9)(delayed(slice_.apply)(row_to_tensor, axis=1) for slice_ in slices) # mildly diabolical one-liner
        rows = [item for sublist in chunks for item in sublist]

    if bins:
        final = torch.stack(rows, dim=0).to(device="cpu", dtype=default_dtype)
    else:
        final = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=0).to(device="cpu", dtype=default_dtype)
    return final


class Trainer(IterableDataset):
    @profile
    def __init__(self, data, bins, batchsize=1024, autoencoder=True, fold=True, training=True, multithread=False, bin_overlap_frac=0, filters=None):
        super().__init__()
        print(len(data))
        self.data = data_tbl_apply_mask(data)
        self.data["npts"] = self.data["w1flux"].apply(len)
        self.data = self.data.sort_values("npts", ascending=False)
        self.bins = bins
        self.autoencoder = autoencoder
        self.batchsize = batchsize
        self.shuffle = False
        self.types = keeptypes
        self.multithread = multithread

        if training:
            self.data = self.data[self.data["type"].isin(self.types)]
        
        # print("starting fold")
        self.data = self.explode_and_fold()

        if filters:
            self.data = self.data[filters]

        # print("starting tensor transformation")
        # print(len(self.data))
        self.tensor = data_tbl_to_nn_input(self.data, bins, thread=multithread, bin_overlap_frac=bin_overlap_frac)
        if len(self.data) != len(self.tensor):
            raise ValueError("Data length mismatch")
        torch.nan_to_num(self.tensor, nan=0.0)
        if training:
            self.label = self.type_to_label(self.data["type"])
        
    def __iter__(self):
        self.i = 0
        if self.shuffle:
            self.tensor = self.tensor[torch.randperm(len(self.tensor))]
        return self
    
    def fold(self, row):
        if pd.isna(row["period"]):
            return (row["mjd"] - row["mjd"].min()) / (row["mjd"].max() - row["mjd"].min())
        t = (row["mjd"] % row["period"]) / row["period"]
        return t 
    
    def explode_and_fold(self):
        def periodfilter(p):
            if p > 170 and p < 190:
                return pd.NA
            return float(p)

        data = self.data
        periodcols = ["peak1", "peak2", "peak3", "peak4", "peak5", "best_sub_1d", "best_sub_10d", "best_sub_50d", "best_sub_1000d"]

        periodschemes = []
        for colname in ["peak1", "peak2", "peak3", "best_sub_1d", "best_sub_10d"]:
            scheme = data.copy()
            scheme["period"] = data[colname]
            scheme["period"] = scheme["period"].apply(periodfilter)
            scheme.dropna(subset=["period"], inplace=True)
            scheme.reset_index(inplace=True)
            scheme.drop_duplicates(subset=["period", "cluster_id"], inplace=True, keep="first")
            scheme.set_index("cluster_id", inplace=True)

            periodschemes.append(scheme)
        exploded = pd.concat(periodschemes)

        exploded["mjd"] = exploded.apply(self.fold, axis=1)
        exploded["period"] = exploded["period"].astype(float)

        # exploded["period_significance"] = exploded.apply(self.periodsignificance, axis=1)
        # sorted = exploded.sort_values("period_significance", ascending=False)
        # sameobj = sorted.groupby("cluster_id")
        # # keep k best periods
        # k = self.keep_pd
        # dropindices = sameobj.cumcount(ascending=False) > (k-1)
        # exploded = sorted[~dropindices]
        return exploded
    
    def type_to_label(self, types: pd.Series):
        n_types = len(self.types)
        vals = types.apply(lambda tp: self.types.index(tp)).values
        rowidx = torch.arange(len(vals))
        label = torch.zeros(len(vals), n_types)
        label[rowidx, vals] = 1.0
        return label
    
    def trim_tensor(self, tensor):
        is_zero = tensor == 0
        full_zero = torch.all(is_zero, dim=2)
        col_has_data = ~torch.all(full_zero, dim=0)
        # find the index of the first true value in col_has_data
        firstzero = torch.argmax(torch.cumsum(col_has_data, dim=0), dim=0) + 1
        return tensor[:, :max(20, firstzero), :]
    
    def random_selection(self, oftype, n):
        indices = torch.where(oftype)[0]
        rand = indices[torch.randperm(len(indices))[:-n]] # leave only n indices
        oftype[rand] = False
        return oftype

    @profile
    def _smooth_(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode="same")
        return y_smooth

    def periodsignificance(self, row):
        width = 0.1 # similar to 10 bins
        p = row["period"]
        if not p > 0:
            return
        t = np.array(row["mjd"])
        y = np.array(row["w1flux"])
        ynorm = (y - np.median(y)) / (np.quantile(y, 0.8) - np.quantile(y, 0.2))

        # Null hyp
        smoothed = self._smooth_(ynorm, int(len(ynorm) * width))
        mse_null = np.mean((ynorm - smoothed) ** 2)

        # Alt hyp
        phase = (t % p) / p
        sorter = np.argsort(phase)
        phase = phase[sorter]
        ynorm = ynorm[sorter]
        smoothed = self._smooth_(ynorm, int(len(ynorm) * width))
        mse_alt = np.mean((ynorm - smoothed) ** 2)

        improvement = mse_null / mse_alt
        return improvement
    
    def augment(self, tensor):
        n = tensor.shape[0]
        bins = tensor.shape[1]
        feat = tensor.shape[2]
        shift = torch.rand(tensor.shape[0], device=device).repeat(bins, 1).t()
        tensor[:, :, 0] += shift

        r = np.random.rand()
        if r < 0.4:
            tensor = torch.flip(tensor, dims=[1])
        r = np.random.rand()
        if r < 0.4:
            tensor[:, :, 0] = -tensor[:, :, 0]
        # r = np.random.rand() + 0.5
        # tensor[:, :, 0] = tensor[:, :, 0] * r
        return tensor

    def __next__(self):
        i = self.i
        t1 = time.time()
        if i >= len(self) - 1:
            raise StopIteration
        
        n = self.batchsize
        n_each = n // len(self.types)  
        label = torch.argmax(self.label, dim=1)

        tensors = []
        labels = []

        for i in range(len(self.types)):
            of_type = label == i
            indexer = self.random_selection(of_type, n_each).to(device)
            tensors.append(self.tensor[indexer])
            labels.append(self.label[indexer])

        input = torch.cat(tensors, dim=0)
        label = torch.cat(labels, dim=0)

        input = self.augment(input)
        t2 = time.time()

        self.i += 1
        return input, label
        
    def __len__(self):
        return ceil(len(self.tensor) / self.batchsize)

def binify(tens, bins):
        y = tens[:, :, 0]
        t = tens[:, :, 1]

        binned = []

        for i in range(bins):
            mask = (t >= i/bins) & (t < (i+1)/bins)
            copied = y.clone()
            copied[~mask] = np.nan
            mean = torch.nanmean(copied, dim=1)
            var = torch.nanmean((copied - mean.unsqueeze(1))**2, dim=1)
            features = torch.stack([mean, var], dim=1)
            binned.append(features)
        
        binned = torch.stack(binned, dim=1)

        # patching with interpolation
        needs_interp = torch.isnan(binned)
        # interpolator = [(binned[:,(i-1) % bins, :] + binned[:,(i+1) % bins, :]) / 2 for i in range(bins)]
        # interpolator = torch.stack(interpolator, dim=1)
        binned[needs_interp] = 0

        return binned

def rebase_phase(binned):
    binned = binned.detach().cpu().numpy()
    min_bin = np.argmin(binned[:, :, 0], axis=1)
    def rebase(batch,t,feature):
        batch = batch.astype(np.int32)
        t = t.astype(np.int32)
        feature = feature.astype(np.int32)
        original_index = (batch, (t + min_bin[batch]) % binned.shape[1], feature)
        return binned[original_index]
        
    permuted = torch.tensor(np.fromfunction(rebase, binned.shape), dtype=default_dtype)
    return permuted.to(device)

class TreeDL(Trainer):
    def __init__(self, data_tbl: pd.DataFrame, morpho_model: torch.nn.Module, training:bool=False, lc_bins=16, bin_overlap_frac=0, filters=None):
        if "type" not in data_tbl.columns and training:
            raise KeyError("flag_tbl must have a 'type' column in training mode")
        
        super().__init__(data_tbl, lc_bins, training=training, bin_overlap_frac=bin_overlap_frac, filters=filters)

        self.morpho_model_batchsize = 4096
        self.morpho_model = morpho_model.to(device)
        self.training = training

        self.featuretbl = None
        self.featuretbl = self.get_feature_tbl()

    def carry_over_flagtbl_features(self): 
        tbl = self.data
        # features stored in the flagtbl are period and crossmatched columns
        cols = ["W1mag", "W2mag", "W3mag", "W4mag", "Jmag", "Hmag", "Kmag", "period", "confidence"]
        tbl.loc[:, "W2mag"] = tbl["W2mag"] - tbl["W1mag"]
        tbl.loc[:, "W3mag"] = tbl["W3mag"] - tbl["W1mag"]
        tbl.loc[:, "W4mag"] = tbl["W4mag"] - tbl["W1mag"]
        tbl.loc[:, "Jmag"] = tbl["Jmag"] - tbl["W1mag"]
        tbl.loc[:, "Hmag"] = tbl["Hmag"] - tbl["W1mag"]
        tbl.loc[:, "Kmag"] = tbl["Kmag"] - tbl["W1mag"]
        tbl.loc[:, "W1mag"] = tbl["W1mag"] - tbl["W1mag"]
        
        return tbl[cols]
    
    def inverse_von_neumann(self, x):
        x = np.array(x)
        n = len(x)
        mu = np.mean(x)
        numer = n * np.sum(np.diff(x)**2)
        denom = (n-1) * np.sum((x - mu)**2)
        return denom / numer
    
    def gaussian_chi_2(self, x):
        x = x - np.mean(x)
        s = np.std(x)
        total = len(x)

        expected_prop = np.abs(scipy.stats.norm.cdf(-x, scale=s) - scipy.stats.norm.cdf(x, scale=s))
        expected_prop = np.sort(expected_prop)
        err = 0
        for n, e_prop in enumerate(expected_prop):
            if e_prop != 0:
                proportion = (n+1) / total
                err += (proportion - e_prop)**2 / e_prop
        return err
    
    def calc_Stetson_I(self, row):
        mag = np.array(row["w1flux"])
        err = np.array(row["w1sigflux"])
        N = len(mag)
        wmean = np.mean(mag)

        d = np.sqrt(1.0 * N / (N - 1)) * (mag - wmean) / err
        P = d[:-1] * d[1:]

        # stetsonI
        stetsonI = np.sum(P)

        return stetsonI
    
    def calc_Stetson_J(self, row):
        mag = np.array(row["w1flux"])
        err = np.array(row["w1sigflux"])
        N = len(mag)
        wmean = np.mean(mag)

        d = np.sqrt(1.0 * N / (N - 1)) * (mag - wmean) / err
        P = d[:-1] * d[1:]

        # stetsonJ
        stetsonJ = np.sum(np.sign(P) * np.sqrt(np.abs(P)))

        return stetsonJ
    
    def calc_Stetson_K(self, row):
        mag = np.array(row["w1flux"])
        err = np.array(row["w1sigflux"])
        N = len(mag)
        wmean = np.mean(mag)

        d = np.sqrt(1.0 * N / (N - 1)) * (mag - wmean) / err
        P = d[:-1] * d[1:]

        stetsonK = np.sum(abs(d)) / N
        stetsonK /= np.sqrt(1.0 / N * np.sum(d**2))

        return stetsonK

    def get_data_features(self):
        tbl = self.data
        i60r = tbl["w1flux"].apply(lambda x: np.quantile(x, 0.6) - np.quantile(x, 0.4))
        i75r = tbl["w1flux"].apply(lambda x: np.quantile(x, 0.75) - np.quantile(x, 0.25))
        i90r = tbl["w1flux"].apply(lambda x: np.quantile(x, 0.9) - np.quantile(x, 0.1))
        median_abs_dev = tbl["w1flux"].apply(lambda x: np.median(np.abs(x - np.median(x))))
        skew = tbl["w1flux"].apply(lambda x: pd.Series(x).skew())
        kurt = tbl["w1flux"].apply(lambda x: pd.Series(x).kurt())
        mean_chi_2 = tbl["w1rchi2"].apply(lambda x: np.mean(x))
        std_chi_2 = tbl["w1rchi2"].apply(lambda x: np.std(x))
        median_uncertainty = tbl["w1sigflux"].apply(lambda x: np.median(x))
        mad_uncertainty = tbl["w1sigflux"].apply(lambda x: np.median(np.abs(x - np.median(x))))
        iqr_unc_ratio = i75r / median_uncertainty
        ivn = tbl["w1flux"].apply(self.inverse_von_neumann)
        gaussian_chi_2 = tbl["w1flux"].apply(self.gaussian_chi_2)
        stetsonI = tbl.apply(self.calc_Stetson_I, axis=1)
        stetsonJ = tbl.apply(self.calc_Stetson_J, axis=1)
        stetsonK = tbl.apply(self.calc_Stetson_K, axis=1)

        # print("Getting period significance")
        sig = tbl.apply(self.periodsignificance, axis=1)
        # print("Done getting period significance")

        features = pd.DataFrame({"i60r": i60r, "i75r": i75r, "i90r": i90r, "median_abs_dev": median_abs_dev, 
                                 "skew": skew, "period_significance": sig, "mean_chi_2": mean_chi_2, "std_chi_2": std_chi_2,
                                 "median_uncertainty": median_uncertainty, "mad_uncertainty": mad_uncertainty, "iqr_unc_ratio": iqr_unc_ratio,
                                 "kurt": kurt, "ivn": ivn, "gaussian_chi_2": gaussian_chi_2, "stetsonI": stetsonI, "stetsonJ": stetsonJ, "stetsonK": stetsonK})
        return features

    def eval_morpho_model(self, morpho_model_input):
        out = self.morpho_model(morpho_model_input.to(device)).detach().cpu().numpy()
        n_features_out = out.shape[1]
        morpho_features = pd.DataFrame({f"feature_{i}": out[:,i] for i in range(n_features_out)})
        return morpho_features

    @profile
    def get_feature_tbl(self):
        t = self.tensor
        physical = self.carry_over_flagtbl_features()
        data_features = self.get_data_features()
        morphos = pd.concat([self.eval_morpho_model(t[self.morpho_model_batchsize*i:self.morpho_model_batchsize*(i+1)]) for i in range(len(t) // self.morpho_model_batchsize + 1)])

        morphos.index = self.data.index

        tables_to_join = [physical, morphos, data_features]

        feature_tbl = pd.DataFrame(index=self.data.index)
        for tbl in tables_to_join:
            for col in tbl.columns:
                feature_tbl[col] = tbl[col]
        if self.training:
            label = self.data["type"].apply(lambda x: self.types.index(x)).astype(int) # map type to int
            feature_tbl["type"] = label
        if len(feature_tbl) != len(self.data):
            raise ValueError("Feature table length mismatch")
        return feature_tbl
    
    def train_valid_test(self, trainfrac=0.8, validfrac=0.1):
        train = []
        valid = []
        test = []

        types = self.featuretbl.groupby("type")

        for type_, group in types:
            group = group.sample(frac=1, random_state=RANDOM_SEED) # shuffle
            n = len(group)
            trainidx = int(n * trainfrac)
            valididx = int(n * (trainfrac + validfrac))
            train.append(group.iloc[:trainidx])
            valid.append(group.iloc[trainidx:valididx])
            test.append(group.iloc[valididx:])
        
        train = pd.concat(train)
        valid = pd.concat(valid)
        test = pd.concat(test)
        return train, valid, test
