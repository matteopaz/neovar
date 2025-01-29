from periodfind import ls, ce, aov
import numpy as np
import pandas as pd
from math import ceil
from line_profiler import profile

def pgram(type, args):
    print("Running {}".format(type))
    nbins = args["n_phase"] if "n_phase" in args else 10
    args = {k: v for k, v in args.items() if k != "n_phase"}
    if type == "ls":
        pg = ls.LombScargle().calc(**args)
    elif type == "ce":
        pg = ce.ConditionalEntropy(n_phase=nbins).calc(**args)
    elif type == "aov":
        pg = aov.AOV(n_phase=nbins).calc(**args)
    else:
        raise ValueError("Invalid periodogram type")
    
    sigs = np.array([[a.significance for a in stats] for stats in pg], dtype=np.float32)
    pds = np.array([[a.params[0] for a in stats] for stats in pg], dtype=np.float32)

    sorter = np.argsort(pds, axis=1)
    pds = np.take_along_axis(pds, sorter, axis=1)
    sigs = np.take_along_axis(sigs, sorter, axis=1)

    # rescale to 0-1
    # sigs = (sigs - np.min(sigs, axis=1)[:, None]) / (np.max(sigs, axis=1) - np.min(sigs, axis=1))[:, None]
    return pds[0], sigs

def peaktable(periods, sigs, n, dist): # actually kind of hard
    # if periods.shape[0] != sigs.shape[0]:
    #     periods = periods[:, None].repeat(sigs.shape[0], axis=1).T # make into a matrix

    # rank = np.argsort(sigs, axis=1)[:, ::-1][:, :1000]
    # sigs = np.take_along_axis(sigs, rank, axis=1).T
    # periods = np.take_along_axis(periods, rank, axis=1).T

    # periods[sigs == 0.0] = np.nan

    # for i in range(periods.shape[0]):
    #     subtractor = np.array(periods[i])
    #     subtractor[np.isnan(subtractor)] = 0
    #     dperiod = np.abs(periods - subtractor)
    #     invalids = (dperiod < dist*subtractor) & (dperiod != 0)
    #     sigs[invalids] = 0.0
    #     periods[invalids] = np.nan

    # sigs = sigs.T
    # periods = periods.T

    # newrank = np.argsort(sigs, axis=1)[:, ::-1][:, :n]
    # periods = np.take_along_axis(periods, newrank, axis=1)
    # periodsigs = np.take_along_axis(sigs, newrank, axis=1)
    # return periods, periodsigs
    peaks = []
    peaks_sig = []
    for s in sigs:
        p = np.array(periods)
        s = np.array(s)
        peaks_obj = []     
        peaks_obj_sig = []

        sort = np.argsort(s)[::-1]
        p = p[sort]
        s = s[sort]

        for i in range(n):
            peaks_obj.append(p[0])
            peaks_obj_sig.append(s[0])
            idxr = np.abs(p - p[0]) > dist*p[0]
            p = p[idxr]
            s = s[idxr]
        peaks.append(peaks_obj)
        peaks_sig.append(peaks_obj_sig)
    return np.array(peaks), np.array(peaks_sig)
        


def plavchan_pgram(times, mags, periods, windowsize):
    # null hyp
    if len(times) != len(mags) or len(times) != len(periods):
        raise ValueError("Length mismatch")
    
    scores = []
    
    for t,m,p in zip(times, mags, periods):
        null_hyp = np.mean((m - np.mean(m))**2)
        scores_obj = []
        for trialperiod in p:
            fold_t = (t % trialperiod) / trialperiod
            argsort = np.argsort(fold_t)
            fold_t = fold_t[argsort]
            fold_m = m[argsort]

            smoothed = np.zeros_like(m)
            for i in range(len(fold_t)):
                box_start = (fold_t[i] - windowsize / 2) % 1
                box_end = (fold_t[i] + windowsize / 2) % 1
                if box_start < box_end:
                    box = (fold_t >= box_start) & (fold_t <= box_end)
                else:
                    box = (fold_t >= box_start) | (fold_t <= box_end)
                smoothed[i] = np.mean(fold_m[box])
            
            mse = np.mean((fold_m - smoothed) ** 2)

            scores_obj.append(null_hyp/mse)
        scores.append(scores_obj)
    
    return np.array(scores)

def excise_aliased_periods(times, periods):
    diff = np.diff(times)

    shortcadence = np.median(diff[diff < 50])
    longcadence = np.median(diff[diff > 50])

    multiples = [1/3, 1/2, 1, 2, 4/3, 3/2, 3, 4]
    excise_radius = 0.085 # frac of the period

    mask = np.zeros_like(periods, dtype=bool)

    for cadence in [shortcadence, longcadence]:
        for m in multiples:
            bad = cadence * m
            bad_start = bad - excise_radius * bad
            bad_end = bad + excise_radius * bad
            mask[(periods < bad_start) | (periods > bad_end)] = True

    return mask
    
@profile
def get_period(dataframe, trialperiods, batchsize=1024, return_pgram=False, peak_resolution_pct=10, **kwargs):

    trialperiods = np.array(trialperiods, dtype=np.float32)
    t_total = list(dataframe["time"].apply(lambda x: np.array(x, dtype=np.float32)).values)
    m_total = list(dataframe["mag"].apply(lambda x: np.array(x, dtype=np.float32)).values)

    batchsize = min(batchsize, len(dataframe))

    chunks = [(t_total[i*batchsize:(i+1)*batchsize], m_total[i*batchsize:(i+1)*batchsize]) for i in range(ceil(len(dataframe) / batchsize))]

    best_periods = []
    for t, m in chunks:
        # excise = excise_aliased_periods(t, trialperiods)
        pds = trialperiods
        if "mask" in kwargs:
            kwargs["mask"] = kwargs["mask"]

        aov_pds = pds[pds<15.0]
        pgram_args_AOV = {
            "times": t,
            "mags": m,
            "periods": aov_pds,
            "period_dts": np.array([0.0], dtype=np.float32),
            "n_stats": len(aov_pds),
            "n_phase": 12,
        }
        
        ls_pds = pds[pds>=15.0]
        pgram_args_LS = {
            "times": t,
            "mags": m,
            "periods": ls_pds,
            "period_dts": np.array([0.0], dtype=np.float32),
            "n_stats": len(ls_pds),
        }

        p_aov, sigaov = pgram("aov", pgram_args_AOV)
        if len(ls_pds) != 0:
            p_ls, sigls = pgram("ls", pgram_args_LS)
            sigls = sigls[:, p_ls >= 15.0] * (1.7 + 0.75*np.log10(p_ls[p_ls >= 15.0]/15.0)) # TOO HIGH when period is very large. should plateu harder
        else:
            p_ls = np.array([[] for _ in range(len(t))])
            sigls = np.array([[] for _ in range(len(t))])
        p_ls = p_ls[p_ls >= 15.0]

        # p = np.concatenate([p_aov, p_ls])
        sig = np.concatenate([sigaov, sigls], axis=1)
        # MASKING for PERIOD REFINEMENT
        if "mask" in kwargs:
            # mask is a list of integers the same length as the number of periods, where each integer corresponds to a mask
            mask = kwargs["mask"]
            if len(mask) != len(pds):
                raise ValueError("Mask length does not match number of periods")
            if np.max(mask)+1 != sig.shape[0]:
                raise ValueError("Mask value out of bounds")
            
            for i in range(sig.shape[0]):
                sig[i, mask != i] = 0.0

        # sig is the final significance array, shaped (n_stars, n_periods)
        excise_matrix = np.array([excise_aliased_periods(ex_times, pds) for ex_times in t])
        print("AAA")
        print(excise_matrix.shape)
        print(np.any(excise_matrix, axis=1))


        print(sig[excise_matrix].shape)

        sig[excise_matrix] = 0.0

        peaks_3, _ = peaktable(pds, sig, 3, 0.1)
        plavchan_sigs = plavchan_pgram(t, m, peaks_3, 0.1)


        best_long_period_idx = np.argmax(sig[:, pds > 30.0], axis=1) if (pds > 30.0).any() else np.zeros(len(sig))
        best_long_period = pds[pds > 30.0][best_long_period_idx] if (pds > 30.0).any() else np.zeros(len(sig))
        best_long_period_sig = sig[:, pds > 30.0][np.arange(len(sig)), best_long_period_idx] if (pds > 30.0).any() else np.zeros(len(sig))


        best = pd.DataFrame({
            "peak1": peaks_3[:, 0],
            "peak1_sig": plavchan_sigs[:, 0],
            "peak2": peaks_3[:, 1],
            "peak2_sig": plavchan_sigs[:, 1],
            "peak3": peaks_3[:, 2],
            "peak3_sig": plavchan_sigs[:, 2],
            "best_long_period": best_long_period,
            "best_long_period_sig": best_long_period_sig
        })

        for col in best.columns:
            best[col] = best[col].apply(lambda x: np.round(x, 7))

        print("Decide done")
        best_periods.append(best)
    
    best_periods = pd.concat(best_periods)
    best_periods["cluster_id"] = dataframe.index
    best_periods.set_index("cluster_id", inplace=True)

    sigls_padded = np.concatenate((np.zeros((sigls.shape[0], sigls.shape[1] - len(p_ls))), sigls), axis=1)
    sigaov_padded = np.concatenate((sigaov, np.zeros((sigaov.shape[0], sigaov.shape[1] - len(p_aov)))), axis=1)
    sigce = np.zeros_like(sigaov)
    pgrams = (pds, (sigls_padded, sigce, sigaov_padded))
    if return_pgram:
        return best_periods, pgrams
    return best_periods