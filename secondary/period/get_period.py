from periodfind import ls, ce, aov
import numpy as np
import pandas as pd
from math import ceil

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
    if periods.shape[0] != sigs.shape[0]:
        periods = periods[:, None].repeat(sigs.shape[0], axis=1).T # make into a matrix

    rank = np.argsort(sigs, axis=1)[:, ::-1][:, :1000]
    sigs = np.take_along_axis(sigs, rank, axis=1).T
    periods = np.take_along_axis(periods, rank, axis=1).T

    for i in range(periods.shape[0]):
        subtractor = periods[i]
        subtractor[(subtractor == False) | (subtractor == np.nan)] = 0
        dperiod = np.abs(periods - subtractor)
        invalids = (dperiod < dist*subtractor) & (dperiod != 0)
        sigs[invalids] = 0.0
        periods[invalids] = np.nan

    sigs = sigs.T
    periods = periods.T

    newrank = np.argsort(sigs, axis=1)[:, ::-1][:, :n]
    periods = np.take_along_axis(periods, newrank, axis=1)
    periodsigs = np.take_along_axis(sigs, newrank, axis=1)
    return periods, periodsigs

def plavchan_pgram(times, mags, periods, windowsize):
    # null hyp
    if len(times) != len(mags) or len(times) != len(periods):
        raise ValueError("Length mismatch")
    
    scores = []
    
    for t,m,p in zip(times, mags, periods):
        var = np.var(m)
        scores_obj = []
        for trialperiod in p:
            fold = (t % trialperiod) / trialperiod
            argsort = np.argsort(fold)
            fold = fold[argsort]
            m = m[argsort]

            filtsize = int(windowsize * len(m))
            filt = np.ones(filtsize) / filtsize
            smoothed = np.convolve(m, filt, mode="same")
            mse = np.mean((m - smoothed) ** 2)
            scores_obj.append(var/mse)
        scores.append(scores_obj)
    
    return np.array(scores)

def decide_periods(periods, ls, ce, aov, times, mags):
    oneday = np.cumsum(periods > 1).tolist().index(1)
    tenday = np.cumsum(periods > 10).tolist().index(1)
    fiftyday = np.cumsum(periods > 50).tolist().index(1)

    # norm
    ls = (ls - np.min(ls, axis=1)[:, None])
    ce = (ce - np.min(ce, axis=1)[:, None])
    aov = (aov - np.min(aov, axis=1)[:, None])

    # peaks_ls_3 = peaktable(periods, pgram, 5, 0.125) 
    # peaks_ls_5 = peaktable(periods, ls, 5, 0.125)
    # peaks_aov_5 = peaktable(periods, aov, 5, 0.125)
    # best_10 = np.concatenate([peaks_ls_5, peaks_aov_5], axis=1)
    # pchan = plavchan_pgram(times, mags, best_10, 0.065)
    # pchan_order = np.argsort(pchan, axis=1)[:, :-6:-1]
    # peaks_5 = np.take_along_axis(best_10, pchan_order, axis=1)
    # peaks_5_sig = np.take_along_axis(pchan, pchan_order, axis=1)
    peaks_5, _ = peaktable(periods, aov, 5, 0.125)
    peaks_5_sig = plavchan_pgram(times, mags, peaks_5, 0.05)

    # tr = go.Scatter(x=periods, y=pgram[0], mode="lines", name="Combined")
    # pdslist = list(periods)
    # indexes = [pdslist.index(p) for p in peaks[0]]

    # ls = go.Scatter(x=periods, y=ls[0], mode="lines", name="LS")
    # ce = go.Scatter(x=periods, y=ce[0], mode="lines", name="CE")
    # aov = go.Scatter(x=periods, y=aov[0], mode="lines", name="AOV")

    # trpeak = go.Scatter(x=peaks[0], y=pgram[0][indexes], mode="markers", name="Peaks", marker=dict(color="red"))

    # fig = go.Figure(data=[tr, trpeak])
    # fig.write_html("testpgram.html")

    # best_sub_1d = periods[np.argmax(ls[:, :oneday] + ce[:, :oneday] + aov[:, :oneday], axis=1)]
    # best_sub_10d = periods[np.argmax(ls[:, oneday:tenday] + ce[:, oneday:tenday] + aov[:, oneday:tenday], axis=1) + oneday]
    # best_sub_50d = periods[np.argmax(ls[:, tenday:fiftyday] + ce[:, tenday:fiftyday] + aov[:, tenday:fiftyday], axis=1) + tenday]
    # best_50plus = periods[np.argmax(ls[:, fiftyday:] + ce[:, fiftyday:] + aov[:, fiftyday:], axis=1) + fiftyday]

    best_found_periods = pd.DataFrame({
        "peak1": peaks_5[:, 0],
        "peak1_sig": peaks_5_sig[:, 0],
        "peak2": peaks_5[:, 1],
        "peak2_sig": peaks_5_sig[:, 1],
        "peak3": peaks_5[:, 2],
        "peak3_sig": peaks_5_sig[:, 2],
    })

    return best_found_periods

def excise_aliased_periods(times, periods):
    # times = np.sort(times, axis=1)
    # diff = np.diff(times, axis=1)
    # diff[diff < 50] = np.nan
    # longcadence = float(np.mean(np.nanmean(diff, axis=1), axis=0))
    longcadence = 177.5 

    multiples = [1/3, 1/2, 1, 2, 4/3, 3/2, 3, 4]
    excise_radius = 0.04 # % of the period

    for m in multiples:
        bad = longcadence * m
        bad_start = bad - excise_radius * bad
        bad_end = bad + excise_radius * bad
        periods = periods[(periods < bad_start) | (periods > bad_end)]

    return periods

def prefer_double(periods, sigs):
    for sig in sigs:
        maxpd = periods[np.argmax(sig)]
        if maxpd > 4:
            continue
        double = maxpd * 2
        idxdist = np.abs(periods - double)
        closest = np.argmin(idxdist)
        sig[closest] *= 2
        print("Doubling! ")
    return sigs
    

def get_period(dataframe, trialperiods, batchsize=1024, return_pgram=False):
    trialperiods = np.array(trialperiods, dtype=np.float32)
    t_total = list(dataframe["time"].apply(lambda x: np.array(x, dtype=np.float32)).values)
    m_total = list(dataframe["mag"].apply(lambda x: np.array(x, dtype=np.float32)).values)

    batchsize = min(batchsize, len(dataframe))

    chunks = [(t_total[i*batchsize:(i+1)*batchsize], m_total[i*batchsize:(i+1)*batchsize]) for i in range(ceil(len(dataframe) / batchsize))]

    best_periods = []
    for t, m in chunks:
        pds = excise_aliased_periods(t, trialperiods)


        pgram_args_AOV = {
            "times": t,
            "mags": m,
            "periods": pds[pds<15.0],
            "period_dts": np.array([0.0], dtype=np.float32),
            "n_stats": len(pds[pds<15.0]) - 1,
            "n_phase": 12,
        }

        pgram_args_CE = {
            "times": t,
            "mags": m,
            "periods": pds,
            "period_dts": np.array([0.0], dtype=np.float32),
            "n_stats": len(pds) - 1,
            "n_phase": 12,
        }

        pgram_args_LS = {
            "times": t,
            "mags": m,
            "periods": np.concat([pds[pds<15.0][::50], pds[pds>15.0]]),
            "period_dts": np.array([0.0], dtype=np.float32),
            "n_stats": len(pds[pds>15.0]) - 1,
        }

        p_aov, sigaov = pgram("aov", pgram_args_AOV)
        p_ls, sigls = pgram("ls", pgram_args_LS)
        sigls = sigls[p_ls > 15.0]
        p_ls = p_ls[p_ls > 15.0]
        sigce = np.zeros_like(sigaov)
        
        p = np.concatenate([p_ls, p_aov])
        sig = np.concatenate([sigaov, sigls], axis=1)

        best = decide_periods(p, np.zeros_like(sig), np.zeros_like(sig), sig, t, m)
        print("Decide done")
        best_periods.append(best)
    
    best_periods = pd.concat(best_periods)
    best_periods["cluster_id"] = dataframe.index
    best_periods.set_index("cluster_id", inplace=True)
    pgrams = (p, (sigls, sigce, sigaov))
    if return_pgram:
        return best_periods, pgrams
    return best_periods