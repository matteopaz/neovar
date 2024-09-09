import torch
from torch.utils.data import IterableDataset
import numpy as np
import plotly.graph_objects as go
from joblib import Parallel, delayed

class GenSet(IterableDataset):
    def __init__(self, batchsize=384, randseed=1):
        super(GenSet).__init__()
        self.counter = 0

        np.random.seed(randseed)
        
        self.batchsize = batchsize

        self.days = 4000
        self.longspacing = 175
        self.shortspacing = 0.115
        self.apparition_frequencies = [10, 11, 12, 13, 13, 10, 14, 14, 14, 14, 15, 15, 16, 16, 20, 21, 24, 35, 54]
        self.next = None
    
    def __iter__(self):
        self.counter = 0
        return self
    
    def __next__(self):
        return self.gen()
    
    def gen(self):
        def getgenfunc(i):
            if i % 5 < 2:
                return self.gen_null
            elif i % 5 == 2:
                return self.gen_nova
            elif i % 5 == 3:
                return self.gen_pulsating_var
            elif i % 5 == 4:
                return self.gen_transit

        pattern = [0,0,1,2,2]

        batchlist = Parallel(n_jobs=1)(delayed(getgenfunc(i))() for i in range(self.batchsize)) # faster
        labels = [pattern[i % 5] for i in range(self.batchsize)]
        onehot = lambda i: [1.0,0.0,0.0] if i == 0 else [0.0,1.0,0.0] if i == 1 else [0.0,0.0,1.0]
        labels = torch.tensor([onehot(i) for i in labels]).cuda()
        batch = torch.nn.utils.rnn.pad_sequence(batchlist, batch_first=True).cuda()
        perm = torch.randperm(batch.shape[0])

        return batch[perm], labels[perm]

    def s(self,bounds): # sample
        return np.random.uniform(bounds[0], bounds[1])

    def g(self,x, std): # gaussian
        return np.random.normal(x, std)

    def baseflux(self): # Distribution stuff
        r = self.s([-4,0.5])
        return 1.5 * 10**r

    def getstd(self,flux):
        r = self.s([-4, -1])
        return 10**r
    
    def apparitionsbeforegap(self):
        return np.random.choice(self.apparition_frequencies)

    def gen_sampling(self, sparse=0):
        x = [0]
        i = 0
        app = self.apparitionsbeforegap()
        while x[-1] < self.days:
            if i % app == 0:
                x.append(x[-1] + self.longspacing)
            else:
                if np.random.random() > 0:
                    x.append(x[-1] + self.shortspacing)
                x.append(x[-1] + self.shortspacing)
            i += 1
        return np.array(x)
    
    def to_datatens(self,t,flux,err):
        if type(flux) == np.ndarray:
            y = torch.tensor(flux, dtype=torch.float32)
        if type(t) == np.ndarray:
            x = torch.tensor(t, dtype=torch.float32)

        NOISE_TO_PRED_ERR_RATIO = self.s([0.4,0.6]) # Important factor. Empirically determined

        sigflux = torch.tensor(err, dtype=torch.float32) * NOISE_TO_PRED_ERR_RATIO
        sigflux += torch.randn_like(sigflux) * 0.3 * err # add some noise to error

        # Augmentations
        # Cosmic Ray Hits
        pct_rays = self.s([0.01, 0.02])
        n_rays = int(pct_rays * len(y))
        indices = np.random.choice(len(y), n_rays, replace=False)

        center = torch.median(y)
        spread = torch.quantile(y, 0.95) - torch.quantile(y, 0.05)
        values = [center + spread * self.s([-1,1]) for i in range(n_rays)]

        y[indices] = torch.tensor(values)
        sigflux[indices] = self.s([0.2, 4]) * sigflux[indices]

        # Random Misses (dropout)
        pct_miss = self.s([0.05, 0.4])
        n_miss = int(pct_miss * len(y))
        indices = np.random.choice(len(y), n_miss, replace=False)
        
        y = y[~np.isin(np.arange(len(y)), indices)]
        sigflux = sigflux[~np.isin(np.arange(len(sigflux)), indices)]
        x = x[~np.isin(np.arange(len(x)), indices)]

       # Shuffle lightcurve
        if np.random.random() > 0.5:
            perm = torch.randperm(len(y))
            y = y[perm]
            sigflux = sigflux[perm]
            x = x[perm]
        
       # shift x times

        # if np.random.random() > 0.2:
        #     x = x + torch.randn_like(x) * (0.1/4000) * (x[-1] - x[0])

      #  Reverse lightcurve
        # if np.random.random() > 0.9:
        #     y = y.flip(0)
        #     sigflux = sigflux.flip(0)
        #     x = x.flip(0)
        
        # flip lightcurve

        # if np.random.random() > 0.9:
        #     y = -y

        centered_y = y - torch.median(y)
        scale_factor = torch.quantile(centered_y, 0.75) - torch.quantile(centered_y, 0.25)
        zscored = centered_y / scale_factor
        sigflux = sigflux / scale_factor

        asin_y = torch.asinh(zscored)
        sigflux = torch.asinh(sigflux)

        x1 = (x - torch.min(x)) / 4000

        out = torch.stack([asin_y, sigflux, x1], dim=0).T.to(torch.float32)
      
        return out
 
    
    def gen_null(self):
        brightness = self.baseflux()
        std = self.getstd(brightness)

        def get_null_func(bright):
            def eval(x):
                return self.g(bright, std), std

            return eval
        light_func = np.vectorize(get_null_func(brightness))
        x = self.gen_sampling(sparse=(np.random.random() > 0))

        y, err = light_func(x)
        return self.to_datatens(x,y,err)
    
    def gen_nova(self):
        brightness = self.baseflux()
        std = self.getstd(brightness)

        def get_nova_func(brightness, amplitude, duration, sharpness):
            sharpness = sharpness*1.5 + 2
            t_0 = self.s([0, 4000 - duration])
            sparse = np.random.random() > 0.8
            sparse_drop_chance = self.s([0.6,1.6])

            def lognormal_raw(x): # The basic curve shape
                leading_coeff = 1 / (x * np.sqrt(2 * np.pi))
                exponent = -0.5 * (np.arccosh(duration / x) - sharpness)**2
                return leading_coeff * np.exp(exponent)
            
            maximum_point = duration*np.exp(-sharpness)
            maximum_value = lognormal_raw(maximum_point)
            end_value = lognormal_raw(duration)
            lognormal = lambda x: brightness + max(0, (amplitude / maximum_value) * (lognormal_raw(x) - end_value)) # Normalize to amplitude

            max_spread_factor_at_peak = 0.5*10**self.s([0,1.25])

            sparse_dist = self.s([180, 450])

            def eval(x):
                spread = std
                if x > t_0 and x < t_0 + duration:
                    val = lognormal(x - t_0)
                    val = self.g(val, spread)
                    if np.abs(x - t_0) < 200:
                        spread *= max_spread_factor_at_peak
                else:
                    val = self.g(brightness, spread)
                
                if sparse and np.abs(x - maximum_point) > sparse_dist: # if far off the nova, make it sparse
                    if np.random.random() < sparse_drop_chance:
                        return np.nan, np.nan
                    
                return val, spread
            return eval

        amplitude = 10**self.s([0.5, 3]) * std
        duration = self.s([500, 3000])
        sharpness = self.s([0.8,1.4]) # ALWAYS BETWEEN 0 and 1

        starfunc = np.vectorize(get_nova_func(brightness, amplitude, duration, sharpness))
        x = self.gen_sampling()
        y, err = starfunc(x)

        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        err = err[~np.isnan(err)]

        zscore = (y - np.median(y)) / (np.quantile(y, 0.75) - np.quantile(y, 0.25))
        if np.max(np.abs(zscore)) < 3:
            return self.gen_nova()

        return self.to_datatens(x,y,err)

    
    def gen_transit(self, returnpd=False):
        
        brightness = self.baseflux()
        std = self.getstd(brightness)

        def get_transit_func(bright, depth, period, smoothing):
            primary_transit_time = 0.25 * period
            secondary_transit_time = 0.75 * period
            phase = self.s([0, period])
            secondary_transit_depth = self.s([0, 1])
            sharpness = period * smoothing * 0.2
            def eval(x):
                xp = (x+ phase) % period

                primary_impact = -depth*np.exp(-1/(sharpness**2) * (xp - primary_transit_time)**2)
                secondary_impact = -secondary_transit_depth*depth*np.exp(-1/(sharpness**2) * (xp - secondary_transit_time)**2)

                val = bright + primary_impact + secondary_impact
                return self.g(val, std), std
                
            return eval
        
        period = self.s([0.1, 3])
        depth = self.s([2.5*std, 30*std])
        smoothing = self.s([0.04, 0.5])
        starfunc = np.vectorize(get_transit_func(brightness, depth, period, smoothing))
        x = self.gen_sampling()
        y, err = starfunc(x)
        if returnpd:
            return (self.to_datatens(x,y,err), period / 4000)
        return self.to_datatens(x,y,err)

            
            
    
    def gen_pulsating_var(self, returnpd=False):
        # ampsrange = [0, 0.75]
        # periodsrange = [0.1, 200] # min and max in days

        # def get_period_func(bright, amps, periods, phases):
        #     amps = bright * np.array(amps)
        #     freqs = 2 * np.pi / np.array(periods)
        #     k = np.random.random()

        #     def eval(x):
        #         val = bright
        #         for A, F, P in zip(amps, freqs, phases):
        #             currentval = A * np.sin(F * (x + P))
        #             val += currentval
        #         return self.g(val, self.getstd(val))
        #     return eval
                
        # r = np.random.random()
        # if r > 0.7:
        #     periods = [self.s([0.075, 1]), self.s([10, 60])]
        # elif r > 0.2:
        #     periods = [self.g(40,15), self.g(35, 30)]
        # else:   
        #     periods = [self.s([1, 100]), self.s([10, 500])]

        
        # periods = np.clip(periods, periodsrange[0], periodsrange[1])

        # brightness = self.baseflux()
        # std = self.getstd(brightness)
        # amps = [self.g(1.75*std/brightness, 0.1), self.g(1.75*std/brightness, 0.1)] # Clip to 2.25 snr
        # amps = np.clip(np.abs(amps), 3*std / brightness, ampsrange[1]) #Clip to 1.5 snr

        # phases = [self.s([0,periods[0]]), self.s([0,periods[1]])]
        # starfunc = np.vectorize(get_period_func(brightness, amps, periods, phases))
        # x = self.gen_sampling()
        # y = starfunc(x)  
        # if returnpd:
        #     return (self.to_datatens(x,y), periods[0] / np.max(x), phases[0] / np.max(x))
        # return self.to_datatens(x,y)   

        brightness = self.baseflux()
        std = self.getstd(brightness)

        order = self.s([-1, 4])
        period = 5**order

        max_amp = 2*std + np.abs(self.g(0, 3*std))

        gridres = 110
        grid = np.zeros(gridres)

        windowsize = gridres // 2.7

        modifier = windowsize * 6
        filt = (1 / np.sqrt(np.pi*modifier)) * np.exp(-(1 / modifier)*np.arange(-windowsize, windowsize)**2)
        # filt[:int(windowsize)] += 0.001
        # if selector > 0.4: # 60% mountainrange waveform
        n_peaks = np.random.randint(2, 4)

        centers = []
        for i in range(n_peaks):
            centers.append(int(self.s([gridres*i / n_peaks + 1, gridres*(i+1) / n_peaks - windowsize/2])))
        heights = [(-1)**k * self.s([0.05, 1]) for k in range(n_peaks)] 
        grid[centers] = heights

        padding = np.zeros(int(windowsize//3))
        grid = np.concatenate((padding, grid, padding))

        grid = np.convolve(grid, filt, mode="same")
        # relative to eachother peaks. Biggest peak can be 4.5 times bigger than smallest
        # Also alternates signs with (-1)**k

        grid = (grid * max_amp / np.max(grid)) + brightness

        def eval(xr):
            x = xr % period
            idx = (len(grid) - 1) * x / period
            # get the two closest points, and interpolate
            u,v = np.floor(idx), np.ceil(idx)
            interp = grid[int(u)] + (grid[int(v)] - grid[int(u)]) * (idx - u)
            return self.g(interp, std), std # FLAG
        
        x = self.gen_sampling()
        y, err = np.vectorize(eval)(x)
        if returnpd:
            return (self.to_datatens(x,y,err), period / 4000, 0)
        return self.to_datatens(x,y,err)