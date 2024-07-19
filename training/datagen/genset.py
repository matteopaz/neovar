import torch
from torch.utils.data import IterableDataset
import numpy as np
import plotly.graph_objects as go
from joblib import Parallel, delayed
from line_profiler import LineProfiler

class GenSet(IterableDataset):
    def __init__(self, epochsize=4000, batchsize=400, randseed=0, valid=False, aggfreq=False, prop=[0.25, 0.25, 0.25, 0.25]):
        super(GenSet).__init__()
        self.valid = valid
        self.counter = 0
        self.aggfreq = aggfreq
        self.prop = np.array(prop)

        np.random.seed(randseed)

        if epochsize % batchsize != 0:
            raise Exception("Epoch size must be divisible by batch size")
        if epochsize % 4 != 0:
            raise Exception("Epoch size must be divisible by 4")
        
        self.epochsize = epochsize
        self.batchsize = batchsize

        self.days = 4000
        self.longspacing = 175
        self.shortspacing = 0.115
        self.apparition_frequencies = [10, 10, 10, 10, 10, 10, 14, 14, 14, 14, 20, 20, 20, 30, 30, 50, 100, 200, 100000]
        self.next = None

        if valid:
            self.valid_data = self.gen()


    def __len__(self):
        return self.epochsize // self.batchsize
    
    def __iter__(self):
        self.counter = 0
        return self
    
    def __next__(self):
        if self.valid and self.counter == 0:
            self.counter += 1
            return self.valid_data
        elif self.valid and self.counter == 1:
            raise StopIteration()
        else:
            return self.gen()
    
    def gen(self):
        n_each = self.batchsize // 4

        r = np.random.random()
        if r > 0.85: # .15 chance of polars
            self.apparition_frequencies = [10, 11, 12, 13, 14, 15, 16, 52, 102, 240, 210, 100000, 500, 50, 110]
            n_each = self.batchsize // 16 # to make room in memory
        else:
            self.apparition_frequencies = [10, 11, 12, 13, 13, 10, 14, 14, 14, 14, 20, 21, 24, 30, 35, 54, 100]


        split = [int(self.batchsize * self.prop[i]) for i in range(4)]
        if np.sum(split) != self.batchsize:
            raise Warning("Invalid Proportion of Examples")
        assert self.batchsize == np.sum(split)

        def getgenfunc(i):
            thresh = np.cumsum(split)
            if i < thresh[0]:
                return self.gen_null
            elif i < thresh[1]:
                return self.gen_nova
            elif i < thresh[2]:
                return self.gen_pulsating_var
            elif i < thresh[3]:
                return self.gen_transit

        
        batchlist = Parallel(n_jobs=1)(delayed(getgenfunc(i))() for i in range(self.batchsize)) # faster
        if not self.aggfreq:
            labels = torch.tensor([[1.0,0.0,0.0,0.0]]*split[0]+ [[0.0,1.0,0.0,0.0]]*split[1]+ [[0.0,0.0,1.0,0.0]]*split[2]+ [[0.0,0.0,0.0,1.0]]*split[3]).cuda()
        else:
            labels = torch.tensor([[1.0,0.0,0.0]]*split[0]+ [[0.0,1.0,0.0]]*split[1]+ [[0.0,0.0,1.0]]*(split[2]+split[3])).cuda()
        batch = torch.nn.utils.rnn.pad_sequence(batchlist, batch_first=True).cuda()
        perm = torch.randperm(batch.shape[0])
        return batch[perm], labels[perm]

    def s(self,bounds): # sample
        return np.random.uniform(bounds[0], bounds[1])

    def g(self,x, std): # gaussian
        return np.random.normal(x, std)

    def baseflux(self): # Distribution stuff
        r = self.s([-6.5,2])
        fluxbound = [0.0002,0.2]
        f = lambda x: (fluxbound[1] - fluxbound[0]) / (1 + np.exp(-3*x)) + fluxbound[0]
        return f(r)

    def getstd(self,flux):
        if flux < 0:
            return -flux / 2
        uncertainty = 0.713 * (0.002*flux)**0.8+0.000018 # Canonically accurate noise vs flux # UNCERTAINTY / FLUX / MAGNITUDE / SNR keywords
        return uncertainty / 0.85
        # mag = -2.5 * np.log10(flux / 309.54)
        # uncertainty = 2.383*10**(-7)*np.exp(0.8461*mag) + 0.02
        # toflux = lambda m: 309.54 * 10**(-m / 2.5)
        # fluxunc = toflux(mag - uncertainty / 2) - toflux(mag + uncertainty /2)
        # return fluxunc / 10 # Sets uncertainty value to 95% confidence error
    
    def apparitionsbeforegap(self):
        return np.random.choice(self.apparition_frequencies)

    def gen_sampling(self):
        x = [0]
        i = 0
        app = self.apparitionsbeforegap()
        while x[-1] < self.days:
            if i % app == 0:
                x.append(x[-1] + self.longspacing)
            else:
                x.append(x[-1] + self.shortspacing)
            i += 1
        return np.array(x)
    
    def to_datatens(self,x,y):
        if type(y) == np.ndarray:
            y = torch.tensor(y)
        if type(x) == np.ndarray:
            x = torch.tensor(x)

        y = torch.abs(y)

        snr = -torch.log10(y) / 3
        if torch.isnan(snr).any():
            raise ValueError("SNR has nan")
        y1 = (y - torch.mean(y)) / torch.std(y)
        y1 = torch.arcsinh(y1)
        # x1 = (x - torch.min(x)) / torch.max(x)
        x1 = (x - torch.min(x)) / 4000
        out = torch.stack([y1, snr, x1], dim=0).T.to(torch.float32)

        # Augmentations

        if np.random.random() > 0.2: # add 2-3 sigma noise, emulating bad frames
            num = int(self.s([5, 0.075*len(x1)]))
            # choose num random x values from x1
            indices = torch.randperm(len(x1) - 1)[:num]
            values = torch.tensor([self.s([1,2.5]) if np.random.random() > 0.5 else self.s([-2.5,-1]) for _ in range(num)])
            out[indices, 0] = values
        
        if np.random.random() > 0.9:
            out = torch.flip(out, [0]) # flip

        if np.random.random() > 0.2: # dropout random time points
            dp_amt = np.random.random() * 0.25 * len(out)
            dp_amt = int(dp_amt)
            indices = torch.sort(torch.randperm(len(out))[dp_amt:]).values
            out = out[indices]

        if np.random.random() > 0.8: # random timeblock
            l = np.random.random() * 0.25 * len(out)
            l = int(l)
            start = int(np.random.random() * (len(out) - l))
            out = torch.concat([out[:start], out[start+l:]], dim=0)
        
        if np.random.random() > 0.2:
            out[:, 0] = out[:,0] * self.s([0.5, 1.5]) # random scaling
            
        return out
 

    
    def gen_null(self):
        def get_null_func(bright):
            std = self.getstd(bright)
            def eval(x):
                return self.g(bright, std)
            return eval

        brightness = self.baseflux()
        transitfunc = np.vectorize(get_null_func(brightness))
        x = self.gen_sampling()
        y = transitfunc(x)
        return self.to_datatens(x,y)
    
    def gen_nova(self):
        decayrange = [0.01, 0.075]

        def get_nova_func(bright, peak, decay):
            pop = int(self.s([0, self.days / 180]))*180 # so it pops on a observation period
            small = bool(np.random.random() > 0.2)
            visiblebefore = bool(np.random.random() > 0.6)

            peak = max(1.7 + np.abs(self.g(0, 0.7)) if small else 
                    self.s([20, 60]), 3.5*self.getstd(bright) / bright) # clipped to 3.5 sigma

            def eval(x):
                if x < pop and (visiblebefore or small):
                    return self.g(bright, self.getstd(bright))
                elif x >= pop:
                    exp = (decay*0.22*(x - pop - 1))
                    val = peak * np.exp(-exp) * bright + bright
                    std = self.getstd(val)
                    if not visiblebefore and val < 1.5*bright and not small: # If not vis before, dissapear once decay below 3x bg
                        return -1.0
                    else:
                        return self.g(val, std)
                else:
                    return -1.0 # Invisible
            return eval

        brightness = self.baseflux()
        decay = self.s(decayrange)
        novafunc = np.vectorize(get_nova_func(brightness, 0, decay))
        x = self.gen_sampling()
        y = novafunc(x)
        idxer = y > 0 # discard invisible
        x = x[idxer]
        y = y[idxer]
        if len(x) <= 15:
            return self.gen_nova()
        return self.to_datatens(x,y)
    
    def gen_transit(self, returnpd=False):
        transitdurationrange = [0.04, 0.15] # Percentage of period domain occupied by transit
        transitdepthrange = [0.04, 0.6] # Percentage dip in flux
        def get_transit_func(bright, period, duration, depth):
            std = self.getstd(bright)
            duration = period * duration

            depth = max(bright * depth, 2.5 * std) # Clip to 2 sigma



            transitstart = self.s([0, period - duration])
            transitend = transitstart + duration

            troughlen = self.s([0.01,0.5]) * duration 
            troughstart = transitstart + (duration - troughlen) / 2
            troughend = transitend - (duration - troughlen) / 2

            secondarybright = self.s([0.2,0.85])

            def eval(xr):
                mod = 1
                x = 2*xr
                if x % 2 * period > period:
                    mod = secondarybright

                x = x % period
                    
                if x < transitstart or x > transitend:
                    val = bright
                    return val
                elif x < troughstart:
                    val = bright - depth * mod * (x - transitstart) / (troughstart - transitstart)
                    return val
                elif x > troughend:
                    val = bright - depth * mod * (1 - (x - troughend) / (transitend - troughend))
                    return val
                else: # in trough
                    val = bright - depth * mod
                    return val
                
            eval.start = transitstart
            eval.end = transitend
                
            def evalwnoise(xr):
                return self.g(eval(xr), std)
            return evalwnoise, eval


        r = np.random.random()
        if r > 0.75: # 25% LP
            period = self.s([10, 25])
        elif r > 0.5: # 25% MP
            period = self.s([2, 10])
        else: # 50% SP
            period = self.s([0.1, 1])

        duration = self.s(transitdurationrange)
        brightness = self.baseflux()
        depth = self.s(transitdepthrange)
        transitfunc, underlying = get_transit_func(brightness, period, duration, depth)
        transitfunc = np.vectorize(transitfunc)
        underlying = np.vectorize(underlying)
        x = self.gen_sampling()
        y = transitfunc(x)
        
        xm = (x % period) / period

        if np.quantile(y, 0.08) > brightness - 2*self.getstd(brightness):
            return self.gen_transit(returnpd=returnpd)
        
        if returnpd:
            return (self.to_datatens(x,y), period / np.max(x))
        else:
            return self.to_datatens(x,y)
    
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

        selector = np.random.random()
        if selector > 0.7: # 30% SP
            period = self.s([0.1, 2])
        elif selector > 0.2: # 50% MP
            period = self.s([2, 100])
        else: # 20% LP
            period = self.s([100, 400])

        selector = np.random.random()
        sineamp = 0
        sineperiod = 1
        sinephase = 1
        if selector > 0.5: # add sine
            sineperiod = self.s([50,800])
            sineamp = self.s([2, 10]) * std / brightness #2-10 snr
            sinephase = self.s([0, period])

        max_amp = self.s([1.7*std/brightness, 0.65])

        gridres = 110
        grid = np.zeros(gridres)

        windowsize = gridres // 2.7

        modifier = windowsize * 6
        filt = (1 / np.sqrt(np.pi*modifier)) * np.exp(-(1 / modifier)*np.arange(-windowsize, windowsize)**2)
        # filt[:int(windowsize)] += 0.001
        # if selector > 0.4: # 60% mountainrange waveform
        n_peaks = int(np.abs(self.g(0,2)) + 2)

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

        grid = grid * max_amp * brightness / np.max(np.abs(grid)) + brightness # rescale to max_amp and add base flux

        def eval(xr):
            x = xr % period
            idx = (len(grid) - 1) * x / period
            # get the two closest points, and interpolate
            u,v = np.floor(idx), np.ceil(idx)
            interp = grid[int(u)] + (grid[int(v)] - grid[int(u)]) * (idx - u)
            val = interp + sineamp * np.sin(2 * np.pi * x / sineperiod + sinephase)
            return self.g(val, self.getstd(val)) # FLAG
        
        x = self.gen_sampling()
        y = np.vectorize(eval)(x)
        if returnpd:
            return (self.to_datatens(x,y), period / 4000, 0)
        return self.to_datatens(x,y)
            


    
# inst = GenSet(epochsize=100, batchsize=4, randseed=0, valid=False)

# ex, l = inst.gen()
# ex = ex[0]
# l = l[0]

# fig = go.Figure()


# fig.add_trace(go.Scatter(x=ex[:,2].cpu().numpy(), y=ex[:,0].cpu().numpy(), mode='markers', name='Lightcurve'))
# print(l)
# fig.write_image("temp.png")