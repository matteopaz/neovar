import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import ptwt
import pywt
from time import perf_counter


device = "cuda" if torch.cuda.is_available() else "cpu"

    
class CNFourierModel(nn.Module):
    def __init__(self, samples, out, learnsamples=False):
        super().__init__()
        self.samples = samples
        self.out = out
        self.cnn1 = nn.Conv1d(3, 24, 12, padding="same")
        self.cnn2 = nn.Conv1d(24, 48, 7, padding="same")
        self.cnn3 = nn.Conv1d(48, 10, 7, padding="same")
        self.cnn4 = nn.Conv1d(10, 1, 5, padding="same")

        # self.appcnn = lambda x, cnn: cnn(x.permute(0,2,1)).permute(0,2,1)

        self.layerone = nn.Linear(samples,samples//2)
        self.layertwo = nn.Linear(samples//2,samples//4)
        self.layerthree = nn.Linear(samples//4, self.out)


        self.dp = nn.Dropout(p=0.0)
        
        self.fdft = FDFT(samples, learning=learnsamples, real=True, remove_zerocomp=False)
        
        self.norm = nn.LayerNorm(samples//4)
        self.av = nn.ReLU()

    def forward(self, x):
        N = x.shape[0]
        L = x.shape[1]

        x = x.permute(0,2,1)
        x = self.cnn1(x)
        x = self.av(x)
        x = self.cnn2(x)
        x = self.av(x)
        x = self.cnn3(x)
        x = self.av(x)
        x = self.cnn4(x)
        x = self.av(x).permute(0,2,1)
        x = x.reshape(N,L,1)

        frequential = self.fdft(x).squeeze(-1)
        frequential = torch.arcsinh(frequential)
        # frequential = torch.arcsinh(torch.mul(frequential, snr))

        # plt.scatter(t[0].detach().cpu().numpy(), sig[0].detach().cpu().numpy())
        # frequential = torch.arcsinh(frequential)
        # plt.plot(frequential[0].detach().cpu().numpy())
        # plt.show()
        # raise Exception()

        # frequential = torch.arcsinh(frequential * std)
        
        layone = self.layerone(frequential)
        layone = self.av(layone)
        layone = self.dp(layone)

        laytwo = self.layertwo(layone)
        laytwo = self.av(laytwo)
        laytwo = self.dp(laytwo)
        laytwo = self.norm(laytwo)

        laythree = self.layerthree(laytwo)


        return laythree

class FCNFourierModel(nn.Module):
    def __init__(self, samples, out, learnsamples=False):
        super().__init__()
        self.samples = samples
        self.out = out
        self.cnn1 = nn.Conv1d(3, 24, 12, padding="same")
        self.cnn2 = nn.Conv1d(24, 48, 7, padding="same")
        self.cnn3 = nn.Conv1d(48, 10, 7, padding="same")
        self.cnn4 = nn.Conv1d(10, 1, 5, padding="same")

        # self.appcnn = lambda x, cnn: cnn(x.permute(0,2,1)).permute(0,2,1)

        self.layerone = nn.Linear(samples,samples//2)
        self.layertwo = nn.Linear(samples//2,samples//4)
        self.layerthree = nn.Linear(samples//4, self.out)


        self.dp = nn.Dropout(p=0.0)
        
        self.fdft = FDFT(samples, learning=learnsamples, real=True, remove_zerocomp=False)
        
        self.norm = nn.LayerNorm(samples//4)
        self.av = nn.ReLU()

    def forward(self, x):
        N = x.shape[0]
        L = x.shape[1]

        frequential = self.fdft(x).squeeze(-1)
        frequential = torch.arcsinh(frequential)

        x = x.reshape(N,self.samples,3)

        x = x.permute(0,2,1)
        x = self.cnn1(x)
        x = self.av(x)
        x = self.cnn2(x)
        x = self.av(x)
        x = self.cnn3(x)
        x = self.av(x)
        x = self.cnn4(x)
        x = self.av(x).permute(0,2,1)

        x = x.reshape(N, self.samples)

        # frequential = torch.arcsinh(torch.mul(frequential, snr))

        # plt.scatter(t[0].detach().cpu().numpy(), sig[0].detach().cpu().numpy())
        # frequential = torch.arcsinh(frequential)
        # plt.plot(frequential[0].detach().cpu().numpy())
        # plt.show()
        # raise Exception()

        # frequential = torch.arcsinh(frequential * std)
        
        layone = self.layerone(frequential)
        layone = self.av(layone)
        layone = self.dp(layone)

        laytwo = self.layertwo(layone)
        laytwo = self.av(laytwo)
        laytwo = self.dp(laytwo)
        laytwo = self.norm(laytwo)

        laythree = self.layerthree(laytwo)


        return laythree

class WCNFourierModel(nn.Module):
    def __init__(self, samples, out, wavelet, learnsamples=False):
        super().__init__()
        self.samples = samples
        self.out = out
        

        self.cnn1 = nn.Conv1d(3, 16, 13, padding="same")
        self.cnn2 = nn.Conv1d(16, 24, 11, padding="same")
        self.cnn3 = nn.Conv1d(24, 4, 9, padding="same")

        self.cnn4 = nn.Conv1d(11, 32, 9, padding="same")
        self.cnn5 = nn.Conv1d(32, 16, 7, padding="same")
        self.cnn6 = nn.Conv1d(16, 4, 5, padding="same")

        self.cnnafter1 = nn.Conv1d(4, 24, samples//64 - 1, padding="same")
        self.cnnafter2 = nn.Conv1d(24, 12, 17, padding="same")
        self.cnnafter3 = nn.Conv1d(12, 1, 11, padding="same")

        self.layerone = nn.Linear(samples,samples//2)
        self.layertwo = nn.Linear(samples//2,samples//4)
        self.layerthree = nn.Linear(samples//4, self.out)

        self.wavelet = pywt.Wavelet(wavelet)

        self.dp = nn.Dropout(p=0.0)
        
        self.fdft = FDFT(samples, learning=learnsamples, real=True, remove_zerocomp=False)
        
        self.norm = nn.LayerNorm(samples//4)
        self.bnorm = nn.BatchNorm1d(3)
        self.av = nn.ReLU()

        for n, param in self.named_parameters():
            if len(param.shape) < 2:
                continue
            torch.nn.init.xavier_uniform_(param)

    def forward(self, x):
        t1 = perf_counter()
        N = x.shape[0]
        L = x.shape[1]
        original = x.permute(0,2,1)

        x = self.cnn1(original)
        x = self.av(x)
        x = self.cnn2(x)
        x = self.dp(x)
        x = self.av(x)
        x = self.cnn3(x)

        wv = ptwt.wavedec(x, self.wavelet, mode="periodic", level=1)
        approx, detail = wv[0], wv[1]
        waveletdec = torch.concat((approx, detail), dim=1)
        waveletdec = waveletdec.repeat_interleave(2, dim=-1)


        waveletdec = waveletdec[:, :, :L]

        x = torch.cat((original, waveletdec), dim=1)


        L = x.shape[-1]

        x = self.cnn4(x)
        x = self.av(x)
        x = self.dp(x)
        x = self.cnn5(x)
        x = self.av(x)
        x = self.dp(x)
        x = self.cnn6(x)
        x = x.permute(0,2,1)
        x = x.reshape(N,L,4)

        fourier = self.fdft(x)
        fourier = torch.arcsinh(fourier)

        freq = fourier.permute(0,2,1) # shape (b, 3, samples)
        freq = self.cnnafter1(freq)
        freq = self.av(freq)
        freq = self.dp(freq)
        freq = self.cnnafter2(freq)
        freq = self.av(freq)
        freq = self.dp(freq)
        freq = self.cnnafter3(freq)
        freq = self.av(freq)
        freq = self.dp(freq)
        freq = freq.permute(0,2,1).squeeze() # now of shape (b, samples)

        
        layone = self.layerone(freq)
        layone = self.av(layone)
        layone = self.dp(layone)

        laytwo = self.layertwo(layone)
        laytwo = self.av(laytwo)
        laytwo = self.dp(laytwo)


        laytwo = self.norm(laytwo)

        laythree = self.layerthree(laytwo)

        return laythree
    
class Wavelet(nn.Module):
    def __init__(self, wavelet, samples):
        super().__init__()
        self.wavelet = wavelet
        self.samples = samples
        self.wave = pywt.Wavelet(wavelet)
        self.pool = nn.MaxPool1d(2)
        self.av = nn.ReLU()

    
    def forward(self, x):
        if len(x.shape) != 2:
            raise Exception("Input must be of shape (b, N)")
        
        B = x.shape[0]
        L = x.shape[1]

        # mw = ptwt.matmul_transform.MatrixWavedec(self.wave, level=1)
        # a, d1 = mw(x)
        # stacked = torch.stack((a, d1), dim=1) # of size (b, 2, N / 2)

        stacked = x.unsqueeze(1)

        inter = self.cnn1(stacked)
        inter = self.av(inter)
        inter = self.pool(inter)
        inter = self.cnn2(inter)
        inter = self.av(inter)
        inter = self.pool(inter)
        inter = self.cnn3(inter)
        out = self.pool(inter) # shape (b, samples, N / 8)
        out = self.cnn4(out)
        # sum across the last dimension
        out = out.sum(dim=-1) # shape (b, samples)
        return out
        

class FUDFT(nn.Module):
    def __init__(self, samples, learning=False, real=False, remove_zerocomp=True):
        super().__init__()
        self.samples = samples
        self.freqs = torch.linspace(start=0, end=1, steps=samples)
        self.zerocomp = remove_zerocomp

        if learning:
            self.freqs = nn.Parameter(self.freqs)
        
        self.out = lambda x: x
        if real:
            self.out = lambda x: x.abs().to(dtype=torch.float32)
        
        
    def make_fourier(self, N, t):
        freqs = torch.mul(self.freqs, N - 1)
        # exponent_rows = (-2 * torch.pi * 1j * freqs / N).view(1,-1,1) # shape 1,samples,1
        # exponent_cols = (t.unsqueeze(1) * (N - 1)).to(torch.cfloat).to(device) # shape b,1,N
        # exponent = torch.matmul(exponent_rows, exponent_cols)
        exponent_rows = (-2 * torch.pi * 1j * freqs / N)
        exponent_cols = (t * (N - 1)).to(torch.cfloat).to(device)
        exponent = torch.einsum("i, bj -> bij", exponent_rows, exponent_cols)


        fourier = torch.exp(exponent)
        return (1 / np.sqrt(N)) * fourier # of shape (b, samples, N)
        
        
    def forward(self, x: torch.Tensor, t):
        if len(x.shape) != 3:
            raise Exception("Input must be of shape (b, N, C)")
            
        fourier_tens = self.make_fourier(x.size(1), t)
        temp = x.to(torch.cfloat)
        if len(temp.shape) == 2:
            transformed = torch.bmm(fourier_tens, temp)
        else:
            transformed = torch.bmm(fourier_tens, temp)
        if self.zerocomp:
            transformed[:,0] = 0
            transformed[:,-1] = 0
        return self.out(transformed)

class FDFT(nn.Module):
    def __init__(self, samples, learning=False, real=False, remove_zerocomp=True):
        super().__init__()
        self.samples = samples
        self.freqs = torch.linspace(start=0, end=1, steps=samples)
        if learning:
            self.freqs = nn.Parameter(self.freqs)
        
        self.zerocomp = remove_zerocomp
        self.out = lambda x: x
        if real:
            self.out = lambda x: x.abs().to(dtype=torch.float32)
        
        
    def make_fourier(self, N):
        freqs = self.freqs * (N - 1) 
        exponent_rows = (-2 * torch.pi * 1j * freqs / N)
        exponent_cols = torch.arange(N).to(device)
        exponent = torch.outer(exponent_rows, exponent_cols)
        fourier = torch.exp(exponent)
        return (1 / np.sqrt(N)) * fourier # of shape (b, samples, N)
        
        
    def forward(self, x: torch.Tensor, dim=-1):
        if len(x.shape) != 3:
            raise Exception("Input must be of shape (b, N, C)")
        fourier_tens = self.make_fourier(x.size(1)).to(device)
        temp = x.to(torch.cfloat)
        transformed = torch.matmul(fourier_tens, temp)
        if self.zerocomp:
            transformed[:,0] = 0
            transformed[:,-1] = 0
        return self.out(transformed)