import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import gaussian_kde
import os
from tqdm import tqdm

# Self-Attention Module
class SelfAttention(nn.Module):
    def __init__(self, e_dim):
        super(SelfAttention, self).__init__()
        self.q = nn.Linear(e_dim, e_dim)
        self.k = nn.Linear(e_dim, e_dim)
        self.v = nn.Linear(e_dim, e_dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.matmul(q, k.transpose(0, 1))
        attn = nn.functional.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

# Autoencoder with Self-Attention
class Autoencoder(nn.Module):
    def __init__(self, inp_dim, enc_dim, out_dim):
        super(Autoencoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(inp_dim, enc_dim),
            nn.ReLU(True),
            nn.Linear(enc_dim, enc_dim),
            SelfAttention(enc_dim),
            nn.ReLU(True)
        )
        self.dec = nn.Sequential(
            nn.Linear(enc_dim, out_dim),
            nn.Sigmoid(),
            nn.Linear(out_dim, out_dim),
        )
        self.dec2 = nn.Sequential(
            nn.Linear(out_dim, inp_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.enc(x)
        out = self.dec(x)
        return out, self.dec2(out)

# Entropy Calculation
def calculate_kde_entropy(d):
    kde = gaussian_kde(d.T)
    dens = kde(d.T)
    return -np.sum(dens * np.log(dens + 1e-10))

# Weight Normalization
def normalize_weights(ent):
    return torch.softmax(torch.tensor(ent), dim=0)

# Weighted Mean Calculation
def weighted_mean_calc(d, w):
    w = w.view(-1, 1)
    return torch.sum(d * w, dim=0).unsqueeze(0)

# Main Processing Function
def process_files(in_f, out_f):
    inp, enc, out = 1000, 640, 300
    ae = Autoencoder(inp, enc, out)
    cr = nn.MSELoss()
    opti = opt.Adam(ae.parameters(), lr=1e-4)
    ae.train()

    files = [f for f in os.listdir(in_f) if f.endswith('.csv')]

    for fn in tqdm(files, desc='Processing Files'):
        d = pd.read_csv(os.path.join(in_f, fn), usecols=range(3, 1003)).values
        t = torch.tensor(d, dtype=torch.float32)
        dl = DataLoader(TensorDataset(t), batch_size=10, shuffle=True)

        for _ in range(10):
            for b in dl:
                inp_b = b[0]
                opti.zero_grad()
                _, out_b = ae(inp_b)
                loss = cr(out_b, inp_b)
                loss.backward()
                opti.step()

        ae.eval()
        with torch.no_grad():
            dec, _ = ae(t)
            ent = np.array([calculate_kde_entropy(r.numpy()) for r in dec])
            w = normalize_weights(ent)
            fm = weighted_mean_calc(dec, w)

        pd.DataFrame(fm.numpy()).to_csv(os.path.join(out_f, fn), index=False)

# Execution
process_files(input_dir, output_dir)
