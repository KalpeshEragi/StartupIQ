"""Small PyTorch text models for Task 7: text CNN and BiLSTM with attention.
Embeddings are learned from scratch (no pretrained vectors are available offline); this is stated in the results."""
import re
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
MAXLEN = 100


def tokenize(t):
    return TOK.findall(str(t).lower())


class Vocab:
    def __init__(self, texts, min_count=2):
        c = Counter(w for t in texts for w in tokenize(t))
        self.itos = ["<pad>", "<unk>"] + [w for w, n in c.items() if n >= min_count]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def ids(self, t, L=MAXLEN):
        x = [self.stoi.get(w, 1) for w in tokenize(t)][:L]
        return x + [0] * (L - len(x))

    def encode(self, texts, L=MAXLEN):
        return torch.tensor([self.ids(t, L) for t in texts])


class TextCNN(nn.Module):
    def __init__(self, V, C, emb=100, nf=64, ks=(2, 3, 4), drop=0.5):
        super().__init__()
        self.ks, self.nf = ks, nf
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, nf, k) for k in ks)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(nf * len(ks), C)

    def pooled(self, e):
        h = e.transpose(1, 2)
        return torch.cat([torch.relu(c(h)).max(dim=2).values for c in self.convs], dim=1)

    def forward(self, x, e=None):
        e = self.emb(x) if e is None else e
        return self.fc(self.drop(self.pooled(e)))


class BiLSTMAttn(nn.Module):
    def __init__(self, V, C, emb=100, hid=64, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.rnn = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hid, 64)
        self.v = nn.Linear(64, 1, bias=False)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * hid, C)

    def forward(self, x, e=None, return_attn=False):
        e = self.emb(x) if e is None else e
        h, _ = self.rnn(e)
        s = self.v(torch.tanh(self.att(h))).squeeze(-1).masked_fill(x == 0, -1e9)
        a = torch.softmax(s, dim=1)
        ctx = (a.unsqueeze(-1) * h).sum(1)
        out = self.fc(self.drop(ctx))
        return (out, a) if return_attn else out


def predict_proba(model, X, bs=256):
    model.eval()
    with torch.no_grad():
        return torch.cat([torch.softmax(model(X[i:i + bs]), 1) for i in range(0, len(X), bs)]).numpy()


def train(model, Xtr, ytr, Xva, yva, class_w, seed, lr=2e-3, wd=1e-2, epochs=20, patience=4, bs=128):
    """AdamW + weighted cross-entropy; early stopping on validation macro-F1; returns best val macro-F1."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(weight=class_w)
    best, state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad(); lossf(model(Xtr[b]), ytr[b]).backward(); opt.step()
        f = f1_score(yva, predict_proba(model, Xva).argmax(1), average="macro")
        if f > best:
            best, bad = f, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state)
    return best
