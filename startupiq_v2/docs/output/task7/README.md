# Task 7: Deep learning (YC industry classification)

## What we did, in plain words
- Task 6 already has a simple model: TF-IDF turns each startup description into word counts, and logistic regression (LR) predicts the industry.
- Task 7 asks: **can a neural network read the same text better?** We tried two:
  - **Text CNN**: slides small word-window detectors (2, 3 and 4 words wide) over the description and keeps the strongest hit of each.
  - **BiLSTM + attention**: reads the sentence forwards and backwards and learns which words to focus on.
- Fair test: same train/validation/test split as Task 5, test scored once, and the choice rule written **before** running.
- Result: **LR won**. Networks need more data than the ~4,300 training texts we have.

## Where the code is
| What | File |
|---|---|
| Models and training loop | `src/startupiq/models/textnn.py` |
| Same split as Task 5 | `src/startupiq/pipelines/yc_split.py` |
| Experiment + decision rule | `src/startupiq/pipelines/task7_yc.py` |
| Images | `src/startupiq/pipelines/output_figures_78.py` |
| Numbers | `docs/results/task7/results_task7.json` |

## Code snippets (copied from the real files)

**1. Turn text into numbers (vocabulary fitted on train only)**
```python
voc = Vocab(tr["text"].tolist(), 2)          # words seen at least twice in TRAIN
Xtr = voc.encode(tr["text"].tolist())        # each text -> 100 word ids (padded)
```

**2. The text CNN (`textnn.py`)**
```python
class TextCNN(nn.Module):
    def __init__(self, V, C, emb=100, nf=64, ks=(2, 3, 4), drop=0.5):
        self.emb = nn.Embedding(V, emb, padding_idx=0)            # word -> vector
        self.convs = nn.ModuleList(nn.Conv1d(emb, nf, k) for k in ks)  # 2/3/4-word detectors
        self.fc = nn.Linear(nf * len(ks), C)                       # -> industry scores
    def pooled(self, e):
        h = e.transpose(1, 2)
        return torch.cat([torch.relu(c(h)).max(dim=2).values for c in self.convs], dim=1)
```

**3. The BiLSTM with attention**
```python
h, _ = self.rnn(e)                                         # read both directions
s = self.v(torch.tanh(self.att(h))).squeeze(-1).masked_fill(x == 0, -1e9)
a = torch.softmax(s, dim=1)                                # attention weight per word
ctx = (a.unsqueeze(-1) * h).sum(1)                         # weighted summary of the text
```

**4. Training: AdamW, class weights, early stopping on validation macro-F1**
```python
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
lossf = nn.CrossEntropyLoss(weight=class_w)                # rare industries count more
...
f = f1_score(yva, predict_proba(model, Xva).argmax(1), average="macro")
if f > best: best, state = f, copy_of_weights              # keep best epoch
else: bad += 1; if bad >= patience: break                  # stop when it stops improving
```

**5. The decision rule (fixed before running; uses validation, not test)**
```python
vf = {k: mf1(yva, v) for k, v in pv.items()}               # validation macro-F1 per model
best = max(vf, key=vf.get); lo = boot(yva, pv[best], mf1)[0]   # lower 95% bound of the best
chosen = next(k for k in ["LR", "CNN", "BiLSTM"] if vf[k] >= lo)  # simplest model inside it
```
Output: `shipped: LR` (validation LR 0.741, CNN 0.695, BiLSTM 0.615, bound 0.671).

**6. Image code (image 1)**
```python
v = [r7["test"][m]["macro_f1"] for m in ms]
ax.bar(ms, v, yerr=[lo, hi], capsize=4)                    # bars with 95% bootstrap intervals
```

---

Question: does a neural network beat the simple TF-IDF + logistic regression (LR) on the same split?
Models: text CNN, BiLSTM with attention (embeddings learned from scratch; no pretrained vectors offline). Same split as Task 5; test scored once.

## Images
1. `1_test_macro_f1.png`: LR 0.689, CNN 0.649, BiLSTM 0.552 (bars = 95% bootstrap CI). Say: "macro-F1 averages all industries equally, so it is the fair score."
2. `2_test_accuracy.png`: LR 0.798, CNN 0.741, BiLSTM 0.650. Say: "LR is best on both scores; the CNN's interval is lower."
3. `3_cnn_random_search.png`: 4 random CNN settings (validation macro-F1 0.59 to 0.70) vs LR's 0.741 dashed. Say: "tuning helped the CNN but none reached LR."
4. `4_cost.png`: LR 0.2M parameters / ~5 s; CNN 1.3M / ~92 s; BiLSTM 1.3M / ~49 s.

## Decision (rule fixed before running)
Ship the simplest model whose validation macro-F1 is at or above the lower 95% bound of the best. LR was best on validation (0.741, bound 0.671), so LR ships.

## Likely questions
- Why did deep learning lose? About 4,300 training texts is small; the labels are driven by clear keywords, which TF-IDF already uses.
- Did you tune enough? Limited: 4 CNN trials, 3 CNN seeds, 2 BiLSTM seeds (time limit of the machine). A bigger search or pretrained vectors could close the gap; not tested.
- Is the test set clean? Yes, used once after the decision.

## Limits
Embeddings from scratch; reduced search/seeds; one split. Results: `docs/results/task7/results_task7.json`.
