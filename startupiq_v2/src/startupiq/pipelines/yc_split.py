"""The exact Task 5 split for YC, shared by Tasks 7 and 8."""
import pandas as pd
from sklearn.model_selection import train_test_split
SEED = 42

def load_split():
    df = pd.read_csv("data/processed/yc_clean.csv")
    df = df[df["usable_text"] & df["industry"].notna()].reset_index(drop=True)
    small = df["industry"].value_counts()[lambda s: s < 50].index.tolist()
    df["label"] = df["industry"].where(~df["industry"].isin(small), "Other (merged)")
    tr_va, te = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=SEED)
    tr, va = train_test_split(tr_va, test_size=0.1765, stratify=tr_va["label"], random_state=SEED)
    return df, tr, va, te
