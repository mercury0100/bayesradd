import torch
import argparse
import matplotlib.pyplot as plt
from load_model import load_model
from bayes_radd import mc_marginal_ppl
import data

import os
import json

os.makedirs("BayesRADD", exist_ok=True)
torch.manual_seed(42)

MODEL_PATH = "JingyangOu/radd-t-dce"
SEQ_LEN    = 1024
BATCH_SIZE = 16
MAX_BATCHES = 1
K_LIST     = [1, 2, 4, 8, 16, 32, 48, 64]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, noise = load_model(MODEL_PATH, device)
model.eval()

def build_loader():
    args = argparse.Namespace(
        cache_dir="data", batch_size=BATCH_SIZE,
        length=SEQ_LEN, valid_dataset="wikitext2", ngpus=1
    )
    return data.get_valid_dataloaders(args, distributed=False)

if __name__ == "__main__":
    ppls = []
    for K in K_LIST:
        ppl = mc_marginal_ppl(
            model, noise, device,
            build_loader(),
            K=K,
            max_batches=MAX_BATCHES,
            tokenwise=True   # ensure per-token PPL
        )
        print(f"K = {K:<2d} → MC marginal PPL ≈ {ppl:.2f}")
        ppls.append(ppl)
    
    with open("BayesRADD/mc_ppl_values.json", "w") as f:
        json.dump({"K": K_LIST, "ppl": ppls}, f)