import math
import numpy as np
from scipy.stats import norm
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from losses import get_loss_fn as _get_loss_fn
from noise_lib import add_noise_lambda, Noise

# ——— 1) It's a corruption ting ————————————————
def corrupt_and_denoise(model, noise, x):
    B, T    = x.shape
    device  = x.device
    MASK_ID = model.config.tokens  # = vocab_size-1

    # sample per-example time
    t      = torch.rand(B, device=device)             
    # 1) get total_noise σ(t)
    sigma  = noise.total_noise(t)                     
    # 2) convert to mask probability p = 1 - exp(-σ)
    Lambda = (1 - torch.exp(-sigma)).clamp(0., 1.)     # [B]
    # 3) mask via repo helper (correct mask token)
    x_t    = add_noise_lambda(x, Lambda, MASK_ID)      # [B,T]
    # 4) denoise
    out    = model(x_t)
    return out.logits if hasattr(out, "logits") else out


# ——— 2.1) K‐sample ensemble statistics ————————————————
def mc_marginal(model, noise, x: torch.LongTensor, K: int):
    """
    Monte‐Carlo marginal over K corrupt+denoise passes.
    Returns:
      mc_p  : [B, T, V_nonmask] renormalized over real vocab
      mc_var: [B, T, V_nonmask]
    """
    B, T = x.shape
    V     = model.config.tokens + 1
    # accumulate full-vocab probs
    sum_p  = torch.zeros(B, T, V, device=x.device)
    sum_p2 = torch.zeros_like(sum_p)

    for _ in range(K):
        logits = corrupt_and_denoise(model, noise, x)    # [B,T,V]
        p      = F.softmax(logits, dim=-1)               # [B,T,V]
        sum_p  += p
        sum_p2 += p * p

    mc_p_full  = sum_p  / K                             # [B,T,V]
    mc_var_full= sum_p2 / K - mc_p_full*mc_p_full       # [B,T,V]

    # drop mask class, renormalize
    mc_p  = mc_p_full[..., :-1]
    mc_var= mc_var_full[..., :-1]
    denom = mc_p.sum(dim=-1, keepdim=True)              # [B,T,1]
    mc_p   = mc_p  / denom
    mc_var = mc_var / denom**2                          # var[p/(∑p)] approx

    return mc_p, mc_var


# ——— 2.2) K‐sample tokenwise ensemble statistics ————————————————
def mc_marginal_tokenwise(model, noise, x: torch.LongTensor, K: int):
    """
    Monte‐Carlo marginal *true‐token* probs and variance.
    Returns:
      mc_p_true : [B, T]
      mc_var_true: [B, T]
    """
    B, T = x.shape
    sum_p  = torch.zeros(B, T, device=x.device)
    sum_p2 = torch.zeros_like(sum_p)

    for _ in range(K):
        logits = corrupt_and_denoise(model, noise, x)    # [B,T,V]
        p      = F.softmax(logits, dim=-1)               # [B,T,V]
        # drop mask & renormalize
        p = p[..., :-1]
        p = p / p.sum(dim=-1, keepdim=True)
        # gather true-token prob
        p_true = p.gather(-1, x.unsqueeze(-1)).squeeze(-1)  # [B,T]
        sum_p  += p_true
        sum_p2 += p_true * p_true

    mc_p_true  = sum_p  / K
    mc_var_true= sum_p2 / K - mc_p_true*mc_p_true

    return mc_p_true, mc_var_true

# —————————————————————————————————————————————————
#  New: MC Mutual Information
# —————————————————————————————————————————————————
@torch.no_grad()
def mc_mutual_information(model, noise, x: torch.LongTensor, K: int):
    """
    Returns
      MI_tok : [B, T] mutual information per token
      MI_seq : [B]   mean over tokens → one score per sequence
    """
    B, T = x.shape
    V    = model.config.tokens + 1

    # accumulate per‐sample probabilities and entropies
    probs = []
    H_per_sample = torch.zeros(B, T, device=x.device)
    for _ in range(K):
        logits = corrupt_and_denoise(model, noise, x)  # [B,T,V]
        p      = F.softmax(logits, dim=-1)             # [B,T,V]
        probs.append(p.unsqueeze(0))
        H_per_sample += -(p * p.clamp(min=1e-12).log()).sum(-1)

    probs = torch.cat(probs, dim=0)                   # [K, B, T, V]
    p_bar = probs.mean(0)                             # [B, T, V]

    # entropy of the mean
    H_bar = -(p_bar * p_bar.clamp(min=1e-12).log()).sum(-1)  # [B, T]
    H_exp = H_per_sample / K                              # [B, T]

    MI_tok = H_bar - H_exp                                # [B, T]
    MI_seq = MI_tok.mean(-1)                             # [B]

    return MI_tok, MI_seq


# ——— 3) Perplexity from marginal probabilities —————————
def compute_ppl_from_mc_p(mc_p: torch.Tensor, x: torch.LongTensor):
    """
    Given mc_p [B,T,V_nonmask] and x [B,T], returns PPL (scalar).
    """
    # true-token prob
    p_true       = mc_p.gather(-1, x.unsqueeze(-1)).squeeze(-1)  # [B,T]
    logp_per_seq = p_true.clamp(min=1e-12).log().sum(dim=-1)     # [B]
    avg_logp     = logp_per_seq.mean().item() / x.size(1)
    return math.exp(-avg_logp)


# ——— 4) Full MC‐marginal pass over dataset ————————
@torch.no_grad()
def mc_marginal_ppl(model, noise, device, dataloader, K,
                    max_batches=None, tokenwise=False):
    total_logp, total_count = 0.0, 0

    length = min(max_batches, len(dataloader)) if max_batches else len(dataloader)
    for i, batch in enumerate(tqdm(dataloader, total=length, desc=f"PPL K={K}")):
        if max_batches is not None and i >= max_batches:
            break

        x = batch["input_ids"].to(device)
        B, T = x.shape

        if tokenwise:
            p_true, _ = mc_marginal_tokenwise(model, noise, x, K)  # [B,T]
            p_true    = p_true.clamp(min=1e-12)
            total_logp   += p_true.log().sum().item()
            total_count += B * T
        else:
            mc_p, _     = mc_marginal(model, noise, x, K)  # [B,T,V']
            batch_ppl   = compute_ppl_from_mc_p(mc_p, x)
            total_logp += math.log(batch_ppl)
            total_count += 1

    if total_count == 0:
        raise ValueError("mc_marginal_ppl: no batches processed")

    if tokenwise:
        return math.exp(-total_logp / total_count)
    else:
        return math.exp(total_logp / total_count)


# ——— 5) Baseline pass over dataset ————————
@torch.no_grad()
def baseline_ppl(model, noise, device, dataloader,
                 max_batches=None, sequence_length=1024):
    base_loss_fn = _get_loss_fn(noise,
                                model.config.tokens+1,
                                train=False,
                                loss_type="lambda_DCE")
    total_nll, total_batches = 0.0, 0
    steps = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    for i, batch in enumerate(tqdm(dataloader, total=steps, desc="Baseline PPL")):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["input_ids"].to(device)
        nll_batch = base_loss_fn(model, x)
        if nll_batch.ndim > 0:
            nll_batch = nll_batch.mean()
        total_nll    += (nll_batch / sequence_length).item()
        total_batches += 1

    avg_nll = total_nll / total_batches
    return math.exp(avg_nll)