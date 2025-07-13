import math
import numpy as np
from scipy.stats import norm
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from losses import get_loss_fn
from noise_lib import add_noise_lambda, Noise

# ——— 1) It's a corruption ting ————————————————
def corrupt_with_mask(x: torch.LongTensor,
                      Lambda: torch.Tensor,
                      MASK_ID: int):
    """
    Randomly mask each position in x with probability Lambda[b].
    Returns:
      x_t  : [B,T] the corrupted inputs
      mask : [B,T] bool mask (True = this position was masked)
    """
    B, T = x.shape
    # expand per-sample probability to per-token
    probs = Lambda.unsqueeze(-1).expand(B, T)
    mask  = torch.bernoulli(probs).bool()
    x_t   = x.masked_fill(mask, MASK_ID)
    return x_t, mask


# ——— 1.1) Pure denoise (no masking inside) —————————————
@torch.no_grad()
def denoise(model, x_t: torch.LongTensor):
    out = model(x_t)
    return out.logits if hasattr(out, "logits") else out  # [B,T,V]


# ——— 2.1) K‐sample ensemble statistics (mask-aware) ————————————————
@torch.no_grad()
def mc_marginal(model, noise, x: torch.LongTensor, K: int):
    B, T = x.shape
    V    = model.config.tokens + 1
    sum_p  = torch.zeros(B, T, V, device=x.device)
    sum_p2 = torch.zeros_like(sum_p)
    sum_mask = torch.zeros(B, T, device=x.device, dtype=torch.float)

    for _ in range(K):
        # 1) sample noise / Lambda
        t      = torch.rand(B, device=x.device)
        sigma  = noise.total_noise(t)
        Lambda = (1 - torch.exp(-sigma)).clamp(0, 1)

        # 2) corrupt and track mask
        x_t, mask = corrupt_with_mask(x, Lambda, model.config.tokens)

        # 3) denoise
        logits = denoise(model, x_t)      # [B,T,V]
        p      = F.softmax(logits, dim=-1)  # [B,T,V]

        # 4) accumulate only at masked positions
        mask_f = mask.float().unsqueeze(-1)  # [B,T,1]
        sum_p  += p * mask_f
        sum_p2 += (p * p) * mask_f
        sum_mask += mask_f.squeeze(-1)

    # avoid division by zero
    sum_mask = sum_mask.clamp(min=1e-6).unsqueeze(-1)  # [B,T,1]

    mc_p_full   = sum_p  / sum_mask                   # [B,T,V]
    mc_var_full = sum_p2 / sum_mask - mc_p_full**2    # [B,T,V]

    # drop MASK class and renormalize
    mc_p   = mc_p_full[..., :-1]
    mc_var = mc_var_full[..., :-1]
    denom  = mc_p.sum(dim=-1, keepdim=True)
    mc_p   = mc_p   / denom
    mc_var = mc_var / denom**2

    return mc_p, mc_var


# ——— 2.2) MC marginal of true-token (mask-aware) ————————
@torch.no_grad()
def mc_marginal_tokenwise(model, noise, x: torch.LongTensor, K: int):
    B, T = x.shape
    sum_p  = torch.zeros(B, T, device=x.device)
    sum_p2 = torch.zeros_like(sum_p)
    sum_mask = torch.zeros_like(sum_p)

    for _ in range(K):
        t      = torch.rand(B, device=x.device)
        sigma  = noise.total_noise(t)
        Lambda = (1 - torch.exp(-sigma)).clamp(0, 1)

        x_t, mask = corrupt_with_mask(x, Lambda, model.config.tokens)

        logits = denoise(model, x_t)       # [B,T,V]
        p      = F.softmax(logits, dim=-1) # [B,T,V]

        # drop MASK class & renormalize per position
        p = p[..., :-1]
        p = p / p.sum(dim=-1, keepdim=True)

        # gather true-token prob, only where mask=True
        p_true = p.gather(-1, x.unsqueeze(-1)).squeeze(-1)  # [B,T]
        sum_p  += p_true * mask.float()
        sum_p2 += (p_true**2) * mask.float()
        sum_mask += mask.float()

    # prevent zero-mask
    sum_mask = sum_mask.clamp(min=1e-6)

    mc_p_true   = sum_p  / sum_mask
    mc_var_true = sum_p2 / sum_mask - mc_p_true**2

    return mc_p_true, mc_var_true

# ——— 3) Baseline pass over dataset ————————
@torch.no_grad()
def zero_shot_ppl(model, noise, device, dataloader,
            max_batches=None, sequence_length=1024):
    """
    Computes PPL by exponentiating the Denoising Cross-Entropy (DCE) loss:
      - This is the single-pass, analytic marginal likelihood under the diffusion model.
      - Matches the RADD paper’s zero-shot PPL in Table 2.
    Args:
      model           : your BayesRADD / diffusion model
      noise           : the noise schedule object
      device          : torch device (“cuda” / “cpu”)
      dataloader      : yields batches with batch["input_ids"] of shape [B, T]
      max_batches     : optional int to limit how many batches to process
      sequence_length : int, length T to normalize per-sequence loss
    Returns:
      ppl (float)
    """
    # build the DCE loss function
    # vocab_size = model.config.tokens+1  (includes the MASK class)
    base_loss_fn = get_loss_fn(
        noise,
        model.config.tokens + 1,
        train=False,
        loss_type="lambda_DCE"
    )

    total_nll, total_batches = 0.0, 0
    steps = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)

    for i, batch in enumerate(tqdm(dataloader, total=steps, desc="DCE PPL")):
        if max_batches is not None and i >= max_batches:
            break

        x = batch["input_ids"].to(device)  # [B, T]
        # base_loss_fn returns per-sequence NLL (averaged over tokens if T>1)
        nll_batch = base_loss_fn(model, x)

        # if it returns a vector, take its mean
        if nll_batch.ndim > 0:
            nll_batch = nll_batch.mean()

        # convert per-sequence loss to per-token, then sum across batches
        total_nll    += (nll_batch / sequence_length).item()
        total_batches += 1

    if total_batches == 0:
        raise ValueError("dce_ppl: no batches processed")

    avg_nll = total_nll / total_batches
    return math.exp(avg_nll)

@torch.no_grad()
def zero_shot_mc_ppl(model,
                              noise,
                              device,
                              dataloader,
                              K: int,
                              mask_rate: float,
                              max_batches: int = None,
                              sequence_length: int = 1024):
    """
    Zero‐shot PPL with both:
      1) Outer diffusion‐schedule masking (via add_noise_lambda)
      2) Inner fixed‐rate Bernoulli masking over the remaining tokens
    
    Args:
      model           : BayesRADD denoiser
      noise           : diffusion noise schedule
      device          : torch.device
      dataloader      : yields batch["input_ids"] of shape [B, T]
      K               : number of MC samples
      mask_rate       : float in [0,1], inner‐mask rate
      max_batches     : cap on number of batches (for quick tests)
      sequence_length : length T for per‐token normalization
    Returns:
      ppl (float)
    """
    model.eval()
    MASK_ID = model.config.tokens
    total_logp, total_count = 0.0, 0

    steps = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    for i, batch in enumerate(tqdm(dataloader, total=steps, desc="MC double‐mask PPL")):
        if max_batches is not None and i >= max_batches:
            break

        # [B, T] token IDs
        x = torch.stack([torch.tensor(seq, device=device) for seq in batch["input_ids"]], dim=0)
        B, T = x.shape

        # accumulate sum of true‐token probs over K
        sum_p = torch.zeros(B, T, device=device)

        for _ in range(K):
            # 1) outer schedule mask
            t0      = torch.rand(B, device=device)
            sigma0  = noise.total_noise(t0)
            Λ0      = (1 - torch.exp(-sigma0)).clamp(0.,1.)    # [B]
            x0      = add_noise_lambda(x, Λ0, MASK_ID)        # [B, T]

            # 2) inner fixed‐rate mask on unmasked slots
            valid    = (x0 != MASK_ID)                       # [B, T]
            rand_m   = torch.rand(B, T, device=device) < mask_rate
            inner_m  = rand_m & valid                        # [B, T]
            xk       = x0.masked_fill(inner_m, MASK_ID)      # [B, T]

            # 3) denoise and get true‐token probs
            out    = model(xk)
            logits = out.logits if hasattr(out, "logits") else out  # [B, T, V]
            p_all  = F.softmax(logits, dim=-1)                       # [B, T, V]
            p_true = p_all.gather(-1, x.unsqueeze(-1)).squeeze(-1)   # [B, T]

            # 4) only score the *outer* masked positions
            outer_m = (x0 == MASK_ID).float()                       # [B, T]
            sum_p  += p_true * outer_m

        # 5) average over K and accumulate log‐probs
        p_hat   = (sum_p / K).clamp(min=1e-12)                     # [B, T]
        # sum log‐prob *once* per masked token
        total_logp  += (p_hat.log() * (x0==MASK_ID).float()).sum().item()
        total_count += (x0==MASK_ID).float().sum().item()

    avg_logp = total_logp / total_count
    return math.exp(-avg_logp)