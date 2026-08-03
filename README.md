# STAR — Scene Token Accumulation via Recurrence (Parallel)

> *A recurrent, GPU-friendly scene-accumulation mechanism — now parallelized and successfully swapped into pretrained GPT-2 in place of self-attention.*

---

## Overview

Hello, everyone who will see this paper and maybe even use it in their own LLMs.

I developed an algorithm that — to my knowledge — has no exact prior art. It started as a purely sequential recurrence (see the **legacy section** near the bottom), which worked but couldn't be parallelized on GPU. That version has since been reformulated into a **fully parallel layer** using cumulative sums instead of a Python loop, and it's this parallel version that is now the main result of this repo: it has been dropped directly into a pretrained GPT‑2's attention slots and trained successfully.

---

## Core Idea

The central question was:

> **What if tokens, instead of attending to each other, contributed differences to a shared "scene" vector that accumulates a representation of what the text describes?**

Each token computes a `delta` — a change it wants to apply to the **scene vector**. The scene vector absorbs these deltas causally, building up a compressed representation of the context seen so far. Every token then reads from the scene to enrich itself.

### Example

Take the sentence: `"Bird was flying around"`

| Step | Token | Scene vector knows… |
|------|-------|----------------------|
| 1 | `Bird` | There is a bird |
| 2 | `was` | Something was |
| 3 | `flying` | A bird was flying |
| 4 | `around` | A bird was flying around |

At step 3, the scene vector has effectively *imagined* a flying bird — not by attending over all past tokens quadratically, and (in the parallel version below) not by looping step-by-step either, but by computing a causal running average of gated token contributions in a single parallel pass.

---

## Mechanism — Parallel Version ✅ *(Parallel successful version)*

```
raw            =  tok_proj(x)                         # what each token wants to write
gate           =  sigmoid( gate_proj(x) )              # how strongly it wants to write it
weighted       =  raw * gate

cum_weighted   =  cumsum(weighted, dim=time)
cum_gate       =  cumsum(gate,     dim=time)

# shift right by one step so token t only ever sees tokens < t (causal)
scene_raw      =  shift_right(cum_weighted) / (shift_right(cum_gate) + eps)
scene          =  LayerNorm(scene_raw)

delta          =  tanh( raw * (1 + scene_proj(scene)) )
out            =  x + W_out( scene + delta )
```

Where:
- **`tok_proj`** — 2-layer MLP that nonlinearly encodes *what the token carries*
- **`gate_proj`** — linear gate deciding *how much* of that content actually gets written into the scene
- **`scene_proj`** — 2-layer MLP that nonlinearly reads *what is currently active in the scene*
- **`cumsum`** — replaces the old step-by-step Python loop with a single parallel prefix-sum over the sequence dimension
- **the right-shift** — guarantees causality: the scene available to token `t` is built only from tokens `0…t-1`
- **`(1 + …)`** — ensures the first token can still write into an empty scene and gradients flow from step 0
- **`tanh`** — bounds each write to prevent explosion over long sequences
- **`LayerNorm`** — keeps the scene magnitude stable regardless of how many tokens have been accumulated

---

## Why This Works

A single linear matrix is just rotation and scaling — it cannot nonlinearly separate `"bird"` from `"motion"`. The 2-layer MLPs inside `tok_proj` and `scene_proj` are what allow the mechanism to extract and combine complex features.

The gate is what makes the parallel formulation possible without losing the spirit of the original recurrence: instead of a strict step-by-step update (which forces sequential computation), each token's contribution is weighted and folded into a **running, causally-masked average** via cumulative sums. That average is exactly what a recurrent accumulation would converge to, but `cumsum` is a single parallel operation — no per-token Python loop, no per-step GPU kernel launch. This is what actually makes the layer GPU-friendly rather than just "sequential but O(n)".

---

## Complexity

| Mechanism | Sequence complexity | Parallel across time steps? |
|-----------|---------------------|------------------------------|
| Multi-Head Attention | O(n²) | Yes (batched matmul) |
| STAR — non-parallel (legacy) | O(n) | **No** — sequential Python loop, one step at a time |
| **STAR — parallel (ours)** | **O(n)** | **Yes** — single `cumsum` / parallel-scan pass |

The parallel version keeps the linear sequence complexity of the original idea while removing the sequential bottleneck that made the legacy layer slow on GPU.

---

## Architecture (tested integration: GPT-2 + STAR)

The parallel layer was tested by direct substitution into a pretrained model rather than by training a small model from scratch:

```
Pretrained GPT-2 (124M, HuggingFace "gpt2")
        ↓
Every GPT2Block's self-attention module
        →  replaced with STARAttentionWrapper(STARLayerParallel)
        ↓
Token/position embeddings, LayerNorms, MLPs, tied lm_head
        →  loaded from pretrained GPT-2 and FROZEN
        ↓
Only the newly-inserted STAR layers are trainable
        (≈42M of the model's 124M parameters, roughly a third)
```

Trained on **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`, `sample-10BT`, streamed), `seq_len=512`, `batch_size=8`, AdamW (`lr=5e-5`, cosine decay to 10%, `weight_decay=0.1`, `grad_clip=1.0`).

Since `use_cache` is disabled, STAR recomputes the full sequence at every generation step — there is no KV cache equivalent yet for this layer.

---

## Results

**(Source: `parallelSTARinGPT2_train_log.txt`, `STARinGPT2_trainscript.ipynb`)**

This run resumed from a checkpoint already at step 1500 and continued training only the STAR attention layers (GPT-2 backbone frozen) toward a planned 20,000-step schedule. Training was manually stopped (`KeyboardInterrupt`) at step ~2598, so these numbers are a **mid-training snapshot**, not a finished run — but they show the parallel layer training stably and successfully inside a real pretrained transformer.

| Step | Validation loss | Validation perplexity |
|------|------------------|------------------------|
| 1501 | 5.050 | 156.0 |
| 1601 | 4.957 | 142.2 |
| 1701 | 4.998 | 148.1 |
| 1801 | 5.023 | 151.9 |
| 1901 | 4.933 | 138.8 |
| 2001 | 4.975 | 144.7 |
| 2101 | 5.002 | 148.7 |
| 2201 | 5.024 | 152.0 |
| 2301 | 4.938 | 139.5 |
| **2401** | **4.893** | **133.4** |
| 2501 | 5.026 | 152.3 |

Best validation loss observed in this run: **4.893** (≈133 perplexity) at step 2401. Per-step training loss over this window ranged roughly 4.47–5.67 (mean ≈5.07), which is expected noise at `batch_size=8` with no gradient accumulation.

Checkpoints were saved at steps 2000 and 2500. No matched MHA-vs-STAR-parallel baseline at this GPT-2 scale has been run yet — that comparison is listed under Status below.

---

## Code — Parallel Layer *(Parallel successful version)*

```python
class STARLayerParallel(nn.Module):
    """Causal token-mixing layer (replaces attention).

    add_residual=True  -> standalone use: returns x + transform(x)
    add_residual=False -> "attention-slot" use: returns only transform(x),
                           letting the surrounding block add its own
                           residual (this is how it's used inside GPT-2
                           blocks, since GPT2Block already does
                           `residual + attn_output` around ln_1).
    """
    def __init__(self, d_model, eps: float = 1e-6, add_residual: bool = True):
        super().__init__()
        self.add_residual = add_residual
        self.tok_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False), nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.scene_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False), nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.scene_norm = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.eps = eps

    def forward(self, x):
        B, T, D = x.shape

        raw = self.tok_proj(x)
        gate = torch.sigmoid(self.gate_proj(x))
        weighted = raw * gate

        cum_weighted = torch.cumsum(weighted, dim=1)
        cum_gate = torch.cumsum(gate, dim=1)

        cum_weighted_excl = torch.zeros_like(cum_weighted)
        cum_gate_excl = torch.zeros_like(cum_gate)
        cum_weighted_excl[:, 1:, :] = cum_weighted[:, :-1, :]
        cum_gate_excl[:, 1:, :] = cum_gate[:, :-1, :]

        scene_raw = cum_weighted_excl / (cum_gate_excl + self.eps)
        scene_before = self.scene_norm(scene_raw)

        delta = torch.tanh(raw * (1.0 + self.scene_proj(scene_before)))
        out = self.W_out(scene_before + delta)
        return x + out if self.add_residual else out


class STARAttentionWrapper(nn.Module):
    """Drop-in replacement for a GPT2Block's `.attn` module."""
    def __init__(self, d_model):
        super().__init__()
        self.star = STARLayerParallel(d_model, add_residual=False)

    def forward(self, hidden_states, *args, **kwargs):
        attn_output = self.star(hidden_states)
        return attn_output, None
```

---

## Status

The parallel layer works and trains stably when substituted into pretrained GPT-2. Next steps:

- [ ] Resume/complete the interrupted run to the full 20,000-step schedule
- [ ] Run a matched MHA-vs-STAR-parallel baseline at the same GPT-2 scale
- [ ] Compare against Mamba and RWKV at the same scale
- [ ] Add a proper CUDA/Triton kernel for the cumsum step and a KV-cache-equivalent for generation

---

## Legacy — Non-Parallel Original Version

> **Non-parallel original version, not optimized and not fully tested.**

This is the original formulation the idea started from: a strict step-by-step recurrence implemented as a Python `for` loop over the sequence. It is what `STARLayer` in `STARLLM.py` uses, and it cannot be parallelized across time steps on GPU — each step must wait for the previous one to finish. It's kept here for reference and history, not as the recommended layer to use.

```python
class STARLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.tok_proj   = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False), nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.scene_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False), nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.W_out      = nn.Linear(d_model, d_model, bias=False)
        self.scene_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        scene   = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        out     = []
        for i in range(T):
            tok   = x[:, i, :]
            delta = torch.tanh(self.tok_proj(tok) * (1.0 + self.scene_proj(scene)))
            scene = self.scene_norm(scene + delta)
            out.append(tok + self.W_out(scene))
        return torch.stack(out, dim=1)
```

### Legacy benchmark (old non-parallel layer only)

This benchmark used the non-parallel layer above, in a small **from-scratch** transformer (**`STARLLM.py`** and **`MHALLM.py`**), *not* the GPT-2 integration described above. It predates the parallel version and is kept for historical comparison only.

Trained on **WikiText-2-raw** for **1 epoch**, `d_model=512`, 4 layers, `seq_len=256`, batch size 16. All hyperparameters and training conditions were identical between the two models.

| Model | Params | Train loss | Val loss | Val perplexity |
|-------|--------|------------|----------|----------------|
| MHA Baseline | ~39M | 6.40 | 6.00 | 403 |
| STAR — non-parallel (legacy) | ~39M | 6.20 | 5.83 | 340 |

This legacy layer showed 15.6% lower perplexity than the MHA baseline at this small scale, but since it cannot be parallelized, it was not a practical layer to keep building on — hence the parallel rewrite above.

---

## Citation

If you use this work, please cite it, I am 16 years old and your citation can help me in future❤

---

*Built from scratch. Mechanism, theory, and experiments by a single maybe not good but researcher.*

