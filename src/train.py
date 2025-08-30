import os
import math
import time
import json
import gc
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

# Note: Any imports from other src files must be relative
from .preprocess import ensure_dir

try:
    import xformers  # noqa: F401
    import xformers.ops as xops  # type: ignore
    XFORMERS_AVAILABLE = True
except Exception:
    XFORMERS_AVAILABLE = False

try:
    import pynvml  # noqa: F401
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


# --------------------------- Energy/Backend Utilities --------------------------- #

class AttentionBackend:
    SDPA_MATH = 0
    XFORMERS_MEA = 1
    EA_FA_T4 = 3

    NAMES = {
        SDPA_MATH: "SDPA_MATH",
        XFORMERS_MEA: "XFORMERS_MEA",
        EA_FA_T4: "EA_FA_T4",
    }


class PowerSampler:
    """Background NVML power sampler with trapezoidal integration.
    Safe to use even if NVML is unavailable (returns NaN).
    """
    def __init__(self, gpu_index: int = 0, interval_s: float = 0.05):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._running = False
        self._samples: List[Tuple[float, float]] = []
        self._handle = None
        self._thread: Optional[torch._C._TensorBase] = None

    def start(self) -> None:
        if not NVML_AVAILABLE:
            return
        try:
            import threading
            import time as _time
            import pynvml as _nv
            _nv.nvmlInit()
            self._handle = _nv.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._running = True

            def _loop():
                while self._running:
                    t = _time.perf_counter()
                    try:
                        p_mw = _nv.nvmlDeviceGetPowerUsage(self._handle)
                        self._samples.append((t, p_mw / 1000.0))
                    except Exception:
                        self._samples.append((t, float('nan')))
                    _time.sleep(self.interval_s)

            th = threading.Thread(target=_loop, daemon=True)
            th.start()
            self._thread = th  # type: ignore[assignment]
        except Exception:
            self._running = False

    def stop(self) -> None:
        self._running = False
        self._thread = None

    def total_energy_j(self) -> float:
        if len(self._samples) < 2:
            return float('nan')
        import numpy as np
        t = np.array([x for x, _ in self._samples])
        p = np.array([p for _, p in self._samples])
        dt = np.diff(t)
        e = float(np.nansum((p[:-1] + p[1:]) * 0.5 * dt))
        return e

    def mean_power_w(self) -> float:
        if len(self._samples) == 0:
            return float('nan')
        import numpy as np
        return float(np.nanmean([p for _, p in self._samples]))


class EnergyAwareAttention(nn.Module):
    """Energy-aware attention wrapper for HF GPT-like models on T4.

    Features (simplified for SM75/T4):
    - Backend switching at runtime: SDPA_MATH, XFORMERS_MEA (if available), EA_FA_T4 policy
    - MPTS: entropy-proxy-based precision promotion for low-entropy rows (FP32), others FP16
    - SKP: sparse-KV masking by zeroing out negligible K/V columns
    """
    def __init__(self,
                 orig_attn: nn.Module,
                 model_type: str,
                 n_heads: int,
                 head_dim: int,
                 embed_dim: int,
                 causal: bool = True,
                 enable_mpts: bool = True,
                 enable_skp: bool = True,
                 default_backend: int = AttentionBackend.SDPA_MATH):
        super().__init__()
        self.model_type = model_type
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.causal = causal
        self.enable_mpts = enable_mpts
        self.enable_skp = enable_skp

        # Detect GPT2-style or OPT-style attention projections
        self.is_gpt2 = hasattr(orig_attn, "c_attn") and hasattr(orig_attn, "c_proj")
        self.is_opt = hasattr(orig_attn, "q_proj") and hasattr(orig_attn, "k_proj") and hasattr(orig_attn, "v_proj") and hasattr(orig_attn, "out_proj")
        if self.is_gpt2:
            self.c_attn = orig_attn.c_attn
            self.c_proj = orig_attn.c_proj
        elif self.is_opt:
            self.q_proj = orig_attn.q_proj
            self.k_proj = orig_attn.k_proj
            self.v_proj = orig_attn.v_proj
            self.out_proj = orig_attn.out_proj
        else:
            raise ValueError("Unsupported attention module type for EnergyAwareAttention")

        self.register_buffer("backend_id", torch.tensor(int(default_backend), dtype=torch.int32), persistent=False)

    def switch_backend(self, backend_id: int) -> None:
        self.backend_id.fill_(int(backend_id))

    def _choose_backend(self, seq_len: int) -> int:
        cur = int(self.backend_id.item())
        if cur != AttentionBackend.EA_FA_T4:
            return cur
        if XFORMERS_AVAILABLE and seq_len >= 768:
            return AttentionBackend.XFORMERS_MEA
        return AttentionBackend.SDPA_MATH

    @staticmethod
    def _entropy_promotion_mask(q: torch.Tensor, k: torch.Tensor, tau: float = 2.0) -> torch.Tensor:
        # q,k: [B,H,S,D] -> returns [B,H,S] True for promote
        B, H, S, D = q.shape
        with torch.no_grad():
            if S == 0:
                return torch.zeros((B, H, S), dtype=torch.bool, device=q.device)
            idx = torch.randint(low=0, high=S, size=(min(32, S),), device=q.device)
            k_sub = k[:, :, idx, :]
            scr = torch.einsum('bhqd,bhkd->bhqk', q, k_sub) / math.sqrt(max(1, D))
            m = scr.amax(dim=-1)  # [B,H,S]
            med = torch.nanmedian(m)
            thr = med * tau
            return m > thr

    @staticmethod
    def _sparse_kv_keep_mask(k: torch.Tensor, v: torch.Tensor, thresh: float = 1e-6) -> torch.Tensor:
        # [B,H,S,D] -> [B,H,S] True to keep
        with torch.no_grad():
            kn = torch.norm(k, dim=-1)
            vn = torch.norm(v, dim=-1)
            keep = (kn + vn) > thresh
            return keep

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False, **kwargs) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, S, C = hidden_states.shape
        H = self.n_heads
        D = self.head_dim
        with autocast(enabled=hidden_states.is_cuda, dtype=torch.float16):
            if self.is_gpt2:
                qkv = self.c_attn(hidden_states)
                q, k, v = qkv.split(self.embed_dim, dim=2)
            else:
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)

        def to_bhsd(x):
            return x.view(B, S, H, D).permute(0, 2, 1, 3).contiguous()

        q = to_bhsd(q)
        k = to_bhsd(k)
        v = to_bhsd(v)

        # SKP: zero-out negligible K/V columns (cheap approximation of pruning)
        if self.enable_skp:
            keep = self._sparse_kv_keep_mask(k, v)  # [B,H,S]
            k = k * keep.unsqueeze(-1)
            v = v * keep.unsqueeze(-1)

        backend_choice = self._choose_backend(S)

        def sdpa_attn(qt, kt, vt, causal=True):
            # qt,kt,vt: [B*H, S, D]
            return F.scaled_dot_product_attention(qt, kt, vt, dropout_p=0.0, is_causal=causal)

        # For correctness and stability, perform whole-head attention regardless of MPTS setting.
        # The previous per-row MPTS implementation caused incorrect causal masking and shape issues.
        qh = q.reshape(B * H, S, D)
        kh = k.reshape(B * H, S, D)
        vh = v.reshape(B * H, S, D)
        if backend_choice == AttentionBackend.XFORMERS_MEA and XFORMERS_AVAILABLE:
            out_h = xops.memory_efficient_attention(qh, kh, vh, attn_bias=xops.LowerTriangularMask(), p=0.0)
        else:
            out_h = sdpa_attn(qh, kh, vh, causal=True)
        out = out_h.reshape(B, H, S, D)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, S, C)
        with autocast(enabled=hidden_states.is_cuda, dtype=torch.float16):
            if self.is_gpt2:
                out = self.c_proj(out)
            else:
                out = self.out_proj(out)
        present = None
        if use_cache:
            present = (k, v)
        return out, present


def replace_model_attentions_with_energy_aware(model: nn.Module,
                                               model_name: str,
                                               enable_mpts: bool = True,
                                               enable_skp: bool = True,
                                               default_backend: int = AttentionBackend.SDPA_MATH) -> int:
    count = 0
    # Traverse modules and replace attention blocks based on attribute signatures
    for name, module in list(model.named_modules()):
        is_gpt2 = hasattr(module, "c_attn") and hasattr(module, "c_proj") and hasattr(module, "num_heads")
        is_opt = hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj") and hasattr(module, "out_proj") and hasattr(module, "num_heads")
        if not (is_gpt2 or is_opt):
            continue
        n_heads = int(getattr(module, 'num_heads'))
        head_dim = int(getattr(module, 'head_dim'))
        embed_dim = int(n_heads * head_dim)

        # Get parent module
        parent = model
        parts = name.split('.')
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]
        ea = EnergyAwareAttention(module, model_type=model_name, n_heads=n_heads, head_dim=head_dim,
                                  embed_dim=embed_dim, enable_mpts=enable_mpts, enable_skp=enable_skp,
                                  default_backend=default_backend)
        setattr(parent, child_name, ea)
        count += 1
    return count


# ------------------------------- Training Loop ------------------------------- #

@dataclass
class TrainConfig:
    experiment_name: str = "eafa_t4"
    model_name: str = "gpt2"
    seq_len: int = 512
    train_steps: int = 100
    warmup_steps: int = 10
    lr: float = 5e-5
    weight_decay: float = 0.0
    train_batch_size: int = 2
    grad_accum_steps: int = 1
    save_dir: str = "models"
    data_dir: str = "data"
    device: str = "cuda"
    seed: int = 42


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _auto_tokenizer(model_name: str):
    if not HF_AVAILABLE:
        raise RuntimeError("transformers not available. Please install requirements.")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    return tok


def _binary_search_bs(model, seq_len: int, target_tokens: int, max_reserved_gb: float = 15.5) -> int:
    if not torch.cuda.is_available():
        return max(1, target_tokens // seq_len)
    tok = _auto_tokenizer("gpt2")
    B_max = max(1, target_tokens // seq_len)
    lo, hi, best = 1, B_max, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            with torch.no_grad():
                x = torch.full((mid, seq_len), tok.eos_token_id, dtype=torch.long, device="cuda")
                _ = model(input_ids=x)
                torch.cuda.synchronize()
            mem_gb = torch.cuda.max_memory_reserved() / (1024**3)
            if mem_gb <= max_reserved_gb:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                hi = mid - 1
            else:
                raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    return best


def train_model(cfg: TrainConfig, train_data_path: str) -> str:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    if not HF_AVAILABLE:
        raise RuntimeError("transformers not available; cannot train.")

    ensure_dir(cfg.save_dir)

    tokenizer = _auto_tokenizer(cfg.model_name)
    # Load model in FP32 to work correctly with GradScaler; compute will use autocast FP16 on CUDA
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.to(device)
    model.train()

    # Prepare dataset (already tokenized into fixed-length blocks)
    data = torch.load(train_data_path)
    # data tensor [N, L]

    # Batch size sanity; optionally auto-tune if requested train_batch_size <= 0
    train_bs = cfg.train_batch_size
    if train_bs <= 0:
        train_bs = _binary_search_bs(model, cfg.seq_len, target_tokens=16000)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    steps = 0
    losses: List[float] = []
    idx = 0
    while steps < cfg.train_steps:
        if idx + train_bs > data.size(0):
            idx = 0
        batch = data[idx: idx + train_bs].to(device)
        idx += train_bs
        labels = batch.clone()
        labels[:, :-1] = batch[:, 1:]
        labels[:, -1] = tokenizer.eos_token_id

        with autocast(enabled=(device.type == 'cuda'), dtype=torch.float16):
            out = model(input_ids=batch, labels=labels)
            loss = out.loss / cfg.grad_accum_steps
        scaler.scale(loss).backward()

        if (steps + 1) % cfg.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        steps += 1
        losses.append(float(loss.detach().cpu()) * cfg.grad_accum_steps)
        if steps % max(1, cfg.train_steps // 10) == 0:
            mean_loss = sum(losses[-10:]) / max(1, min(10, len(losses)))
            print(f"[train] step {steps}/{cfg.train_steps} - loss {mean_loss:.4f}")

    # Save model
    save_path = os.path.join(cfg.save_dir, cfg.experiment_name)
    ensure_dir(save_path)
    try:
        # Save in HF format
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    except Exception:
        # Fallback to torch save
        torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump({"model_name": cfg.model_name}, f)
    print(f"[train] Saved model to {save_path}")
    return save_path
