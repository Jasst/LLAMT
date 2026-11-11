import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# 1. Типизированные эмбеддинги
# ─────────────────────────────────────────────────────────────
class TypedEmbedding(nn.Module):
    def __init__(self, vocab_size, d, n_types=3):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.type_emb = nn.Embedding(n_types, d)
    def forward(self, ids, type_ids):
        return self.tok(ids) + self.type_emb(type_ids)

# ─────────────────────────────────────────────────────────────
# 2. Router
# ─────────────────────────────────────────────────────────────
class Router(nn.Module):
    def __init__(self, d, n_budgets=4, k=32):
        super().__init__()
        self.n_budgets, self.k = n_budgets, k
        self.net = nn.Sequential(nn.Linear(d, 256), nn.GELU(),
                                 nn.Linear(256, n_budgets + 1))
    def forward(self, x):
        out = self.net(x)
        budget_logits = out[..., :self.n_budgets]
        score = out[..., -1]
        k_eff = min(self.k, score.size(-1))          # fix: не падать на коротком вводе
        _, topk_idx = score.topk(k_eff, dim=-1)
        return budget_logits, topk_idx.sort(-1).values

# ─────────────────────────────────────────────────────────────
# 3. Локальное внимание
# ─────────────────────────────────────────────────────────────
class LocalAttention(nn.Module):
    def __init__(self, d, n_heads, window=256):
        super().__init__()
        self.h, self.dh, self.window = n_heads, d // n_heads, window
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
    def forward(self, x):
        B, N, _ = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dh).unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        idx = torch.arange(N, device=x.device)
        band = (idx[None, :] - idx[:, None]).abs() <= self.window // 2
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        scores = scores.masked_fill(~band, float('-inf'))
        out = scores.softmax(-1) @ v
        return self.o(out.transpose(1, 2).reshape(B, N, -1))

# ─────────────────────────────────────────────────────────────
# 4. Разреженное глобальное внимание
# ─────────────────────────────────────────────────────────────
class SparseGlobalAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
    def forward(self, x, topk_idx):
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(x).view(B, N, 2, self.h, self.dh).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        idx = topk_idx[:, None, :, None].expand(-1, self.h, -1, self.dh)
        k_sel = torch.gather(k, 2, idx)
        v_sel = torch.gather(v, 2, idx)
        scores = (q @ k_sel.transpose(-1, -2)) / math.sqrt(self.dh)
        out = scores.softmax(-1) @ v_sel
        return self.o(out.transpose(1, 2).reshape(B, N, -1))

# ─────────────────────────────────────────────────────────────
# 5. Память (статичная, обучаемая — не рабочая)
# ─────────────────────────────────────────────────────────────
class MemoryBank(nn.Module):
    """
    Слоты памяти — обучаемый словарь, НЕ обновляются по ходу инференса.
    (В прошлой версии был write_gate, который нигде не сохранялся.
     Для стабильности чата оставляем только чтение.)
    """
    def __init__(self, n_slots, d, n_heads):
        super().__init__()
        self.M = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.n_slots, self.h, self.dh = n_slots, n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
    def forward(self, x):
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(self.M).view(self.n_slots, 2, self.h, self.dh).unbind(1)
        k, v = k.transpose(0, 1), v.transpose(0, 1)
        attn = ((q @ k.transpose(-1, -2)) / math.sqrt(self.dh)).softmax(-1)
        read = self.o((attn @ v).transpose(1, 2).reshape(B, N, -1))
        return read

# ─────────────────────────────────────────────────────────────
# 6. MoE
# ─────────────────────────────────────────────────────────────
class MoE(nn.Module):
    def __init__(self, d, n_experts=8, top_k=2, ffn_mult=4):
        super().__init__()
        self.E, self.top_k = n_experts, top_k
        self.gate = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(),
                          nn.Linear(ffn_mult * d, d))
            for _ in range(n_experts)
        ])
    def forward(self, x):
        logits = self.gate(x)
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self.top_k, -1)
        topv = topv / topv.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            w = (topv * (topi == e)).sum(-1)
            m = w > 0
            if m.any():
                out[m] += w[m, None] * expert(x[m])
        one_hot = F.one_hot(topi, self.E).sum(1).float()
        aux = self.E * (one_hot.mean(0) * probs.mean(0)).sum()
        return out, aux

# ─────────────────────────────────────────────────────────────
# 7. Адаптивный блок
# ─────────────────────────────────────────────────────────────
class AdaptiveBlock(nn.Module):
    def __init__(self, d, n_heads, window, n_slots, n_experts):
        super().__init__()
        self.n1, self.n2, self.n3, self.n4 = [nn.LayerNorm(d) for _ in range(4)]
        self.local = LocalAttention(d, n_heads, window)
        self.mem = MemoryBank(n_slots, d, n_heads)
        self.glob = SparseGlobalAttention(d, n_heads)
        self.moe = MoE(d, n_experts)
    def forward(self, x, topk_idx, mask):
        x = x + mask * self.local(self.n1(x))
        x = x + mask * self.mem(self.n2(x))
        x = x + mask * self.glob(self.n3(x), topk_idx)
        B, N, D = x.shape
        m, aux = self.moe(self.n4(x).reshape(-1, D))
        x = x + mask * m.reshape(B, N, D)
        return x, aux

# ─────────────────────────────────────────────────────────────
# 8. Латентный цикл с halting
# ─────────────────────────────────────────────────────────────
class LatentLoop(nn.Module):
    def __init__(self, d, n_heads, window, n_slots, n_experts, max_steps=4):
        super().__init__()
        self.block = AdaptiveBlock(d, n_heads, window, n_slots, n_experts)
        self.halt = nn.Linear(d, 1)
        self.max_steps = max_steps
    def forward(self, x, topk_idx):
        B, N, _ = x.shape
        alive = torch.ones(B, N, 1, device=x.device)
        halt_probs, expected = [], torch.zeros(B, N, device=x.device)
        for t in range(self.max_steps):
            x, _ = self.block(x, topk_idx, torch.ones_like(alive))  # fix: 2 значения, не 3
            h = torch.sigmoid(self.halt(x)) * alive
            halt_probs.append(h)
            expected += (t + 1) * h.squeeze(-1)
            alive = alive * (1 - h)
            if alive.max() < 1e-3:
                break
        return x, halt_probs, expected

# ─────────────────────────────────────────────────────────────
# 9. Полная LAMT
# ─────────────────────────────────────────────────────────────
class LAMT(nn.Module):
    def __init__(self, vocab, d=512, n_heads=8, n_layers=8, window=256,
                 n_slots=64, n_experts=8, k_global=32, max_latent=4, n_budgets=4):
        super().__init__()
        self.embed = TypedEmbedding(vocab, d)
        self.router = Router(d, n_budgets, k_global)
        self.blocks = nn.ModuleList([
            AdaptiveBlock(d, n_heads, window, n_slots, n_experts)
            for _ in range(n_layers)
        ])
        self.latent = LatentLoop(d, n_heads, window, n_slots, n_experts, max_latent)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.tok.weight

    def forward(self, ids, type_ids=None):
        if type_ids is None:
            type_ids = torch.zeros_like(ids)
        x = self.embed(ids, type_ids)
        budget_logits, topk_idx = self.router(x)
        budget = budget_logits.argmax(-1) + 1
        aux_acc = 0.0
        for l, block in enumerate(self.blocks):
            mask = (budget >= l + 1).float().unsqueeze(-1)
            x, aux = block(x, topk_idx, mask)
            aux_acc = aux_acc + aux
        x, halt_probs, expected = self.latent(x, topk_idx)
        return {
            'logits': self.head(self.norm(x)),
            'budget': budget,
            'halt_probs': halt_probs,
            'expected_steps': expected,
            'aux_moe': aux_acc / len(self.blocks),
            'router_logits': budget_logits,
        }