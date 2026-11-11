import json, time, argparse, threading
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

from core.tokenizer_utils import ensure_tokenizer
from core.data import TextDataset, ChatJsonlDataset, collate_fn


# ─────── утилиты для сохранения ───────

def _infer_heads(model) -> int:
    from core.lamt_model import LocalAttention
    for m in model.modules():
        if isinstance(m, LocalAttention):
            return m.h
    return 8


def _infer_window(model) -> int:
    from core.lamt_model import LocalAttention
    for m in model.modules():
        if isinstance(m, LocalAttention):
            return m.window
    return 256


def save_checkpoint(model, opt, sched, step, loss, out_dir, tokenizer_vocab,
                    save_full=True, save_best=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_full:
        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "step": step,
            "loss": loss,
        }, out_dir / f"step_{step:07d}.pt")

    best_dir = out_dir / "best"
    best_dir.mkdir(exist_ok=True)
    meta_p = best_dir / "meta.json"
    prev_best = float("inf")
    if meta_p.exists():
        try:
            prev_best = json.loads(meta_p.read_text(encoding="utf-8"))["loss"]
        except Exception:
            pass

    if save_best and loss < prev_best:
        save_file({k: v.contiguous() for k, v in model.state_dict().items()},
                  best_dir / "model.safetensors")
        (best_dir / "config.json").write_text(json.dumps({
            "vocab":     tokenizer_vocab,
            "d":         model.embed.tok.embedding_dim,
            "n_heads":   _infer_heads(model),
            "n_layers":  len(model.blocks),
            "window":    _infer_window(model),
            "n_slots":   model.blocks[0].mem.n_slots,
            "n_experts": model.blocks[0].moe.E,
            "k_global":  model.router.k,
            "max_latent": model.latent.max_steps,
            "n_budgets": model.router.n_budgets,
        }, indent=2), encoding="utf-8")
        meta_p.write_text(json.dumps({"step": step, "loss": loss}),
                          encoding="utf-8")

    with (out_dir / "training_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "step": step, "loss": loss,
            "lr": sched.get_last_lr()[0] if sched else 0.0,
            "ts": time.time(),
        }) + "\n")


def load_resume_weights(model, path: str):
    """Загрузить веса из safetensors (для fine-tune поверх best)."""
    p = Path(path)
    if not p.exists():
        return False
    sd = load_file(p)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"↻ Resume from {p.name}: "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    return True


class TrainingLock:
    """TRAINING.lock: UI читает его наличие."""
    def __init__(self, ckpt_dir="checkpoints"):
        self.p = Path(ckpt_dir) / "TRAINING.lock"

    def __enter__(self):
        self.p.parent.mkdir(parents=True, exist_ok=True)
        self.p.write_text(str(__import__("os").getpid()), encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self.p.unlink(missing_ok=True)


# ─────── loss ───────

def compute_loss(out, labels, model, target_steps=2.0,
                 w_ponder=0.01, w_moe=0.01, w_budget=1e-3, w_mem=1e-3):
    V = out["logits"].size(-1)
    ce = F.cross_entropy(
        out["logits"][:, :-1].reshape(-1, V),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    ponder = ((out["expected_steps"] - target_steps) ** 2).mean()
    budget = out["budget"].float().mean() / 4.0
    reg = 0.0
    for m in model.modules():
        if hasattr(m, "M"):
            MtM = m.M @ m.M.T
            I = torch.eye(MtM.size(0), device=MtM.device)
            reg = reg + ((MtM - I) ** 2).mean()
    total = ce + w_ponder * ponder + w_moe * out["aux_moe"] \
           + w_budget * budget + w_mem * reg
    return total, {"ce": ce.item(), "ponder": ponder.item(),
                   "moe": out["aux_moe"].item(), "budget": budget.item(),
                   "steps": out["expected_steps"].mean().item()}


# ─────── полный цикл ───────

def run_training(
    corpus: str,
    ckpt_dir: str = "checkpoints",
    tok_source: str = "gpt2",
    tok_corpus: str = None,
    tok_vocab: int = 32000,
    d: int = 512, n_heads: int = 8, n_layers: int = 8, window: int = 256,
    n_slots: int = 64, n_experts: int = 8, k_global: int = 32,
    max_latent: int = 4, n_budgets: int = 4,
    max_len: int = 512,
    batch_size: int = 8,
    lr: float = 3e-4,
    steps: int = 10_000,
    warmup_steps: int = 500,
    save_every: int = 500,
    grad_clip: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    amp: bool = True,
    data_kind: str = "txt",              # NEW: "txt" | "jsonl"
    resume_from: Optional[str] = None,   # NEW: путь к safetensors
):
    from core.lamt_model import LAMT

    ckpt_path = Path(ckpt_dir)
    best_dir = ckpt_path / "best"

    vocab = ensure_tokenizer(best_dir, source=tok_source,
                             corpus=tok_corpus, vocab_size=tok_vocab)

    model = LAMT(vocab=vocab, d=d, n_heads=n_heads, n_layers=n_layers,
                 window=window, n_slots=n_slots, n_experts=n_experts,
                 k_global=k_global, max_latent=max_latent,
                 n_budgets=n_budgets)

    if resume_from is None:
        default_best = best_dir / "model.safetensors"
        if default_best.exists():
            resume_from = str(default_best)
    if resume_from:
        load_resume_weights(model, resume_from)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 LAMT: {n_params/1e6:.1f}M параметров, vocab={vocab}, "
          f"data_kind={data_kind}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(best_dir / "tokenizer")

    if data_kind == "jsonl":
        ds = ChatJsonlDataset(corpus, tok, max_len=max_len)
    else:
        ds = TextDataset(corpus, tok, max_len=max_len)
    print(f"📚 Примеров в датасете: {len(ds)}")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0,
                        drop_last=len(ds) >= batch_size,
                        pin_memory=(device != "cpu"))

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.02)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device != "cpu")

    model.train()
    step, running_loss = 0, 0.0
    t0 = time.time()

    with TrainingLock(ckpt_dir):
        while step < steps:
            for batch in loader:
                ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda",
                                    enabled=amp and device != "cpu"):
                    out = model(ids)
                    loss, logs = compute_loss(out, labels, model)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()

                running_loss += loss.item()
                step += 1

                if step % 50 == 0:
                    avg = running_loss / 50
                    speed = step / max(time.time() - t0, 1e-6)
                    print(f"step {step:6d} | loss {avg:.4f} | "
                          f"ce {logs['ce']:.4f} | steps={logs['steps']:.2f} | "
                          f"{speed:.1f} it/s")
                    running_loss = 0.0

                if step % save_every == 0:
                    save_checkpoint(model, opt, sched, step, loss.item(),
                                    ckpt_path, vocab, save_best=True)

                if step >= steps:
                    break

    save_checkpoint(model, opt, sched, step, loss.item(),
                    ckpt_path, vocab, save_best=True)
    print(f"✅ Готово. Лучший чекпоинт: {best_dir}/")


def start_training_thread(corpus: str, data_kind: str = "jsonl",
                          steps: int = 500, **kwargs) -> threading.Thread:
    """
    Запуск run_training в фоновом потоке.
    Возвращает поток (его можно проверить .is_alive()).
    """
    def _run():
        try:
            run_training(corpus=corpus, data_kind=data_kind,
                         steps=steps, **kwargs)
        except Exception as e:
            print(f"[train] ❌ {type(e).__name__}: {e}")

    th = threading.Thread(target=_run, daemon=True, name="lamt-trainer")
    th.start()
    return th


if __name__ == "__main__":
    ap = argparse.ArgumentParser("LAMT training")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--kind", choices=["txt", "jsonl"], default="txt")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--tok-source", default="gpt2")
    ap.add_argument("--tok-corpus", default=None)
    ap.add_argument("--tok-vocab", type=int, default=32000)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--n-slots", type=int, default=64)
    ap.add_argument("--n-experts", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume-from", default=None)
    a = ap.parse_args()

    run_training(
        corpus=a.corpus, data_kind=a.kind, ckpt_dir=a.ckpt_dir,
        tok_source=a.tok_source, tok_corpus=a.tok_corpus,
        tok_vocab=a.tok_vocab, d=a.d, n_heads=a.n_heads,
        n_layers=a.n_layers, window=a.window, n_slots=a.n_slots,
        n_experts=a.n_experts, max_len=a.max_len,
        batch_size=a.batch_size, lr=a.lr, steps=a.steps,
        save_every=a.save_every, resume_from=a.resume_from,
    )