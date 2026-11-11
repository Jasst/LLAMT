"""Проверка LAMT-чекпоинта: val-loss/ppl + живой сэмпл."""
import json, argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from core.data import TextDataset, ChatJsonlDataset, collate_fn
from core.lamt_model import LAMT
from core.backends import LAMTBackend
from safetensors.torch import load_file
from transformers import AutoTokenizer


def evaluate(ckpt_dir, val_path, data_kind="txt", max_len=512, batch_size=8):
    ckpt = Path(ckpt_dir)
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    model = LAMT(**cfg)
    missing, unexpected = model.load_state_dict(
        load_file(ckpt / "model.safetensors"), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Чекпоинт не сходится: missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    tok = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = (ChatJsonlDataset(val_path, tok, max_len=max_len)
          if data_kind == "jsonl"
          else TextDataset(val_path, tok, max_len=max_len))
    loader = DataLoader(ds, batch_size=batch_size,
                        collate_fn=collate_fn, num_workers=0)

    total_n, total_loss = 0, 0.0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(ids)["logits"]
            ce = torch.nn.functional.cross_entropy(
                out[:, :-1].reshape(-1, out.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100, reduction="sum")
            n = (labels[:, 1:] != -100).sum().item()
            total_loss += ce.item()
            total_n += n

    avg = total_loss / max(total_n, 1)
    ppl = torch.exp(torch.tensor(avg)).item()
    print(f"🔍 val-loss={avg:.4f}  ppl={ppl:.2f}  tokens={total_n}")
    return ppl


def sample(ckpt_dir, prompts):
    b = LAMTBackend()
    b.load(ckpt_dir, device="cuda")
    for p in prompts:
        print(f"\n### user\n{p}\n### assistant")
        for tok in b.chat_stream(
                [{"role": "user", "content": p}],
                temperature=0.7, max_tokens=200):
            print(tok, end="", flush=True)
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--val", required=True)
    ap.add_argument("--kind", choices=["txt", "jsonl"], default="txt")
    ap.add_argument("--sample", nargs="*", default=None)
    a = ap.parse_args()
    evaluate(a.ckpt_dir, a.val, a.kind)
    if a.sample:
        sample(a.ckpt_dir, a.sample)