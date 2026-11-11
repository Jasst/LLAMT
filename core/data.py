from pathlib import Path
from typing import Union
import json
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Обычный корпус: .txt, по одному документу на строку."""
    def __init__(self, path: Union[str, Path], tokenizer, max_len: int = 512):
        self.tok = tokenizer
        self.max_len = max_len
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        self.lines = [ln.strip() for ln in lines if ln.strip()]
        if not self.lines:
            raise ValueError(f"Пустой корпус: {path}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        enc = self.tok(
            self.lines[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]
        labels = ids.clone()
        if self.tok.pad_token_id is not None:
            labels[labels == self.tok.pad_token_id] = -100
        return {"input_ids": ids, "labels": labels}


class ChatJsonlDataset(Dataset):
    """
    Дистиллят: {"messages": [...]} по строке.
    Промпт маскируется -100, обучаем ТОЛЬКО ответы ассистента.
    """
    def __init__(self, path: Union[str, Path], tokenizer, max_len: int = 512):
        self.tok = tokenizer
        self.max_len = max_len
        self.samples = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "messages" in rec and rec["messages"]:
                self.samples.append(rec["messages"])
        if not self.samples:
            raise ValueError(f"Пустой дистиллят: {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        msgs = self.samples[idx]
        ids_all, labels = [], []
        for m in msgs:
            piece = self.tok.apply_chat_template(
                [m], tokenize=False, add_generation_prompt=False)
            ids = self.tok(piece, add_special_tokens=False).input_ids
            ids_all.extend(ids)
            if m["role"] == "assistant":
                labels.extend(ids)
            else:
                labels.extend([-100] * len(ids))

        ids_all = ids_all[: self.max_len]
        labels = labels[: self.max_len]
        pad = self.max_len - len(ids_all)
        pad_id = self.tok.pad_token_id or 0
        ids_all += [pad_id] * pad
        labels += [-100] * pad
        return {
            "input_ids": torch.tensor(ids_all, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels":    torch.stack([b["labels"]    for b in batch]),
    }