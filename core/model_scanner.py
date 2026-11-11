from pathlib import Path
from typing import List, Dict
import re

DEFAULT_MODELS_DIR = Path(r"E:\models")

_QUANT_RE = re.compile(
    r"(?i)\b(i?q\d+(?:_[a-z0-9]+)*|f16|f32|bf16|iq\d+(?:_[a-z0-9]+)*)\b"
)

def scan_gguf(root: Path = DEFAULT_MODELS_DIR) -> List[Dict]:
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for p in root.rglob("*.gguf"):
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        m = _QUANT_RE.search(p.name)
        quant = m.group(1).upper() if m else "?"
        out.append({
            "path": str(p),
            "name": p.name,
            "parent": p.parent.name,
            "size_gb": round(size / (1024**3), 2),
            "quant": quant,
        })
    out.sort(key=lambda x: (x["parent"].lower(), x["name"].lower()))
    return out

def format_label(item: Dict) -> str:
    q = item.get("quant", "?")
    return f"[{item['parent']}] {item['name']}  ·  {q}  ({item['size_gb']} GB)"