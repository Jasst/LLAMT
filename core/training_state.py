import json
from pathlib import Path
from typing import Dict, List, Optional

CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

def get_state() -> Dict:
    """Возвращает актуальный прогресс обучения для UI."""
    if not CKPT_DIR.exists():
        return {"status": "no_dir", "checkpoints": [], "log": [], "best": None}

    checkpoints = sorted(CKPT_DIR.glob("step_*.pt"))
    best_meta_p = CKPT_DIR / "best" / "meta.json"
    best = json.loads(best_meta_p.read_text()) if best_meta_p.exists() else None

    log_lines = []
    log_p = CKPT_DIR / "training_log.jsonl"
    if log_p.exists():
        tail = log_p.read_text(encoding="utf-8").strip().splitlines()[-200:]
        for line in tail:
            try: log_lines.append(json.loads(line))
            except Exception: pass

    running = (CKPT_DIR / "TRAINING.lock").exists()
    return {
        "status": "running" if running else "idle",
        "checkpoints": [p.name for p in checkpoints[-10:]],
        "last_step": checkpoints[-1].stem if checkpoints else None,
        "best": best,
        "log": log_lines,
    }