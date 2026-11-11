import json
from pathlib import Path
from .backends import LlamaBackend


def generate_pairs(backend: LlamaBackend, prompts, out_path: str,
                   system: str = None, temperature: float = 0.8,
                   max_tokens: int = 768, seed: int = 42):
    """Дистилляция большой модели в JSONL через фиксированный seed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    teacher = backend.info().get("file", "?") if backend.is_loaded() else "?"
    with out_path.open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts, 1):
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
            kw = dict(temperature=temperature, max_tokens=max_tokens)
            if seed is not None and seed >= 0:
                kw["seed"] = seed + i
            answer = backend.chat(msgs, **kw)
            f.write(json.dumps({
                "messages": msgs + [{"role": "assistant", "content": answer}],
                "teacher": teacher,
            }, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(prompts)}] {prompt[:60]}...")


def dump_session_to_jsonl(session_id: str, out_path: str):
    """Одну активную ветку сессии → JSONL."""
    from . import storage
    path = storage.get_active_path(session_id)
    if len(path) < 2:
        return 0
    msgs = ([{"role": "system", "content": path[0]["content"]}]
            if path[0]["role"] == "system" else []) + \
           [{"role": n["role"], "content": n["content"]}
            for n in path if n["role"] != "system"]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    return 1


def dump_all_sessions_to_jsonl(out_path: str) -> int:
    """Все сессии БД → JSONL (перезапись). Возвращает число диалогов."""
    from . import storage
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for s in storage.list_sessions():
            path = storage.get_active_path(s["id"])
            if len(path) < 2:
                continue
            sys_ = next((p["content"] for p in path if p["role"] == "system"), None)
            body = [{"role": p["role"], "content": p["content"]}
                    for p in path if p["role"] != "system"]
            if body[-1]["role"] != "assistant":
                continue
            msgs = ([{"role": "system", "content": sys_}] if sys_ else []) + body
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_exchange_to_jsonl(session_id: str, out_path: str) -> bool:
    """
    Дозаписывает последнюю пару (user, assistant) из активной ветки.
    Возвращает True, если записано.
    """
    from . import storage
    path = storage.get_active_path(session_id)
    if len(path) < 2:
        return False
    if path[-2]["role"] != "user" or path[-1]["role"] != "assistant":
        return False
    if not path[-1]["content"].strip():
        return False

    sys_ = next((p["content"] for p in path if p["role"] == "system"), None)
    msgs = ([{"role": "system", "content": sys_}] if sys_ else []) + \
           [{"role": "user", "content": path[-2]["content"]},
            {"role": "assistant", "content": path[-1]["content"]}]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    return True