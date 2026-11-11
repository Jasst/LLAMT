"""CLI LLAMT: chat и distill."""
import sys, argparse
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.model_scanner import scan_gguf, DEFAULT_MODELS_DIR
    from core.backends import LlamaBackend
    from core.distill import generate_pairs
else:
    from .model_scanner import scan_gguf, DEFAULT_MODELS_DIR
    from .backends import LlamaBackend
    from .distill import generate_pairs

__all__ = ["LlamaBackend", "scan_gguf", "generate_pairs", "DEFAULT_MODELS_DIR"]


def _pick_model() -> str:
    models = scan_gguf()
    if not models:
        print(f"❌ В {DEFAULT_MODELS_DIR} не найдено *.gguf")
        sys.exit(1)
    print("Найдено моделей:")
    for i, m in enumerate(models, 1):
        print(f"  {i:2d}. [{m['parent']}] {m['name']}  ({m['size_gb']} GB)")
    return models[int(input("Выбери номер: ")) - 1]["path"]


def cli_chat():
    path = _pick_model()
    backend = LlamaBackend()
    print(f"⏳ Загрузка {path} ...")
    backend.load(path)
    print("✅ Готово. Ctrl+C для выхода.")
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    while True:
        try:
            user = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        if not user.strip():
            continue
        history.append({"role": "user", "content": user})
        print("Bot: ", end="", flush=True)
        buf = ""
        for tok in backend.chat_stream(history):
            print(tok, end="", flush=True)
            buf += tok
        print()
        history.append({"role": "assistant", "content": buf})


def cli_distill():
    path = _pick_model()
    backend = LlamaBackend()
    backend.load(path)
    prompts_file = input("Файл с промптами (по строке): ").strip()
    prompts = [l.strip() for l in Path(prompts_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = input("Куда сохранить JSONL [data/distill.jsonl]: ").strip() or "data/distill.jsonl"
    generate_pairs(backend, prompts, out,
                   system="Ты — эксперт. Отвечай развёрнуто и по делу.")
    print(f"✅ {len(prompts)} пар → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["chat", "distill"], default="chat", nargs="?")
    a = ap.parse_args()
    {"chat": cli_chat, "distill": cli_distill}[a.cmd]()