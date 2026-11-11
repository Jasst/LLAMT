import os, sys, json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Generator, Optional

try:
    import torch
except ImportError:
    torch = None


# ── Карта GGUF file_type → читаемое имя квантизации ──
GGUF_FILE_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "IQ3_XXS",
    22: "IQ1_S", 23: "IQ4_NL", 24: "IQ3_S", 25: "IQ2_S",
    26: "IQ4_XS", 27: "I8", 28: "I16", 29: "I32",
    30: "I64", 31: "F64", 32: "IQ1_M",
    33: "BF16", 34: "Q4_0_4_4", 35: "Q4_0_4_8",
    36: "Q4_0_8_8", 37: "TQ1_0", 38: "TQ2_0",
}

def _guess_quant_from_name(name: str) -> str:
    """Fallback: тянем метку квантизации из имени файла (q4_k_m, Q5_K_S, IQ4_XS…)."""
    import re
    m = re.search(r"(?i)\b(i?q\d+(?:_[a-z0-9]+)*|f16|f32|bf16)\b", name or "")
    return m.group(1).upper() if m else "?"


class ChatBackend(ABC):
    @abstractmethod
    def load(self, target: str, **kw): ...
    @abstractmethod
    def is_loaded(self) -> bool: ...
    @abstractmethod
    def info(self) -> Dict: ...
    @abstractmethod
    def chat_stream(self, messages: List[Dict], **kw) -> Generator[str, None, None]: ...

    def chat(self, messages: List[Dict], **kw) -> str:
        return "".join(self.chat_stream(messages, **kw))

    def unload(self):
        pass


# ─────────────────────────────────────────────────────────────────────
#  GGUF (llama.cpp): полный набор параметров загрузки и сэмплинга
# ─────────────────────────────────────────────────────────────────────
class LlamaBackend(ChatBackend):
    def __init__(self):
        self.llm = None
        self.path = None
        self._meta: Dict = {}
        self._quant = "?"
        self._load_kwargs: Dict = {}

    # -------- загрузка --------
    def load(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        flash_attn: bool = False,
        kv_cache_type: str = "f16",       # f16 | q8_0 | q4_0 | q4_1
        use_mmap: bool = True,
        use_mlock: bool = False,
        offload_kqv: bool = True,
        rope_freq_base: float = 0.0,      # 0 = из модели
        rope_freq_scale: float = 0.0,     # 0 = авто
        seed: int = -1,
        chat_format: Optional[str] = None,
    ):
        if sys.platform == "win32":
            for d in (r"C:\llama.cpp\build\bin\Release",
                      r"C:\llama.cpp\build\bin"):
                if Path(d).exists():
                    os.add_dll_directory(d)
                    break

        from llama_cpp import Llama
        if self.llm is not None:
            self.unload()

        kv = (kv_cache_type or "f16").lower()
        kwargs = dict(
            model_path=model_path,
            n_ctx=int(n_ctx),
            n_batch=int(n_batch),
            n_gpu_layers=int(n_gpu_layers),
            n_threads=int(n_threads) if n_threads else (os.cpu_count() or 4),
            n_threads_batch=int(n_threads_batch) if n_threads_batch
                             else (n_threads or os.cpu_count() or 4),
            flash_attn=bool(flash_attn),
            use_mmap=bool(use_mmap),
            use_mlock=bool(use_mlock),
            offload_kqv=bool(offload_kqv),
            seed=int(seed),
            verbose=False,
            type_k=kv if kv in ("f16", "q8_0", "q4_0", "q4_1") else "f16",
            type_v=kv if kv in ("f16", "q8_0", "q4_0", "q4_1") else "f16",
        )
        if rope_freq_base and rope_freq_base > 0:
            kwargs["rope_freq_base"] = float(rope_freq_base)
        if rope_freq_scale and rope_freq_scale > 0:
            kwargs["rope_freq_scale"] = float(rope_freq_scale)
        if chat_format:
            kwargs["chat_format"] = chat_format

        # У старых версий llama_cpp нет части аргументов — отсеиваем.
        try:
            self.llm = Llama(**kwargs)
        except TypeError:
            legacy = {k: v for k, v in kwargs.items()
                      if k in ("model_path", "n_ctx", "n_batch",
                               "n_gpu_layers", "n_threads",
                               "seed", "verbose")}
            self.llm = Llama(**legacy)

        self.path = model_path
        self._load_kwargs = {
            "n_ctx": int(n_ctx),
            "n_batch": int(n_batch),
            "n_gpu_layers": int(n_gpu_layers),
            "n_threads": kwargs["n_threads"],
            "n_threads_batch": kwargs["n_threads_batch"],
            "flash_attn": bool(flash_attn),
            "kv_cache_type": kv,
            "use_mmap": bool(use_mmap),
            "use_mlock": bool(use_mlock),
            "offload_kqv": bool(offload_kqv),
            "rope_freq_base": float(rope_freq_base),
            "rope_freq_scale": float(rope_freq_scale),
            "seed": int(seed),
        }
        self._meta = getattr(self.llm, "metadata", {}) or {}
        self._quant = self._detect_quant()

    def _detect_quant(self) -> str:
        m = self._meta or {}
        ft = m.get("general.file_type")
        if isinstance(ft, int) and ft in GGUF_FILE_TYPE:
            return GGUF_FILE_TYPE[ft]
        # некоторые модели кладут строку прямо в general.quantization_version
        q = m.get("general.quantization_version")
        if isinstance(q, str) and q:
            return q.upper()
        return _guess_quant_from_name(Path(self.path).name if self.path else "")

    # -------- состояние --------
    def is_loaded(self):
        return self.llm is not None

    def unload(self):
        if self.llm is not None:
            try:
                del self.llm
            finally:
                self.llm = None
                self._meta = {}
                self._load_kwargs = {}

    def info(self) -> Dict:
        if not self.llm:
            return {}
        m = getattr(self.llm, "metadata", {}) or self._meta or {}
        arch = m.get("general.architecture", "?")
        n = m.get("general.parameter_count", "?")
        if isinstance(n, int):
            n = f"{n/1e9:.2f}B" if n >= 1e9 else f"{n/1e6:.1f}M"
        n_ctx_train = m.get(f"{arch}.context_length", "?")
        n_ctx = self._load_kwargs.get("n_ctx", n_ctx_train)
        return {
            "type": "GGUF",
            "arch": arch,
            "params": str(n),
            "quant": self._quant,
            "ctx": n_ctx,
            "ctx_train": n_ctx_train,
            "file": Path(self.path).name if self.path else "?",
            "n_gpu_layers": self._load_kwargs.get("n_gpu_layers", -1),
            "flash_attn": self._load_kwargs.get("flash_attn", False),
            "kv_cache_type": self._load_kwargs.get("kv_cache_type", "f16"),
        }

    # -------- стриминг --------
    def chat_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        repeat_penalty: float = 1.1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        max_tokens: int = 1024,
        seed: int = -1,
    ):
        if not self.llm:
            raise RuntimeError("GGUF model not loaded")

        kw = dict(
            messages=messages,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repeat_penalty=float(repeat_penalty),
            frequency_penalty=float(frequency_penalty),
            presence_penalty=float(presence_penalty),
            max_tokens=int(max_tokens),
            stream=True,
            seed=int(seed) if seed is not None and seed >= 0 else None,
        )
        # min_p / typical_p есть в новых версиях llama_cpp — передаём мягко
        try:
            if min_p and min_p > 0:
                kw["min_p"] = float(min_p)
            if typical_p and typical_p != 1.0:
                kw["typical_p"] = float(typical_p)
            if mirostat_mode in (1, 2):
                kw["mirostat_mode"] = int(mirostat_mode)
                kw["mirostat_tau"] = float(mirostat_tau)
                kw["mirostat_eta"] = float(mirostat_eta)
        except Exception:
            pass

        stream = self.llm.create_chat_completion(**kw)
        for c in stream:
            d = c["choices"][0].get("delta", {})
            if d.get("content"):
                yield d["content"]


# ─────────────────────────────────────────────────────────────────────
#  LAMT (без изменений по логике — компактно)
# ─────────────────────────────────────────────────────────────────────
class LAMTBackend(ChatBackend):
    def __init__(self):
        self.model = None
        self.tok = None
        self.device = "cpu"
        self.ckpt_dir = None

    def load(self, ckpt_dir: str, device: str = "cuda", dtype="bfloat16"):
        if torch is None:
            raise RuntimeError("torch не установлен")
        from safetensors.torch import load_file
        from core.lamt_model import LAMT
        from transformers import AutoTokenizer

        ckpt = Path(ckpt_dir)
        if not (ckpt / "model.safetensors").exists():
            raise FileNotFoundError(
                f"{ckpt}/model.safetensors not found — сначала обучи модель "
                f"(`python -m core.train --corpus ...`)")

        tok_dir = ckpt / "tokenizer"
        if not ((tok_dir / "tokenizer.json").exists()
                or (tok_dir / "tokenizer_config.json").exists()):
            raise FileNotFoundError(f"Токенизатор не найден в {tok_dir}")

        cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
        model = LAMT(**cfg)
        sd = load_file(ckpt / "model.safetensors")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(
                f"В чекпоинте нет весов ({len(missing)} шт.): {missing[:5]}…")
        if unexpected:
            raise RuntimeError(
                f"В чекпоинте лишние веса ({len(unexpected)} шт.): {unexpected[:5]}…")

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}[dtype]
        self.model = model.to(self.device, dtype=dt).eval()
        self.tok = AutoTokenizer.from_pretrained(tok_dir)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.ckpt_dir = str(ckpt)

    def is_loaded(self):
        return self.model is not None

    def unload(self):
        if self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
        self.tok = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def info(self) -> Dict:
        if not self.model:
            return {}
        p = sum(x.numel() for x in self.model.parameters())
        return {
            "type": "LAMT", "arch": "lamt",
            "params": f"{p/1e9:.2f}B" if p >= 1e9 else f"{p/1e6:.1f}M",
            "quant": "bf16", "ctx": "?",
            "file": Path(self.ckpt_dir).name if self.ckpt_dir else "?",
        }

    def _encode(self, messages):
        if getattr(self.tok, "chat_template", None):
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        else:
            parts = [f"### {m['role']}\n{m['content']}" for m in messages]
            text = "\n".join(parts) + "\n### assistant\n"
        return self.tok(text, return_tensors="pt").input_ids

    def chat_stream(self, messages, temperature=0.7, top_p=0.9, top_k=40,
                    repeat_penalty=1.1, max_tokens=1024, seed=-1, **_):
        if not self.model:
            raise RuntimeError("LAMT model not loaded")
        if seed is not None and seed >= 0:
            torch.manual_seed(seed)

        with torch.no_grad():
            x = self._encode(messages).to(self.device)
            generated, eos = [], (self.tok.eos_token_id or -1)

            for _ in range(max_tokens):
                out = self.model(x)["logits"][:, -1, :].float()[0]
                if repeat_penalty != 1.0 and generated:
                    for prev in set(generated[-64:]):
                        out[prev] /= repeat_penalty
                if temperature > 0:
                    out = out / temperature
                if top_k > 0:
                    v, _ = out.topk(top_k)
                    out[out < v[-1]] = -float("inf")
                if top_p < 1.0:
                    s, idx = out.sort(descending=True)
                    cum = s.softmax(-1).cumsum(-1)
                    mask = cum > top_p
                    mask[1:] = mask[:-1].clone()
                    mask[0] = False
                    s[mask] = -float("inf")
                    out.scatter_(0, idx, s)

                nid = torch.multinomial(out.softmax(-1), 1).item()
                if nid == eos:
                    break
                generated.append(nid)
                yield self.tok.decode([nid], skip_special_tokens=True)
                x = torch.cat(
                    [x, torch.tensor([[nid]], device=self.device)], dim=1)