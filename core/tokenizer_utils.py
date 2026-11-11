from pathlib import Path
from typing import Optional, Union


# ChatML-подобный шаблон: не требует спецтокенов, работает с любым
# токенизатором (gpt2, свой BPE, HF-совместимые).
CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}### system\n{{ m['content'] }}\n"
    "{% elif m['role'] == 'user' %}### user\n{{ m['content'] }}\n"
    "{% elif m['role'] == 'assistant' %}### assistant\n{{ m['content'] }}\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}### assistant\n{% endif %}"
)


def attach_chat_template(tok_dir: Union[str, Path]) -> None:
    """Гарантирует, что у сохранённого токенизатора есть chat_template."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    tok.chat_template = CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(tok_dir)


def ensure_tokenizer(
    ckpt_dir: Union[str, Path],
    source: str = "gpt2",
    corpus: Optional[Union[str, Path]] = None,
    vocab_size: int = 32000,
    force: bool = False,
) -> int:
    """
    Гарантирует, что <ckpt_dir>/tokenizer/ содержит HF-токенизатор
    с приклеенным chat_template. Возвращает размер словаря.
    """
    ckpt_dir = Path(ckpt_dir)
    tok_dir = ckpt_dir / "tokenizer"

    ready = (tok_dir / "tokenizer.json").exists() \
            or (tok_dir / "tokenizer_config.json").exists()
    if ready and not force:
        attach_chat_template(tok_dir)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok_dir)
        print(f"✅ Токенизатор найден: {tok_dir}  (vocab={len(tok)})")
        return len(tok)

    tok_dir.mkdir(parents=True, exist_ok=True)

    if corpus:
        vocab = _train_bpe(Path(corpus), tok_dir, vocab_size)
        attach_chat_template(tok_dir)
        print(f"✅ Обучен свой BPE: {tok_dir}  (vocab={vocab})")
        return vocab

    from transformers import AutoTokenizer
    print(f"⏳ Скачиваю токенизатор '{source}' ...")
    tok = AutoTokenizer.from_pretrained(source)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(tok_dir)
    attach_chat_template(tok_dir)
    print(f"✅ Готовый токенизатор сохранён: {tok_dir}  (vocab={len(tok)})")
    return len(tok)


def _train_bpe(corpus: Path, out_dir: Path, vocab_size: int) -> int:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast

    if not corpus.exists():
        raise FileNotFoundError(f"Корпус не найден: {corpus}")

    tok = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|padding|>"],
        show_progress=True,
    )
    tok.train(files=[str(corpus)], trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|padding|>",
    )
    fast.save_pretrained(out_dir)
    return len(fast)