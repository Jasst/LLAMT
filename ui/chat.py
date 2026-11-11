import sys, threading
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import pandas as pd
from core.model_scanner import scan_gguf, format_label, DEFAULT_MODELS_DIR
from core.backends import LlamaBackend, LAMTBackend
from core import storage, training_state, distill
from core.train import start_training_thread

storage.init_db()

BACKEND = None
STOP = threading.Event()
_TRAIN_THREAD = None
_AUTO_STATE = {"trained_on": 0}
DEFAULT_SYSTEM = ("You are a helpful, concise assistant. "
                  "Answer in the user's language.")
CURSOR = "▌"
AUTO_DISTILL = "data/auto_distill.jsonl"


def _backend_badge() -> str:
    if BACKEND is None or not BACKEND.is_loaded():
        return '<span class="badge badge-off">⚪ Модель не загружена</span>'
    info = BACKEND.info()
    bits = [f"<b>{info.get('type', '?')}</b>"]
    q = info.get("quant")
    if q and q != "?":
        bits.append(f'<span class="badge-quant">{q}</span>')
    for k in ("params", "ctx"):
        if info.get(k) not in (None, "?"):
            bits.append(str(info[k]))
    return '<span class="badge badge-on">🟢 ' + " · ".join(bits) + "</span>"


def _train_badge() -> str:
    st = training_state.get_state()
    if st["status"] == "running":
        return '<span class="badge badge-on">🟢 обучение идёт</span>'
    if st["status"] == "no_dir":
        return '<span class="badge badge-off">⚪ checkpoints/ нет</span>'
    return '<span class="badge badge-idle">⚪ тренировка idle</span>'


# ─── сессии ───
def _sessions_map():
    return {f"💬 {s['title']}": s["id"] for s in storage.list_sessions()}


def new_chat():
    sid = storage.create_session()
    storage.add_node(sid, None, "system", DEFAULT_SYSTEM)
    labels = list(_sessions_map().keys())
    title = f"💬 {storage.get_session(sid)['title']}"
    return sid, gr.update(choices=labels, value=title), []


def switch_chat(label, current_sid):
    if not label:
        return current_sid, []
    sid = _sessions_map().get(label)
    if not sid:
        return current_sid, []
    path = storage.get_active_path(sid)
    hist = [{"role": n["role"], "content": n["content"], "_node": n["id"]}
            for n in path if n["role"] != "system"]
    return sid, hist


def delete_current(sid):
    if sid:
        storage.delete_session(sid)
    labels = list(_sessions_map().keys())
    return (gr.update(choices=labels, value=labels[0] if labels else None),
            None, [])


# ─── backend ───
def scan_models():
    items = scan_gguf()
    labels = [format_label(m) for m in items]
    mapping = {format_label(m): m["path"] for m in items}
    return gr.update(choices=labels,
                     value=labels[0] if labels else None), mapping


def load_gguf_backend(label, mapping, n_ctx, n_batch, n_gpu, n_threads,
                      flash_attn, kv_cache, use_mmap, use_mlock,
                      offload_kqv, rope_base, rope_scale, seed):
    global BACKEND
    STOP.set()
    if not label or not mapping:
        return _backend_badge(), "⚠️ Сначала просканируй папку"
    path = mapping.get(label)
    if not path:
        return _backend_badge(), "⚠️ Файл не найден"
    try:
        BACKEND = LlamaBackend()
        BACKEND.load(
            path,
            n_ctx=int(n_ctx), n_batch=int(n_batch),
            n_gpu_layers=int(n_gpu),
            n_threads=int(n_threads) if n_threads else None,
            flash_attn=bool(flash_attn),
            kv_cache_type=kv_cache,
            use_mmap=bool(use_mmap),
            use_mlock=bool(use_mlock),
            offload_kqv=bool(offload_kqv),
            rope_freq_base=float(rope_base or 0),
            rope_freq_scale=float(rope_scale or 0),
            seed=int(seed),
        )
        inf = BACKEND.info()
        return (_backend_badge(),
                f"✅ GGUF: `{Path(path).name}`\n"
                f"`{inf.get('arch')}` · **{inf.get('quant')}** · "
                f"{inf.get('params')} · ctx {inf.get('ctx')}")
    except Exception as e:
        BACKEND = None
        return _backend_badge(), f"❌ {type(e).__name__}: {e}"


def load_lamt_backend(ckpt_path, dtype):
    global BACKEND
    STOP.set()
    try:
        BACKEND = LAMTBackend()
        BACKEND.load(ckpt_path, dtype=dtype)
        return _backend_badge(), f"✅ LAMT: `{ckpt_path}`"
    except Exception as e:
        BACKEND = None
        return _backend_badge(), f"❌ {type(e).__name__}: {e}"


def unload_backend():
    global BACKEND
    STOP.set()
    if BACKEND is not None:
        try:
            BACKEND.unload()
        except Exception:
            pass
    BACKEND = None
    return _backend_badge(), "Модель выгружена"


# ─── чат ───
def user_submit(message, history, sid):
    if not message.strip():
        return "", history, sid, gr.update()

    dd_update = gr.update()
    if not sid:
        sid = storage.create_session()
        storage.add_node(sid, None, "system", DEFAULT_SYSTEM)
        labels = list(_sessions_map().keys())
        title = f"💬 {storage.get_session(sid)['title']}"
        dd_update = gr.update(choices=labels, value=title)

    history = history or []
    path = storage.get_active_path(sid)
    parent = path[-1]["id"] if path else None
    nid = storage.add_node(sid, parent, "user", message.strip())
    history.append({"role": "user", "content": message.strip(), "_node": nid})
    return "", history, sid, dd_update


def bot_respond(history, sid, system_prompt,
                temperature, top_p, top_k, min_p, typical_p,
                repeat_penalty, frequency_penalty, presence_penalty,
                mirostat_mode, mirostat_tau, mirostat_eta,
                max_tokens, seed):
    if BACKEND is None or not BACKEND.is_loaded():
        history = (history or []) + [{
            "role": "assistant",
            "content": "⚠️ Модель не загружена. Открой вкладку **Модель**.",
        }]
        yield history
        return

    if not history or history[-1]["role"] != "user":
        yield history
        return

    msgs = ([{"role": "system", "content": system_prompt}]
            if system_prompt else []) + \
           [{"role": m["role"], "content": m["content"]} for m in history]

    user_node_id = history[-1].get("_node")
    asst_nid = storage.add_node(sid, user_node_id, "assistant", "")
    history.append({"role": "assistant", "content": CURSOR, "_node": asst_nid})

    STOP.clear()
    buf = ""
    try:
        for tok in BACKEND.chat_stream(
                msgs,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                min_p=float(min_p),
                typical_p=float(typical_p),
                repeat_penalty=float(repeat_penalty),
                frequency_penalty=float(frequency_penalty),
                presence_penalty=float(presence_penalty),
                mirostat_mode=int(mirostat_mode),
                mirostat_tau=float(mirostat_tau),
                mirostat_eta=float(mirostat_eta),
                max_tokens=int(max_tokens),
                seed=int(seed)):
            if STOP.is_set():
                buf += "\n\n_⏹ остановлено_"
                break
            buf += tok
            history[-1]["content"] = buf + CURSOR
            yield history
        history[-1]["content"] = buf
        yield history
    finally:
        storage.update_node_content(asst_nid, buf)


def regenerate(history, sid, *gen_args):
    history = history or []
    if len(history) < 2 or history[-1]["role"] != "assistant":
        yield history
        return
    history = history[:-1]
    yield from bot_respond(history, sid, *gen_args)


def switch_branch(history, sid, delta):
    if not history or len(history) < 2:
        return history
    last = history[-1]
    siblings = storage.get_siblings(last.get("_node"))
    if len(siblings) < 2:
        return history
    ids = [s["id"] for s in siblings]
    i = ids.index(last["_node"])
    new_id = ids[(i + delta) % len(ids)]
    storage.set_selected_child(siblings[i]["parent_id"], new_id)
    path = storage.get_active_path(sid)
    return [{"role": n["role"], "content": n["content"], "_node": n["id"]}
            for n in path if n["role"] != "system"]


# ─── авто-дистилляция / обучение ───
def _count_lines(path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def _resolve_resume():
    best = Path("checkpoints/best/model.safetensors")
    return str(best) if best.exists() else None


def distill_current_session(sid, out_path):
    if not sid:
        return "⚠️ Нет активной сессии"
    ok = distill.dump_session_to_jsonl(sid, out_path)
    if not ok:
        return "⚠️ В сессии нет пары user→assistant"
    return f"✅ Дистиллят дописан: `{out_path}` ({_count_lines(out_path)} строк)"


def distill_all_sessions(out_path):
    try:
        n = distill.dump_all_sessions_to_jsonl(out_path)
        return f"✅ Собрано диалогов: **{n}** → `{out_path}`"
    except Exception as e:
        return f"❌ {type(e).__name__}: {e}"


def train_now(corpus, steps):
    global _TRAIN_THREAD
    if _TRAIN_THREAD is not None and _TRAIN_THREAD.is_alive():
        return "⚠️ Обучение уже идёт (см. бейдж 🟢 в шапке)"

    corpus = (corpus or "").strip()
    if not corpus:
        return (f"⚠️ Укажи путь к файлу датасета "
                f"(например, `{AUTO_DISTILL}`).")

    p = Path(corpus)

    # 1) Папку создаём всегда — иначе open('a') в distill упадёт
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return f"❌ Не удалось создать папку `{p.parent}`: {e}"

    # 2) Если файла нет — для стандартного авто-дистиллята
    #    попробуем собрать его из уже сохранённых сессий.
    if not p.exists():
        is_auto = (p.name == "auto_distill.jsonl")
        if is_auto:
            try:
                n = distill.dump_all_sessions_to_jsonl(str(p))
            except Exception as e:
                return (f"❌ Не удалось создать `{corpus}`: "
                        f"{type(e).__name__}: {e}")
            if n == 0:
                return (
                    f"📭 Файл `{corpus}` создан, но в нём пока нет данных.\n\n"
                    f"**Что делать (любой вариант):**\n"
                    f"1. Включи «🔁 Авто-обучение» в этой вкладке — каждая "
                    f"пара вопрос→ответ будет автоматически дописываться "
                    f"в файл.\n"
                    f"2. Нажми «📥 Дистилл. сессию» — сохранит текущий чат.\n"
                    f"3. Нажми «📚 Дистилл. всё» — сохранит все чаты.\n\n"
                    f"Затем снова «🚀 Обучить сейчас»."
                )
        else:
            return (
                f"⚠️ Файл не найден: `{corpus}`\n\n"
                f"Укажи путь к существующему JSONL-датасету "
                f"или используй стандартный `{AUTO_DISTILL}`."
            )

    # 3) Пустой файл (0 строк) — обучение упадёт, лучше сказать прямо
    n = _count_lines(corpus)
    if n == 0:
        return (
            f"⚠️ Датасет `{corpus}` пуст (0 строк).\n\n"
            f"Пообщайся с моделью при включённом авто-обучении "
            f"или сделай дистилляцию через кнопки "
            f"«📥 Дистилл. сессию» / «📚 Дистилл. всё»."
        )

    resume = _resolve_resume()
    _AUTO_STATE["trained_on"] = n
    _TRAIN_THREAD = start_training_thread(
        corpus, data_kind="jsonl", steps=int(steps),
        resume_from=resume)
    extra = f", resume from `{Path(resume).name}`" if resume else ", с нуля"
    return (f"🚀 Обучение запущено на `{corpus}` "
            f"({n} пар, {int(steps)} steps{extra})")


def _post_reply_hook(sid, auto_on, auto_n, steps):
    if not auto_on or not sid:
        return read_training()
    ok = distill.append_exchange_to_jsonl(sid, AUTO_DISTILL)
    if not ok:
        return read_training()
    n = _count_lines(AUTO_DISTILL)
    if n - _AUTO_STATE["trained_on"] >= int(auto_n):
        if not (_TRAIN_THREAD and _TRAIN_THREAD.is_alive()):
            _AUTO_STATE["trained_on"] = n
            start_training_thread(AUTO_DISTILL, data_kind="jsonl",
                                  steps=int(steps),
                                  resume_from=_resolve_resume())
    return read_training()


def read_training():
    st = training_state.get_state()
    empty = pd.DataFrame({"step": [], "loss": []})
    if st["status"] == "no_dir":
        md = ("Папка `checkpoints/` не найдена. Запусти первое обучение:\n\n"
              "```\npython -m core.train --corpus data/corpus.txt\n```")
        return md, empty, _train_badge()

    badge = "🟢 идёт" if st["status"] == "running" else "⚪️ не запущено"
    last = st.get("last_step") or "—"
    best = st.get("best")
    best_s = (f"step **{best['step']}**, loss **{best['loss']:.4f}**"
              if best else "—")
    n_distill = _count_lines(AUTO_DISTILL)
    md = (f"**Статус:** {badge}  \n"
          f"**Последний чекпоинт:** `{last}`  \n"
          f"**Лучший:** {best_s}  \n"
          f"**Пар в авто-дистилляте:** {n_distill}  \n"
          f"**Обучено на:** {_AUTO_STATE['trained_on']}")

    log = st["log"]
    if log:
        table = "step | loss | lr\n---|---|---\n" + "\n".join(
            f"{r['step']} | {r['loss']:.4f} | {r['lr']:.2e}"
            for r in log[-10:])
        md += "\n\n**Хвост лога:**\n\n" + table

    df = (pd.DataFrame([{"step": r["step"], "loss": r["loss"]}
                        for r in log[-200:]]) if log else empty)
    return md, df, _train_badge()


# ─── CSS ───
_css_path = Path(__file__).with_name("style.css")
CSS = _css_path.read_text(encoding="utf-8") if _css_path.exists() else ""


# ─── интерфейс ───
with gr.Blocks(theme=gr.themes.Soft(primary_hue="slate"),
               css=CSS, title="LLAMT") as demo:
    session_state = gr.State(None)
    mapping_state = gr.State({})

    with gr.Row(elem_classes="header-row"):
        with gr.Column(scale=1, min_width=140):
            gr.Markdown("## 🦙 LLAMT", elem_classes="logo")
        with gr.Column(scale=6):
            backend_badge = gr.HTML(_backend_badge(),
                                    elem_classes="badge-wrap")
        with gr.Column(scale=1, min_width=160):
            train_badge = gr.HTML(_train_badge(),
                                  elem_classes="badge-wrap")

    with gr.Row():
        with gr.Column(scale=1, min_width=340, elem_classes="sidebar"):
            with gr.Tabs():

                with gr.Tab("💬 Чаты"):
                    sessions_dd = gr.Dropdown(label="", choices=[],
                                              interactive=True)
                    with gr.Row():
                        new_btn = gr.Button("➕ Новый", size="sm",
                                            variant="primary")
                        del_btn = gr.Button("🗑 Удалить", size="sm")
                    rename_tb = gr.Textbox(placeholder="новое имя чата",
                                           show_label=False)
                    rename_btn = gr.Button("✏️ Переименовать", size="sm",
                                           variant="secondary")

                # ── МОДЕЛЬ: расширенные параметры загрузки GGUF ──
                with gr.Tab("🧩 Модель"):
                    gr.Markdown(f"*GGUF: `{DEFAULT_MODELS_DIR}`*")
                    with gr.Group():
                        gr.Markdown("**GGUF (llama.cpp)**")
                        model_dd = gr.Dropdown(label="", choices=[],
                                               interactive=True)
                        with gr.Row():
                            scan_btn = gr.Button("🔄 Скан", size="sm")
                            gguf_load = gr.Button("📥 Загрузить", size="sm",
                                                  variant="primary")

                        with gr.Accordion("Параметры загрузки", open=True):
                            with gr.Row():
                                n_ctx = gr.Slider(512, 131072, 4096, step=512,
                                                  label="n_ctx (контекст)")
                                n_batch = gr.Slider(64, 2048, 512, step=64,
                                                    label="n_batch")
                            with gr.Row():
                                n_gpu = gr.Slider(-1, 100, -1, step=1,
                                                  label="n_gpu_layers")
                                n_threads = gr.Number(
                                    value=0, precision=0, label="n_threads (0=auto)")
                            flash_attn = gr.Checkbox(
                                value=False, label="Flash Attention")
                            kv_cache = gr.Dropdown(
                                label="KV-cache type",
                                choices=["f16", "q8_0", "q4_0", "q4_1"],
                                value="f16", interactive=True)
                            with gr.Row():
                                use_mmap = gr.Checkbox(value=True, label="use_mmap")
                                use_mlock = gr.Checkbox(value=False, label="use_mlock")
                                offload_kqv = gr.Checkbox(value=True, label="offload_kqv")
                            with gr.Row():
                                rope_base = gr.Number(
                                    value=0, precision=0,
                                    label="rope_freq_base (0=auto)")
                                rope_scale = gr.Number(
                                    value=0, precision=0,
                                    label="rope_freq_scale (0=auto)")
                            seed = gr.Number(-1, precision=0, label="seed")

                    with gr.Group():
                        gr.Markdown("**LAMT (свой чекпоинт)**")
                        ckpt_path = gr.Textbox(label="папка",
                                               value="checkpoints/best")
                        dtype_dd = gr.Dropdown(
                            label="dtype",
                            choices=["bfloat16", "float16", "float32"],
                            value="bfloat16", interactive=True)
                        lamt_load = gr.Button("📥 Загрузить LAMT", size="sm",
                                              variant="primary")

                    unload_btn = gr.Button("⏏ Выгрузить модель",
                                           size="sm", variant="stop")
                    load_status = gr.Markdown("")

                # ── ГЕНЕРАЦИЯ: все сэмплинг-параметры ──
                with gr.Tab("🎛 Генерация"):
                    with gr.Accordion("Сэмплинг", open=True):
                        with gr.Row():
                            temperature = gr.Slider(0, 2, 0.7, 0.05,
                                                    label="temperature")
                            top_p = gr.Slider(0, 1, 0.9, 0.01, label="top_p")
                        with gr.Row():
                            top_k = gr.Slider(1, 200, 40, 1, label="top_k")
                            min_p = gr.Slider(0, 1, 0.05, 0.01, label="min_p")
                        typical_p = gr.Slider(0, 1, 1.0, 0.01, label="typical_p")
                    with gr.Accordion("Штрафы", open=True):
                        with gr.Row():
                            repeat_penalty = gr.Slider(1.0, 2.0, 1.1, 0.01,
                                                       label="repeat_penalty")
                            frequency_penalty = gr.Slider(0, 2, 0.0, 0.05,
                                                          label="frequency_penalty")
                            presence_penalty = gr.Slider(0, 2, 0.0, 0.05,
                                                         label="presence_penalty")
                    with gr.Accordion("Mirostat (mode=1/2)", open=False):
                        with gr.Row():
                            mirostat_mode = gr.Radio(
                                choices=[0, 1, 2], value=0, label="mode")
                            mirostat_tau = gr.Slider(0.1, 10, 5.0, 0.1,
                                                     label="tau")
                            mirostat_eta = gr.Slider(0.01, 1, 0.1, 0.01,
                                                     label="eta")
                    with gr.Row():
                        max_tokens = gr.Slider(64, 16384, 1024, 64,
                                               label="max_tokens")
                        gen_seed = gr.Number(-1, precision=0,
                                             label="seed (−1=random)")
                    system_prompt = gr.Textbox(value=DEFAULT_SYSTEM, lines=4,
                                               label="system prompt")

                with gr.Tab("📊 Обучение"):
                    with gr.Group():
                        gr.Markdown("**Авто-дистилляция**")
                        auto_on = gr.Checkbox(
                            value=False,
                            label="🔁 Авто-обучение после ответов ассистента")
                        auto_n = gr.Slider(
                            1, 50, 5, step=1,
                            label="обучать после N новых пар")
                        auto_steps = gr.Slider(
                            100, 5000, 500, step=100,
                            label="шагов на запуск (fine-tune)")

                    with gr.Group():
                        gr.Markdown("**Ручное управление**")
                        distill_path = gr.Textbox(
                            value=AUTO_DISTILL, label="файл авто-дистиллята")
                        with gr.Row():
                            distill_sess_btn = gr.Button(
                                "📥 Дистилл. сессию", size="sm")
                            distill_all_btn = gr.Button(
                                "📚 Дистилл. всё", size="sm")
                        train_now_btn = gr.Button(
                            "🚀 Обучить сейчас", size="sm", variant="primary")
                        manual_out = gr.Markdown("")

                    refresh_train = gr.Button("🔄 Обновить статус",
                                              size="sm", variant="secondary")
                    train_md = gr.Markdown("*нажми «Обновить»*")
                    loss_plot = gr.LinePlot(x="step", y="loss", height=200,
                                            show_legend=False)

        with gr.Column(scale=4, elem_classes="main"):
            chatbot = gr.Chatbot(type="messages", height=560,
                                 show_copy_button=True,
                                 avatar_images=(None, "🦙"), label="")
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Сообщение…  Enter — отправить",
                    show_label=False, lines=1, max_lines=6,
                    scale=9, autofocus=True)
                send_btn = gr.Button("▸", variant="primary", scale=1,
                                     min_width=48)
            with gr.Row(elem_classes="controls-row"):
                regen_btn = gr.Button("🔁 Перегенерировать", size="sm",
                                      variant="secondary", scale=2)
                prev_btn = gr.Button("◀", size="sm", variant="secondary",
                                     scale=1)
                next_btn = gr.Button("▶", size="sm", variant="secondary",
                                     scale=1)
                stop_btn = gr.Button("⏹ Стоп", size="sm",
                                     elem_classes="gr-button-stop", scale=1)
                clear_btn = gr.Button("🗑", size="sm", variant="secondary",
                                      scale=1)

    gen_inputs = [chatbot, session_state, system_prompt,
                  temperature, top_p, top_k, min_p, typical_p,
                  repeat_penalty, frequency_penalty, presence_penalty,
                  mirostat_mode, mirostat_tau, mirostat_eta,
                  max_tokens, gen_seed]
    auto_inputs = [session_state, auto_on, auto_n, auto_steps]

    demo.load(
        lambda: (
            gr.update(choices=list(_sessions_map().keys())),
            None, "", _backend_badge(), _train_badge(),
        ),
        outputs=[sessions_dd, session_state, rename_tb,
                 backend_badge, train_badge])

    new_btn.click(new_chat,
                  outputs=[session_state, sessions_dd, chatbot])
    sessions_dd.change(switch_chat, [sessions_dd, session_state],
                       [session_state, chatbot])
    rename_btn.click(
        lambda sid, name: (storage.rename_session(sid, name or "чат"),
                           list(_sessions_map().keys())),
        inputs=[session_state, rename_tb],
        outputs=[rename_tb, sessions_dd])
    del_btn.click(delete_current, session_state,
                  outputs=[sessions_dd, session_state, chatbot])

    scan_btn.click(scan_models, outputs=[model_dd, mapping_state])
    gguf_load.click(
        load_gguf_backend,
        [model_dd, mapping_state, n_ctx, n_batch, n_gpu, n_threads,
         flash_attn, kv_cache, use_mmap, use_mlock, offload_kqv,
         rope_base, rope_scale, seed],
        [backend_badge, load_status])
    lamt_load.click(load_lamt_backend,
                    [ckpt_path, dtype_dd],
                    [backend_badge, load_status])
    unload_btn.click(unload_backend, None, [backend_badge, load_status])

    msg.submit(user_submit, [msg, chatbot, session_state],
               [msg, chatbot, session_state, sessions_dd]).then(
        bot_respond, gen_inputs, chatbot).then(
        _post_reply_hook, auto_inputs,
        [train_md, loss_plot, train_badge])
    send_btn.click(user_submit, [msg, chatbot, session_state],
                   [msg, chatbot, session_state, sessions_dd]).then(
        bot_respond, gen_inputs, chatbot).then(
        _post_reply_hook, auto_inputs,
        [train_md, loss_plot, train_badge])

    regen_btn.click(regenerate, gen_inputs, chatbot).then(
        _post_reply_hook, auto_inputs,
        [train_md, loss_plot, train_badge])
    prev_btn.click(lambda h, s: switch_branch(h, s, -1),
                   [chatbot, session_state], chatbot)
    next_btn.click(lambda h, s: switch_branch(h, s, +1),
                   [chatbot, session_state], chatbot)
    stop_btn.click(lambda: STOP.set(), queue=False)
    clear_btn.click(lambda: [], None, chatbot)

    distill_sess_btn.click(
        distill_current_session,
        [session_state, distill_path],
        manual_out)
    distill_all_btn.click(
        distill_all_sessions, distill_path, manual_out)
    train_now_btn.click(
        train_now, [distill_path, auto_steps], manual_out)
    refresh_train.click(
        read_training, outputs=[train_md, loss_plot, train_badge])

    try:
        auto = gr.Timer(15)
        auto.tick(read_training,
                  outputs=[train_md, loss_plot, train_badge])
    except AttributeError:
        pass


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        inbrowser=True, show_error=True, server_name="127.0.0.1")