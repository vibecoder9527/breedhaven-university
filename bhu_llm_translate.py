# -*- coding: utf-8 -*-
"""
Breedhaven University LLM 翻譯工具（Windows）

掃描 content/default/ 與 src/shared/i18n/en/，
對照 content/<lang_id>/ 已有翻譯，把還沒翻的句子分批送給 OpenAI 相容 LLM，
再寫回正確的語言資料夾結構。

不需要額外套件。雙擊「啟動翻譯.bat」，或：

    python bhu_llm_translate.py
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import ssl
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

APP_NAME = "Breedhaven LLM 翻譯"
CONFIG_NAME = "bhu_llm_config.json"

# 內容裡通常要翻譯的欄位名（大小寫不敏感比對）
TEXT_KEYS = {
    "name", "title", "description", "definition", "bio", "text", "label",
    "summary", "prompt", "example", "dialogue", "speech", "quirk", "secret",
    "opening", "narration", "direction", "tip", "tips", "question", "topic",
    "syllabus", "skill", "college", "subject", "course", "activity",
    "location", "region", "stall", "item", "job", "society", "event",
    "holiday", "festival", "fair", "room", "outfit", "expression",
    "caption", "message", "hint", "note", "flavor", "lore", "entry",
    "body", "content", "line", "reply", "response", "instruction", "blurb",
    "highlights", "lines", "ask", "answer", "appearance", "offer",
    "welcome", "nannyHello", "arrears", "paid", "evicted", "guidance", "clothes"
}

# 絕對不要翻的欄位
SKIP_KEYS = {
    "id", "code", "number", "path", "url", "src", "image", "sprite",
    "icon", "sound", "audio", "file", "filename", "type", "kind",
    "flag", "flags", "tag", "tags", "key", "keys", "ref", "refs",
    "locationid", "traitids", "studentids", "teacherid", "professor",
    "major", "level", "hour", "hours", "day", "days", "week",
    "x", "y", "price", "cost", "energy", "stat", "stats", "value",
    "min", "max", "seed", "version", "schema",
}

# UI namespace
UI_NAMESPACES = (
    "core", "scene", "llm", "game", "comfy", "editor",
    "shell", "screens", "panels",
)


@dataclass
class StringItem:
    key: str          # 用於快取的唯一鍵
    text: str         # 原文
    context: str      # 給 LLM 的上下文（檔案／欄位路徑）
    kind: str         # "ui" 或 "content"


@dataclass
class Settings:
    repo_root: str = "."
    lang_id: str = "zh-TW"
    lang_name: str = "繁體中文（台灣）"
    base_url: str = "https://api.openai.com/v1/chat/completions"
    api_keys: str = ""
    model: str = ""
    target_lang: str = "繁體中文（台灣）"
    temperature: float = 0.05
    timeout: int = 120
    batch_size: int = 15
    max_tokens: int = 4096
    glossary: str = "glossary.json"
    use_glossary: bool = True
    opencc: bool = True
    translate_ui: bool = True
    translate_content: bool = True

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def load(cls, path):
        obj = cls()
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
        return obj

    def save(self, path):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def config_path():
    return os.path.join(app_dir(), CONFIG_NAME)


def cache_path(settings: Settings) -> str:
    return os.path.join(app_dir(), f"translation_cache_{settings.lang_id}.json")


# ---------------------------------------------------------------------------
# 抽取待譯字串
# ---------------------------------------------------------------------------

def _normalize_key(key: str) -> str:
    """把 camelCase / snake_case / kebab 都變成小寫連續字串方便比對。"""
    # 先把駝峰拆開：nannyHello → nanny_hello
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return s.lower().replace("_", "").replace("-", "")


def _is_text_key(key: str) -> bool:
    k = _normalize_key(key)
    norm_text = {_normalize_key(s) for s in TEXT_KEYS}
    # TEXT_KEYS 優先：明確列出的一定翻
    if k in norm_text:
        return True
    if k in {_normalize_key(s) for s in SKIP_KEYS}:
        return False
    # 只擋明顯的識別欄位：以 Id/Ids/Path/Url/Src 結尾（駝峰或底線）
    if re.search(r"(Id|Ids|Path|Url|Src)$", key) or re.search(r"_(id|ids|path|url|src)$", key, re.I):
        return False
    return any(tk in k for tk in ("name", "title", "desc", "bio", "text", "def"))


def _worth(s: str, force: bool = False) -> bool:
    t = s.strip()
    if len(t) < 2:
        return False
    if t.isdigit():
        return False
    # TEXT_KEYS 明確列出的欄位，單字也要翻（如 answer: "Iscara"）
    if force:
        return True
    if re.fullmatch(r"[A-Za-z0-9_\-./]+", t):
        return False
    # 至少要有字母或中日韓字
    if not re.search(r"[A-Za-z\u00C0-\u024F\u3040-\u30FF\u3400-\u9FFF]", t):
        return False
    return True


def walk_content(obj: Any, path: str, out: list[StringItem], prefix: str, parent_key: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            is_text = _is_text_key(k)
            if isinstance(v, str) and is_text and _worth(v, force=is_text):
                out.append(StringItem(f"{prefix}:{p}", v, p, "content"))
            elif isinstance(v, list) and is_text:
                # 陣列裡的字串也要翻，例如 wrong / lines / keywords
                for i, item in enumerate(v):
                    if isinstance(item, str) and _worth(item, force=True):
                        ip = f"{p}[{i}]"
                        out.append(StringItem(f"{prefix}:{ip}", item, ip, "content"))
                    else:
                        walk_content(item, f"{p}[{i}]", out, prefix, k)
            else:
                walk_content(v, p, out, prefix, k)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            ident = ""
            if isinstance(item, dict):
                for idk in ("id", "code", "number"):
                    if idk in item:
                        ident = str(item[idk])
                        break
            p = f"{path}[{ident or i}]"
            force = _is_text_key(parent_key)
            if isinstance(item, str) and force and _worth(item, force=True):
                out.append(StringItem(f"{prefix}:{p}", item, p, "content"))
            else:
                walk_content(item, p, out, prefix, parent_key)


def extract_ui(repo: Path) -> list[StringItem]:
    items = []
    ui_root = repo / "src" / "shared" / "i18n" / "en"
    if not ui_root.is_dir():
        return items
    for ns in UI_NAMESPACES:
        f = ui_root / f"{ns}.json"
        if not f.is_file():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for k, v in data.items():
            if isinstance(v, str) and _worth(v) and k != "llm.writeIn":
                items.append(StringItem(f"ui:{ns}.{k}", v, f"{ns}.{k}", "ui"))
    return items


def extract_content(repo: Path) -> list[StringItem]:
    items = []
    content = repo / "content" / "default"
    if not content.is_dir():
        return items
    # 頂層 JSON
    for f in sorted(content.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        walk_content(data, "", items, f"content:{f.stem}")
    # love-interests
    li = content / "love-interests"
    if li.is_dir():
        for f in sorted(li.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            walk_content(data, "", items, f"content:love-interests/{f.stem}")
    return items


def extract_all(settings: Settings) -> list[StringItem]:
    repo = Path(settings.repo_root).resolve()
    items = []
    if settings.translate_ui:
        items.extend(extract_ui(repo))
    if settings.translate_content:
        items.extend(extract_content(repo))
    # 去重（同原文只留第一個，但保留所有 key 以便回寫）
    return items


# ---------------------------------------------------------------------------
# 快取與回寫
# ---------------------------------------------------------------------------

def load_cache(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def save_cache(path: str, cache: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def apply_translations(settings: Settings, cache: dict, items: list[StringItem], log):
    """把快取裡的譯文寫回 content/<lang_id>/ 結構。"""
    repo = Path(settings.repo_root).resolve()
    lang_dir = repo / "content" / settings.lang_id
    lang_dir.mkdir(parents=True, exist_ok=True)

    # language.json
    (lang_dir / "language.json").write_text(
        json.dumps({"name": settings.lang_name}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # UI strings
    ui_out: dict[str, dict[str, str]] = {}
    for it in items:
        if it.kind != "ui" or it.key not in cache:
            continue
        ns, key = it.key[3:].split(".", 1)  # ui:ns.key
        ui_out.setdefault(ns, {})[key] = cache[it.key]
    strings_dir = lang_dir / "strings"
    strings_dir.mkdir(exist_ok=True)
    for ns, table in ui_out.items():
        (strings_dir / f"{ns}.json").write_text(
            json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Content：依檔案分組，重建最小 JSON（只含已譯欄位 + id）
    # 簡化處理：對每個 content:xxx 鍵，收集已譯字串，寫成對應檔案
    by_file: dict[str, list[tuple[str, str]]] = {}
    for it in items:
        if it.kind != "content" or it.key not in cache:
            continue
        # content:love-interests/anwen:path 或 content:traits:path
        rest = it.key[8:]  # 去掉 "content:"
        file_part, _, field_path = rest.partition(":")
        by_file.setdefault(file_part, []).append((field_path, cache[it.key]))

    for file_part, pairs in by_file.items():
        # 讀英文原檔當骨架，只覆蓋已譯欄位
        src_path = None
        if file_part.startswith("love-interests/"):
            src_path = repo / "content" / "default" / "love-interests" / f"{file_part.split('/',1)[1]}.json"
            out_path = lang_dir / "love-interests" / f"{file_part.split('/',1)[1]}.json"
        else:
            src_path = repo / "content" / "default" / f"{file_part}.json"
            out_path = lang_dir / f"{file_part}.json"
        if not src_path or not src_path.is_file():
            continue
        try:
            data = json.loads(src_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # 用路徑覆蓋（簡易版：只處理一層或常見結構）
        for field_path, translated in pairs:
            _set_by_path(data, field_path, translated)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"已寫回 {lang_dir}")


def _set_by_path(obj, path: str, value):
    """依路徑設定值。支援 a.b[code].c 形式，陣列用 id/code/number 匹配。"""
    if not path:
        return
    parts = re.findall(r"([^.\[\]]+)|\[([^\]]+)\]", path)
    cur = obj
    for i, (key, idx) in enumerate(parts[:-1]):
        token = key or idx
        if key:
            if not isinstance(cur, dict) or key not in cur:
                return
            cur = cur[key]
        else:
            # [ident]
            if not isinstance(cur, list):
                return
            found = None
            for item in cur:
                if isinstance(item, dict):
                    for idk in ("id", "code", "number"):
                        if str(item.get(idk, "")) == token:
                            found = item
                            break
                    if found is not None:
                        break
            if found is None:
                # 退回數字索引
                try:
                    found = cur[int(token)]
                except Exception:
                    return
            cur = found
    # 最後一層
    last_key, last_idx = parts[-1]
    token = last_key or last_idx
    if last_key:
        if isinstance(cur, dict) and last_key in cur:
            if isinstance(cur[last_key], str):
                cur[last_key] = value
            elif isinstance(cur[last_key], list):
                # 不該走到這裡，陣列項目用 [i] 處理
                pass
    else:
        if isinstance(cur, list):
            try:
                idx = int(token)
                if 0 <= idx < len(cur) and isinstance(cur[idx], str):
                    cur[idx] = value
            except Exception:
                pass


# ---------------------------------------------------------------------------
# LLM 呼叫（沿用你 Ren'Py 工具的風格）
# ---------------------------------------------------------------------------

def load_glossary(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return {str(k): str(v) for k, v in data.items() if k and v} if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_messages(batch: list[StringItem], settings: Settings, glossary: dict):
    gloss = ""
    if glossary:
        blob = "\n".join(it.text for it in batch)
        hits = [(k, v) for k, v in glossary.items() if k in blob]
        if hits:
            gloss = "\n".join(f"- {k} => {v}" for k, v in hits[:60])
    sys = (
        f"你是遊戲在地化譯者。把給定句子譯成{settings.target_lang}。\n"
        "規則：\n"
        "1. 只輸出一個 JSON 陣列，不要說明、不要 markdown。\n"
        "2. 每項是 {\"i\": 編號, \"t\": \"譯文\"}，編號必須與輸入一致。\n"
        "3. 保留語氣、角色口吻。介面文字要簡潔。\n"
        "4. 角色名、地名除非詞彙表指定，否則可音譯或保留。\n"
        "5. {player} 這種佔位符原樣保留。\n"
    )
    if gloss:
        sys += "詞彙表（必須沿用）：\n" + gloss + "\n"
    items = [{"i": n, "ctx": it.context, "text": it.text} for n, it in enumerate(batch)]
    user = "待譯 JSON：\n" + json.dumps(items, ensure_ascii=False)
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def parse_llm_json(content: str):
    content = (content or "").strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    start = content.find("[")
    end = content.rfind("]")
    if start >= 0 and end > start:
        try:
            data = json.loads(content[start:end + 1])
            if isinstance(data, list):
                out = {}
                for item in data:
                    if isinstance(item, dict) and "i" in item and "t" in item:
                        out[int(item["i"])] = str(item["t"])
                if out:
                    return out
        except json.JSONDecodeError:
            pass
    raise ValueError("回應裡沒有有效 JSON 陣列：" + content[:200].replace("\n", " "))


def call_llm(messages, settings: Settings, key_index: int) -> str:
    keys = [k.strip() for k in re.split(r"[,;\n]", settings.api_keys) if k.strip()]
    if not keys:
        raise RuntimeError("尚未填 API key")
    if not settings.model.strip():
        raise RuntimeError("尚未填模型名稱")
    api_key = keys[key_index % len(keys)]
    payload = {
        "model": settings.model.strip(),
        "messages": messages,
        "temperature": float(settings.temperature),
        "max_tokens": int(settings.max_tokens),
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        settings.base_url.strip(),
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=int(settings.timeout), context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"HTTP {e.code}：{detail}") from e
    data = json.loads(raw)
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content.strip():
        for k in ("reasoning_content", "reasoning", "output"):
            if isinstance(msg.get(k), str) and msg[k].strip():
                content = msg[k]
                break
    return content


_opencc = None
_opencc_failed = False


def to_traditional(text: str, enabled: bool) -> str:
    global _opencc, _opencc_failed
    if not enabled or not text:
        return text
    if _opencc_failed:
        return text
    if _opencc is None:
        try:
            import opencc
            try:
                _opencc = opencc.OpenCC("s2twp")
            except Exception:
                _opencc = opencc.OpenCC("s2twp.json")
        except Exception:
            _opencc_failed = True
            return text
    return _opencc.convert(text)


def translate_batch(batch, settings, glossary, key_index):
    messages = build_messages(batch, settings, glossary)
    last_err = None
    for attempt in range(2):
        try:
            content = call_llm(messages, settings, key_index + attempt)
            parsed = parse_llm_json(content)
            result = {}
            for n, it in enumerate(batch):
                if n not in parsed:
                    raise ValueError(f"缺少編號 {n}")
                text = to_traditional(parsed[n].strip(), settings.opencc)
                if not text or text == it.text:
                    continue
                result[it.key] = text
            return result
        except Exception as e:
            last_err = e
            time.sleep(1.2)
    raise last_err


def run_translation(settings: Settings, log, stop_event):
    repo = Path(settings.repo_root).resolve()
    if not (repo / "content" / "default").is_dir():
        raise FileNotFoundError(f"找不到 content/default：{repo}")

    items = extract_all(settings)
    cache = load_cache(cache_path(settings))
    todo = [it for it in items if it.key not in cache]
    glossary = load_glossary(settings.glossary) if settings.use_glossary else {}

    if settings.opencc:
        probe = to_traditional("软件", True)
        if probe == "软件":
            log("沒有裝 OpenCC，簡體不會自動轉繁體。可執行：py -3 -m pip install opencc-python-reimplemented")
        else:
            log("OpenCC 簡轉繁已啟用")

    log(f"掃描到 {len(items)} 句，快取已有 {len(cache)}，待翻 {len(todo)}")
    if not todo:
        log("沒有待翻條目。")
        apply_translations(settings, cache, items, log)
        return

    done = 0
    key_index = 0
    batch_size = max(1, int(settings.batch_size))
    for start in range(0, len(todo), batch_size):
        if stop_event.is_set():
            log("已停止。進度已寫入快取。")
            break
        batch = todo[start:start + batch_size]
        log(f"送出第 {start+1}–{start+len(batch)} 句")
        try:
            translated = translate_batch(batch, settings, glossary, key_index)
        except Exception as e:
            log(f"  這批失敗，改一句一句重試：{e}")
            translated = {}
            for one in batch:
                if stop_event.is_set():
                    break
                try:
                    translated.update(translate_batch([one], settings, glossary, key_index))
                except Exception as one_err:
                    log(f"  跳過：{one.text[:40]}（{one_err}）")
        key_index += 1
        for it in batch:
            if it.key in translated:
                cache[it.key] = translated[it.key]
                done += 1
        save_cache(cache_path(settings), cache)
        log(f"  已寫入快取，本輪完成 {len(batch)}，累計新譯 {done}")

    apply_translations(settings, cache, items, log)
    log(f"結束。新譯 {done} 句，快取共 {len(cache)} 條。")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    settings = Settings.load(config_path())
    root = tk.Tk()
    root.title(APP_NAME)
    root.geometry("900x680")
    try:
        root.option_add("*Font", ("Microsoft JhengHei UI", 10))
    except Exception:
        pass

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(1, weight=1)

    fields = {}

    def row(r, label, key, browse=None):
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=3)
        var = tk.StringVar(value=str(getattr(settings, key)))
        fields[key] = var
        ttk.Entry(frm, textvariable=var).grid(row=r, column=1, sticky="ew", pady=3)
        if browse:
            def pick(k=key, kind=browse):
                if kind == "dir":
                    p = filedialog.askdirectory()
                else:
                    p = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("全部", "*.*")])
                if p:
                    fields[k].set(p)
            ttk.Button(frm, text="瀏覽", command=pick).grid(row=r, column=2, padx=4)

    row(0, "遊戲根目錄（含 content/）", "repo_root", "dir")
    row(1, "語言 id（資料夾名）", "lang_id")
    row(2, "語言顯示名稱", "lang_name")
    row(3, "API 位址", "base_url")
    row(4, "API key（多把用逗號）", "api_keys")
    row(5, "模型", "model")
    row(6, "目標語言描述", "target_lang")
    row(7, "詞彙表(Json)", "glossary", "json")

    opt = ttk.Frame(frm)
    opt.grid(row=8, column=0, columnspan=3, sticky="ew", pady=4)
    for label, key, width in (
        ("每批句數", "batch_size", 6),
        ("溫度", "temperature", 6),
        ("逾時秒", "timeout", 6),
    ):
        ttk.Label(opt, text=label).pack(side="left")
        var = tk.StringVar(value=str(getattr(settings, key)))
        fields[key] = var
        ttk.Entry(opt, textvariable=var, width=width).pack(side="left", padx=(2, 10))

    ui_var = tk.BooleanVar(value=bool(settings.translate_ui))
    content_var = tk.BooleanVar(value=bool(settings.translate_content))
    gloss_var = tk.BooleanVar(value=bool(settings.use_glossary))
    oc_var = tk.BooleanVar(value=bool(settings.opencc))
    ttk.Checkbutton(frm, text="翻譯 UI 字串（src/shared/i18n/en）", variable=ui_var).grid(
        row=9, column=0, columnspan=3, sticky="w")
    ttk.Checkbutton(frm, text="翻譯遊戲內容（content/default）", variable=content_var).grid(
        row=10, column=0, columnspan=3, sticky="w")
    ttk.Checkbutton(frm, text="套用詞彙表", variable=gloss_var).grid(
        row=11, column=0, columnspan=3, sticky="w")
    ttk.Checkbutton(frm, text="OpenCC 簡轉繁（台灣，s2twp）", variable=oc_var).grid(
        row=12, column=0, columnspan=3, sticky="w")

    status = tk.StringVar(value="就緒")
    ttk.Label(frm, textvariable=status).grid(row=13, column=0, columnspan=3, sticky="w", pady=4)

    logbox = tk.Text(frm, height=16, wrap="word")
    logbox.grid(row=14, column=0, columnspan=3, sticky="nsew", pady=4)
    frm.rowconfigure(14, weight=1)

    log_q = queue.Queue()
    stop_event = threading.Event()
    worker = {"t": None}

    def log(msg):
        log_q.put(msg)

    def drain():
        while True:
            try:
                msg = log_q.get_nowait()
            except queue.Empty:
                break
            logbox.insert("end", msg + "\n")
            logbox.see("end")
        root.after(200, drain)

    def collect():
        s = Settings()
        s.repo_root = fields["repo_root"].get().strip() or "."
        s.lang_id = fields["lang_id"].get().strip() or "zh-TW"
        s.lang_name = fields["lang_name"].get().strip() or "繁體中文（台灣）"
        s.base_url = fields["base_url"].get().strip()
        s.api_keys = fields["api_keys"].get().strip()
        s.model = fields["model"].get().strip()
        s.target_lang = fields["target_lang"].get().strip() or s.lang_name
        s.glossary = fields["glossary"].get().strip()
        s.batch_size = int(float(fields["batch_size"].get()))
        s.temperature = float(fields["temperature"].get())
        s.timeout = int(float(fields["timeout"].get()))
        s.translate_ui = bool(ui_var.get())
        s.translate_content = bool(content_var.get())
        s.use_glossary = bool(gloss_var.get())
        s.opencc = bool(oc_var.get())
        return s

    def scan():
        try:
            s = collect()
            items = extract_all(s)
            cache = load_cache(cache_path(s))
            todo = [it for it in items if it.key not in cache]
            log(f"掃描完成：共 {len(items)}，快取 {len(cache)}，待翻 {len(todo)}")
            for it in todo[:10]:
                log(f"  [{it.kind}] {it.context}: {it.text[:50]}")
            if len(todo) > 10:
                log(f"  …其餘 {len(todo)-10} 句")
            status.set(f"待翻 {len(todo)} 句")
        except Exception as e:
            messagebox.showerror(APP_NAME, str(e))

    def start():
        if worker["t"] and worker["t"].is_alive():
            status.set("翻譯進行中")
            return
        try:
            s = collect()
        except Exception as e:
            messagebox.showerror(APP_NAME, f"設定格式不對：{e}")
            return
        s.save(config_path())
        stop_event.clear()
        status.set("翻譯中…")

        def job():
            try:
                run_translation(s, log, stop_event)
            except Exception:
                log(traceback.format_exc())
            log_q.put("__DONE__")

        worker["t"] = threading.Thread(target=job, daemon=True)
        worker["t"].start()

    def stop():
        stop_event.set()
        status.set("停止中…")

    def watch_done():
        content = logbox.get("1.0", "end")
        if "__DONE__" in content:
            logbox.delete("1.0", "end")
            logbox.insert("1.0", content.replace("__DONE__\n", "").replace("__DONE__", ""))
            status.set("完成或已停止")
        root.after(400, watch_done)

    def export_csv():
        try:
            s = collect()
            p = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
                initialfile="pending.csv",
            )
            if not p:
                return
            export_pending(s, p)
            log(f"已匯出待翻條目到 {p}")
            status.set("匯出完成")
        except Exception as e:
            messagebox.showerror(APP_NAME, str(e))

    def apply_now():
        try:
            s = collect()
            items = extract_all(s)
            cache = load_cache(cache_path(s))
            apply_translations(s, cache, items, log)
            status.set("已套用到語言資料夾")
        except Exception as e:
            messagebox.showerror(APP_NAME, str(e))

    btns = ttk.Frame(frm)
    btns.grid(row=15, column=0, columnspan=3, sticky="ew", pady=6)
    ttk.Button(btns, text="只掃描", command=scan).pack(side="left", padx=4)
    ttk.Button(btns, text="開始翻譯", command=start).pack(side="left", padx=4)
    ttk.Button(btns, text="停止", command=stop).pack(side="left", padx=4)
    ttk.Button(btns, text="匯出待翻 CSV", command=export_csv).pack(side="left", padx=4)
    ttk.Button(btns, text="套用到語言資料夾", command=apply_now).pack(side="left", padx=4)

    root.after(200, drain)
    root.after(400, watch_done)
    root.mainloop()


def export_pending(settings: Settings, out_path: str):
    """把待翻條目匯出成 CSV。"""
    import csv
    items = extract_all(settings)
    cache = load_cache(cache_path(settings))
    todo = [it for it in items if it.key not in cache]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["類型", "上下文", "原文", "快取鍵"])
        for it in todo:
            w.writerow([it.kind, it.context, it.text, it.key])
    print(f"已匯出 {len(todo)} 筆到 {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Breedhaven University LLM 翻譯")
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--export", metavar="CSV", help="把待翻條目匯出成 CSV")
    args = parser.parse_args()
    settings = Settings.load(config_path())

    if args.export:
        export_pending(settings, args.export)
        return

    if args.scan:
        items = extract_all(settings)
        cache = load_cache(cache_path(settings))
        todo = [it for it in items if it.key not in cache]
        print(f"共 {len(items)}，快取 {len(cache)}，待翻 {len(todo)}")
        for it in todo[:20]:
            print(f"{it.kind}\t{it.context}\t{it.text[:60]}")
        return

    if args.cli:
        stop = threading.Event()
        run_translation(settings, print, stop)
        settings.save(config_path())
        return

    launch_gui()


if __name__ == "__main__":
    main()
