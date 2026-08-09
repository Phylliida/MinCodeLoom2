"""Minimal multi-model chat webui with branching history.

Talks to one of several `llama-server` instances hosted via SimpleBot.
Each model id maps to the SimpleBot proxy port for that one-at-a-time
worker, so the chosen backend is spun up on demand and torn down when
something else needs the GPU. Each conversation remembers which model
it's using via `set_model` events in chats.jsonl.

Run:  python webui.py
Then visit http://localhost:8081

Event types in chats.jsonl:
  {"type":"msg",       "id":..., "conversation_id":..., "role":..., "content":..., "parent_id":..., "ts":...}
  {"type":"select",    "conversation_id":..., "msg_id":..., "ts":...}
  {"type":"extend",    "conversation_id":..., "msg_id":..., "content_append":..., "ts":...}
  {"type":"set_model", "conversation_id":..., "model":..., "ts":...}
  {"type":"set_workdir", "conversation_id":..., "workdir":..., "ts":...}
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path
from threading import Event

from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).parent
LOG_PATH = Path(os.environ.get("TALKIE_LOG", ROOT / "chats.jsonl"))

# Talkie-specific stop strings — used only when the active model is "talkie"
# AND eos is disabled (so we can still bail on these textual markers).
TALKIE_STOP_TOKENS = ["<|end|>", "<|user|>", "<|assistant|>", "<|system|>", "<|endoftext|>"]

# Available backends. Each value is the SimpleBot proxy URL for that
# one-at-a-time worker (listen port = backend port + 1). Override any of
# them with the corresponding env var if you ever run the manager on a
# different host.
MODELS = {
    "talkie": {
        "label": "talkie-1930-13b",
        "url": os.environ.get("TALKIE_LLAMA_URL", "http://127.0.0.1:8061").rstrip("/"),
        "thinking": False,
        # Talkie-family model: emits textual stop markers (see TALKIE_STOP_TOKENS)
        # that need special handling on both /completion and /v1/chat/completions.
        "talkie": True,
    },
    "talkie-catwoman": {
        "label": "talkie-1930-catwoman",
        # Same talkie-1930-13b-it GGUF as "talkie", but the one-at-a-time worker
        # steers it with the catwoman union control vector (scale 3, layers 16-24)
        # and runs greedy (--temp 0, flash-attn off) for deterministic steering.
        # SimpleBot proxy port = worker port (3291) + 1.
        "url": os.environ.get("TALKIE_CATWOMAN_LLAMA_URL", "http://127.0.0.1:3292").rstrip("/"),
        "thinking": False,
        "talkie": True,
    },
    "gemma": {
        "label": "gemma-4-31B-it",
        "url": os.environ.get("GEMMA_LLAMA_URL", "http://127.0.0.1:8051").rstrip("/"),
        # Gemma-4-31B-it exposes reasoning via the chat template's
        # `enable_thinking` kwarg. llama-server then extracts it into
        # delta.reasoning_content for us to forward.
        "thinking": True,
        # Stock gemma-it template understands OpenAI-style function calling,
        # so tool use goes through real tool_calls/tool messages instead of
        # the textual <bash> fallback used by the talkie models.
        "native_tools": True,
    },
    "gemma-softened": {
        "label": "gemma-4-31B-softened",
        "url": os.environ.get("GEMMA_SOFTENED_LLAMA_URL", "http://127.0.0.1:8065").rstrip("/"),
        "thinking": False,
        "native_tools": True,
    },
    "gemma-reality-dream": {
        "label": "gemma-4-31B-reality-dream",
        "url": os.environ.get("GEMMA_REALITY_DREAM_LLAMA_URL", "http://127.0.0.1:8075").rstrip("/"),
        # Base gemma-4-31B-it steered with the "reality" control vector at
        # scale -4 (imagination/dream pole). It's a creative variant, so keep
        # the reasoning channel off like its gemma-softened sibling.
        "thinking": False,
        "native_tools": True,
    },
    "gemma-AM": {
        "label": "gemma-4-31B-AM",
        "url": os.environ.get("GEMMA_REALITY_DREAM_LLAMA_URL", "http://127.0.0.1:8143").rstrip("/"),
        # Base gemma-4-31B-it steered with the "reality" control vector at
        # scale -4 (imagination/dream pole). It's a creative variant, so keep
        # the reasoning channel off like its gemma-softened sibling.
        "thinking": True,
        "native_tools": True,
    },
}
DEFAULT_MODEL = "talkie"

# --- bash tool ---------------------------------------------------------------
# When the UI sends "tools": true, the model may emit <bash>...</bash> to run
# a shell command. The command runs inside a bwrap sandbox whose working
# directory is BASH_WORKDIR below (this is the knob to edit, or set
# TALKIE_BASH_WORKDIR). After the call, the server appends
# <output>...</output> to the same assistant message and lets the model
# continue, up to MAX_TOOL_CALLS per user message.
BASH_WORKDIR = os.environ.get("TALKIE_BASH_WORKDIR", str(Path.home() / "prog"))
BASH_TIMEOUT_S = int(os.environ.get("TALKIE_BASH_TIMEOUT", "120"))
BASH_MAX_OUTPUT = 4000   # chars of combined stdout+stderr fed back to the model
MAX_TOOL_CALLS = 8       # tool calls per user message, then generation stops
# PATH inside the sandbox. $HOME is not wholesale-bound (only the entries
# below), so host PATH entries under ~ would dangle; point at system paths.
BASH_PATH = "/run/current-system/sw/bin:/usr/bin:/bin"

# Bind spec: (flag, source, dest or None=same as source). Entries whose
# source doesn't exist are skipped so the sandbox still builds on machines
# that lack some of these.
BWRAP_BINDS = [
    ("--bind", "~/.elan", None),
    ("--bind", "~/.local/share/uv", None),
    ("--bind", "~/.cache", None),
    ("--bind", "~/.local/bin", None),
    ("--bind", "~/.venv", None),
    ("--bind", "~/.cargo", None),
    ("--dev-bind", "/dev", None),
    ("--bind", "~/.rustup", None),
    ("--bind", "~/.gitconfig", None),
    ("--ro-bind", "/nix", None),
    ("--ro-bind", "/run", None),
    ("--ro-bind", "/etc", None),
    ("--bind", "/tmp", None),
    ("--ro-bind", "/usr/bin/env", "/usr/bin/env"),
    ("--ro-bind", "/run/current-system/sw/bin/sh", "/bin/sh"),
]

BASH_TOOL_PROMPT = """\
You can run shell commands with a bash tool.
To run a command, write it exactly like this, with no text after the closing tag:
<bash>command goes here</bash>
The command runs in a sandbox with working directory {workdir}. The command's \
output will then appear as <output>...</output>, after which you continue your \
reply — you may make further tool calls the same way. Only one command per \
<bash> block. Never write <output> blocks yourself."""

WEB_TOOL_PROMPT = """\
You can read web pages with a web tool.
To fetch a page, write its URL exactly like this, with no text after the closing tag:
<web>https://example.com/page</web>
The server opens the page in a headless browser and the page's text then \
appears as <output>...</output>, after which you continue your reply — you \
may make further tool calls the same way. Only one URL per <web> block. \
Never write <output> blocks yourself."""


def _bash_tool_def(workdir: str) -> dict:
    """OpenAI-style function definition for models with native tool calling."""
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command in a sandboxed shell with working "
                f"directory {workdir}. Returns the command's exit code and "
                "combined stdout/stderr (truncated)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to run.",
                    },
                },
                "required": ["command"],
            },
        },
    }


def _web_tool_def() -> dict:
    """OpenAI-style function definition for the web (page fetch) tool."""
    return {
        "type": "function",
        "function": {
            "name": "web",
            "description": (
                "Fetch a web page in a headless browser and return its text "
                "content (truncated). Use this to read public web pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The http(s) URL of the page to fetch.",
                    },
                },
                "required": ["url"],
            },
        },
    }


def extract_tool_call(round_text: str, bash: bool = True, web: bool = False):
    """Pull a tool call out of a generation round.

    Returns ("bash", command), ("web", url), or None if there's no call.
    A call counts only if nothing but whitespace follows the block in this
    round — the stop list makes llama-server halt right after the closing
    tag, so any trailing prose means the model was quoting the syntax, not
    calling it. Only tags for enabled tools are considered.
    """
    best = None
    for kind, enabled in (("bash", bash), ("web", web)):
        if not enabled:
            continue
        m = re.search(rf"<{kind}>(.*?)(?:</{kind}>|$)", round_text, re.DOTALL)
        if not m or round_text[m.end():].strip():
            continue
        if best is None or m.start() < best[0]:
            best = (m.start(), kind, m.group(1).strip())
    if best and best[2]:
        return best[1], best[2]
    return None


def run_bash(cmd: str, workdir: str | None = None) -> str:
    """Run `cmd` in the bwrap sandbox; return combined output for the model."""
    wd = Path(workdir or BASH_WORKDIR).expanduser()
    if not wd.is_dir():
        return f"error: workdir {wd} does not exist"
    argv = ["bwrap"]
    home = Path.home()
    for flag, src, dst in BWRAP_BINDS:
        s = Path(src).expanduser()
        if not s.exists():
            continue
        argv += [flag, str(s), dst or str(s)]
    argv += ["--bind", str(wd), str(wd)]
    argv += [
        "--proc", "/proc",
        "--unshare-pid",
        "--setenv", "HOME", str(home),
        "--setenv", "PATH", BASH_PATH,
        "--setenv", "TERM", "dumb",
        "--chdir", str(wd),
        "--die-with-parent",
        "--new-session",
        "/bin/sh", "-c", cmd,
    ]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=BASH_TIMEOUT_S)
        out = p.stdout or ""
        if p.stderr:
            out += ("\n" if out and not out.endswith("\n") else "") + p.stderr
        out = f"[exit {p.returncode}]\n" + out
    except subprocess.TimeoutExpired:
        out = f"[timed out after {BASH_TIMEOUT_S}s]\n"
    except FileNotFoundError:
        return "error: bwrap is not installed on the host"
    if len(out) > BASH_MAX_OUTPUT:
        out = out[:BASH_MAX_OUTPUT] + f"\n[...truncated {len(out) - BASH_MAX_OUTPUT} chars]"
    return out


# --- web tool ---------------------------------------------------------------
# When the UI sends "web": true, the model may fetch a page with
# <web>url</web> (textual models) or the web() function (native tool models).
# The server fetches it in headless chromium via zendriver — same approach as
# the housing-search scrapers — and feeds the page text back as
# <output>...</output>. Runs on the host (not in the bwrap sandbox: it needs
# network and a browser), so approve mode gates it too.
WEB_TIMEOUT_S = int(os.environ.get("TALKIE_WEB_TIMEOUT", "90"))
WEB_MAX_OUTPUT = 6000   # chars of page text fed back to the model


def _find_chrome() -> str | None:
    """Hardcoded nix-store chromium, else newest chromium in the store, else PATH."""
    hardcoded = "/nix/store/kvy6drb6mr45j7vjhl6dpy13c7kb66kj-chromium-148.0.7778.167/bin/chromium"
    if os.path.exists(hardcoded):
        return hardcoded
    candidates = sorted(glob.glob("/nix/store/*-chromium-*/bin/chromium"))
    if candidates:
        return candidates[-1]
    return shutil.which("chromium") or shutil.which("google-chrome")


def run_web(url: str) -> str:
    """Fetch `url` in headless chromium via zendriver; return text for the model."""
    url = url.strip()
    if not re.match(r"https?://", url):
        return f"error: not an http(s) url: {url!r}"
    try:
        import zendriver as zd
    except ImportError:
        return "error: zendriver is not installed on the server (pip install zendriver)"
    chrome = _find_chrome()
    if not chrome:
        return "error: no chromium binary found on the server"

    async def _fetch() -> tuple[str, str]:
        browser = await zd.start(
            browser_executable_path=chrome,
            headless=True,
            browser_args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        try:
            page = await browser.get(url)
            await asyncio.sleep(5)  # let scripts settle / lazy content load
            title = await page.evaluate("document.title") or ""
            text = await page.evaluate(
                "document.body ? document.body.innerText : ''"
            ) or ""
            return str(title), str(text)
        finally:
            await browser.stop()

    async def _main() -> tuple[str, str]:
        return await asyncio.wait_for(_fetch(), WEB_TIMEOUT_S)

    try:
        title, text = asyncio.run(_main())
    except (asyncio.TimeoutError, TimeoutError):
        return f"[timed out after {WEB_TIMEOUT_S}s fetching {url}]"
    except Exception as e:
        return f"error fetching {url}: {e}"
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return (f"[{title}] {url}\n[page had no text content — it may be "
                f"blocked, empty, or require interaction]")
    out = f"[{title}] {url}\n{text}"
    if len(out) > WEB_MAX_OUTPUT:
        out = out[:WEB_MAX_OUTPUT] + f"\n[...truncated {len(out) - WEB_MAX_OUTPUT} chars]"
    return out


def _with_tool_prompt(messages: list[dict], workdir: str | None = None,
                      bash: bool = True, web: bool = False) -> list[dict]:
    """Prepend the tool instructions, merging into an existing system message."""
    parts = []
    if bash:
        parts.append(BASH_TOOL_PROMPT.format(workdir=workdir or BASH_WORKDIR))
    if web:
        parts.append(WEB_TOOL_PROMPT)
    prompt = "\n\n".join(parts)
    out = list(messages)
    if out and out[0]["role"] == "system":
        out[0] = {"role": "system", "content": out[0]["content"] + "\n\n" + prompt}
    else:
        out.insert(0, {"role": "system", "content": prompt})
    return out


# Max tokens to generate when the request doesn't specify one (the frontend
# currently never does). -1 tells llama-server to keep going until the model
# emits EOS (or, with eos disabled, until the context window fills) — i.e. no
# artificial truncation. The real ceiling is each worker's --ctx-size.
DEFAULT_MAX_NEW_TOKENS = -1

# UI-injected wrapping for reasoning content. The browser already knows how
# to render <think>...</think> as a collapsible spoiler.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove the UI's <think>...</think> wrappers from a stored message.

    Reasoning is a render artifact, not part of the conversation the model
    should see when continuing or generating a new turn. The chat template
    strips channels from prior turns natively, but only ones in the model's
    native format — not our wrapped form.
    """
    return _THINK_BLOCK_RE.sub("", text)

CANCELS: dict[str, Event] = {}
RESPONSES: dict[str, object] = {}
# request_id -> {"event": Event, "approved": bool} for pending tool approvals.
APPROVALS: dict[str, dict] = {}
# How long a tool call waits for approval before counting as rejected.
APPROVE_TIMEOUT_S = int(os.environ.get("TALKIE_APPROVE_TIMEOUT", "600"))


def _gated_run(rid: str, cancel: Event, fn) -> str:
    """Run fn() after explicit user approval (POST /approve resolves the wait).

    The caller emits the tool_propose SSE event first. Rejection, /stop, and
    timeout all come back as output text for the model rather than raising.
    """
    approval = {"event": Event(), "approved": False}
    APPROVALS[rid] = approval
    try:
        waited = 0.0
        while not approval["event"].wait(0.5):
            waited += 0.5
            if cancel.is_set():
                return "[interrupted — the call was not run]"
            if waited >= APPROVE_TIMEOUT_S:
                return "[approval timed out — the call was not run]"
        if not approval["approved"]:
            return "[rejected by user — the call was not run]"
        return fn()
    finally:
        APPROVALS.pop(rid, None)


app = Flask(__name__)


def append_event(entry: dict) -> None:
    entry.setdefault("ts", time.time())
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def conversation_model(cid: str) -> str:
    """Latest model chosen for this conversation, or the default."""
    if not LOG_PATH.exists():
        return DEFAULT_MODEL
    chosen = DEFAULT_MODEL
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "set_model" or e.get("conversation_id") != cid:
                continue
            m = e.get("model")
            if m in MODELS:
                chosen = m
    return chosen


def conversation_workdir(cid: str) -> str:
    """Latest bash workdir chosen for this conversation, or the default."""
    if not LOG_PATH.exists():
        return BASH_WORKDIR
    chosen = BASH_WORKDIR
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") != "set_workdir" or e.get("conversation_id") != cid:
                continue
            w = e.get("workdir")
            if w:
                chosen = w
    return chosen


def resolve_model(requested: str | None, cid: str) -> tuple[str, dict]:
    """Pick a valid model id given the request body and conversation state."""
    if requested and requested in MODELS:
        return requested, MODELS[requested]
    m = conversation_model(cid)
    return m, MODELS[m]


def load_conversation(cid: str) -> tuple[list[dict], list[dict]]:
    """Return (msgs, selects) for one conversation, migrating legacy rows.

    Legacy rows (no `type` field, no `id`) are assigned synthetic ids and a
    linear parent_id chain in file order so old chats remain navigable.
    """
    if not LOG_PATH.exists():
        return [], []
    msgs: list[dict] = []
    selects: list[dict] = []
    extends: dict[str, list[tuple[float, str]]] = defaultdict(list)
    legacy_last_id: str | None = None
    legacy_n = 0
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("conversation_id") != cid:
                continue
            etype = e.get("type", "msg")
            if etype == "msg":
                if "id" in e:
                    msgs.append(
                        {
                            "id": e["id"],
                            "role": e["role"],
                            "content": e["content"],
                            "parent_id": e.get("parent_id"),
                            "ts": e.get("ts", 0.0),
                            "tool": e.get("tool"),
                        }
                    )
                else:
                    nid = f"legacy-{cid[:8]}-{legacy_n}"
                    legacy_n += 1
                    msgs.append(
                        {
                            "id": nid,
                            "role": e["role"],
                            "content": e["content"],
                            "parent_id": legacy_last_id,
                            "ts": e.get("ts", 0.0),
                        }
                    )
                    legacy_last_id = nid
            elif etype == "select":
                selects.append({"msg_id": e["msg_id"], "ts": e.get("ts", 0.0)})
            elif etype == "extend":
                extends[e["msg_id"]].append(
                    (e.get("ts", 0.0), e.get("content_append", ""))
                )
    for m in msgs:
        adds = extends.get(m["id"])
        if not adds:
            continue
        adds.sort(key=lambda x: x[0])
        m["content"] = m["content"] + "".join(t for _, t in adds)
    return msgs, selects


def active_thread(msgs: list[dict], selects: list[dict]) -> list[dict]:
    """Compute the active thread from root following selected children."""
    children: dict[str | None, list[dict]] = defaultdict(list)
    for m in msgs:
        children[m["parent_id"]].append(m)
    for k in children:
        children[k].sort(key=lambda x: x["ts"])

    select_ts: dict[str, float] = {}
    for s in selects:
        ts = s["ts"]
        mid = s["msg_id"]
        if ts > select_ts.get(mid, -1):
            select_ts[mid] = ts

    def active_child(parent_id):
        kids = children.get(parent_id, [])
        if not kids:
            return None
        best = None
        best_ts = -1.0
        for k in kids:
            ts = select_ts.get(k["id"], -1.0)
            if ts > best_ts:
                best_ts = ts
                best = k
        return best if best_ts >= 0 else kids[-1]

    out = []
    cur: str | None = None
    while True:
        c = active_child(cur)
        if c is None:
            break
        sibs = children[c["parent_id"]]
        idx = next(i for i, s in enumerate(sibs) if s["id"] == c["id"])
        out.append(
            {
                "id": c["id"],
                "role": c["role"],
                "content": c["content"],
                "parent_id": c["parent_id"],
                "ts": c["ts"],
                "tool": c.get("tool"),
                "siblings": [s["id"] for s in sibs],
                "sibling_index": idx,
            }
        )
        cur = c["id"]
    return out


def thread_to_messages(msgs: list[dict], parent_id: str | None) -> list[dict]:
    """Build an OpenAI-style messages list by walking parent_id to the root.

    Prior assistant turns have their UI reasoning wrappers stripped, so the
    backend only sees the actual response text. Tool history is rebuilt into
    the native shape: a tool_call msg merges into the preceding assistant
    message as a `tool_calls` entry (or opens a fresh assistant message if
    the round had no text), and a tool msg becomes a role:"tool" result.
    """
    by_id = {m["id"]: m for m in msgs}
    path = []
    cur = parent_id
    while cur is not None:
        m = by_id.get(cur)
        if m is None:
            break
        path.append(m)
        cur = m["parent_id"]
    path.reverse()
    out = []
    for m in path:
        role = m["role"]
        if role in ("user", "system"):
            out.append({"role": role, "content": m["content"]})
        elif role == "assistant":
            out.append({"role": "assistant", "content": strip_think_blocks(m["content"])})
        elif role == "tool_call":
            name = m.get("tool") or "bash"
            arg_key = "url" if name == "web" else "command"
            call = {
                "id": m["id"],
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps({arg_key: m["content"]}),
                },
            }
            if out and out[-1]["role"] == "assistant":
                if not out[-1]["content"]:
                    out[-1]["content"] = None
                out[-1].setdefault("tool_calls", []).append(call)
            else:
                out.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m["parent_id"], "content": m["content"]})
    return out


def _common_sampling(temperature: float, max_new_tokens: int) -> dict:
    """Sampling params shared by /completion and /v1/chat/completions."""
    return {
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        # llama.cpp also accepts n_predict, but max_tokens is the OpenAI name.
        "n_predict": max_new_tokens,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
    }


def _chat_completions_request(model_url: str, messages: list[dict],
                              temperature: float, max_new_tokens: int,
                              disable_eos: bool, model_id: str,
                              tools: bool = False, web: bool = False,
                              tool_defs: list | None = None):
    """Open a streaming POST to llama-server's /v1/chat/completions endpoint."""
    body = {
        "messages": messages,
        "stream": True,
        "cache_prompt": True,
        **_common_sampling(temperature, max_new_tokens),
    }
    if MODELS[model_id].get("thinking"):
        body["chat_template_kwargs"] = {"enable_thinking": True}
    if disable_eos:
        body["ignore_eos"] = True
        if MODELS[model_id].get("talkie"):
            # Talkie's textual markers aren't real EOG tokens to all builds —
            # suppress them at the logit level too so ignore_eos actually frees
            # the model from the chat-template boundary.
            body["logit_bias"] = [[t, False] for t in TALKIE_STOP_TOKENS]
    if tool_defs:
        # Native function calling (OpenAI-style tools array).
        body["tools"] = tool_defs
    else:
        # Textual fallback: halt generation right after a tool call.
        stops = (["</bash>"] if tools else []) + (["</web>"] if web else [])
        if stops:
            body["stop"] = [*(body.get("stop") or []), *stops]

    req = urllib.request.Request(
        f"{model_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    return urllib.request.urlopen(req)


def _apply_template(model_url: str, messages: list[dict], model_id: str) -> str:
    """Ask llama-server to render `messages` through the GGUF chat template.

    Used by the continuation path so we can extend an existing assistant
    message without re-implementing each model's template in Python.
    """
    payload = {"messages": messages}
    if MODELS[model_id].get("thinking"):
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    req = urllib.request.Request(
        f"{model_url}/apply-template",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body["prompt"]


def _completion_request(model_url: str, prompt: str, temperature: float,
                        max_new_tokens: int, disable_eos: bool, model_id: str,
                        tools: bool = False, web: bool = False):
    """Streaming POST to llama-server's /completion endpoint (for continuation)."""
    body = {
        "prompt": prompt,
        "stream": True,
        "cache_prompt": True,
        **_common_sampling(temperature, max_new_tokens),
    }
    if disable_eos:
        body["ignore_eos"] = True
        if MODELS[model_id].get("talkie"):
            body["logit_bias"] = [[t, False] for t in TALKIE_STOP_TOKENS]
            body["stop"] = []
        else:
            body["stop"] = []
    elif MODELS[model_id].get("talkie"):
        # Talkie was originally trained to stop on these literals; keep that
        # behavior for raw /completion calls so a continuation still ends cleanly.
        body["stop"] = TALKIE_STOP_TOKENS
    stops = (["</bash>"] if tools else []) + (["</web>"] if web else [])
    if stops:
        # Halt generation right after a tool call so we can execute it.
        body["stop"] = [*(body.get("stop") or []), *stops]

    req = urllib.request.Request(
        f"{model_url}/completion",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    return urllib.request.urlopen(req)


def _iter_sse_payloads(response):
    """Yield decoded JSON payloads from an llama-server SSE response.

    Stops when an OpenAI-style `[DONE]` sentinel is seen, otherwise lets the
    caller decide based on each payload's contents.
    """
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        payload_str = line[5:].lstrip()
        if not payload_str:
            continue
        if payload_str == "[DONE]":
            return
        try:
            yield json.loads(payload_str)
        except json.JSONDecodeError:
            continue


def _extract_delta(payload: dict) -> tuple[str, str, bool, list]:
    """Pull the next text chunk, its kind, done flag, and any tool-call deltas.

    kind is "reasoning" (delta.reasoning_content from llama-server's thinking
    extraction) or "content" (the actual reply tokens). /completion has no
    reasoning channel, so chunks are always "content" there. tool_calls is the
    raw delta.tool_calls list (empty when absent).

    /v1/chat/completions: {"choices":[{"delta":{"content"|"reasoning_content":"...","tool_calls":[...]},"finish_reason":...}]}
    /completion:          {"content":"...","stop":bool,...}
    """
    if "choices" in payload:
        choices = payload.get("choices") or []
        if not choices:
            return "", "content", False, []
        choice = choices[0]
        delta = choice.get("delta") or {}
        done = choice.get("finish_reason") is not None
        tool_calls = delta.get("tool_calls") or []
        content_text = delta.get("content") or ""
        if content_text:
            return content_text, "content", done, tool_calls
        reasoning_text = delta.get("reasoning_content") or ""
        if reasoning_text:
            return reasoning_text, "reasoning", done, tool_calls
        return "", "content", done, tool_calls
    return payload.get("content", "") or "", "content", bool(payload.get("stop")), []


def _stream_loop(open_response, start_event: dict, persist):
    """SSE generator wrapping any llama-server streaming response.

    `open_response()` is a zero-arg callable that returns an open
    http.client.HTTPResponse-like object. `persist(full_text)` is called
    exactly once at end of stream (or on cancel) with whatever was streamed.
    """
    rid = uuid.uuid4().hex
    cancel = Event()
    CANCELS[rid] = cancel

    full = ""
    persisted = False

    def do_persist():
        nonlocal persisted
        if persisted:
            return
        persisted = True
        persist(full)

    response = None
    try:
        evt = dict(start_event)
        evt["request_id"] = rid
        yield "data: " + json.dumps(evt) + "\n\n"

        try:
            response = open_response()
        except Exception as e:
            yield "data: " + json.dumps(
                {"type": "delta", "text": f"\n[error: {e}]"}
            ) + "\n\n"
            do_persist()
            yield "data: " + json.dumps({"type": "done"}) + "\n\n"
            return

        RESPONSES[rid] = response

        current_kind = "content"

        def emit(text: str):
            nonlocal full
            full += text
            return "data: " + json.dumps({"type": "delta", "text": text}) + "\n\n"

        # If /stop closes the response mid-read, urllib raises AttributeError
        # ('NoneType' object has no attribute 'peek') from the chunked reader,
        # or IncompleteRead / OSError on socket teardown. Treat any of those
        # as a clean end-of-stream.
        try:
            for payload in _iter_sse_payloads(response):
                if cancel.is_set():
                    break
                text, kind, done, _tool_calls = _extract_delta(payload)
                if text:
                    # Inject <think>/</think> boundaries inline whenever the
                    # stream transitions between reasoning and content. The
                    # browser's spoiler state machine handles the rest.
                    if kind != current_kind:
                        if current_kind == "reasoning":
                            yield emit(THINK_CLOSE)
                        if kind == "reasoning":
                            yield emit(THINK_OPEN)
                        current_kind = kind
                    yield emit(text)
                if done:
                    break
        except (AttributeError, ValueError, OSError):
            pass

        # If the stream ended mid-think (no content followed reasoning,
        # e.g. cancel during thinking), still emit a closer so the persisted
        # text has matched tags.
        if current_kind == "reasoning":
            yield emit(THINK_CLOSE)

        do_persist()
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"
    except GeneratorExit:
        cancel.set()
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        do_persist()
        raise
    finally:
        CANCELS.pop(rid, None)
        RESPONSES.pop(rid, None)
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        do_persist()


def stream_generation(
    cid: str,
    parent_id: str,
    temperature: float,
    max_new_tokens: int,
    auto_select: bool,
    model_id: str,
    disable_eos: bool = False,
    approve: bool = False,  # accepted for call-site uniformity; no tools here
    tools: bool = False,    # ditto
    web: bool = False,      # ditto
):
    """Stream a fresh assistant message under parent_id."""
    msgs, _ = load_conversation(cid)
    messages = thread_to_messages(msgs, parent_id)
    asst_id = uuid.uuid4().hex
    model_url = MODELS[model_id]["url"]

    def persist(full):
        append_event(
            {
                "type": "msg",
                "id": asst_id,
                "conversation_id": cid,
                "role": "assistant",
                "content": full,
                "parent_id": parent_id,
                "model": model_id,
            }
        )
        if auto_select:
            append_event(
                {"type": "select", "conversation_id": cid, "msg_id": asst_id}
            )

    return _stream_loop(
        open_response=lambda: _chat_completions_request(
            model_url, messages, temperature, max_new_tokens, disable_eos, model_id
        ),
        start_event={
            "type": "start",
            "assistant_id": asst_id,
            "parent_id": parent_id,
            "model": model_id,
        },
        persist=persist,
    )


def stream_agent(
    cid: str,
    parent_id: str,
    temperature: float,
    max_new_tokens: int,
    auto_select: bool,
    model_id: str,
    disable_eos: bool = False,
    approve: bool = False,
    prelude: tuple = (),
    tools: bool = True,
    web: bool = False,
):
    """Agentic variant of stream_generation with the bash and/or web tools.

    Round 0 goes through /v1/chat/completions with the tool system prompt and
    the enabled tools' closing tags as stop strings. If the round ended in a
    tool call, we run it (run_bash / run_web), append <output>...</output> to
    the same assistant message, and continue it via /completion — exactly like
    stream_extension — until the model stops calling tools or MAX_TOOL_CALLS
    is hit.
    """
    msgs, _ = load_conversation(cid)
    workdir = conversation_workdir(cid)
    messages = _with_tool_prompt(
        thread_to_messages(msgs, parent_id), workdir, bash=tools, web=web
    )
    asst_id = uuid.uuid4().hex
    model_url = MODELS[model_id]["url"]

    rid = uuid.uuid4().hex
    cancel = Event()
    CANCELS[rid] = cancel

    full = ""
    persisted = False
    response = None

    def do_persist():
        nonlocal persisted
        if persisted:
            return
        persisted = True
        append_event(
            {
                "type": "msg",
                "id": asst_id,
                "conversation_id": cid,
                "role": "assistant",
                "content": full,
                "parent_id": parent_id,
                "model": model_id,
            }
        )
        if auto_select:
            append_event(
                {"type": "select", "conversation_id": cid, "msg_id": asst_id}
            )

    current_kind = "content"

    def emit(text: str):
        nonlocal full
        full += text
        return "data: " + json.dumps({"type": "delta", "text": text}) + "\n\n"

    try:
        yield "data: " + json.dumps(
            {
                "type": "start",
                "assistant_id": asst_id,
                "parent_id": parent_id,
                "model": model_id,
                "request_id": rid,
                "tools": True,
            }
        ) + "\n\n"

        # Messages persisted before this stream started (e.g. a re-run tool
        # result from /rerun_tool) so the browser can bubble them too.
        for ev in prelude:
            yield "data: " + json.dumps(ev) + "\n\n"

        for round_n in range(MAX_TOOL_CALLS + 1):
            if cancel.is_set():
                break
            if round_n == 0:
                open_response = lambda: _chat_completions_request(
                    model_url, messages, temperature, max_new_tokens,
                    disable_eos, model_id, tools=tools, web=web,
                )
            else:
                # Continuation round: render the template up to the assistant
                # opener, then replay what this message already contains.
                base_prompt = _apply_template(model_url, messages, model_id)
                prompt = base_prompt + strip_think_blocks(full)
                open_response = lambda p=prompt: _completion_request(
                    model_url, p, temperature, max_new_tokens,
                    disable_eos, model_id, tools=tools, web=web,
                )

            try:
                response = open_response()
            except Exception as e:
                yield emit(f"\n[error: {e}]")
                break
            RESPONSES[rid] = response

            round_text = ""

            def emit_round(text: str):
                nonlocal round_text
                round_text += text
                return emit(text)

            # Same teardown-tolerant read as _stream_loop: /stop closing the
            # socket mid-read surfaces as AttributeError/ValueError/OSError.
            try:
                for payload in _iter_sse_payloads(response):
                    if cancel.is_set():
                        break
                    text, kind, done, _tool_calls = _extract_delta(payload)
                    if text:
                        if kind != current_kind:
                            if current_kind == "reasoning":
                                yield emit_round(THINK_CLOSE)
                            if kind == "reasoning":
                                yield emit_round(THINK_OPEN)
                            current_kind = kind
                        yield emit_round(text)
                    if done:
                        break
            except (AttributeError, ValueError, OSError):
                pass
            try:
                response.close()
            except Exception:
                pass
            response = None
            RESPONSES.pop(rid, None)

            call = None if cancel.is_set() else extract_tool_call(
                round_text, bash=tools, web=web
            )
            if not call:
                break
            kind, arg = call
            run = (lambda: run_bash(arg, workdir)) if kind == "bash" \
                else (lambda: run_web(arg))
            if approve:
                yield "data: " + json.dumps(
                    {"type": "tool_propose", "content": arg}
                ) + "\n\n"
                result = _gated_run(rid, cancel, run)
            else:
                result = run()
            # The round text usually ends mid-block ("...<bash>cmd") because
            # the stop string is consumed; close it before the output.
            close_tag = f"</{kind}>"
            closer = "" if close_tag in round_text else close_tag + "\n"
            yield emit(closer + f"<output>\n{result}</output>\n")

        if current_kind == "reasoning":
            yield emit(THINK_CLOSE)

        do_persist()
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"
    except GeneratorExit:
        cancel.set()
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        do_persist()
        raise
    finally:
        CANCELS.pop(rid, None)
        RESPONSES.pop(rid, None)
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        do_persist()


def stream_agent_native(
    cid: str,
    parent_id: str,
    temperature: float,
    max_new_tokens: int,
    auto_select: bool,
    model_id: str,
    disable_eos: bool = False,
    approve: bool = False,
    prelude: tuple = (),
    tools: bool = True,
    web: bool = False,
):
    """Agentic loop for models with native function calling (native_tools).

    Every round is a plain /v1/chat/completions with the enabled tool
    definitions; when the model answers with tool_calls, the call and its
    output are persisted as their own messages (roles tool_call / tool, with a
    "tool" field recording which tool was called), appended to the OpenAI
    message list, and the loop continues — no /completion continuation needed.
    The browser learns about the intermediate messages via tool_msg SSE events
    so it can bubble them separately from the streaming reply.
    """
    msgs, _ = load_conversation(cid)
    workdir = conversation_workdir(cid)
    messages = thread_to_messages(msgs, parent_id)
    tool_defs = []
    if tools:
        tool_defs.append(_bash_tool_def(workdir))
    if web:
        tool_defs.append(_web_tool_def())
    model_url = MODELS[model_id]["url"]

    rid = uuid.uuid4().hex
    cancel = Event()
    CANCELS[rid] = cancel

    response = None
    parent = parent_id
    pending_text = ""      # current round's streamed text, not yet persisted
    persisted_final = False
    current_kind = "content"

    def persist_msg(role: str, content: str, tool: str | None = None) -> str:
        nonlocal parent
        mid = uuid.uuid4().hex
        ev = {
            "type": "msg",
            "id": mid,
            "conversation_id": cid,
            "role": role,
            "content": content,
            "parent_id": parent,
            "model": model_id,
        }
        if tool:
            ev["tool"] = tool
        append_event(ev)
        parent = mid
        return mid

    def do_persist_final():
        # The reply's final (or partial, on cancel) text, as an assistant msg.
        nonlocal persisted_final
        if persisted_final:
            return
        persisted_final = True
        mid = persist_msg("assistant", pending_text)
        if auto_select:
            append_event(
                {"type": "select", "conversation_id": cid, "msg_id": mid}
            )

    def emit(text: str):
        return "data: " + json.dumps({"type": "delta", "text": text}) + "\n\n"

    try:
        yield "data: " + json.dumps(
            {
                "type": "start",
                "parent_id": parent_id,
                "model": model_id,
                "request_id": rid,
                "tools": True,
                "native_tools": True,
            }
        ) + "\n\n"

        # Messages persisted before this stream started (e.g. a re-run tool
        # result from /rerun_tool) so the browser can bubble them too.
        for ev in prelude:
            yield "data: " + json.dumps(ev) + "\n\n"

        for round_n in range(MAX_TOOL_CALLS + 1):
            if cancel.is_set():
                break
            open_response = lambda: _chat_completions_request(
                model_url, messages, temperature, max_new_tokens,
                disable_eos, model_id, tool_defs=tool_defs,
            )
            try:
                response = open_response()
            except Exception as e:
                pending_text += f"\n[error: {e}]"
                yield emit(f"\n[error: {e}]")
                break
            RESPONSES[rid] = response

            call_id = None
            call_name = None
            call_args = ""

            # Same teardown-tolerant read as _stream_loop: /stop closing the
            # socket mid-read surfaces as AttributeError/ValueError/OSError.
            try:
                for payload in _iter_sse_payloads(response):
                    if cancel.is_set():
                        break
                    text, kind, done, tcs = _extract_delta(payload)
                    for c in tcs:
                        if c.get("id"):
                            call_id = c["id"]
                        fn = c.get("function") or {}
                        if fn.get("name"):
                            call_name = fn["name"]
                        call_args += fn.get("arguments") or ""
                    if text:
                        if kind != current_kind:
                            if current_kind == "reasoning":
                                pending_text += THINK_CLOSE
                                yield emit(THINK_CLOSE)
                            if kind == "reasoning":
                                pending_text += THINK_OPEN
                                yield emit(THINK_OPEN)
                            current_kind = kind
                        pending_text += text
                        yield emit(text)
                    if done:
                        break
            except (AttributeError, ValueError, OSError):
                pass
            try:
                response.close()
            except Exception:
                pass
            response = None
            RESPONSES.pop(rid, None)

            if cancel.is_set() or not call_args:
                break  # normal round (or interrupt) — pending_text is final

            try:
                parsed = json.loads(call_args)
            except json.JSONDecodeError:
                parsed = {}
            name = call_name or ("bash" if tools else "web")
            arg = (parsed.get("url" if name == "web" else "command") or "").strip()
            if not arg and name == "bash":
                arg = call_args.strip()
            if not arg:
                break
            enabled = (name == "bash" and tools) or (name == "web" and web)

            # Persist the round: optional text, then the call, then the result.
            if pending_text.strip():
                persist_msg("assistant", pending_text)
                pending_text = ""
            elif current_kind == "reasoning":
                # Round was pure reasoning — keep the think block with the
                # final message rather than dropping it.
                pending_text += THINK_CLOSE
                yield emit(THINK_CLOSE)
                current_kind = "content"

            tc_id = persist_msg("tool_call", arg, tool=name)
            yield "data: " + json.dumps(
                {"type": "tool_msg", "role": "tool_call", "id": tc_id,
                 "content": arg, "tool": name}
            ) + "\n\n"

            if not enabled:
                # Model called a tool that isn't enabled in this chat.
                output = f"[error: the {name} tool is not enabled in this chat]"
            else:
                run = (lambda: run_bash(arg, workdir)) if name == "bash" \
                    else (lambda: run_web(arg))
                if approve:
                    yield "data: " + json.dumps(
                        {"type": "tool_propose", "content": arg}
                    ) + "\n\n"
                    output = _gated_run(rid, cancel, run)
                else:
                    output = run()
            to_id = persist_msg("tool", output)
            yield "data: " + json.dumps(
                {"type": "tool_msg", "role": "tool", "id": to_id, "content": output}
            ) + "\n\n"

            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id or tc_id,
                            "type": "function",
                            "function": {"name": name, "arguments": call_args},
                        }
                    ],
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": call_id or tc_id, "content": output}
            )

        if current_kind == "reasoning":
            pending_text += THINK_CLOSE
            yield emit(THINK_CLOSE)

        do_persist_final()
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"
    except GeneratorExit:
        cancel.set()
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        do_persist_final()
        raise
    finally:
        CANCELS.pop(rid, None)
        RESPONSES.pop(rid, None)
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        do_persist_final()


def stream_extension(
    cid: str,
    asst_id: str,
    temperature: float,
    max_new_tokens: int,
    model_id: str,
    disable_eos: bool = False,
):
    """Continue an existing assistant message; persist as an `extend` event.

    We can't use /v1/chat/completions here because that endpoint always
    closes the assistant turn after the last message. Instead we ask
    llama-server to render the chat template up to the assistant-turn
    opener (via /apply-template), then append the existing content and
    let /completion keep writing.
    """
    msgs, _ = load_conversation(cid)
    by_id = {m["id"]: m for m in msgs}
    asst = by_id.get(asst_id)
    if asst is None or asst["role"] != "assistant":
        raise ValueError("not found or not assistant msg")
    parent_messages = thread_to_messages(msgs, asst["parent_id"])
    model_url = MODELS[model_id]["url"]
    # /apply-template defaults to add_generation_prompt=true → returned
    # prompt ends with the model's assistant-turn opener. We append the
    # existing assistant content (with UI reasoning wrappers stripped,
    # since the model's native format differs) so it picks up where the
    # actual reply left off.
    base_prompt = _apply_template(model_url, parent_messages, model_id)
    prompt = base_prompt + strip_think_blocks(asst["content"])

    def persist(full):
        if not full:
            return
        append_event(
            {
                "type": "extend",
                "conversation_id": cid,
                "msg_id": asst_id,
                "content_append": full,
                "model": model_id,
            }
        )

    return _stream_loop(
        open_response=lambda: _completion_request(
            model_url, prompt, temperature, max_new_tokens, disable_eos, model_id
        ),
        start_event={
            "type": "start",
            "assistant_id": asst_id,
            "extending": True,
            "model": model_id,
        },
        persist=persist,
    )


@app.route("/")
def index():
    return send_from_directory(ROOT, "webui.html")


@app.route("/history")
def history():
    cid = request.args.get("conversation_id", "")
    if not cid:
        return jsonify([])
    msgs, selects = load_conversation(cid)
    return jsonify(active_thread(msgs, selects))


@app.route("/conversations")
def conversations():
    if not LOG_PATH.exists():
        return jsonify([])
    seen: dict[str, dict] = {}
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type", "msg") != "msg":
                continue
            cid = e.get("conversation_id")
            if not cid:
                continue
            if cid not in seen:
                seen[cid] = {"id": cid, "first_user_msg": None, "ts": e.get("ts")}
            if seen[cid]["first_user_msg"] is None and e.get("role") == "user":
                seen[cid]["first_user_msg"] = (e.get("content") or "")[:80]
    return jsonify(sorted(seen.values(), key=lambda x: x["ts"] or 0, reverse=True))


@app.route("/select", methods=["POST"])
def select():
    data = request.get_json()
    append_event(
        {
            "type": "select",
            "conversation_id": data["conversation_id"],
            "msg_id": data["msg_id"],
        }
    )
    return ""


@app.route("/stop", methods=["POST"])
def stop():
    data = request.get_json() or {}
    rid = data.get("request_id")
    ev = CANCELS.get(rid)
    if ev:
        ev.set()
    resp = RESPONSES.get(rid)
    if resp is not None:
        try:
            resp.close()
        except Exception:
            pass
    return ""


@app.route("/approve", methods=["POST"])
def approve():
    """Resolve a pending tool_propose: {request_id, approved: bool}."""
    data = request.get_json() or {}
    a = APPROVALS.get(data.get("request_id"))
    if a:
        a["approved"] = bool(data.get("approved"))
        a["event"].set()
    return ""


@app.route("/models")
def list_models():
    """Return available model ids and human-readable labels for the UI dropdown."""
    return jsonify(
        {
            "default": DEFAULT_MODEL,
            "models": [{"id": k, "label": v["label"]} for k, v in MODELS.items()],
        }
    )


@app.route("/model", methods=["GET", "POST"])
def conversation_model_route():
    """GET ?conversation_id=...  → current model for that conversation.
    POST {conversation_id, model} → persist a set_model event.
    """
    if request.method == "GET":
        cid = request.args.get("conversation_id", "")
        if not cid:
            return jsonify({"model": DEFAULT_MODEL})
        return jsonify({"model": conversation_model(cid)})

    data = request.get_json() or {}
    cid = data.get("conversation_id")
    model = data.get("model")
    if not cid or model not in MODELS:
        return jsonify({"error": "invalid conversation_id or model"}), 400
    append_event({"type": "set_model", "conversation_id": cid, "model": model})
    return jsonify({"ok": True, "model": model})


@app.route("/workdir", methods=["GET", "POST"])
def conversation_workdir_route():
    """GET ?conversation_id=...  → bash tool workdir for that conversation.
    POST {conversation_id, workdir} → persist a set_workdir event.
    The workdir must be an existing directory; it is stored expanded+absolute.
    """
    if request.method == "GET":
        cid = request.args.get("conversation_id", "")
        if not cid:
            return jsonify({"workdir": BASH_WORKDIR})
        return jsonify({"workdir": conversation_workdir(cid)})

    data = request.get_json() or {}
    cid = data.get("conversation_id")
    wd = Path(data.get("workdir") or "").expanduser()
    if not cid or not wd.is_dir():
        return jsonify({"error": "invalid conversation_id or workdir (must be an existing directory)"}), 400
    append_event({"type": "set_workdir", "conversation_id": cid, "workdir": str(wd)})
    return jsonify({"ok": True, "workdir": str(wd)})


def _generation_fn(model_id: str, tools: bool, web: bool = False):
    """Pick the streaming generator: plain, textual-tool, or native-tool."""
    if not tools and not web:
        return stream_generation
    if MODELS[model_id].get("native_tools"):
        return stream_agent_native
    return stream_agent


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    cid = data["conversation_id"]
    parent_id = data.get("parent_id")
    user_msg = data["message"]
    temp = float(data.get("temperature", 0.7))
    mnt = int(data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    no_eos = bool(data.get("disable_eos", False))
    model_id, _ = resolve_model(data.get("model"), cid)

    user_id = uuid.uuid4().hex
    append_event(
        {
            "type": "msg",
            "id": user_id,
            "conversation_id": cid,
            "role": "user",
            "content": user_msg,
            "parent_id": parent_id,
            "model": model_id,
        }
    )

    tools = bool(data.get("tools", False))
    web = bool(data.get("web", False))
    gen_fn = _generation_fn(model_id, tools, web)
    return Response(
        gen_fn(cid, user_id, temp, mnt,
               auto_select=False, model_id=model_id, disable_eos=no_eos,
               approve=bool(data.get("approve", False)), tools=tools, web=web),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/branch_user", methods=["POST"])
def branch_user():
    """Create an alternate user message (sibling of an existing user msg) and
    generate a fresh assistant reply for it. Auto-selects the new user msg so
    it becomes the active branch."""
    data = request.get_json()
    cid = data["conversation_id"]
    orig_id = data["user_msg_id"]
    new_content = data["new_content"]
    temp = float(data.get("temperature", 0.7))
    mnt = int(data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    model_id, _ = resolve_model(data.get("model"), cid)

    msgs, _ = load_conversation(cid)
    by_id = {m["id"]: m for m in msgs}
    orig = by_id.get(orig_id)
    if orig is None or orig["role"] != "user":
        return jsonify({"error": "not found or not user msg"}), 404

    no_eos = bool(data.get("disable_eos", False))

    new_user_id = uuid.uuid4().hex
    append_event(
        {
            "type": "msg",
            "id": new_user_id,
            "conversation_id": cid,
            "role": "user",
            "content": new_content,
            "parent_id": orig["parent_id"],
            "model": model_id,
        }
    )
    append_event({"type": "select", "conversation_id": cid, "msg_id": new_user_id})

    tools = bool(data.get("tools", False))
    web = bool(data.get("web", False))
    gen_fn = _generation_fn(model_id, tools, web)
    return Response(
        gen_fn(cid, new_user_id, temp, mnt,
               auto_select=False, model_id=model_id, disable_eos=no_eos,
               approve=bool(data.get("approve", False)), tools=tools, web=web),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/edit_assistant", methods=["POST"])
def edit_assistant():
    """Create an assistant-message sibling with custom content. No generation."""
    data = request.get_json()
    cid = data["conversation_id"]
    orig_id = data["assistant_msg_id"]
    new_content = data["new_content"]

    msgs, _ = load_conversation(cid)
    by_id = {m["id"]: m for m in msgs}
    orig = by_id.get(orig_id)
    if orig is None or orig["role"] != "assistant":
        return jsonify({"error": "not found or not assistant msg"}), 404

    new_id = uuid.uuid4().hex
    append_event(
        {
            "type": "msg",
            "id": new_id,
            "conversation_id": cid,
            "role": "assistant",
            "content": new_content,
            "parent_id": orig["parent_id"],
        }
    )
    append_event({"type": "select", "conversation_id": cid, "msg_id": new_id})
    return jsonify({"id": new_id})


@app.route("/branch_tool_call", methods=["POST"])
def branch_tool_call():
    """Edit a tool_call: sibling with new command, re-run it, then regenerate
    the downstream reply. Like branch_user but for tool turns. The command is
    executed without an approval prompt — the user just wrote it themselves."""
    data = request.get_json()
    cid = data["conversation_id"]
    orig_id = data["tool_call_msg_id"]
    new_content = data["new_content"]
    temp = float(data.get("temperature", 0.7))
    mnt = int(data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    no_eos = bool(data.get("disable_eos", False))
    appr = bool(data.get("approve", False))
    model_id, _ = resolve_model(data.get("model"), cid)

    msgs, _ = load_conversation(cid)
    by_id = {m["id"]: m for m in msgs}
    orig = by_id.get(orig_id)
    if orig is None or orig["role"] != "tool_call":
        return jsonify({"error": "not found or not tool_call msg"}), 404

    tool_name = orig.get("tool") or "bash"
    new_id = uuid.uuid4().hex
    append_event(
        {
            "type": "msg",
            "id": new_id,
            "conversation_id": cid,
            "role": "tool_call",
            "content": new_content,
            "parent_id": orig["parent_id"],
            "model": model_id,
            "tool": tool_name,
        }
    )
    append_event({"type": "select", "conversation_id": cid, "msg_id": new_id})

    if tool_name == "web":
        output = run_web(new_content)
    else:
        output = run_bash(new_content, conversation_workdir(cid))
    tool_id = uuid.uuid4().hex
    append_event(
        {
            "type": "msg",
            "id": tool_id,
            "conversation_id": cid,
            "role": "tool",
            "content": output,
            "parent_id": new_id,
            "model": model_id,
        }
    )
    append_event({"type": "select", "conversation_id": cid, "msg_id": tool_id})

    return Response(
        _generation_fn(model_id, True, bool(data.get("web", False)))(
            cid, tool_id, temp, mnt, auto_select=True, model_id=model_id,
            disable_eos=no_eos, approve=appr, tools=True,
            web=bool(data.get("web", False)),
            prelude=({"type": "tool_msg", "role": "tool", "id": tool_id,
                      "content": output},),
        ),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/rerun_tool", methods=["POST"])
def rerun_tool():
    """Re-execute a tool_call into a sibling tool msg, then regenerate the
    downstream reply. Explicit user action, so no approval prompt."""
    data = request.get_json()
    cid = data["conversation_id"]
    orig_id = data["tool_call_msg_id"]
    temp = float(data.get("temperature", 0.7))
    mnt = int(data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    no_eos = bool(data.get("disable_eos", False))
    appr = bool(data.get("approve", False))
    model_id, _ = resolve_model(data.get("model"), cid)

    msgs, _ = load_conversation(cid)
    by_id = {m["id"]: m for m in msgs}
    orig = by_id.get(orig_id)
    if orig is None or orig["role"] != "tool_call":
        return jsonify({"error": "not found or not tool_call msg"}), 404

    tool_name = orig.get("tool") or "bash"
    if tool_name == "web":
        output = run_web(orig["content"])
    else:
        output = run_bash(orig["content"], conversation_workdir(cid))
    tool_id = uuid.uuid4().hex
    append_event(
        {
            "type": "msg",
            "id": tool_id,
            "conversation_id": cid,
            "role": "tool",
            "content": output,
            "parent_id": orig_id,
            "model": model_id,
        }
    )
    append_event({"type": "select", "conversation_id": cid, "msg_id": tool_id})

    return Response(
        _generation_fn(model_id, True, bool(data.get("web", False)))(
            cid, tool_id, temp, mnt, auto_select=True, model_id=model_id,
            disable_eos=no_eos, approve=appr, tools=True,
            web=bool(data.get("web", False)),
            prelude=({"type": "tool_msg", "role": "tool", "id": tool_id,
                      "content": output},),
        ),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/edit_tool", methods=["POST"])
def edit_tool():
    """Create a tool-message sibling with custom content. No execution."""
    data = request.get_json()
    cid = data["conversation_id"]
    orig_id = data["tool_msg_id"]
    new_content = data["new_content"]

    msgs, _ = load_conversation(cid)
    by_id = {m["id"]: m for m in msgs}
    orig = by_id.get(orig_id)
    if orig is None or orig["role"] != "tool":
        return jsonify({"error": "not found or not tool msg"}), 404

    new_id = uuid.uuid4().hex
    append_event(
        {
            "type": "msg",
            "id": new_id,
            "conversation_id": cid,
            "role": "tool",
            "content": new_content,
            "parent_id": orig["parent_id"],
        }
    )
    append_event({"type": "select", "conversation_id": cid, "msg_id": new_id})
    return jsonify({"id": new_id})


@app.route("/regenerate", methods=["POST"])
def regenerate():
    data = request.get_json()
    cid = data["conversation_id"]
    asst_id = data["assistant_id"]
    temp = float(data.get("temperature", 0.7))
    mnt = int(data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    model_id, _ = resolve_model(data.get("model"), cid)

    msgs, _ = load_conversation(cid)
    by_id = {m["id"]: m for m in msgs}
    asst = by_id.get(asst_id)
    if asst is None or asst["role"] != "assistant":
        return jsonify({"error": "not found or not assistant"}), 404
    parent_id = asst["parent_id"]

    no_eos = bool(data.get("disable_eos", False))

    gen_fn = _generation_fn(model_id, bool(data.get("tools", False)),
                            bool(data.get("web", False)))
    return Response(
        gen_fn(cid, parent_id, temp, mnt,
               auto_select=True, model_id=model_id, disable_eos=no_eos,
               approve=bool(data.get("approve", False)),
               tools=bool(data.get("tools", False)), web=bool(data.get("web", False))),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/continue", methods=["POST"])
def continue_msg():
    """Resume generating from where an existing assistant reply left off."""
    data = request.get_json()
    cid = data["conversation_id"]
    asst_id = data["assistant_msg_id"]
    temp = float(data.get("temperature", 0.7))
    mnt = int(data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    no_eos = bool(data.get("disable_eos", False))
    model_id, _ = resolve_model(data.get("model"), cid)

    try:
        gen = stream_extension(cid, asst_id, temp, mnt, model_id=model_id, disable_eos=no_eos)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    return Response(
        gen,
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("TALKIE_PORT", "8081")), threaded=True)
