# talkie webui & API

A Flask server (`webui.py`) that loads the GPTQ model once and exposes:

1. A browser chat UI at `/`
2. A small HTTP/SSE API for programmatic use
3. An append-only JSONL log (`chats.jsonl`) shared by both

A stdlib-only Python client (`talkie_client.py`) is provided.

The browser UI and any clients hit the **same** server, so an agent's
conversations appear in the sidebar instantly and you can branch / regenerate /
edit them from either side.

The `png` button in the header renders the active thread to a downloadable
PNG, drawn on a canvas in the same visual style. A modal previews the bake
live with knobs for column count (newspaper-style split for long chats),
column width, render scale (1–3×), theme (current / light / dark / black —
pure grayscale, no tan), whether to include thinking blocks, and how many of
the first messages to skip (render only k–end).

## Run

```
python webui.py
```

Listens on `http://127.0.0.1:8000`. The model takes ~30 s to load; the listener
opens once it's ready. Generation runs at ~10 tok/s (no KV cache).

## HTTP API

All bodies are JSON. Streaming endpoints return Server-Sent Events.

| Endpoint | Body | Returns |
|---|---|---|
| `GET /` | — | the browser UI (HTML) |
| `GET /history?conversation_id=<id>` | — | active thread (JSON list) |
| `GET /conversations` | — | sidebar listing (JSON list) |
| `POST /chat` | `{conversation_id, parent_id, message, temperature?, max_new_tokens?}` | SSE stream |
| `POST /regenerate` | `{conversation_id, assistant_id, temperature?, max_new_tokens?}` | SSE stream |
| `POST /continue` | `{conversation_id, assistant_msg_id, temperature?, max_new_tokens?}` | SSE stream |
| `POST /branch_user` | `{conversation_id, user_msg_id, new_content, temperature?, max_new_tokens?}` | SSE stream |
| `POST /edit_assistant` | `{conversation_id, assistant_msg_id, new_content}` | `{"id":"<new_msg_id>"}` |
| `POST /branch_tool_call` | `{conversation_id, tool_call_msg_id, new_content, …}` | SSE stream (re-runs + regenerates) |
| `POST /rerun_tool` | `{conversation_id, tool_call_msg_id, …}` | SSE stream (re-runs + regenerates) |
| `POST /edit_tool` | `{conversation_id, tool_msg_id, new_content}` | `{"id":"<new_msg_id>"}` |
| `POST /approve` | `{request_id, approved}` | empty 200 |
| `POST /select` | `{conversation_id, msg_id}` | empty 200 |
| `POST /stop` | `{request_id}` | empty 200 |
| `GET/POST /model` | `?conversation_id=` / `{conversation_id, model}` | per-chat backend |
| `GET/POST /workdir` | `?conversation_id=` / `{conversation_id, workdir}` | per-chat bash workdir |
| `GET /usage` | `?conversation_id=` | token totals (all branches + active thread) + ctx size |

Edits are non-destructive: `branch_user` and `edit_assistant` create a **new sibling**
with the provided content and select it. The original is still on disk and reachable
via the ◀ button or `/select`.

`parent_id` may be `null` for the first message in a new conversation.
Defaults: `temperature=0.7`, `max_new_tokens=1200`. All four streaming endpoints
also accept `disable_eos: bool` (default `false`); when `true` the model ignores
stop tokens and runs until `max_new_tokens` or you hit `/stop`. Useful for letting
the model continue past `<|end|>` and start writing as the user, etc. Total
context is capped at `max_position_embeddings=2048`, so very long histories may
need a smaller cap or a fresh conversation.

### SSE event types

Each event is one `data: <json>` line followed by a blank line.

```
{"type":"start","request_id":"<hex>","assistant_id":"<hex>","parent_id":"<id>"}
{"type":"delta","text":"…"}
{"type":"tool_msg","role":"tool_call|tool","id":"<hex>","content":"…"}   // native tool runs only
{"type":"tool_propose","content":"<command>"}   // approve mode: server parked, POST /approve
{"type":"done","assistant_id":"<hex>"}
```

`assistant_id` is reserved on `start`, so the client knows the new message id
before the row is committed. Use `request_id` for `/stop`.

### Active-thread rule

For any parent (including the synthetic `null` root), among its children the
**active** child is whichever has the most recent `select` event; if none has
any `select`, it falls back to the most recent child by `ts`. The active
thread is built by walking from `null` and following the active child at each
step.

### Curl examples

```bash
# Start a new conversation
curl -N -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"demo","parent_id":null,"message":"Greet me."}'

# See what's there
curl 'http://127.0.0.1:8000/history?conversation_id=demo'

# Re-roll the last assistant reply (creates a sibling, auto-selects it)
ASST=$(curl -s 'http://127.0.0.1:8000/history?conversation_id=demo' \
  | python -c 'import json,sys; print(json.load(sys.stdin)[-1]["id"])')
curl -N -X POST http://127.0.0.1:8000/regenerate \
  -H 'Content-Type: application/json' \
  -d "{\"conversation_id\":\"demo\",\"assistant_id\":\"$ASST\"}"

# Swap back to the previous variant
PREV=$(curl -s 'http://127.0.0.1:8000/history?conversation_id=demo' \
  | python -c 'import json,sys; m=json.load(sys.stdin)[-1]; print(m["siblings"][0])')
curl -X POST http://127.0.0.1:8000/select \
  -H 'Content-Type: application/json' \
  -d "{\"conversation_id\":\"demo\",\"msg_id\":\"$PREV\"}"
```

## Tool calling (bash)

The `tools` header toggle (persisted per browser) makes generate requests carry
`"tools": true` (`/chat`, `/branch_user`, `/regenerate`). Two protocols, picked
per model via the `native_tools` flag in `MODELS`:

- **Native** (gemma family): the bash function is sent as an OpenAI-style
  `tools` definition. When the model answers with `tool_calls`, the call and
  its output are persisted as **separate messages** (roles `tool_call` and
  `tool`, rendered as their own collapsible monospace bubbles, labeled
  `bash`/`output`), appended to the message list as proper
  `assistant.tool_calls` + `role:"tool"` entries, and the loop continues —
  up to `MAX_TOOL_CALLS` (8) rounds. History is rebuilt into the same OpenAI
  shape on later turns (`thread_to_messages`).
- **Textual fallback** (talkie models, whose template has no tool support):
  the model is told via system prompt to write `<bash>command</bash>`;
  generation stops after the tag, the output is appended as
  `<output>...</output>` inside the same assistant message.

Either way the command runs in a **bwrap sandbox** (`run_bash`). During a
native tool run the server emits `tool_msg` SSE events so the browser can
bubble the call and result separately from the streaming reply.

**Approval mode.** The `approve` header toggle (shown when tools are on)
makes every tool call pause before execution: the server emits a
`tool_propose` SSE event and parks until the browser POSTs `/approve`
(`{request_id, approved}`). Rejection, `/stop`, or a timeout
(`APPROVE_TIMEOUT_S`, 600 s) all come back to the model as output text like
`[rejected by user — the command was not run]`, and the loop continues.

**Tool turns are first-class.** `tool_call` and `tool` messages carry action
buttons like any other turn:

- `tool_call` ✎ — edit the command: creates a sibling (`/branch_tool_call`),
  **re-runs it** (no approval prompt — you wrote it), and regenerates the
  downstream reply. ↻ — re-run the same command (`/rerun_tool`) into a
  sibling result + regenerate.
- `tool` ✎ — replace the output with custom text (`/edit_tool`, sibling, no
  execution). ↻ — re-run the parent command (`/rerun_tool`).

All variants stay on disk as siblings, so ◀▶ navigation works over tool
turns exactly like over user/assistant branches.

Config at the top of `webui.py`:

- `BASH_WORKDIR` (env `TALKIE_BASH_WORKDIR`) — the **default** sandbox working
  directory, bound read-write. Each conversation can override it: when tools
  are on, a `dir` text field appears in the header showing this chat's workdir.
  Editing it POSTs to `/workdir`, which validates the directory exists and
  appends a `set_workdir` event; the value is used for that conversation's
  tool calls and injected into its tool system prompt.
- `BWRAP_BINDS` — what's visible inside (a few tool dirs under `$HOME` rw,
  `/nix`, `/etc`, `/run` ro, `/proc`, `/tmp`; missing sources are skipped).
  `$HOME` itself is **not** wholesale-bound.
- `BASH_TIMEOUT_S` (120), `BASH_MAX_OUTPUT` (4000 chars fed back),
  `BASH_PATH` (PATH inside the sandbox).

`/continue` does not take the tool path. Caveat: the protocol is textual — if
the model writes `<bash>...</bash>` as prose with nothing after it, it *will*
be executed.

## Web tool (zendriver)

The `web` header toggle (off by default, persisted per browser) makes generate
requests carry `"web": true`. It's independent of the bash toggle — either,
both, or neither can be on.

- **Native**: a second function `web({url})` is added to the `tools` array.
- **Textual**: the system prompt also documents `<web>url</web>`, and `</web>`
  becomes a stop string alongside `</bash>`.

The server fetches the page in headless chromium via zendriver (`run_web`,
same pattern as the housing-search scrapers: `document.body.innerText` after
a short settle delay) and feeds the text back as the tool output. Unlike bash
this runs **on the host**, not in the bwrap sandbox (it needs network and a
browser) — approval mode gates web fetches the same way. `tool_call` messages
for it carry `"tool": "web"` so history rebuilds the right function call and
the bubble is labeled `web`; ✎/↻ re-run the fetch instead of a command.

Config: `WEB_TIMEOUT_S` (env `TALKIE_WEB_TIMEOUT`, 90), `WEB_MAX_OUTPUT`
(6000 chars fed back). Chromium is located by `_find_chrome()` (hardcoded
nix-store path, else newest in the store, else PATH). Needs the `zendriver`
pip package and a chromium binary on the server.

## Storage: `chats.jsonl`

Append-only. Two event types:

```jsonc
// Message — role is user|assistant|system, or tool_call|tool for native tool
// runs (tool_call content is the command, tool content is the output, and a
// tool msg's parent_id is its tool_call msg)
{"type":"msg","id":"<uuid>","conversation_id":"<id>","role":"user|assistant|system|tool_call|tool",
 "content":"…","parent_id":"<id|null>","ts":<float>}

// Sibling selection
{"type":"select","conversation_id":"<id>","msg_id":"<id>","ts":<float>}

// Per-conversation bash tool workdir (latest one wins)
{"type":"set_workdir","conversation_id":"<id>","workdir":"/abs/path","ts":<float>}

// Continue — appended to msg.content on read, in ts order
{"type":"extend","conversation_id":"<id>","msg_id":"<id>","content_append":"…","ts":<float>}
```

Legacy rows (no `type` field, no `id` — written by an earlier version) are
migrated on read into a linear parent chain with synthetic ids
`legacy-<cid8>-<n>`, so old chats remain navigable as a single thread.

Editing the file by hand is supported:

- Append a `msg` row to splice content in.
- Append a `select` row to pin a sibling.
- To "delete" a branch, point the parent's `select` at a different sibling — the
  unused branch stays on disk but drops out of the active thread.

## Python client (`talkie_client.py`)

Stdlib only — no `pip install` required.

```python
from talkie_client import Talkie

t = Talkie()                                 # new conversation
print(t.send("Greet me formally."))          # blocks, returns full reply

for chunk in t.stream("Compose a sonnet about a typewriter."):
    print(chunk, end="", flush=True)         # token-by-token

t.regenerate()                               # re-roll the last reply
t.continue_msg()                             # extend last reply (appends in place)
t.history()                                  # list of msgs on the active thread

# Edits — non-destructive, create siblings you can ◀▶ between
hist = t.history()
user_id = hist[0]["id"]
t.edit_user(user_id, "Greet me, but in French.")    # branches + regenerates
t.edit_assistant(hist[1]["id"], "Bonjour, monsieur.")  # custom assistant content
t.conversations()                            # everything in the sidebar

# Resume something visible in the webui
existing = Talkie(conversation_id="abc123…")
existing.send("And now in French.")
```

`Talkie.stop()` is safe to call from another thread to interrupt an in-flight
`stream()` / `send()`. Whatever was generated so far is persisted as the
assistant message.

`stream(...)` and `send(...)` accept `temperature=…` and `max_new_tokens=…` kwargs.

### CLI

```bash
# One-shot
python talkie_client.py "Greet me formally."

# Interactive REPL — same conversation across turns
python talkie_client.py --repl

# Resume an existing conversation by id (find ids via --list or the sidebar)
python talkie_client.py --list
python talkie_client.py --cid <conversation_id> --repl

# Sampling knobs
python talkie_client.py --temperature 0.9 --max-new-tokens 800 "…"
```

REPL commands:

- `/regen` — re-roll the previous assistant reply
- `/history` — print the active thread
- `/stop` — interrupt (best-effort; works while a reply is streaming)
- Ctrl-D — quit

## Chat template

The server inserts these tokens; the client sends raw text. For reference:

```
<|system|>{system}<|end|>
<|user|>{user 1}<|end|>
<|assistant|>{reply 1}<|end|>
<|user|>{user 2}<|end|>
<|assistant|>
```

Generation stops at any of `<|end|>`, `<|user|>`, `<|assistant|>`, `<|system|>`,
`<|endoftext|>`. The SSE stream strips them; saved `content` does not contain them.
