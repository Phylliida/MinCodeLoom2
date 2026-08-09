# MinCodeLoom2

Coding Loom Chat Thingy Idk

A self-contained copy of the talkie loom chat webui (multi-model chat with
branching history, backed by SimpleBot's one-at-a-time llama-server workers).

## Run

```
pip install -r requirements.txt   # just flask
python webui.py                   # or ./run.sh
```

Then visit http://localhost:8081

- `webui.py` — Flask server (chat UI + HTTP/SSE API, bash tool sandbox,
  optional zendriver web-fetch tool via the `web` header toggle)
- `webui.html` — the browser UI, served at `/`
- `chats.jsonl` — append-only branching conversation log
- `talkie_client.py` — stdlib-only Python client / CLI for the same server
- `WEBUI.md` — full API docs
- `run.sh` + `port.txt` + `health_path.txt` — SimpleBot persistent-route glue
