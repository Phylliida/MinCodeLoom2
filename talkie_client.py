"""Tiny stdlib-only client for the talkie webui.

Talks to the same server as the browser (default http://localhost:8000), so
agent conversations appear in the webui sidebar and can be regenerated,
branched, or deleted from either side.

Programmatic use:

    from talkie_client import Talkie
    t = Talkie()                              # new conversation
    print(t.send("Greet me."))                # blocks, returns full reply
    for chunk in t.stream("Compose a sonnet."):
        print(chunk, end="", flush=True)
    t.regenerate()                            # re-roll the last reply
    t.history()                               # list of msgs on the active thread

    # Resume an existing conversation
    t = Talkie(conversation_id="abc123...")

CLI:

    # one-shot, prints the streamed reply
    python talkie_client.py "Greet me."

    # interactive REPL — shares the same conversation across turns
    python talkie_client.py --repl

    # resume a conversation visible in the webui sidebar
    python talkie_client.py --cid <conversation_id> --repl

    # list recent conversations
    python talkie_client.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from typing import Iterator


class Talkie:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        conversation_id: str | None = None,
    ):
        self.base = base_url.rstrip("/")
        self.conversation_id = conversation_id or uuid.uuid4().hex
        self.tip: str | None = None             # id of last msg in active thread
        self.last_assistant: str | None = None  # id of last assistant reply
        self._request_id: str | None = None     # current /stop target
        if conversation_id:
            self._sync_from_history()

    def _post(self, path: str, payload: dict):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib.request.urlopen(req)

    def _get_json(self, path: str):
        with urllib.request.urlopen(self.base + path) as r:
            return json.load(r)

    def _consume(self, resp) -> Iterator[str]:
        """Yield delta strings from an SSE response; track ids as they arrive."""
        for line in resp:
            line = line.decode("utf-8", "replace").rstrip()
            if not line.startswith("data:"):
                continue
            data = json.loads(line[5:].strip())
            t = data.get("type")
            if t == "start":
                self._request_id = data["request_id"]
                self.last_assistant = data["assistant_id"]
                self.tip = data["assistant_id"]
            elif t == "delta":
                yield data["text"]
            elif t == "done":
                self._request_id = None
                return

    def _sync_from_history(self) -> None:
        h = self.history()
        if not h:
            return
        self.tip = h[-1]["id"]
        for m in reversed(h):
            if m["role"] == "assistant":
                self.last_assistant = m["id"]
                break

    def stream(self, message: str, **gen_kwargs) -> Iterator[str]:
        """Send a user message and yield reply chunks as they stream."""
        payload = {
            "conversation_id": self.conversation_id,
            "parent_id": self.tip,
            "message": message,
            **gen_kwargs,
        }
        return self._consume(self._post("/chat", payload))

    def send(self, message: str, **gen_kwargs) -> str:
        """Blocking single-turn — returns the full reply."""
        return "".join(self.stream(message, **gen_kwargs))

    def stream_regenerate(self, **gen_kwargs) -> Iterator[str]:
        if not self.last_assistant:
            raise RuntimeError("no prior assistant reply to regenerate")
        payload = {
            "conversation_id": self.conversation_id,
            "assistant_id": self.last_assistant,
            **gen_kwargs,
        }
        return self._consume(self._post("/regenerate", payload))

    def regenerate(self, **gen_kwargs) -> str:
        return "".join(self.stream_regenerate(**gen_kwargs))

    def stream_edit_user(self, user_msg_id: str, new_content: str, **gen_kwargs) -> Iterator[str]:
        """Branch a user message into a sibling with new_content and stream the
        fresh assistant reply."""
        payload = {
            "conversation_id": self.conversation_id,
            "user_msg_id": user_msg_id,
            "new_content": new_content,
            **gen_kwargs,
        }
        return self._consume(self._post("/branch_user", payload))

    def edit_user(self, user_msg_id: str, new_content: str, **gen_kwargs) -> str:
        return "".join(self.stream_edit_user(user_msg_id, new_content, **gen_kwargs))

    def stream_continue(self, assistant_msg_id: str, **gen_kwargs) -> Iterator[str]:
        """Resume generating where an assistant reply left off. Yields appended chunks."""
        payload = {
            "conversation_id": self.conversation_id,
            "assistant_msg_id": assistant_msg_id,
            **gen_kwargs,
        }
        return self._consume(self._post("/continue", payload))

    def continue_msg(self, assistant_msg_id: str | None = None, **gen_kwargs) -> str:
        aid = assistant_msg_id or self.last_assistant
        if not aid:
            raise RuntimeError("no assistant message to continue")
        return "".join(self.stream_continue(aid, **gen_kwargs))

    def edit_assistant(self, assistant_msg_id: str, new_content: str) -> str:
        """Create an assistant-message sibling with new_content. Returns its id."""
        payload = {
            "conversation_id": self.conversation_id,
            "assistant_msg_id": assistant_msg_id,
            "new_content": new_content,
        }
        with self._post("/edit_assistant", payload) as r:
            data = json.load(r)
        new_id = data["id"]
        self.last_assistant = new_id
        self._sync_from_history()
        return new_id

    def stop(self) -> None:
        """Interrupt the in-flight generation. Safe to call from another thread."""
        rid = self._request_id
        if not rid:
            return
        try:
            self._post("/stop", {"request_id": rid}).read()
        except urllib.error.URLError:
            pass

    def select(self, msg_id: str) -> None:
        """Make `msg_id` the active sibling at its level. Updates tip."""
        self._post(
            "/select",
            {"conversation_id": self.conversation_id, "msg_id": msg_id},
        ).read()
        self._sync_from_history()

    def history(self) -> list[dict]:
        """The currently-active thread of this conversation."""
        return self._get_json(f"/history?conversation_id={self.conversation_id}")

    def conversations(self) -> list[dict]:
        return self._get_json("/conversations")


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("message", nargs="?", help="one-shot message; omit for REPL")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--cid", help="resume an existing conversation_id")
    ap.add_argument("--repl", action="store_true", help="interactive mode")
    ap.add_argument("--list", action="store_true", help="list conversations and exit")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--disable-eos", action="store_true",
                    help="ignore stop tokens; run until --max-new-tokens or /stop")
    args = ap.parse_args()

    if args.list:
        t = Talkie(args.url)
        for c in t.conversations():
            preview = (c.get("first_user_msg") or "(empty)")[:60]
            print(f"{c['id']}  {preview}")
        return

    t = Talkie(args.url, conversation_id=args.cid)
    print(f"[conversation_id={t.conversation_id}]", file=sys.stderr)

    gen = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
           "disable_eos": args.disable_eos}

    def speak(stream_iter):
        if args.no_stream:
            print("".join(stream_iter))
        else:
            for chunk in stream_iter:
                print(chunk, end="", flush=True)
            print()

    if args.message and not args.repl:
        speak(t.stream(args.message, **gen))
        return

    print(
        "REPL — Ctrl-D to quit. Commands: /regen, /continue, /history, /stop (best-effort)",
        file=sys.stderr,
    )
    if args.message:
        speak(t.stream(args.message, **gen))
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line == "/regen":
            try:
                speak(t.stream_regenerate(**gen))
            except RuntimeError as e:
                print(f"[{e}]", file=sys.stderr)
            continue
        if line == "/continue":
            try:
                speak(t.stream_continue(t.last_assistant, **gen))
            except (RuntimeError, TypeError) as e:
                print(f"[{e}]", file=sys.stderr)
            continue
        if line == "/history":
            for m in t.history():
                role = m["role"].upper()
                content = m["content"].replace("\n", " ⏎ ")
                if len(content) > 100:
                    content = content[:97] + "..."
                sib = ""
                if len(m.get("siblings", [])) > 1:
                    sib = f"  [{m['sibling_index']+1}/{len(m['siblings'])}]"
                print(f"  {role}: {content}{sib}")
            continue
        if line == "/stop":
            t.stop()
            continue
        try:
            speak(t.stream(line, **gen))
        except KeyboardInterrupt:
            t.stop()
            print("\n[stopped]", file=sys.stderr)


if __name__ == "__main__":
    _cli()
