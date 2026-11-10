#!/usr/bin/env python3
"""freellm-chat: a tiny terminal chat client for a local FreeLLMAPI router.

Only the Python standard library is used. Works with any OpenAI-compatible
endpoint that supports /chat/completions with streaming.
"""
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:3001/v1"
HELP = """Commands:
  /help          show this help
  /reset         clear the conversation history
  /model NAME    switch model (e.g. auto, auto:fast, auto:smart)
  /exit          quit (Ctrl+D also works)"""


def load_config():
    cfg = {"base_url": DEFAULT_URL, "api_key": "", "model": "auto"}
    path = os.environ.get("FREELLM_CONFIG", "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    cfg["base_url"] = os.environ.get("FREELLM_BASE_URL", cfg["base_url"]).rstrip("/")
    cfg["api_key"] = os.environ.get("FREELLM_API_KEY", cfg["api_key"])
    cfg["model"] = os.environ.get("FREELLM_MODEL", cfg["model"])
    return cfg


def stream_chat(cfg, messages, info):
    """Yield text chunks. Fills info['routed'] with the X-Routed-Via header."""
    body = json.dumps(
        {"model": cfg["model"], "messages": messages, "stream": True}
    ).encode("utf-8")
    req = urllib.request.Request(
        cfg["base_url"] + "/chat/completions",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + cfg["api_key"],
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        info["routed"] = resp.headers.get("X-Routed-Via")
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            text = (choices[0].get("delta") or {}).get("content")
            if text:
                yield text


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    cfg = load_config()
    if not cfg["api_key"]:
        print("No API key found. Copy config.example.json to config.json and "
              "fill in your unified FreeLLMAPI key, or set FREELLM_API_KEY.")
        return 1

    print("freellm-chat -> %s (model: %s)" % (cfg["base_url"], cfg["model"]))
    print("Type /help for commands.\n")
    messages = []

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/help":
            print(HELP)
            continue
        if user == "/reset":
            messages = []
            print("History cleared.")
            continue
        if user.startswith("/model"):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                cfg["model"] = parts[1].strip()
                print("Model set to %s" % cfg["model"])
            else:
                print("Current model: %s" % cfg["model"])
            continue

        messages.append({"role": "user", "content": user})
        info = {}
        reply = []
        try:
            print("ai> ", end="", flush=True)
            for text in stream_chat(cfg, messages, info):
                reply.append(text)
                print(text, end="", flush=True)
            print()
            if info.get("routed"):
                print("[served by %s]" % info["routed"])
        except urllib.error.HTTPError as e:
            print("\nHTTP %s: %s" % (e.code, e.read().decode("utf-8", "replace")[:300]))
            messages.pop()
            continue
        except urllib.error.URLError as e:
            print("\nCannot reach %s (%s). Is FreeLLMAPI running?" % (cfg["base_url"], e.reason))
            messages.pop()
            continue
        except KeyboardInterrupt:
            print("\n[interrupted]")

        if reply:
            messages.append({"role": "assistant", "content": "".join(reply)})
        else:
            messages.pop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
