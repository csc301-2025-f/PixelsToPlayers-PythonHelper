# entry point to python app
# handles url and calls other functions to start the main logic

import sys
import os
import json
import urllib.parse
from pathlib import Path
from datetime import datetime

APP_NAME = "PixelsToPlayers"
PROTOCOL = "PixelsToPlayers"

def log_event(data: dict):
    try:
        log_dir = Path(os.getenv("LOCALAPPDATA", ".")) / APP_NAME
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "last_invocation.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass

def handle(path : str, qs : dict):
    # TODO: main logic, i.e. showing window for recording and running background processes here

    pass

def main():
    # Only act if we were launched with a URL-like first arg
    if len(sys.argv) < 2 or "://" not in sys.argv[1]:
        # No URL → exit quietly
        return 0

    url = sys.argv[1]
    parsed = urllib.parse.urlparse(url)

    # Basic sanity: scheme check
    if parsed.scheme.lower() != PROTOCOL:
        # Not our protocol → ignore politely
        return 0

    # Path & query
    path = (parsed.path or "").lstrip("/")
    qs = urllib.parse.parse_qs(parsed.query)

    # TODO: this is how the webapp will start the python client and with args
    # url in format PixelsToPlayers://path?key1=value1&key2=value2&key2=value3
    # path = "path"
    # qs = {"key1" : ["value1], "key2" : ["value2", "value3"]}

    result = None
    status = "ok"
    error = None

    try:
        handle(path, qs)
    except Exception as e:
        status = "error"
        error = repr(e)

    # Minimal diagnostic log (useful for --windowed builds)
    log_event({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "url": url,
        "scheme": parsed.scheme,
        "path": path,
        "query": qs,
        "status": status,
        "result": result,
        "error": error,
    })

    # If running with a console (e.g., during dev), print something
    if sys.stdout and sys.stdout.isatty():
        if error:
            print("Error:", error)
        else:
            print(result or "")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
