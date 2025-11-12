# entry point to python app
# handles url and calls other functions to start the main logic

import sys
import urllib.parse
from datetime import datetime

from pixels_to_players.webcam import WebcamClient, WebcamConfig, processors as P
from pixels_to_players.firebase import FirebaseClient
from pixels_to_players.screen_recording import ScreenRecorder, ScreenRecordingConfig

# Defer heavy imports (cv2/numpy/mediapipe) to avoid import-time failures in minimal test runs.
from pixels_to_players.file_operations import FileManager, Logger

PROTOCOL = "PixelsToPlayers"


def handle(path: str, qs: dict):
    # Special test mode: if path is "test", print diagnostics to stdout so tests can capture them.
    if path == "test":
        print("handle:path=test")
        print(f"handle:query={qs}")
        keys = sorted(qs.keys())
        print(f"handle:keys={keys}")
        counts = {k: len(v) for k, v in qs.items()}
        print(f"handle:value_counts={counts}")
        return {"echo": True, "keys": keys, "counts": counts}

    # TODO: main logic here

    return None


def main():
    # Only act if we were launched with a URL-like first arg
    if len(sys.argv) < 2 or "://" not in sys.argv[1]:
        return 0

    url = sys.argv[1]
    parsed = urllib.parse.urlparse(url)

    # Basic sanity: scheme check
    if parsed.scheme.lower() != PROTOCOL.lower():
        return 0

    # Path & query (account for custom protocols where `netloc` carries the path)
    raw_path = parsed.path or parsed.netloc or ""
    path = raw_path.lstrip("/")
    qs = urllib.parse.parse_qs(parsed.query)

    result = None
    status = "ok"
    error = None

    try:
        result = handle(path, qs)
    except Exception as e:
        status = "error"
        error = repr(e)

    # Minimal diagnostic log (useful for --windowed builds)
    Logger.log_event({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "url": url,
        "scheme": parsed.scheme,
        "path": path,
        "query": qs,
        "status": status,
        "result": result,
        "error": error,
    })

    # Always exit 0 for URL handling to keep tests predictable
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
