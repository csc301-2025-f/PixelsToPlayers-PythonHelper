# entry point to python app
# handles url and calls other functions to start the main logic

import sys
import urllib.parse
from datetime import datetime

from pixels_to_players.webcam import WebcamClient, WebcamConfig, processors as P
from pixels_to_players.file_operations import FileManager, Logger
from pixels_to_players.firebase import FirebaseClient
from pixels_to_players.screen_recording import ScreenRecorder, ScreenRecordingConfig

PROTOCOL = "PixelsToPlayers"


def handle(path: str, qs: dict):
    # TODO: main logic, i.e. showing window for recording and running background processes here.
    # webpage should send the current user session to the client somehow

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

    # If running with a console (e.g., during dev), print something
    if sys.stdout and sys.stdout.isatty():
        if error:
            print("Error:", error)
        else:
            print(result or "")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
