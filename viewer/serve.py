"""Serves the viewer on localhost and opens it in Chrome. Ctrl+C to stop.

Browsers won't load JS modules from file://, so the page needs a server.

    python viewer/serve.py              # http://127.0.0.1:8000
    python viewer/serve.py --port 8080 --no-browser
"""

import argparse
import functools
import subprocess
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parent

# usual Windows install spots, if none exist it opens the default browser
CHROME_PATHS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe",
]


class ViewerFiles(SimpleHTTPRequestHandler):
    # the Windows registry can map .js to text/plain, and Chrome won't run modules served as that
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
    }

    def log_request(self, code="-", size="-"):
        pass  # don't print every file request (errors still print)


def open_in_chrome(url: str) -> None:
    chrome = next((p for p in CHROME_PATHS if p.exists()), None)
    if chrome:
        subprocess.Popen([chrome, url])
    else:
        print("Chrome not found, opening the default browser instead.")
        webbrowser.open(url)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-browser", action="store_true", help="only serve, don't open a browser"
    )
    args = parser.parse_args()

    handler = functools.partial(ViewerFiles, directory=str(VIEWER_DIR))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as e:
        print(
            f"Couldn't listen on port {args.port} ({e.strerror}), try another with --port.",
            file=sys.stderr,
        )
        return 1

    url = f"http://127.0.0.1:{args.port}/"
    print(f"Viewer at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        open_in_chrome(url)
    with server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
