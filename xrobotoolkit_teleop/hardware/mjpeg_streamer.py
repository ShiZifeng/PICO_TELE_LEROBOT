"""
Lightweight HTTP MJPEG streaming server for robot camera feeds.

Provides per-camera MJPEG streams and a simple multi-view HTML page.
No external dependencies beyond Python stdlib.

Usage:
    streamer = MJPEGStreamServer(port=8080)
    streamer.start()

    # In your main loop, push latest frames (BGR numpy arrays):
    streamer.update("left_arm", left_frame_bgr)
    streamer.update("right_arm", right_frame_bgr)
    streamer.update("head", head_frame_bgr)

    streamer.stop()

PICO headset: open http://<pc-ip>:8080 in PICO browser.
"""

import html
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger("mjpeg_streamer")

MJPEG_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SO-101 Cameras</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #111; color: #eee; font-family: monospace; }
  h1 { text-align: center; padding: 12px 0; font-size: 1.2em; color: #4af; }
  .grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; padding: 8px; }
  .camera { flex: 1 1 300px; max-width: 500px; min-width: 280px; background: #1a1a1a;
            border: 1px solid #333; border-radius: 6px; overflow: hidden; }
  .camera .label { background: #222; padding: 6px 12px; font-size: 0.9em;
                   border-bottom: 1px solid #333; }
  .camera img { width: 100%; display: block; }
  .fps { color: #8f8; font-size: 0.8em; float: right; }
</style>
</head>
<body>
<h1>🤖 SO-101 Cameras</h1>
<div class="grid">
{CAMERA_CELLS}
</div>
<script>
// Auto-reload image streams if they stop
document.querySelectorAll('img').forEach(img => {
  img.addEventListener('error', () => {
    setTimeout(() => { img.src = img.src.replace(/t=\d+/, 't=' + Date.now()); }, 2000);
  });
});
</script>
</body>
</html>"""

CAMERA_CELL_HTML = """  <div class="camera">
    <div class="label">📷 {name}<span class="fps" id="fps_{id}"></span></div>
    <img src="/stream/{id}" alt="{name}" id="img_{id}">
  </div>"""


class _MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP handler: / → index, /stream/<name> → MJPEG, /snapshot/<name> → JPEG."""

    server_version = "SO101-MJPEG/1.0"

    def log_message(self, fmt, *args):
        logger.debug("HTTP %s", fmt % args)

    def _send_index(self):
        cells = []
        for camera_id in sorted(self.server.streamer._frames.keys()):
            name = camera_id.replace("_", " ").title()
            cells.append(CAMERA_CELL_HTML.format(name=html.escape(name), id=camera_id))
        html_content = MJPEG_INDEX_HTML.replace("{CAMERA_CELLS}", "\n".join(cells) if cells else "<p>No cameras connected.</p>")
        self._send_html(html_content)

    def _send_html(self, content: str, code: int = 200):
        body = content.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _stream_mjpeg(self, camera_id: str):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--mjpeg")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        streamer = self.server.streamer
        last_seq = -1
        try:
            while streamer._running:
                frame_data, seq = streamer._frames.get(camera_id, (None, -1))
                if frame_data is None or seq == last_seq:
                    time.sleep(0.03)
                    continue
                last_seq = seq
                try:
                    self.wfile.write(b"--mjpeg\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame_data)}\r\n\r\n".encode())
                    self.wfile.write(frame_data)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
        except Exception:
            pass

    def _send_snapshot(self, camera_id: str):
        streamer = self.server.streamer
        frame_data, _ = streamer._frames.get(camera_id, (None, -1))
        if frame_data is None:
            self.send_error(404, "No frame available")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", len(frame_data))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(frame_data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send_index()
        elif path.startswith("/stream/"):
            camera_id = path[len("/stream/"):]
            self._stream_mjpeg(camera_id)
        elif path.startswith("/snapshot/"):
            camera_id = path[len("/snapshot/"):]
            self._send_snapshot(camera_id)
        else:
            self.send_error(404, "Not found")


class MJPEGStreamServer:
    """
    Lightweight MJPEG streaming HTTP server.

    Runs in a background thread. Call update() to push frames,
    start()/stop() to control lifecycle.
    """

    def __init__(self, port: int = 8080, jpeg_quality: int = 70):
        self._port = port
        self._jpeg_quality = jpeg_quality
        self._frames: Dict[str, tuple] = {}  # camera_id → (jpeg_bytes, seq)
        self._seq: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._running = False
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def update(self, camera_id: str, frame: np.ndarray):
        """Push a new frame (BGR numpy array) for a camera."""
        if frame is None:
            return
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not ok:
            return
        self.update_raw(camera_id, buf.tobytes())

    def update_raw(self, camera_id: str, jpeg_bytes: bytes):
        """Push already-encoded JPEG bytes for a camera (avoid re-encoding)."""
        if jpeg_bytes is None:
            return
        with self._lock:
            seq = self._seq.get(camera_id, 0) + 1
            self._seq[camera_id] = seq
            self._frames[camera_id] = (jpeg_bytes, seq)

    @property
    def url(self) -> str:
        import socket
        host = socket.gethostbyname(socket.gethostname())
        return f"http://{host}:{self._port}"

    def start(self):
        if self._running:
            return
        self._running = True
        self._httpd = HTTPServer(("0.0.0.0", self._port), _MJPEGHandler)
        self._httpd.streamer = self
        self._httpd.timeout = 0.5
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        logger.info("MJPEG stream server started at http://0.0.0.0:%d", self._port)

    def stop(self):
        self._running = False
        if self._httpd:
            self._httpd.shutdown()
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("MJPEG stream server stopped.")
