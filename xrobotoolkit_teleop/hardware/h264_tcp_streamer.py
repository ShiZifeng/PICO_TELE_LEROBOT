"""
H.264 TCP video streamer for PICO APK Remote Vision.

Encodes camera frames to H.264 via FFmpeg and streams over TCP,
compatible with XRoboToolkit-Native-Video-Viewer built into the PICO APK.

Two-port protocol (matching XRoboToolkit-Orin-Video-Sender):
  - Port 13579 (control): PC listens, PICO sends CameraRequest binary.
    Format: [optional 4B packet_len BE][4B command_len LE][command_str][4B data_len LE][CameraRequest]
    CameraRequest: 0xCA 0xFE + version=1 + 7×int32 + camera_str + ip_str
  - Port 12345 (video):  PC connects to PICO, sends framed H.264.
    Format: [4B frame_size BE][H.264 NAL unit data]

Usage:
    streamer = H264TCPStreamer(control_port=13579, video_port=12345)
    streamer.start()

    # Push BGR frames (numpy arrays) from your camera pipeline
    streamer.update("left_arm", left_frame_bgr)
    streamer.update("right_arm", right_frame_bgr)

    streamer.stop()

PICO APK: Remote Vision → ZEDMINI → Listen
"""

import logging
import socket
import struct
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("h264_streamer")

# ── CameraRequest binary constants ─────────────────────────────
CAMERA_REQUEST_MAGIC = b"\xCA\xFE"
CAMERA_REQUEST_VERSION = 1
# int32 fields: width, height, fps, bitrate, enableMvHevc, renderMode, port

# ── Known PICO commands (from official XRoboToolkit-Orin-Video-Sender) ──

# Text protocol commands (main_zed_asio.cpp)
CMD_START_STREAM = "startrobotcamerastream"
CMD_STOP_STREAM = "stoprobotcamerastream"
CMD_LOOPTEST_PREFIX = "looptest"
CMD_MEDIA_DECODER_PREFIX = "mediadecoder"

# Additional recognized commands
CMD_OPEN_VARIANTS = {CMD_START_STREAM, "opencamera", "open_camera", "start"}
CMD_CLOSE_VARIANTS = {CMD_STOP_STREAM, "closecamera", "close_camera", "stop"}


def _read_int32_le(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _read_compact_string(data: bytes, offset: int) -> Tuple[str, int]:
    """Read a length-prefixed (1-byte len) string, return (str, new_offset)."""
    length = data[offset]
    offset += 1
    s = data[offset : offset + length].decode("utf-8", errors="replace")
    return s, offset + length


def parse_camera_request(data: bytes) -> dict:
    """Parse PICO CameraRequest binary protocol into a dict.

    Returns dict with keys: width, height, fps, bitrate, enable_mv_hevc,
    render_mode, port, camera, ip.  Returns empty dict on parse failure.
    """
    if len(data) < 10:
        logger.warning("CameraRequest too short: %d bytes", len(data))
        return {}

    offset = 0
    if data[offset : offset + 2] != CAMERA_REQUEST_MAGIC:
        logger.warning("CameraRequest bad magic: %r", data[:2])
        return {}
    offset += 2

    version = data[offset]
    offset += 1
    if version != CAMERA_REQUEST_VERSION:
        logger.warning("CameraRequest unsupported version: %d", version)
        return {}

    if offset + 28 > len(data):
        logger.warning("CameraRequest too short for int fields")
        return {}

    result = {
        "width": _read_int32_le(data, offset),
        "height": _read_int32_le(data, offset + 4),
        "fps": _read_int32_le(data, offset + 8),
        "bitrate": _read_int32_le(data, offset + 12),
        "enable_mv_hevc": _read_int32_le(data, offset + 16),
        "render_mode": _read_int32_le(data, offset + 20),
        "port": _read_int32_le(data, offset + 24),
    }
    offset += 28

    try:
        result["camera"], offset = _read_compact_string(data, offset)
        result["ip"], offset = _read_compact_string(data, offset)
    except (IndexError, UnicodeDecodeError) as e:
        logger.warning("CameraRequest string parse error: %s", e)

    return result


def parse_control_message(data: bytes) -> Tuple[str, bytes]:
    """Parse NetworkDataProtocol: [4B cmd_len LE][cmd][4B data_len LE][data].

    Returns (command_string, payload_bytes).
    """
    if len(data) < 8:
        return ("", b"")
    cmd_len = struct.unpack_from("<i", data, 0)[0]
    offset = 4
    if cmd_len < 0 or offset + cmd_len > len(data):
        return ("", b"")
    command = data[offset : offset + cmd_len].decode("utf-8", errors="replace").rstrip("\x00")
    offset += cmd_len
    if offset + 4 > len(data):
        return (command, b"")
    data_len = struct.unpack_from("<i", data, offset)[0]
    offset += 4
    if data_len < 0 or offset + data_len > len(data):
        return (command, b"")
    return command, data[offset : offset + data_len]


def _control_message_size(buf: bytearray, offset: int = 0) -> Optional[int]:
    """Return a complete inner control-message size, or None if incomplete/invalid."""
    if len(buf) - offset < 8:
        return None
    cmd_len = struct.unpack_from("<i", buf, offset)[0]
    if not (0 < cmd_len <= 256):
        return None
    data_offset = offset + 4 + cmd_len
    if data_offset + 4 > len(buf):
        return None
    data_len = struct.unpack_from("<i", buf, data_offset)[0]
    if not (0 <= data_len <= 65536):
        return None
    total = 4 + cmd_len + 4 + data_len
    if offset + total > len(buf):
        return None
    return total


# ── H.264 Annex B → framed Access Units ────────────────────────

# NAL unit type (lower 5 bits of first byte after start code)
_NAL_TYPE_SLICE_IDR = 5
_NAL_TYPE_SLICE_NON_IDR = 1
_NAL_TYPE_SEI = 6
_NAL_TYPE_SPS = 7
_NAL_TYPE_PPS = 8
_NAL_TYPE_AUD = 9

# VCL NAL types (slice data — start of a new Access Unit)
_VCL_NAL_TYPES = {_NAL_TYPE_SLICE_IDR, _NAL_TYPE_SLICE_NON_IDR}


class _AnnexBToFramed:
    """Convert H.264 Annex B byte stream to Access-Unit-framed packets.

    Each Access Unit (1 video frame) = [optional prefix NALs] + [one VCL NAL]
    Output: 4-byte big-endian length prefix + complete Annex-B AU data.

    Keep Annex-B start codes inside the payload. Android MediaCodec can use
    these SPS/PPS/IDR start codes directly, while the TCP frame length simply
    marks packet boundaries for Remote Vision.
    """

    def __init__(self):
        self._buf = bytearray()
        self._start_code = b"\x00\x00\x00\x01"
        self._current_au = bytearray()  # NALs being accumulated for current AU

    def _nal_type(self, nal_data: bytes) -> int:
        """Get NAL unit type (lower 5 bits of header byte)."""
        return nal_data[0] & 0x1F if nal_data else 0

    def feed(self, data: bytes) -> List[bytes]:
        """Feed raw Annex B data, return outer-length-framed Annex-B AUs."""
        self._buf.extend(data)
        frames = []

        while True:
            idx = self._buf.find(self._start_code)
            if idx < 0:
                break
            next_idx = self._buf.find(self._start_code, idx + 4)
            if next_idx < 0:
                break  # need more data

            nal = bytes(self._buf[idx + 4 : next_idx])
            self._buf = self._buf[next_idx:]

            if not nal:
                continue

            nal_type = self._nal_type(nal)
            nal_annexb = self._start_code + nal

            if nal_type in _VCL_NAL_TYPES:
                # VCL: complete current AU with all accumulated prefix NALs + this slice
                self._current_au.extend(nal_annexb)
                frames.append(
                    struct.pack(">I", len(self._current_au)) + bytes(self._current_au)
                )
                self._current_au.clear()
            else:
                # Non-VCL (SPS/PPS/SEI/AUD): accumulate for next AU
                self._current_au.extend(nal_annexb)

        return frames

    def flush(self) -> List[bytes]:
        """Flush any remaining NAL(s) in buffer and the last AU."""
        frames = []
        # Process trailing NAL in buffer (no following start code)
        if self._buf:
            idx = self._buf.find(self._start_code)
            if idx >= 0:
                nal = bytes(self._buf[idx + 4 :])
                if nal:
                    nal_type = self._nal_type(nal)
                    nal_annexb = self._start_code + nal
                    if nal_type in _VCL_NAL_TYPES:
                        self._current_au.extend(nal_annexb)
                        frames.append(
                            struct.pack(">I", len(self._current_au)) + bytes(self._current_au)
                        )
                        self._current_au.clear()
                    else:
                        self._current_au.extend(nal_annexb)
            self._buf.clear()
        # Flush any accumulated (non-VCL only) AU
        if self._current_au:
            frames.append(
                struct.pack(">I", len(self._current_au)) + bytes(self._current_au)
            )
            self._current_au.clear()
        return frames


# ── Main streamer class ───────────────────────────────────────


class H264TCPStreamer:
    """Encodes camera frames to H.264 and streams over TCP to PICO APK.

    Multiple camera feeds are composited into a grid matching the output
    resolution.  If PICO sends a CameraRequest, the encoder is reconfigured
    to the requested resolution/fps/bitrate before the video socket connects.
    """

    def __init__(
        self,
        control_port: int = 13579,
        video_port: int = 12345,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        bitrate: int = 8_000_000,
        preset: str = "superfast",
    ):
        self._control_port = control_port
        self._video_port = video_port
        self._width = width
        self._height = height
        self._fps = fps
        self._bitrate = bitrate
        self._preset = preset

        self._frames: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._ffmpeg_lock = threading.Lock()
        self._running = False

        self._ffmpeg: Optional[subprocess.Popen] = None
        self._control_server: Optional[socket.socket] = None
        self._pico_addr: Optional[Tuple[str, int]] = None
        self._pico_sock: Optional[socket.socket] = None
        self._pico_ready = threading.Event()
        self._pico_config: dict = {}

        self._control_thread: Optional[threading.Thread] = None
        self._feed_thread: Optional[threading.Thread] = None
        self._send_thread: Optional[threading.Thread] = None

        # Apply PICO-requested resolution (set via control channel)
        self._active_width = width
        self._active_height = height
        self._frame_updates = 0
        self._sent_packets = 0
        self._last_frame_log = 0.0
        self._last_status_log = 0.0

    # ── Public API ──────────────────────────────────────────────

    @property
    def url(self) -> str:
        host = socket.gethostbyname(socket.gethostname())
        return f"tcp://{host}:{self._control_port}"

    @property
    def connected(self) -> bool:
        return self._pico_ready.is_set()

    @property
    def pico_config(self) -> dict:
        return dict(self._pico_config)

    def update(self, camera_id: str, frame: np.ndarray):
        """Push a new BGR frame for a named camera."""
        if frame is None:
            return
        with self._lock:
            self._frames[camera_id] = frame.copy()
            self._frame_updates += 1
            updates = self._frame_updates
            camera_count = len(self._frames)
        now = time.monotonic()
        if updates == 1 or now - self._last_frame_log > 15.0:
            logger.debug(
                "H.264 input frames: %d camera(s), latest=%s shape=%s updates=%d",
                camera_count,
                camera_id,
                tuple(frame.shape),
                updates,
            )
            self._last_frame_log = now

    def start(self):
        """Launch FFmpeg, control server, and background threads."""
        if self._running:
            return
        self._running = True

        self._start_ffmpeg()
        self._start_control_server()
        self._start_threads()

        logger.info(
            "H.264 TCP streamer started — %dx%d@%dfps %dMbps",
            self._width, self._height, self._fps, self._bitrate // 1_000_000,
        )
        logger.info("  Control : tcp://0.0.0.0:%d (waiting PICO CameraRequest)", self._control_port)
        logger.info("  Video   : → PICO:%d (will connect on request)", self._video_port)

    def stop(self):
        """Shut down encoder, connections, and threads."""
        self._running = False
        self._pico_ready.clear()
        self._close_pico_connection()
        self._stop_ffmpeg()
        self._stop_control_server()
        logger.info("H.264 TCP streamer stopped.")

    # ── FFmpeg ───────────────────────────────────────────────────

    def _start_ffmpeg(self):
        self._stop_ffmpeg()
        cmd = [
            "ffmpeg",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._width}x{self._height}",
            "-r", str(self._fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", self._preset,
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-level:v", "4.0",
            "-pix_fmt", "yuv420p",
            "-b:v", str(self._bitrate),
            "-maxrate", str(self._bitrate),
            "-bufsize", str(self._bitrate * 2),
            "-g", str(self._fps),           # keyframe every 1s
            "-keyint_min", str(self._fps),
            "-refs", "1",                   # single reference frame
            "-bf", "0",
            "-x264-params", "repeat-headers=1",  # SPS/PPS with every IDR
            "-an",
            "-f", "h264",
            "pipe:1",
        ]
        with self._ffmpeg_lock:
            self._ffmpeg = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        logger.info(
            "FFmpeg H.264 encoder ready: %dx%d@%dfps %dMbps",
            self._width,
            self._height,
            self._fps,
            self._bitrate // 1_000_000,
        )

    def _apply_camera_request(self, config: dict):
        """Honor PICO requested encoder parameters when they are valid."""
        width = int(config.get("width") or self._width)
        height = int(config.get("height") or self._height)
        fps = int(config.get("fps") or self._fps)
        bitrate = int(config.get("bitrate") or self._bitrate)

        if width <= 0 or height <= 0 or fps <= 0 or bitrate <= 0:
            logger.warning("Ignoring invalid CameraRequest encoder config: %s", config)
            return

        if (width, height, fps, bitrate) == (self._width, self._height, self._fps, self._bitrate):
            return

        self._width = width
        self._height = height
        self._active_width = width
        self._active_height = height
        self._fps = fps
        self._bitrate = bitrate
        self._sent_packets = 0
        logger.info(
            "Reconfiguring H.264 encoder from PICO request: %dx%d@%dfps %dMbps",
            width,
            height,
            fps,
            bitrate // 1_000_000,
        )
        self._start_ffmpeg()

    # ── Control server (port 13579) ─────────────────────────────

    def _start_control_server(self):
        self._control_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._control_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._control_server.bind(("0.0.0.0", self._control_port))
        self._control_server.listen(1)
        self._control_server.settimeout(1.0)

    def _start_threads(self):
        self._control_thread = threading.Thread(
            target=self._control_loop, daemon=True, name="h264-ctrl"
        )
        self._feed_thread = threading.Thread(
            target=self._feed_loop, daemon=True, name="h264-feed"
        )
        self._send_thread = threading.Thread(
            target=self._send_loop, daemon=True, name="h264-send"
        )
        self._control_thread.start()
        self._feed_thread.start()
        self._send_thread.start()

    def _control_loop(self):
        """Accept PICO control connection, handle text or binary commands."""
        while self._running:
            try:
                client, addr = self._control_server.accept()
                logger.info("PICO control connected: %s:%d", addr[0], addr[1])
                pico_ip = addr[0]
                client.settimeout(1.0)

                buf = bytearray()
                while self._running:
                    try:
                        chunk = client.recv(4096)
                        if not chunk:
                            logger.info("PICO control disconnected (EOF)")
                            break
                        buf.extend(chunk)
                    except socket.timeout:
                        continue
                    except (ConnectionResetError, BrokenPipeError, OSError) as e:
                        logger.info("PICO control disconnected: %s", e)
                        break

                    # Process complete messages from buffer
                    buf = self._process_control_buffer(buf, pico_ip)

                self._safe_close(client)
                logger.info("PICO control session ended")

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error("Control accept error: %s", e)
                time.sleep(0.5)

    def _process_control_buffer(self, buf: bytearray, pico_ip: str) -> bytearray:
        """Try to parse complete messages from buf. Returns remaining bytes."""
        while len(buf) >= 4 and self._running:
            # Binary protocol, no envelope:
            #   [4B cmd_len LE][cmd][4B data_len LE][data]
            inner_size = _control_message_size(buf, 0)
            if inner_size is not None:
                msg = bytes(buf[:inner_size])
                buf = buf[inner_size:]
                command, payload = parse_control_message(msg)
                self._handle_command(command.lower(), payload, pico_ip)
                continue

            # Some XRoboToolkit builds wrap the inner message with a 4-byte
            # big-endian packet length:
            #   [4B packet_len BE][4B cmd_len LE][cmd][4B data_len LE][data]
            if len(buf) >= 8:
                wrapped_size = _control_message_size(buf, 4)
                if wrapped_size is not None:
                    total = 4 + wrapped_size
                    if len(buf) < total:
                        break
                    msg = bytes(buf[4:total])
                    buf = buf[total:]
                    command, payload = parse_control_message(msg)
                    self._handle_command(command.lower(), payload, pico_ip)
                    continue

                # If byte 4 looks like a plausible LE command length but the
                # full payload has not arrived, keep waiting for the rest.
                possible_cmd_len = struct.unpack_from("<i", buf, 4)[0]
                if 0 < possible_cmd_len <= 256:
                    data_offset = 8 + possible_cmd_len
                    if len(buf) < data_offset + 4:
                        break
                    possible_data_len = struct.unpack_from("<i", buf, data_offset)[0]
                    if 0 <= possible_data_len <= 65536 and len(buf) < data_offset + 4 + possible_data_len:
                        break

            # Text protocol: read until newline or use all available as one command.
            text = buf.decode("utf-8", errors="replace")
            nl = text.find("\n")
            if nl >= 0:
                command = text[:nl].strip()
                buf = buf[nl + 1:]
            else:
                command = text.strip()
                if len(command) > 64:
                    logger.warning("Control: large text (%dB) without newline, skipping", len(command))
                    buf.clear()
                    break
                buf.clear()

            if command:
                self._handle_command(command.lower(), b"", pico_ip)
            else:
                continue

        return buf

    def _handle_command(self, command: str, payload: bytes, pico_ip: str):
        """Handle a parsed control command (lowercased)."""
        logger.info("PICO command: %r (payload %dB)", command, len(payload))

        if command in CMD_OPEN_VARIANTS:
            self._handle_open(payload, pico_ip)
        elif command in CMD_CLOSE_VARIANTS:
            self._handle_close()
        elif command.startswith(CMD_LOOPTEST_PREFIX):
            logger.info("Control: LOOPTEST received, ignoring")
        elif command.startswith(CMD_MEDIA_DECODER_PREFIX):
            logger.info("Control: MediaDecoder info: %s", command)
        else:
            logger.info("Control: unknown command %r, treating as open", command)
            if payload:
                self._handle_open(payload, pico_ip)
            else:
                self._connect_video_to_pico(pico_ip, self._video_port)

    def _handle_open(self, payload: bytes, pico_ip: str):
        """Handle camera open command."""
        if payload and len(payload) >= 4:
            config = parse_camera_request(payload)
            if config:
                self._pico_config = config
                video_port = config.get("port", self._video_port)
                logger.info(
                    "CameraRequest: %dx%d@%dfps %dMbps → %s:%d",
                    config.get("width", 0), config.get("height", 0),
                    config.get("fps", 0), config.get("bitrate", 0) // 1_000_000,
                    pico_ip, video_port,
                )
                self._apply_camera_request(config)
                self._connect_video_to_pico(pico_ip, video_port)
                return
        # No valid binary payload — connect with defaults
        self._connect_video_to_pico(pico_ip, self._video_port)

    def _handle_close(self):
        """Handle camera close command."""
        logger.info("Control: close camera")
        self._pico_ready.clear()
        self._close_pico_connection()

    def _connect_video_to_pico(self, pico_ip: str, video_port: int):
        """Connect to PICO's video port and mark ready."""
        self._close_pico_connection()

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(5.0)
            sock.connect((pico_ip, video_port))
            sock.settimeout(None)
            self._pico_sock = sock
            self._pico_addr = (pico_ip, video_port)
            self._pico_ready.set()
            logger.info("Video connected → PICO %s:%d ✓", pico_ip, video_port)
        except Exception as e:
            logger.error("Failed to connect video to PICO %s:%d: %s", pico_ip, video_port, e)
            self._pico_ready.clear()

    # ── Feed loop ───────────────────────────────────────────────

    def _feed_loop(self):
        """Composite camera frames and feed to FFmpeg at target FPS."""
        interval = 1.0 / self._fps
        while self._running:
            t0 = time.monotonic()
            now = t0
            if now - self._last_status_log > 10.0:
                with self._lock:
                    frame_names = sorted(self._frames.keys())
                    updates = self._frame_updates
                logger.debug(
                    "H.264 status: pico=%s frames=%s frame_updates=%d sent_packets=%d",
                    "connected" if self._pico_ready.is_set() else "waiting_control",
                    frame_names or "none",
                    updates,
                    self._sent_packets,
                )
                self._last_status_log = now
            if self._pico_ready.is_set():
                grid = self._composite()
                with self._ffmpeg_lock:
                    ffmpeg = self._ffmpeg
                if grid is not None and ffmpeg and ffmpeg.stdin:
                    try:
                        ffmpeg.stdin.write(grid.tobytes())
                        ffmpeg.stdin.flush()
                    except (BrokenPipeError, OSError):
                        logger.error("FFmpeg stdin closed — restart required")
                        break
            elapsed = time.monotonic() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _composite(self) -> Optional[np.ndarray]:
        """Arrange all camera frames into a single grid at output resolution."""
        with self._lock:
            if not self._frames:
                return None
            items = list(self._frames.items())

        n = len(items)
        if n == 1:
            _, frame = items[0]
            return cv2.resize(frame, (self._width, self._height))

        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        cell_w = self._width // cols
        cell_h = self._height // rows

        grid = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        for i, (_cam_id, frame) in enumerate(items):
            r, c = divmod(i, cols)
            resized = cv2.resize(frame, (cell_w, cell_h))
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            grid[y1:y2, x1:x2] = resized
        return grid

    # ── Send loop ───────────────────────────────────────────────

    def _send_loop(self):
        """Read H.264 from FFmpeg stdout, frame it, send to PICO."""
        converter = _AnnexBToFramed()
        drain_cycles = 0

        while self._running:
            with self._ffmpeg_lock:
                ffmpeg = self._ffmpeg
            if not ffmpeg or not ffmpeg.stdout:
                time.sleep(0.01)
                continue

            if not self._pico_ready.is_set():
                drain_cycles += 1
                if drain_cycles > 50:  # ~500ms
                    converter = _AnnexBToFramed()  # reset converter
                    drain_cycles = 0
                time.sleep(0.05)
                continue

            drain_cycles = 0

            try:
                chunk = ffmpeg.stdout.read(4096)
                if not chunk:
                    time.sleep(0.005)
                    continue
            except Exception:
                time.sleep(0.01)
                continue

            # Convert Annex B → framed (4B BE length prefix)
            frames = converter.feed(chunk)
            for framed_nal in frames:
                pico_sock = self._pico_sock
                if pico_sock is None or not self._pico_ready.is_set():
                    break
                try:
                    pico_sock.sendall(framed_nal)
                    self._sent_packets += 1
                except (BrokenPipeError, ConnectionResetError, OSError):
                    logger.warning("PICO video connection lost")
                    self._pico_ready.clear()
                    self._close_pico_connection()
                    break

    # ── Cleanup ─────────────────────────────────────────────────

    def _close_pico_connection(self):
        if self._pico_sock:
            self._safe_close(self._pico_sock)
            self._pico_sock = None
            self._pico_addr = None
        self._pico_ready.clear()

    @staticmethod
    def _safe_close(sock: socket.socket):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass

    def _stop_ffmpeg(self):
        with self._ffmpeg_lock:
            ffmpeg = self._ffmpeg
            self._ffmpeg = None
        if not ffmpeg:
            return
        for pipe in (ffmpeg.stdin, ffmpeg.stdout):
            try:
                pipe.close()
            except Exception:
                pass
        try:
            ffmpeg.wait(timeout=3)
        except subprocess.TimeoutExpired:
            ffmpeg.kill()
            ffmpeg.wait(timeout=2)

    def _stop_control_server(self):
        if self._control_server:
            self._safe_close(self._control_server)
            self._control_server = None
