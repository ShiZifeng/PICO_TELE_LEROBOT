"""
Real-time joint angle + safety clip visualization for XRIKController.

Usage:
    viz = DebugIKVisualizer(joint_names, save_path="/tmp/ik_debug.npz")
    viz.start()
    # In IK loop:
    viz.update(snapshot)  # snapshot from ctrl.target_debug_snapshot()
    viz.stop()            # auto-saves to save_path

Output .npz contains:
    times: wall-clock seconds array
    raw:    (n_joints, n_ticks) raw IK target degrees
    filt:   (n_joints, n_ticks) filtered target degrees
    pub:    (n_joints, n_ticks) published (clipped) target degrees
    obs:    (n_joints, n_ticks) observed joint degrees
    safety: (n_ticks,) bool safety-pause flags
    joint_names: list of joint name strings
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("ik_viz")


class DebugIKVisualizer:
    """Real-time plot + auto-save IK debug data to .npz."""

    def __init__(
        self,
        joint_names: List[str],
        history_s: float = 5.0,
        update_hz: float = 10.0,
        save_path: Optional[str] = None,
    ):
        self._joint_names = list(joint_names)
        self._n_joints = len(joint_names)
        self._history_s = history_s
        self._update_hz = update_hz
        self._save_path = save_path

        self._lock = threading.Lock()
        self._latest: Optional[dict] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Full recording (append-only, saved on stop)
        self._rec_times: List[float] = []
        self._rec_raw: List[List[float]] = [[] for _ in joint_names]
        self._rec_filt: List[List[float]] = [[] for _ in joint_names]
        self._rec_pub: List[List[float]] = [[] for _ in joint_names]
        self._rec_obs: List[List[float]] = [[] for _ in joint_names]
        self._rec_safety: List[bool] = []

    def update(self, snapshot: dict):
        """Push a new snapshot from the IK loop (thread-safe). Also records to buffer."""
        with self._lock:
            self._latest = snapshot
            self._record_unlocked(snapshot)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._plot_loop, daemon=True, name="ik-viz")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._save()

    def _save(self):
        """Save recorded data to .npz and metadata .json."""
        if not self._rec_times or self._save_path is None:
            return

        base = os.path.splitext(self._save_path)[0]
        npz_path = f"{base}.npz"
        json_path = f"{base}.json"

        with self._lock:
            raw = np.array(self._rec_raw, dtype=np.float32)
            filt = np.array(self._rec_filt, dtype=np.float32)
            pub = np.array(self._rec_pub, dtype=np.float32)
            obs = np.array(self._rec_obs, dtype=np.float32)
            safety = np.array(self._rec_safety, dtype=bool)
            times = np.array(self._rec_times, dtype=np.float64)

        np.savez_compressed(npz_path, times=times, raw=raw, filt=filt, pub=pub,
                            obs=obs, safety=safety)

        with open(json_path, "w") as f:
            json.dump({
                "joint_names": self._joint_names,
                "n_ticks": len(times),
                "duration_s": round(float(times[-1] - times[0]), 3) if len(times) > 1 else 0,
                "saved_at": datetime.now().isoformat(),
            }, f, indent=2)

        logger.info("IK debug data saved: %s (%d ticks, %.1fs)",
                     npz_path, len(times), times[-1] - times[0] if len(times) > 1 else 0)

    # ── Plot loop ───────────────────────────────────────────────

    def _plot_loop(self):
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
        except Exception as e:
            logger.warning("matplotlib not available for IK viz: %s", e)
            # Still record data even if no display
            while self._running:
                time.sleep(1 / self._update_hz)
            return

        cols = min(3, self._n_joints)
        rows = (self._n_joints + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)
        fig.suptitle("IK Joint Targets — raw | filtered | published | observed", fontsize=12)
        axes = axes.flatten()

        lines = {}
        for i, jn in enumerate(self._joint_names):
            ax = axes[i]
            ax.set_title(jn, fontsize=9)
            ax.set_ylabel("deg")
            ax.grid(True, alpha=0.3)
            lines[jn] = {
                "raw": ax.plot([], [], "r-", alpha=0.4, lw=1, label="raw")[0],
                "filt": ax.plot([], [], "b-", alpha=0.6, lw=1, label="filt")[0],
                "pub": ax.plot([], [], "g-", lw=1.5, label="pub")[0],
                "obs": ax.plot([], [], "k--", alpha=0.5, lw=1, label="obs")[0],
            }
        for i in range(self._n_joints, len(axes)):
            axes[i].set_visible(False)

        safety_ax = fig.add_axes([0.92, 0.02, 0.06, 0.04])
        safety_ax.set_xticks([])
        safety_ax.set_yticks([])
        safety_patch = safety_ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="green"))
        safety_text = safety_ax.text(0.5, 0.5, "SAFE", ha="center", va="center", fontsize=7, fontweight="bold")
        axes[0].legend(loc="upper right", fontsize=6, ncol=4)
        fig.tight_layout(rect=[0, 0, 0.90, 0.95])

        def _animate(_frame):
            t_now = time.perf_counter()

            with self._lock:
                if len(self._rec_times) < 2:
                    return []

                cutoff = t_now - self._history_s
                start = 0
                for i, rt in enumerate(self._rec_times):
                    if rt >= cutoff:
                        start = i
                        break

                rel_times = [rt - t_now for rt in self._rec_times[start:]]
                raw_window = [list(v[start:]) for v in self._rec_raw]
                filt_window = [list(v[start:]) for v in self._rec_filt]
                pub_window = [list(v[start:]) for v in self._rec_pub]
                obs_window = [list(v[start:]) for v in self._rec_obs]
                safety_paused = bool(self._rec_safety and self._rec_safety[-1])

            artists = []

            for ji, jn in enumerate(self._joint_names):
                for key, window in [("raw", raw_window), ("filt", filt_window),
                                    ("pub", pub_window), ("obs", obs_window)]:
                    y = window[ji]
                    n = min(len(rel_times), len(y))
                    lines[jn][key].set_data(rel_times[:n], y[:n])
                    artists.append(lines[jn][key])
                ax = lines[jn]["raw"].axes
                ax.relim()
                ax.autoscale_view()

            if safety_paused:
                safety_patch.set_facecolor("red")
                safety_text.set_text("PAUSE")
            else:
                safety_patch.set_facecolor("green")
                safety_text.set_text("SAFE")
            artists.extend([safety_patch, safety_text])
            return artists

        interval_ms = int(1000 / self._update_hz)
        _ = FuncAnimation(fig, _animate, interval=interval_ms, blit=True, cache_frame_data=False)
        plt.show(block=True)
        self._running = False

    def _record(self, snap: dict):
        """Append snapshot to recording buffers (called from IK loop)."""
        with self._lock:
            self._record_unlocked(snap)

    def _record_unlocked(self, snap: dict):
        """Append snapshot to recording buffers. Caller must hold self._lock."""
        if snap is None:
            return
        t = time.perf_counter()
        self._rec_times.append(t)
        raw = snap.get("raw_target_deg", {})
        filt = snap.get("filtered_target_deg", {})
        pub = snap.get("published_target_deg", {})
        obs = snap.get("observed_deg", {})

        for ji, jn in enumerate(self._joint_names):
            self._rec_raw[ji].append(raw.get(jn, np.nan))
            self._rec_filt[ji].append(filt.get(jn, np.nan))
            self._rec_pub[ji].append(pub.get(jn, np.nan))
            self._rec_obs[ji].append(obs.get(jn, np.nan))
        self._rec_safety.append(snap.get("safety_paused", False))
