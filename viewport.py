"""3-D viewport: rendering, camera, perturbation, traces, overlays, selection highlight.

Uses the MuJoCo Python renderer (mujoco.Renderer) for offscreen rendering and
paints the result onto a PySide6 QWidget.  Thread-safe via an RLock shared
with SimWorker — render() acquires the lock and caches overlay data so
paintEvent() never touches MjData directly.

All MuJoCo enum access uses safe accessors from constants.py to handle
version differences gracefully.
"""

import numpy as np
from collections import deque

from PySide6.QtWidgets import QWidget, QSizePolicy, QMenu
from PySide6.QtCore import Qt, Signal, QRect, QPoint
from PySide6.QtGui import QImage, QPainter, QFont, QColor, QPen, QBrush, QLinearGradient, QPolygon

import mujoco
from constants import (
    VIS_SHADOW, VIS_CONTACTPOINT, VIS_CONTACTFORCE,
    VIS_JOINT, VIS_ACTUATOR, VIS_SELECT,
    VIS_FLAG_DEFAULTS, safe_vis_flag, safe_enum,
)
from widgets import ToastLabel, log


class MujocoViewport(QWidget):
    """Offscreen-rendered MuJoCo viewport widget (thread-safe)."""

    body_focused = Signal(int)
    nan_detected = Signal()          # NEW: emitted when NaN found in sim

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)

        # MuJoCo objects
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.renderer: mujoco.Renderer | None = None
        self.cam = mujoco.MjvCamera()
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

        # ── SAFE visualization defaults ──
        # Each flag is only set if it exists in this MuJoCo version
        for _name, _flag in VIS_FLAG_DEFAULTS:
            try:
                self.vopt.flags[_flag] = True
            except (IndexError, TypeError):
                log.debug(f"Could not set vis flag {_name}")

        # Thread lock (set by MainWindow)
        self._lock = None

        # Render state
        self._image: QImage | None = None
        self._render_w, self._render_h = 0, 0
        self._img_data_ref: bytes | None = None
        self._render_scale = 1.0

        # Cached overlay data (written in render(), read lock-free in paintEvent)
        self._cached_time_str = ""
        self._cached_has_nan = False
        self._cached_selected_pos = None
        self._cached_selected_name = ""

        # Mouse / interaction state
        self._last_pos = None
        self._active_btn = None
        self._ctrl_held = False
        self._shift_held = False
        self._perturbing = False
        self._selected_body = -1
        self._paused = False

        # Traces
        self._show_traces = False
        self._traces: dict[int, deque] = {}
        self._trace_interval = 4
        self._trace_counter = 0
        self._max_trace_len = 400

        # Overlays
        self._show_fps_overlay = True
        self._show_axis_overlay = True
        self._show_grid_overlay = False
        self._show_info_overlay = True
        self._show_sim_info = True

        # Camera follow
        self._follow_body_id = -1
        self._camera_bookmarks: dict[str, tuple] = {}

        self._highlight_body = -1
        self._step_count = 0
        self._render_error_count = 0
        self._fps = 0.0
        self._nan_notified = False  # prevent repeated signals

        self._toast = ToastLabel(self)

    # ── Lock setup ────────────────────────────────────────────

    def set_lock(self, lock):
        self._lock = lock

    def set_render_scale(self, scale: float):
        self._render_scale = scale
        self._init_renderer()
        self.render()

    # ── Model setup ───────────────────────────────────────────

    def set_model(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.pert = mujoco.MjvPerturb()
        self._traces = {}
        self._trace_counter = 0
        self._selected_body = -1
        self._follow_body_id = -1
        self._render_error_count = 0
        self._highlight_body = -1
        self._step_count = 0
        self._nan_notified = False
        try:
            if self._lock:
                self._lock.acquire()
            mujoco.mj_forward(model, data)
            log.info("mj_forward completed on new model")
        except Exception as e:
            log.error(f"mj_forward failed on new model: {e}")
        finally:
            if self._lock:
                self._lock.release()
        self._init_renderer()
        self.reset_camera()
        self.render()

    def _init_renderer(self):
        if self.model is None:
            return
        w = int(max(self.width(), 64) * self._render_scale)
        h = int(max(self.height(), 64) * self._render_scale)
        self._render_w, self._render_h = w, h
        try:
            self.model.vis.global_.offwidth = w
            self.model.vis.global_.offheight = h
        except Exception as e:
            log.warning(f"Could not set offscreen buffer size: {e}")
        try:
            self.renderer = mujoco.Renderer(self.model, height=h, width=w)
            log.info(f"Renderer initialized {w}×{h} (Scale: {self._render_scale}x)")
        except Exception as e:
            log.error(f"Failed to create renderer: {e}")
            self.renderer = None

    # ── Camera helpers ────────────────────────────────────────

    def reset_camera(self):
        if self.model is None:
            return
        self.cam = mujoco.MjvCamera()
        try:
            mujoco.mjv_defaultFreeCamera(self.model, self.cam)
        except Exception as e:
            log.warning(f"Camera reset failed: {e}")
        self._follow_body_id = -1

    def set_camera_preset(self, azimuth, elevation):
        if self.model is None:
            return
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        self.render()

    def save_camera_bookmark(self, name: str):
        self._camera_bookmarks[name] = (
            self.cam.lookat.copy(),
            float(self.cam.distance),
            float(self.cam.azimuth),
            float(self.cam.elevation),
        )
        log.info(f"Camera bookmark saved: {name}")

    def load_camera_bookmark(self, name: str):
        if name not in self._camera_bookmarks:
            log.warning(f"Camera bookmark not found: {name}")
            return
        lookat, dist, az, el = self._camera_bookmarks[name]
        self.cam.lookat = lookat
        self.cam.distance = dist
        self.cam.azimuth = az
        self.cam.elevation = el
        self.render()
        log.info(f"Camera bookmark loaded: {name}")

    def get_camera_bookmarks(self) -> dict:
        return dict(self._camera_bookmarks)

    def delete_camera_bookmark(self, name: str):
        if name in self._camera_bookmarks:
            del self._camera_bookmarks[name]

    def set_highlight_body(self, body_id: int):
        self._highlight_body = body_id
        self.render()

    def set_follow_body(self, body_id: int):
        self._follow_body_id = body_id
        if body_id >= 0:
            log.info(f"Following body {body_id}")
        else:
            log.info("Follow mode disabled")

    def focus_on_body(self, body_id: int):
        if self.model is None or body_id < 0 or body_id >= self.model.nbody:
            return
        if self._lock:
            self._lock.acquire()
        try:
            pos = self.data.xipos[body_id].copy()
        finally:
            if self._lock:
                self._lock.release()
        self.cam.lookat = pos
        self.cam.distance = max(2.0, self.model.stat.extent * 0.8)
        self.cam.azimuth = 135.0
        self.cam.elevation = -20.0
        self._highlight_body = body_id
        self.render()

    # ── Rendering (thread-safe) ───────────────────────────────

    def render(self):
        if self.renderer is None or self.model is None:
            return
        if self._lock:
            self._lock.acquire()
        try:
            self._render_internal()
        finally:
            if self._lock:
                self._lock.release()

    def _render_internal(self):
        try:
            if self.data is not None and np.any(np.isnan(self.data.qpos)):
                self._render_error_count += 1
                if not self._nan_notified:
                    self._nan_notified = True
                    self._cached_has_nan = True
                    # Emit signal via QTimer to avoid cross-thread issues
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(0, self.nan_detected.emit)
                self._cache_overlay_data()
                self.update()
                return

            # Reset NaN notification flag if sim is healthy
            if self._nan_notified and self.data is not None:
                if not np.any(np.isnan(self.data.qpos)):
                    self._nan_notified = False

            if 0 <= self._follow_body_id < self.model.nbody:
                pos = self.data.xipos[self._follow_body_id]
                if np.all(np.isfinite(pos)):
                    self.cam.lookat = pos.copy()

            # Safe highlight/select flag handling
            if 0 <= self._highlight_body < self.model.nbody:
                if VIS_SELECT is not None:
                    try:
                        self.vopt.flags[VIS_SELECT] = True
                    except (IndexError, TypeError):
                        pass
                if self.pert.select != self._highlight_body:
                    self.pert.select = self._highlight_body

            # Resolve cat bit safely
            cat_all = safe_enum("mjtCatBit", "mjCAT_ALL")
            if cat_all is None:
                cat_all = 7  # fallback: ALL = mjCAT_STATIC | mjCAT_DYNAMIC | mjCAT_DECOR

            mujoco.mjv_updateScene(
                self.model, self.data, self.vopt,
                self.pert, self.cam, cat_all,
                self.renderer.scene,
            )

            if self._show_traces:
                self._add_trace_geoms()

            pixels = self.renderer.render()
            pixels = np.ascontiguousarray(pixels)
            if pixels.ndim != 3 or pixels.shape[0] == 0 or pixels.shape[1] == 0:
                return

            h, w, ch = pixels.shape
            if ch == 4:
                pixels = np.ascontiguousarray(pixels[:, :, :3])
                ch = 3
            if ch != 3:
                return

            bytes_per_line = 3 * w
            img_data = bytes(pixels.data)
            qimg = QImage(img_data, w, h, bytes_per_line, QImage.Format_RGB888)
            if qimg.isNull():
                self._image = None
                return
            self._image = qimg.copy()
            self._img_data_ref = img_data
            self._render_error_count = 0
        except Exception as e:
            self._render_error_count += 1
            if self._render_error_count <= 5:
                log.error(f"Render error ({self._render_error_count}): {e}")

        self._cache_overlay_data()
        self.update()

    def _cache_overlay_data(self):
        if self.data is None:
            self._cached_time_str = ""
            self._cached_has_nan = False
            self._cached_selected_pos = None
            self._cached_selected_name = ""
            return
        self._cached_has_nan = bool(np.any(np.isnan(self.data.qpos)))
        self._cached_time_str = f"t={self.data.time:.3f}s  step={self._step_count}"
        if self._selected_body > 0 and self._selected_body < self.model.nbody:
            pos = self.data.xipos[self._selected_body]
            self._cached_selected_pos = pos.copy() if np.all(np.isfinite(pos)) else None
            self._cached_selected_name = (
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, self._selected_body
                ) or f"body_{self._selected_body}"
            )
        else:
            self._cached_selected_pos = None
            self._cached_selected_name = ""

    def _add_trace_geoms(self):
        scene = self.renderer.scene
        for body_id, positions in self._traces.items():
            if body_id <= 0 or len(positions) < 2:
                continue
            n = len(positions)
            for idx, pos in enumerate(positions):
                if scene.ngeom >= scene.maxgeom:
                    return
                alpha = 0.15 + 0.5 * (idx / max(n - 1, 1))
                rgba = np.array([0.9, 0.6, 0.1, alpha], dtype=np.float32)
                size_arr = np.array([0.008, 0, 0], dtype=np.float64)
                pos_arr = np.array(pos, dtype=np.float64)
                mat_arr = np.eye(3, dtype=np.float64).flatten()
                try:
                    geom_type = safe_enum("mjtGeom", "mjGEOM_SPHERE", 6)
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        geom_type,
                        size_arr, pos_arr, mat_arr, rgba,
                    )
                    scene.ngeom += 1
                except Exception:
                    return

    # ── Trace recording ───────────────────────────────────────

    def record_trace(self):
        if not self._show_traces or self.model is None:
            return
        self._trace_counter += 1
        if self._trace_counter % self._trace_interval != 0:
            return
        for i in range(1, self.model.nbody):
            pos = self.data.xipos[i].copy()
            if i not in self._traces:
                self._traces[i] = deque(maxlen=self._max_trace_len)
            self._traces[i].append(pos)

    def clear_traces(self):
        self._traces = {}
        self._trace_counter = 0
        self._toast.show_message("Traces cleared")

    def toggle_traces(self):
        self._show_traces = not self._show_traces
        if not self._show_traces:
            self._traces = {}
            self._trace_counter = 0
        self._toast.show_message(f"Traces {'ON' if self._show_traces else 'OFF'}")
        log.info(f"Body traces {'enabled' if self._show_traces else 'disabled'}")

    def increment_step_count(self, n: int = 1):
        self._step_count += n

    # ── Selection ─────────────────────────────────────────────

    def _try_select(self, pos):
        if self.renderer is None or self.model is None:
            return
        rect = self._get_render_rect()
        if rect is None:
            return
        dx, dy, dw, dh = rect
        relx = (pos.x() - dx) / max(dw, 1)
        rely = (pos.y() - dy) / max(dh, 1)
        aspectratio = dw / max(dh, 1)
        if not (0 <= relx <= 1 and 0 <= rely <= 1):
            return
        try:
            result = mujoco.mjv_select(
                self.model, self.data, self.vopt,
                aspectratio, relx, rely, self.renderer.scene,
            )
            body_id = -1
            selpnt = np.zeros(3)

            # Handle different return types across MuJoCo versions
            if isinstance(result, (tuple, list)):
                if len(result) >= 5:
                    selpnt = np.asarray(result[0], dtype=np.float64)
                    body_id = int(result[4])
                elif len(result) >= 2:
                    selpnt = np.asarray(result[0], dtype=np.float64)
                    body_id = int(result[-1])
            elif isinstance(result, (int, np.integer)):
                body_id = int(result)
            # Some versions return a namedtuple-like object
            elif hasattr(result, 'bodyid'):
                body_id = int(result.bodyid)
                if hasattr(result, 'point'):
                    selpnt = np.asarray(result.point, dtype=np.float64)

            if body_id > 0:
                self.pert.select = body_id
                self._selected_body = body_id
                self._highlight_body = body_id
                try:
                    self.pert.selectpos[:] = selpnt[:3]
                except Exception:
                    pass
                self.body_focused.emit(body_id)
                name = (
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    or f"body_{body_id}"
                )
                self._toast.show_message(f"Selected: {name}")
            else:
                self.pert.select = 0
                self._selected_body = -1
                self._highlight_body = -1
                self._toast.show_message("Selection cleared")
        except Exception as e:
            log.error(f"Selection error: {e}")

    # ── Painting (lock-free, uses cached data) ────────────────

    @staticmethod
    def _draw_text_with_bg(painter, x, y, text, font, text_color, bg_color=None, padding=5, radius=4):
        if bg_color is None:
            bg_color = QColor(10, 10, 20, 180)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()
        bg_rect = QRect(
            x - padding,
            y - metrics.ascent() - padding // 2,
            text_width + padding * 2,
            text_height + padding,
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(bg_rect, radius, radius)
        painter.setPen(text_color)
        painter.drawText(x, y, text)
        return bg_rect

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0f0f14"))

        if self._image is not None and not self._image.isNull():
            scaled = self._image.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawImage(x, y, scaled)

            if self._paused:
                grad = QLinearGradient(0, 0, 0, self.height())
                grad.setColorAt(0, QColor(0, 0, 0, 0))
                grad.setColorAt(0.35, QColor(0, 0, 0, 120))
                grad.setColorAt(0.65, QColor(0, 0, 0, 120))
                grad.setColorAt(1, QColor(0, 0, 0, 0))
                painter.fillRect(self.rect(), QBrush(grad))
                self._draw_text_with_bg(
                    painter, self.width() // 2 - 100, self.height() // 2 - 10,
                    "⏸  PAUSED", QFont("Segoe UI", 28, QFont.Bold),
                    QColor(255, 255, 255, 230),
                    QColor(0, 0, 0, 160), padding=16, radius=12,
                )
                self._draw_text_with_bg(
                    painter, self.width() // 2 - 100, self.height() // 2 + 30,
                    "Press Space to resume", QFont("Segoe UI", 12),
                    QColor(200, 200, 220, 180),
                    QColor(0, 0, 0, 120), padding=8,
                )

            if self._show_sim_info and self.model is not None:
                self._draw_text_with_bg(
                    painter, 10, self.height() - 14,
                    self._cached_time_str, QFont("Consolas", 10),
                    QColor(180, 200, 255, 220),
                    QColor(0, 0, 0, 160),
                )

            if self._show_info_overlay and self._selected_body > 0 and self.model is not None:
                name = self._cached_selected_name or f"body_{self._selected_body}"
                panel_x, panel_y = 8, 8
                panel_w, panel_h = 220, 56 if not self._perturbing else 70
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(10, 14, 30, 190)))
                painter.drawRoundedRect(panel_x, panel_y, panel_w, panel_h, 8, 8)
                painter.setPen(QPen(QColor(122, 162, 247, 100), 1))
                painter.drawRoundedRect(panel_x, panel_y, panel_w, panel_h, 8, 8)
                self._draw_text_with_bg(
                    painter, panel_x + 8, panel_y + 18,
                    f"● {name}", QFont("Segoe UI", 12, QFont.Bold),
                    QColor(122, 200, 255, 240),
                    QColor(0, 0, 0, 0), padding=0,
                )
                if self._cached_selected_pos is not None:
                    p = self._cached_selected_pos
                    self._draw_text_with_bg(
                        painter, panel_x + 8, panel_y + 36,
                        f"pos: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})",
                        QFont("Consolas", 9),
                        QColor(180, 195, 230, 200),
                        QColor(0, 0, 0, 0), padding=0,
                    )
                if self._perturbing:
                    self._draw_text_with_bg(
                        painter, panel_x + 8, panel_y + 54,
                        "✋ Dragging…", QFont("Segoe UI", 10),
                        QColor(158, 206, 106, 230),
                        QColor(0, 0, 0, 0), padding=0,
                    )

            if self._show_fps_overlay and self._fps > 0:
                fps_text = f"{self._fps:.0f} FPS"
                fps_color = (
                    QColor(158, 206, 106, 230) if self._fps >= 30
                    else QColor(224, 175, 104, 230) if self._fps >= 15
                    else QColor(247, 118, 142, 230)
                )
                self._draw_text_with_bg(
                    painter, self.width() - 90, 18,
                    fps_text, QFont("Consolas", 11, QFont.Bold),
                    fps_color, QColor(10, 10, 20, 170),
                )

            if self._show_axis_overlay:
                self._draw_axis_indicator(painter)

            if self._show_grid_overlay and self.model is not None:
                self._draw_grid_overlay(painter)

            if self._follow_body_id >= 0 and self.model is not None:
                name = (
                    mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, self._follow_body_id
                    ) or f"body_{self._follow_body_id}"
                )
                self._draw_text_with_bg(
                    painter, 10, self.height() - 34,
                    f"📷 Following: {name}", QFont("Segoe UI", 9),
                    QColor(158, 206, 106, 210),
                    QColor(10, 10, 20, 160),
                )
        else:
            painter.setPen(QColor(200, 200, 220))
            painter.setFont(QFont("Segoe UI", 16))
            if self._cached_has_nan:
                painter.drawText(
                    self.rect(), Qt.AlignCenter,
                    "⚠  Simulation diverged (NaN detected)\n"
                    "Press R to reset the simulation",
                )
            elif self.model is not None and self.renderer is not None:
                painter.drawText(
                    self.rect(), Qt.AlignCenter,
                    "Rendering failed — check Log tab",
                )
            elif self.model is not None:
                painter.drawText(
                    self.rect(), Qt.AlignCenter,
                    "Renderer not available — check Log tab",
                )
            else:
                painter.drawText(
                    self.rect(), Qt.AlignCenter,
                    "Load a model to begin\nDrag & drop an XML file here",
                )
        painter.end()

    # ── Overlay drawing helpers ───────────────────────────────

    def _draw_axis_indicator(self, painter: QPainter):
        cx, cy = self.width() - 60, self.height() - 60
        length = 38
        az = np.radians(getattr(self.cam, 'azimuth', 0))
        el = np.radians(getattr(self.cam, 'elevation', 0))
        dx = np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
        dy = np.array([np.cos(az), np.sin(az), 0])
        dz = np.array([-np.sin(az) * np.sin(el), np.cos(az) * np.sin(el), np.cos(el)])

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(10, 14, 30, 160)))
        painter.drawEllipse(cx - length - 8, cy - length - 8,
                            (length + 8) * 2, (length + 8) * 2)
        painter.setPen(QPen(QColor(80, 90, 130, 80), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(cx - length - 8, cy - length - 8,
                            (length + 8) * 2, (length + 8) * 2)

        font = QFont("Consolas", 9, QFont.Bold)
        for direction, color, label in [
            (dx, QColor(255, 120, 140, 240), "X"),
            (dy, QColor(140, 220, 120, 240), "Y"),
            (dz, QColor(120, 170, 255, 240), "Z"),
        ]:
            ex = cx + int(direction[0] * length)
            ey = cy - int(direction[2] * length)
            painter.setPen(QPen(color, 2.5))
            painter.drawLine(cx, cy, ex, ey)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(ex - 3, ey - 3, 6, 6)
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(ex + (5 if direction[0] >= 0 else -15), ey - 4, label)

    def _draw_grid_overlay(self, painter: QPainter):
        rect = self._get_render_rect()
        if rect is None:
            return
        dx, dy, dw, dh = rect
        painter.setPen(QPen(QColor(255, 255, 255, 15), 1))
        step = max(dw, dh) // 8
        for i in range(1, 8):
            painter.drawLine(dx + i * step, dy, dx + i * step, dy + dh)
        for i in range(1, 8):
            painter.drawLine(dx, dy + i * step, dx + dw, dy + i * step)
        painter.setPen(QPen(QColor(255, 255, 255, 25), 1))
        cx, cy = dx + dw // 2, dy + dh // 2
        painter.drawLine(cx, dy, cx, dy + dh)
        painter.drawLine(dx, cy, dx + dw, cy)

    def _get_render_rect(self):
        if self._image is None or self._image.isNull():
            return None
        iw, ih = self._render_w, self._render_h
        ww, wh = self.width(), self.height()
        scale = min(ww / max(iw, 1), wh / max(ih, 1))
        dw, dh = int(iw * scale), int(ih * scale)
        dx, dy = (ww - dw) // 2, (wh - dh) // 2
        return dx, dy, dw, dh

    # ── Resize ────────────────────────────────────────────────

    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        if (
            self.model
            and self.renderer
            and (abs(w - self._render_w / self._render_scale) > 8
                 or abs(h - self._render_h / self._render_scale) > 8)
        ):
            self._init_renderer()
            self.render()
        super().resizeEvent(event)

    # ── Keyboard modifiers ────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            self._ctrl_held = True
        elif event.key() == Qt.Key_Shift:
            self._shift_held = True
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self._ctrl_held = False
        elif event.key() == Qt.Key_Shift:
            self._shift_held = False
        super().keyReleaseEvent(event)

    # ── Mouse interaction ─────────────────────────────────────

    def mousePressEvent(self, event):
        self._last_pos = event.position()
        self._active_btn = event.button()
        if event.button() == Qt.LeftButton and (
            self._ctrl_held or event.modifiers() & Qt.ControlModifier
        ):
            self._try_select(event.position())
            self._perturbing = True
            return

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._try_select(event.position())

    def mouseReleaseEvent(self, event):
        self._last_pos = None
        self._active_btn = None
        if self._perturbing:
            self._perturbing = False
            if self.model is not None:
                if self._lock:
                    self._lock.acquire()
                try:
                    self.data.xfrc_applied[:] = 0
                finally:
                    if self._lock:
                        self._lock.release()
            self.render()

    def mouseMoveEvent(self, event):
        if self._last_pos is None or self.model is None:
            return
        pos = event.position()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()

        if self._perturbing and self._selected_body > 0 and self.renderer is not None:
            try:
                move_v = safe_enum("mjtMouse", "mjMOUSE_MOVE_V")
                if move_v is None:
                    move_v = 2  # fallback value
                mujoco.mjv_movePerturb(
                    self.model, self.data, self.vopt,
                    move_v, dx, dy, self.renderer.scene, self.pert,
                )
                if self.pert.select > 0:
                    body_id = self.pert.select
                    if self._lock:
                        self._lock.acquire()
                    try:
                        self.data.xfrc_applied[body_id, :3] = self.pert.force.copy()
                        self.data.xfrc_applied[body_id, 3:] = self.pert.torque.copy()
                    finally:
                        if self._lock:
                            self._lock.release()
            except Exception:
                pass
            self._last_pos = pos
            self.render()
            return

        if self._active_btn == Qt.LeftButton and self._shift_held:
            action = safe_enum("mjtMouse", "mjMOUSE_MOVE_V") or 2
        elif self._active_btn == Qt.LeftButton:
            action = safe_enum("mjtMouse", "mjMOUSE_ROTATE_V") or 0
        elif self._active_btn == Qt.MiddleButton:
            action = safe_enum("mjtMouse", "mjMOUSE_MOVE_V") or 2
        elif self._active_btn == Qt.RightButton:
            action = safe_enum("mjtMouse", "mjMOUSE_ZOOM") or 3
        else:
            return

        if self.renderer is not None:
            try:
                cat_all = safe_enum("mjtCatBit", "mjCAT_ALL", 7)
                mujoco.mjv_updateScene(
                    self.model, self.data, self.vopt,
                    self.pert, self.cam, cat_all, self.renderer.scene,
                )
                mujoco.mjv_moveCamera(
                    self.model, action, dx, dy, self.renderer.scene, self.cam,
                )
            except Exception as e:
                log.debug(f"Camera move error: {e}")
        self._last_pos = pos
        self.render()

    def wheelEvent(self, event):
        if self.model is None or self.renderer is None:
            return
        delta = event.angleDelta().y() / 120.0
        try:
            cat_all = safe_enum("mjtCatBit", "mjCAT_ALL", 7)
            zoom_action = safe_enum("mjtMouse", "mjMOUSE_ZOOM") or 3
            mujoco.mjv_updateScene(
                self.model, self.data, self.vopt,
                self.pert, self.cam, cat_all, self.renderer.scene,
            )
            mujoco.mjv_moveCamera(
                self.model, zoom_action, 0, -delta * 30,
                self.renderer.scene, self.cam,
            )
        except Exception:
            pass
        self.render()

    # ── Context menu ──────────────────────────────────────────

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #1a1b26; color: #c0caf5; "
            "border: 1px solid #3d59a1; font-size: 13px; }"
            "QMenu::item { padding: 6px 28px; }"
            "QMenu::item:selected { background-color: #3d59a1; }"
        )
        menu.addAction("Fit Camera", lambda: (self.reset_camera(), self.render()))
        menu.addSeparator()

        a_fps = menu.addAction("FPS Overlay")
        a_fps.setCheckable(True)
        a_fps.setChecked(self._show_fps_overlay)
        a_fps.toggled.connect(lambda v: setattr(self, '_show_fps_overlay', v))

        a_axis = menu.addAction("Axis Indicator")
        a_axis.setCheckable(True)
        a_axis.setChecked(self._show_axis_overlay)
        a_axis.toggled.connect(lambda v: setattr(self, '_show_axis_overlay', v))

        a_grid = menu.addAction("Grid Overlay")
        a_grid.setCheckable(True)
        a_grid.setChecked(self._show_grid_overlay)
        a_grid.toggled.connect(lambda v: setattr(self, '_show_grid_overlay', v))

        a_sim = menu.addAction("Sim Info Overlay")
        a_sim.setCheckable(True)
        a_sim.setChecked(self._show_sim_info)
        a_sim.toggled.connect(lambda v: setattr(self, '_show_sim_info', v))

        if self._camera_bookmarks:
            menu.addSeparator()
            bm_menu = menu.addMenu("📷 Bookmarks")
            for name in list(self._camera_bookmarks.keys()):
                bm_menu.addAction(name, lambda n=name: self.load_camera_bookmark(n))
        menu.exec(event.globalPos())

    # ── Drag & drop ───────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith('.xml'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith('.xml'):
                mw = self.window()
                if hasattr(mw, '_load_model_path'):
                    mw._load_model_path(path)
                break