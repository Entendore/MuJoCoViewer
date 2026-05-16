"""3-D viewport: rendering, camera, perturbation, traces, overlays."""

import numpy as np
from collections import deque

from PySide6.QtWidgets import QWidget, QSizePolicy, QMenu
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QFont, QColor, QPen

import mujoco
from widgets import ToastLabel, log


class MujocoViewport(QWidget):
    body_focused = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)

        self.model, self.data = None, None
        self.renderer = None
        self.cam = mujoco.MjvCamera()
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

        self._image = None
        self._render_w, self._render_h = 0, 0
        self._last_pos = None
        self._active_btn = None
        self._ctrl_held = False
        self._shift_held = False
        self._perturbing = False
        self._selected_body = -1
        self._paused = False

        self._show_traces = False
        self._traces = {}
        self._trace_interval = 4
        self._trace_counter = 0
        self._max_trace_len = 400

        self._show_fps_overlay = True
        self._show_axis_overlay = True
        self._show_grid_overlay = False
        self._show_info_overlay = True

        self._follow_body_id = -1

        self._toast = ToastLabel(self)
        self._render_error_count = 0
        self._fps = 0.0

    def set_model(self, model, data):
        self.model, self.data = model, data
        self.pert = mujoco.MjvPerturb()
        self._traces = {}
        self._trace_counter = 0
        self._selected_body = -1
        self._follow_body_id = -1
        self._render_error_count = 0
        try:
            mujoco.mj_forward(model, data)
            log.info("mj_forward completed on new model")
        except Exception as e:
            log.error(f"mj_forward failed on new model: {e}")
        self._init_renderer()
        self.reset_camera()
        self.render()

    def _init_renderer(self):
        if self.model is None:
            return
        w, h = max(self.width(), 64), max(self.height(), 64)
        self._render_w, self._render_h = w, h
        try:
            self.model.vis.global_.offwidth = int(w)
            self.model.vis.global_.offheight = int(h)
        except Exception as e:
            log.warning(f"Could not set offscreen buffer size: {e}")
        try:
            self.renderer = mujoco.Renderer(self.model, height=h, width=w)
            log.info(f"Renderer initialized {w}×{h}")
        except Exception as e:
            log.error(f"Failed to create renderer: {e}")
            self.renderer = None

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

    def set_follow_body(self, body_id):
        self._follow_body_id = body_id
        if body_id >= 0:
            log.info(f"Following body {body_id}")
        else:
            log.info("Follow mode disabled")

    def focus_on_body(self, body_id):
        if self.model is None or body_id < 0 or body_id >= self.model.nbody:
            return
        pos = self.data.xipos[body_id].copy()
        self.cam.lookat = pos
        self.cam.distance = max(2.0, self.model.stat.extent * 0.8)
        self.cam.azimuth = 135.0
        self.cam.elevation = -20.0
        self.render()

    def render(self):
        if self.renderer is None or self.model is None:
            return
        try:
            if self._follow_body_id >= 0 and self._follow_body_id < self.model.nbody:
                pos = self.data.xipos[self._follow_body_id].copy()
                self.cam.lookat = pos

            mujoco.mjv_updateScene(
                self.model, self.data, self.vopt,
                self.pert, self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
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
            img_data = pixels.tobytes()
            qimg = QImage(img_data, w, h, bytes_per_line, QImage.Format_RGB888)
            if qimg.isNull():
                self._image = None
                return
            self._image = qimg.copy()
            self._render_error_count = 0
        except Exception as e:
            self._render_error_count += 1
            if self._render_error_count <= 5:
                log.error(f"Render error ({self._render_error_count}): {e}")
        self.update()

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
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        size_arr, pos_arr, mat_arr, rgba,
                    )
                    scene.ngeom += 1
                except Exception:
                    return

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
        if relx < 0 or relx > 1 or rely < 0 or rely > 1:
            return
        try:
            result = mujoco.mjv_select(
                self.model, self.data, self.vopt,
                aspectratio, relx, rely, self.renderer.scene,
            )
            body_id = -1
            selpnt = np.zeros(3)
            if isinstance(result, (tuple, list)):
                if len(result) >= 5:
                    body_id = int(result[4])
                    selpnt = np.asarray(result[0])
                elif len(result) >= 2:
                    body_id = int(result[-1])
                    selpnt = np.asarray(result[0])
            elif isinstance(result, (int, np.integer)):
                body_id = int(result)
            if body_id > 0:
                self.pert.select = body_id
                self._selected_body = body_id
                try:
                    self.pert.selectpos[:] = selpnt[:3]
                except Exception:
                    pass
                self.body_focused.emit(body_id)
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                ) or f"body_{body_id}"
                self._toast.show_message(f"Selected: {name}")
            else:
                self.pert.select = 0
                self._selected_body = -1
                self._toast.show_message("Selection cleared")
        except Exception as e:
            log.error(f"Selection error: {e}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0f0f14"))

        if self._image is not None and not self._image.isNull():
            scaled = self._image.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawImage(x, y, scaled)

            if self._paused:
                painter.fillRect(self.rect(), QColor(0, 0, 0, 60))
                painter.setPen(QColor(255, 255, 255, 180))
                painter.setFont(QFont("Segoe UI", 28, QFont.Bold))
                painter.drawText(self.rect(), Qt.AlignCenter, "⏸  PAUSED")
                painter.setPen(QColor(255, 255, 255, 100))
                painter.setFont(QFont("Segoe UI", 12))
                r = self.rect()
                r.setTop(r.top() + 30)
                painter.drawText(r, Qt.AlignCenter, "Press Space to resume")

            if self._show_info_overlay and self._selected_body > 0 and self.model is not None:
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, self._selected_body
                ) or f"body_{self._selected_body}"
                painter.setPen(QColor(122, 162, 247, 220))
                painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
                painter.drawText(12, 24, f"🔵 {name}")
                if self._perturbing:
                    painter.setPen(QColor(158, 206, 106, 200))
                    painter.setFont(QFont("Segoe UI", 10))
                    painter.drawText(12, 42, "Dragging…")

            if self._show_fps_overlay and self._fps > 0:
                painter.setPen(QColor(122, 162, 247, 180))
                painter.setFont(QFont("Consolas", 10))
                painter.drawText(self.width() - 80, 20, f"{self._fps:.0f} FPS")

            if self._show_axis_overlay:
                self._draw_axis_indicator(painter)

            if self._show_grid_overlay and self.model is not None:
                self._draw_grid_overlay(painter)

            if self._follow_body_id >= 0 and self.model is not None:
                painter.setPen(QColor(158, 206, 106, 180))
                painter.setFont(QFont("Segoe UI", 9))
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, self._follow_body_id
                ) or f"body_{self._follow_body_id}"
                painter.drawText(12, self.height() - 12, f"📷 Following: {name}")
        else:
            painter.setPen(QColor(200, 200, 220))
            painter.setFont(QFont("Segoe UI", 16))
            if self.model is not None and self.renderer is not None:
                painter.drawText(self.rect(), Qt.AlignCenter, "Rendering failed — check Log tab")
            elif self.model is not None:
                painter.drawText(self.rect(), Qt.AlignCenter, "Renderer not available — check Log tab")
            else:
                painter.drawText(self.rect(), Qt.AlignCenter, "Load a model to begin\nDrag & drop an XML file here")
        painter.end()

    def _draw_axis_indicator(self, painter):
        cx, cy = self.width() - 50, self.height() - 50
        length = 30
        az = np.radians(getattr(self.cam, 'azimuth', 0))
        el = np.radians(getattr(self.cam, 'elevation', 0))

        dx = np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
        dy = np.array([np.cos(az), np.sin(az), 0])
        dz = np.array([-np.sin(az) * np.sin(el), np.cos(az) * np.sin(el), np.cos(el)])

        axes = [
            (dx, QColor(247, 118, 142, 200), "X"),
            (dy, QColor(158, 206, 106, 200), "Y"),
            (dz, QColor(122, 162, 247, 200), "Z"),
        ]

        painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
        painter.drawEllipse(cx - length - 2, cy - length - 2, (length + 2) * 2, (length + 2) * 2)

        font = QFont("Consolas", 8, QFont.Bold)
        painter.setFont(font)
        for direction, color, label in axes:
            ex = cx + int(direction[0] * length)
            ey = cy - int(direction[2] * length)
            painter.setPen(QPen(color, 2))
            painter.drawLine(cx, cy, ex, ey)
            painter.drawText(ex + 2, ey - 2, label)

    def _draw_grid_overlay(self, painter):
        rect = self._get_render_rect()
        if rect is None:
            return
        dx, dy, dw, dh = rect
        painter.setPen(QPen(QColor(255, 255, 255, 15), 1))
        step = max(dw, dh) // 8
        for i in range(1, 8):
            x = dx + i * step
            painter.drawLine(x, dy, x, dy + dh)
        for i in range(1, 8):
            y = dy + i * step
            painter.drawLine(dx, y, dx + dw, y)

    def _get_render_rect(self):
        if self._image is None or self._image.isNull():
            return None
        iw, ih = self._render_w, self._render_h
        ww, wh = self.width(), self.height()
        scale = min(ww / max(iw, 1), wh / max(ih, 1))
        dw, dh = int(iw * scale), int(ih * scale)
        dx, dy = (ww - dw) // 2, (wh - dh) // 2
        return dx, dy, dw, dh

    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        if (
            self.model
            and self.renderer
            and (abs(w - self._render_w) > 8 or abs(h - self._render_h) > 8)
        ):
            self._render_w, self._render_h = max(w, 64), max(h, 64)
            try:
                self.model.vis.global_.offwidth = self._render_w
                self.model.vis.global_.offheight = self._render_h
            except Exception:
                pass
            try:
                self.renderer = mujoco.Renderer(
                    self.model, height=self._render_h, width=self._render_w
                )
            except Exception as e:
                log.error(f"Renderer resize failed: {e}")
                self.renderer = None
        super().resizeEvent(event)

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

    def mousePressEvent(self, event):
        self._last_pos = event.position()
        self._active_btn = event.button()
        if event.button() == Qt.LeftButton and (
            self._ctrl_held or event.modifiers() & Qt.ControlModifier
        ):
            self._try_select(event.position())
            self._perturbing = True
            return
        if event.button() == Qt.LeftButton and event.flags() & Qt.MouseEventCreatedDoubleClick:
            self._try_select(event.position())

    def mouseReleaseEvent(self, event):
        self._last_pos = None
        self._active_btn = None
        if self._perturbing:
            self._perturbing = False
            if self.model is not None:
                self.data.xfrc_applied[:] = 0
            self.render()

    def mouseMoveEvent(self, event):
        if self._last_pos is None or self.model is None:
            return
        pos = event.position()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()

        if self._perturbing and self._selected_body > 0 and self.renderer is not None:
            try:
                mujoco.mjv_movePerturb(
                    self.model, self.data, self.vopt,
                    mujoco.mjtMouse.mjMOUSE_MOVE_V,
                    dx, dy, self.renderer.scene, self.pert,
                )
                if self.pert.select > 0:
                    body_id = self.pert.select
                    self.data.xfrc_applied[body_id, :3] = self.pert.force.copy()
                    self.data.xfrc_applied[body_id, 3:] = self.pert.torque.copy()
            except Exception:
                pass
            self._last_pos = pos
            self.render()
            return

        if self._active_btn == Qt.LeftButton and self._shift_held:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._active_btn == Qt.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self._active_btn == Qt.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._active_btn == Qt.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return

        if self.renderer is not None:
            try:
                mujoco.mjv_updateScene(
                    self.model, self.data, self.vopt,
                    self.pert, self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL,
                    self.renderer.scene,
                )
                mujoco.mjv_moveCamera(
                    self.model, action, dx, dy,
                    self.renderer.scene, self.cam
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
            mujoco.mjv_updateScene(
                self.model, self.data, self.vopt,
                self.pert, self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.renderer.scene,
            )
            mujoco.mjv_moveCamera(
                self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                0, -delta * 30, self.renderer.scene, self.cam,
            )
        except Exception:
            pass
        self.render()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background-color: #1a1b26; color: #a9b1d6; border: 1px solid #3d59a1; }"
                           "QMenu::item:selected { background-color: #3d59a1; }")
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
        menu.exec(event.globalPos())

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