"""Configuration tabs: XML editor with find/replace, render options, and camera bookmarks panel."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QPlainTextEdit, QComboBox, QCheckBox,
    QLineEdit, QSpinBox, QInputDialog,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
import mujoco
from constants import VIS_FLAGS, CAMERA_PRESETS, LABEL_MODES, safe_vis_flag
from viewport import MujocoViewport
from widgets import log


class XMLEditor(QWidget):
    """Live XML editor with find/replace, validation, and apply/revert."""

    apply_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._find_bar = QWidget()
        self._find_bar.setVisible(False)
        fb_lay = QHBoxLayout(self._find_bar)
        fb_lay.setContentsMargins(0, 0, 0, 0)
        fb_lay.setSpacing(4)

        fb_lay.addWidget(QLabel("Find:"))
        self._find_input = QLineEdit()
        self._find_input.setPlaceholderText("Search text…")
        self._find_input.returnPressed.connect(self._find_next)
        fb_lay.addWidget(self._find_input)

        btn_find_next = QPushButton("▼")
        btn_find_next.setFixedWidth(28)
        btn_find_next.setToolTip("Find next")
        btn_find_next.clicked.connect(self._find_next)
        fb_lay.addWidget(btn_find_next)

        btn_find_prev = QPushButton("▲")
        btn_find_prev.setFixedWidth(28)
        btn_find_prev.setToolTip("Find previous")
        btn_find_prev.clicked.connect(self._find_prev)
        fb_lay.addWidget(btn_find_prev)

        self._match_lbl = QLabel("")
        self._match_lbl.setProperty("class", "dim")
        self._match_lbl.setFixedWidth(60)
        fb_lay.addWidget(self._match_lbl)

        fb_lay.addWidget(QLabel("Replace:"))
        self._replace_input = QLineEdit()
        self._replace_input.setPlaceholderText("Replace with…")
        fb_lay.addWidget(self._replace_input)

        btn_replace = QPushButton("Replace")
        btn_replace.clicked.connect(self._replace_one)
        fb_lay.addWidget(btn_replace)

        btn_replace_all = QPushButton("All")
        btn_replace_all.clicked.connect(self._replace_all)
        fb_lay.addWidget(btn_replace_all)

        btn_close_find = QPushButton("✕")
        btn_close_find.setFixedWidth(24)
        btn_close_find.clicked.connect(lambda: self._find_bar.setVisible(False))
        fb_lay.addWidget(btn_close_find)

        layout.addWidget(self._find_bar)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Load a model to view XML…")
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor.setFont(QFont("Consolas", 11))
        layout.addWidget(self.editor)

        self.status_lbl = QLabel("")
        self.status_lbl.setProperty("class", "dim")

        btn_row = QHBoxLayout()

        btn_find = QPushButton("🔍 Find")
        btn_find.setFixedWidth(70)
        btn_find.clicked.connect(self._show_find_bar)
        btn_row.addWidget(btn_find)

        self._wrap_cb = QCheckBox("Wrap")
        self._wrap_cb.setChecked(False)
        self._wrap_cb.toggled.connect(
            lambda v: self.editor.setLineWrapMode(
                QPlainTextEdit.WidgetWidth if v else QPlainTextEdit.NoWrap
            )
        )
        btn_row.addWidget(self._wrap_cb)

        btn_row.addWidget(self.status_lbl)
        btn_row.addStretch()

        self._cursor_lbl = QLabel("")
        self._cursor_lbl.setProperty("class", "dim")
        btn_row.addWidget(self._cursor_lbl)

        self.revert_btn = QPushButton("⟲  Revert")
        self.revert_btn.clicked.connect(self._revert)
        btn_row.addWidget(self.revert_btn)

        self.apply_btn = QPushButton("✔  Apply Changes")
        self.apply_btn.clicked.connect(self.apply_requested.emit)
        self.apply_btn.setStyleSheet(
            "QPushButton { background-color: #2b3f7a; border-color: #3d59a1; }"
            "QPushButton:hover { background-color: #3d59a1; }"
        )
        btn_row.addWidget(self.apply_btn)
        layout.addLayout(btn_row)

        self._saved_xml = ""
        self.editor.cursorPositionChanged.connect(self._update_cursor_pos)

    def set_xml(self, xml_string: str):
        self._saved_xml = xml_string
        self.editor.setPlainText(xml_string)
        self.status_lbl.setText("")
        self.status_lbl.setProperty("class", "dim")

    def set_validation(self, valid: bool, message: str = ""):
        if valid:
            self.status_lbl.setText(f"✓ {message}" if message else "✓ Valid")
            self.status_lbl.setProperty("class", "success")
        else:
            self.status_lbl.setText(f"✗ {message}" if message else "✗ Error")
            self.status_lbl.setProperty("class", "error")
        self.status_lbl.style().unpolish(self.status_lbl)
        self.status_lbl.style().polish(self.status_lbl)

    def _show_find_bar(self):
        self._find_bar.setVisible(True)
        self._find_input.setFocus()
        self._find_input.selectAll()

    def _update_cursor_pos(self):
        cursor = self.editor.textCursor()
        line = cursor.blockNumber() + 1
        col = cursor.columnNumber() + 1
        total_lines = self.editor.document().blockCount()
        self._cursor_lbl.setText(f"Ln {line}:{col} / {total_lines}")

    def _revert(self):
        self.editor.setPlainText(self._saved_xml)
        self.status_lbl.setText("Reverted")
        self.status_lbl.setProperty("class", "dim")

    def _find_next(self):
        text = self._find_input.text()
        if not text:
            return
        cursor = self.editor.document().find(text, self.editor.textCursor())
        if cursor.isNull():
            cursor = self.editor.document().find(text)
        if not cursor.isNull():
            self.editor.setTextCursor(cursor)
        self._update_match_count(text)

    def _find_prev(self):
        text = self._find_input.text()
        if not text:
            return
        cursor = self.editor.document().find(
            text, self.editor.textCursor(),
            self.editor.document().FindBackward,
        )
        if cursor.isNull():
            cursor = self.editor.document().find(
                text, self.editor.document().end(),
                self.editor.document().FindBackward,
            )
        if not cursor.isNull():
            self.editor.setTextCursor(cursor)
        self._update_match_count(text)

    def _update_match_count(self, text):
        if not text:
            self._match_lbl.setText("")
            return
        content = self.editor.toPlainText()
        count = content.count(text)
        self._match_lbl.setText(f"{count} matches")

    def _replace_one(self):
        text = self._find_input.text()
        replacement = self._replace_input.text()
        cursor = self.editor.textCursor()
        if cursor.selectedText() == text:
            cursor.insertText(replacement)
        self._find_next()

    def _replace_all(self):
        text = self._find_input.text()
        replacement = self._replace_input.text()
        if not text:
            return
        content = self.editor.toPlainText()
        count = content.count(text)
        self.editor.setPlainText(content.replace(text, replacement))
        self.status_lbl.setText(f"Replaced {count} occurrences")


# ═══════════════════════════════════════════════════════════════
#  Render Options Panel
# ═══════════════════════════════════════════════════════════════

class RenderOptionsPanel(QWidget):
    """Camera, overlay, trace, visualization, and rendering options."""

    def __init__(self, viewport: MujocoViewport, parent=None):
        super().__init__(parent)
        self.viewport = viewport
        self._checkboxes: list = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # ── Camera ──
        cam_group = QGroupBox("Camera")
        cam_lay = QVBoxLayout(cam_group)

        self.cam_combo = QComboBox()
        self.cam_combo.addItem("Free Camera")
        self.cam_combo.currentIndexChanged.connect(self._on_camera_changed)
        cam_lay.addWidget(self.cam_combo)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        for name, az, el in CAMERA_PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_combo)
        cam_lay.addLayout(preset_row)

        follow_row = QHBoxLayout()
        follow_row.addWidget(QLabel("Follow:"))
        self.follow_combo = QComboBox()
        self.follow_combo.addItem("None")
        self.follow_combo.currentIndexChanged.connect(self._on_follow_changed)
        follow_row.addWidget(self.follow_combo)
        cam_lay.addLayout(follow_row)

        btn_row = QHBoxLayout()
        fit_btn = QPushButton("Fit to Scene")
        fit_btn.clicked.connect(self._fit_camera)
        btn_row.addWidget(fit_btn)
        center_btn = QPushButton("Center")
        center_btn.clicked.connect(self._center_camera)
        btn_row.addWidget(center_btn)
        cam_lay.addLayout(btn_row)

        bm_row = QHBoxLayout()
        bm_row.addWidget(QLabel("Bookmark:"))
        self.bm_name_input = QLineEdit()
        self.bm_name_input.setPlaceholderText("name…")
        self.bm_name_input.setMaximumWidth(100)
        bm_row.addWidget(self.bm_name_input)

        save_bm_btn = QPushButton("💾")
        save_bm_btn.setToolTip("Save current camera as bookmark")
        save_bm_btn.setFixedWidth(28)
        save_bm_btn.clicked.connect(self._save_bookmark)
        bm_row.addWidget(save_bm_btn)

        self.bm_combo = QComboBox()
        self.bm_combo.setMaximumWidth(100)
        bm_row.addWidget(self.bm_combo)

        load_bm_btn = QPushButton("📂")
        load_bm_btn.setToolTip("Load selected bookmark")
        load_bm_btn.setFixedWidth(28)
        load_bm_btn.clicked.connect(self._load_bookmark)
        bm_row.addWidget(load_bm_btn)

        del_bm_btn = QPushButton("🗑")
        del_bm_btn.setToolTip("Delete selected bookmark")
        del_bm_btn.setFixedWidth(28)
        del_bm_btn.clicked.connect(self._delete_bookmark)
        bm_row.addWidget(del_bm_btn)
        cam_lay.addLayout(bm_row)

        layout.addWidget(cam_group)

        # ── Overlays ──
        overlay_group = QGroupBox("Overlays")
        overlay_lay = QVBoxLayout(overlay_group)

        self.fps_overlay_cb = QCheckBox("FPS Counter")
        self.fps_overlay_cb.setChecked(True)
        self.fps_overlay_cb.toggled.connect(
            lambda v: setattr(self.viewport, '_show_fps_overlay', v)
        )
        overlay_lay.addWidget(self.fps_overlay_cb)

        self.axis_overlay_cb = QCheckBox("Axis Indicator")
        self.axis_overlay_cb.setChecked(True)
        self.axis_overlay_cb.toggled.connect(
            lambda v: setattr(self.viewport, '_show_axis_overlay', v)
        )
        overlay_lay.addWidget(self.axis_overlay_cb)

        self.grid_overlay_cb = QCheckBox("Grid Overlay")
        self.grid_overlay_cb.setChecked(False)
        self.grid_overlay_cb.toggled.connect(
            lambda v: setattr(self.viewport, '_show_grid_overlay', v)
        )
        overlay_lay.addWidget(self.grid_overlay_cb)

        self.info_overlay_cb = QCheckBox("Selection Info")
        self.info_overlay_cb.setChecked(True)
        self.info_overlay_cb.toggled.connect(
            lambda v: setattr(self.viewport, '_show_info_overlay', v)
        )
        overlay_lay.addWidget(self.info_overlay_cb)

        self.sim_info_overlay_cb = QCheckBox("Sim Time Info")
        self.sim_info_overlay_cb.setChecked(True)
        self.sim_info_overlay_cb.toggled.connect(
            lambda v: setattr(self.viewport, '_show_sim_info', v)
        )
        overlay_lay.addWidget(self.sim_info_overlay_cb)

        layout.addWidget(overlay_group)

        # ── Traces ──
        trace_group = QGroupBox("Body Traces")
        trace_lay = QVBoxLayout(trace_group)

        self.trace_cb = QCheckBox("Show Traces")
        self.trace_cb.setChecked(False)
        self.trace_cb.toggled.connect(self._on_trace_toggled)
        trace_lay.addWidget(self.trace_cb)

        trace_settings = QHBoxLayout()
        trace_settings.addWidget(QLabel("Interval:"))
        self.trace_interval_spin = QSpinBox()
        self.trace_interval_spin.setRange(1, 50)
        self.trace_interval_spin.setValue(4)
        self.trace_interval_spin.setToolTip("Record trace every N frames")
        self.trace_interval_spin.valueChanged.connect(
            lambda v: setattr(self.viewport, '_trace_interval', v)
        )
        trace_settings.addWidget(self.trace_interval_spin)

        trace_settings.addWidget(QLabel("Max:"))
        self.trace_max_spin = QSpinBox()
        self.trace_max_spin.setRange(50, 2000)
        self.trace_max_spin.setValue(400)
        self.trace_max_spin.setSingleStep(100)
        self.trace_max_spin.setToolTip("Maximum trace points per body")
        self.trace_max_spin.valueChanged.connect(
            lambda v: setattr(self.viewport, '_max_trace_len', v)
        )
        trace_settings.addWidget(self.trace_max_spin)
        trace_lay.addLayout(trace_settings)

        clear_traces_btn = QPushButton("Clear Traces")
        clear_traces_btn.clicked.connect(self.viewport.clear_traces)
        trace_lay.addWidget(clear_traces_btn)

        layout.addWidget(trace_group)

        # ── Visualization flags (safe — VIS_FLAGS only contains available flags) ──
        vis_group = QGroupBox("Visualization")
        vis_lay = QVBoxLayout(vis_group)
        for name, flag in VIS_FLAGS:
            cb = QCheckBox(name)
            try:
                cb.setChecked(self.viewport.vopt.flags[flag])
            except (IndexError, TypeError):
                cb.setChecked(False)
                cb.setEnabled(False)
            cb.toggled.connect(self._make_flag_cb(flag))
            vis_lay.addWidget(cb)
            self._checkboxes.append(cb)

        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Labels:"))
        self.label_combo = QComboBox()
        for name, val in LABEL_MODES:
            self.label_combo.addItem(name)
        self.label_combo.currentIndexChanged.connect(self._on_label_changed)
        label_row.addWidget(self.label_combo)
        vis_lay.addLayout(label_row)

        layout.addWidget(vis_group)

        # ── Geometry Groups ──
        geom_group = QGroupBox("Geometry Groups")
        geom_lay = QGridLayout(geom_group)
        for g in range(6):
            cb = QCheckBox(f"Group {g}")
            cb.setChecked(True)
            cb.toggled.connect(self._make_geom_cb(g))
            geom_lay.addWidget(cb, g // 3, g % 3)
        layout.addWidget(geom_group)

        # ── Rendering quality / stereo / resolution ──
        trans_group = QGroupBox("Rendering")
        trans_lay = QVBoxLayout(trans_group)

        trans_lay.addWidget(QLabel("Stereo:"))
        self.scheme_combo = QComboBox()
        self.scheme_combo.addItems(["None", "Side-by-Side", "Quad Buffered"])
        self.scheme_combo.currentIndexChanged.connect(self._on_scheme_changed)
        trans_lay.addWidget(self.scheme_combo)

        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Low", "Medium", "High"])
        self.quality_combo.setCurrentIndex(1)
        self.quality_combo.currentIndexChanged.connect(self._on_quality_changed)
        quality_row.addWidget(self.quality_combo)
        trans_lay.addLayout(quality_row)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Render Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["1x (Native)", "1.5x", "2x (High Res)"])
        self.scale_combo.setCurrentIndex(0)
        self.scale_combo.currentIndexChanged.connect(self._on_scale_changed)
        scale_row.addWidget(self.scale_combo)
        trans_lay.addLayout(scale_row)

        layout.addWidget(trans_group)

        layout.addStretch()

    # ── Camera population ─────────────────────────────────────

    def populate_cameras(self, model):
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        self.cam_combo.addItem("Free Camera")
        if model is not None:
            for i in range(model.ncam):
                self.cam_combo.addItem(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                    or f"camera_{i}"
                )
        self.cam_combo.blockSignals(False)
        self.cam_combo.setCurrentIndex(0)

        self.follow_combo.blockSignals(True)
        self.follow_combo.clear()
        self.follow_combo.addItem("None")
        if model is not None:
            for i in range(1, model.nbody):
                name = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                    or f"body_{i}"
                )
                self.follow_combo.addItem(name)
        self.follow_combo.blockSignals(False)
        self.follow_combo.setCurrentIndex(0)

    # ── Camera callbacks ──────────────────────────────────────

    def _on_camera_changed(self, idx):
        if idx <= 0:
            self.viewport.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewport.cam.fixedcamid = -1
        else:
            self.viewport.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewport.cam.fixedcamid = idx - 1
        self.viewport.render()

    def _on_preset_changed(self, idx):
        if 0 <= idx < len(CAMERA_PRESETS):
            _, az, el = CAMERA_PRESETS[idx]
            self.viewport.set_camera_preset(az, el)

    def _on_follow_changed(self, idx):
        if idx <= 0:
            self.viewport.set_follow_body(-1)
        else:
            self.viewport.set_follow_body(idx)

    def _fit_camera(self):
        self.viewport.reset_camera()
        self.cam_combo.setCurrentIndex(0)
        self.viewport.render()

    def _center_camera(self):
        if self.viewport.model is not None:
            self.viewport.cam.lookat = self.viewport.model.stat.center.copy()
            self.viewport.render()

    # ── Camera bookmark actions ───────────────────────────────

    def _save_bookmark(self):
        name = self.bm_name_input.text().strip()
        if not name:
            name, ok = QInputDialog.getText(self, "Bookmark Name", "Enter name:")
            if not ok or not name:
                return
        self.viewport.save_camera_bookmark(name)
        self._refresh_bookmark_combo()
        self.bm_name_input.clear()

    def _load_bookmark(self):
        name = self.bm_combo.currentText()
        if name:
            self.viewport.load_camera_bookmark(name)

    def _delete_bookmark(self):
        name = self.bm_combo.currentText()
        if name:
            self.viewport.delete_camera_bookmark(name)
            self._refresh_bookmark_combo()

    def _refresh_bookmark_combo(self):
        self.bm_combo.clear()
        for name in self.viewport.get_camera_bookmarks():
            self.bm_combo.addItem(name)

    # ── Trace callback ────────────────────────────────────────

    def _on_trace_toggled(self, checked):
        self.viewport._show_traces = checked
        if not checked:
            self.viewport._traces = {}
            self.viewport._trace_counter = 0
        self.viewport.render()

    # ── Stereo / label / quality / scale callbacks ────────────

    def _on_scheme_changed(self, idx):
        if self.viewport.renderer is None:
            return
        try:
            if idx == 0:
                self.viewport.renderer.scene.stereo = 0
            elif idx == 1:
                self.viewport.renderer.scene.stereo = 1
        except Exception:
            pass
        self.viewport.render()

    def _on_label_changed(self, idx):
        if 0 <= idx < len(LABEL_MODES):
            _, val = LABEL_MODES[idx]
            self.viewport.vopt.label = val
            self.viewport.render()

    def _on_quality_changed(self, idx):
        """Toggle shadow flag based on quality — uses safe accessor."""
        shadow_flag = safe_vis_flag("mjVIS_SHADOW")
        if shadow_flag is not None:
            try:
                self.viewport.vopt.flags[shadow_flag] = (idx > 0)
            except (IndexError, TypeError):
                pass
        self.viewport.render()

    def _on_scale_changed(self, idx):
        scales = [1.0, 1.5, 2.0]
        if 0 <= idx < len(scales):
            self.viewport.set_render_scale(scales[idx])

    # ── Visualization flag / geom group closures ──────────────

    def _make_flag_cb(self, flag):
        def cb(checked):
            try:
                self.viewport.vopt.flags[flag] = checked
            except (IndexError, TypeError):
                pass
            self.viewport.render()
        return cb

    def _make_geom_cb(self, group):
        def cb(checked):
            self.viewport.vopt.geomgroup[group] = checked
            self.viewport.render()
        return cb