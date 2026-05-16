"""Main application window — assembles all panels and runs the sim loop."""

import time
import os
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QToolBar, QComboBox,
    QSplitter, QScrollArea, QMessageBox, QTabWidget, QSizePolicy,
    QSpinBox,
)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QAction, QKeySequence, QShortcut

import mujoco
from examples import EXAMPLES
from widgets import ShortcutsDialog, LogPanel, log, log_emitter
from viewport import MujocoViewport
from joint_panel import JointPanel
from actuator_panel import ActuatorPanel
from scene_panels import BodyTreePanel, EnergyPanel, ContactsPanel
from config_panels import XMLEditor, RenderOptionsPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MuJoCo Viewer")
        self.resize(1280, 800)
        self.model, self.data = None, None
        self.playing = False
        self.speed_factor = 1.0
        self._sim_time_accumulator = 0.0
        self._current_xml = EXAMPLES["Demo Scene"]
        self._current_file_path = ""

        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.last_real_time = time.time()
        self.last_sim_time = 0.0
        self.rtf = 0.0

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_shortcuts()

        # ── Connect logger to GUI panel ──
        log_emitter.log_signal.connect(self.log_panel.append_log)
        log.info("MuJoCo Viewer starting up")
        log.info(f"MuJoCo version: {mujoco.__version__}")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

        self._load_xml_string(self._current_xml)

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.viewport = MujocoViewport()
        splitter.addWidget(self.viewport)

        right = QWidget()
        right.setMinimumWidth(260)
        right.setMaximumWidth(440)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.joint_panel = JointPanel()
        jscroll = QScrollArea()
        jscroll.setWidgetResizable(True)
        jscroll.setWidget(self.joint_panel)
        self.tabs.addTab(jscroll, "Joints")

        self.actuator_panel = ActuatorPanel()
        ascroll = QScrollArea()
        ascroll.setWidgetResizable(True)
        ascroll.setWidget(self.actuator_panel)
        self.tabs.addTab(ascroll, "Actuators")

        self.body_panel = BodyTreePanel()
        self.body_panel.focus_body.connect(self._focus_body)
        self.tabs.addTab(self.body_panel, "Bodies")

        self.energy_panel = EnergyPanel()
        escroll = QScrollArea()
        escroll.setWidgetResizable(True)
        escroll.setWidget(self.energy_panel)
        self.tabs.addTab(escroll, "Energy")

        self.contacts_panel = ContactsPanel()
        self.tabs.addTab(self.contacts_panel, "Contacts")

        self.xml_editor = XMLEditor()
        self.xml_editor.apply_requested.connect(self._apply_xml)
        self.tabs.addTab(self.xml_editor, "XML Editor")

        self.options_panel = RenderOptionsPanel(self.viewport)
        oscroll = QScrollArea()
        oscroll.setWidgetResizable(True)
        oscroll.setWidget(self.options_panel)
        self.tabs.addTab(oscroll, "Options")

        # ── Log tab ──
        self.log_panel = LogPanel()
        self.tabs.addTab(self.log_panel, "Log")

        right_lay.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 320])
        self.setCentralWidget(splitter)

        self.status_time = QLabel("Time: 0.000s")
        self.status_step = QLabel("Step: 0")
        self.status_rtf = QLabel("RTF: —")
        self.status_fps = QLabel("FPS: —")
        self.status_info = QLabel("")
        self.statusBar().addPermanentWidget(self.status_time)
        self.statusBar().addPermanentWidget(self.status_step)
        self.statusBar().addPermanentWidget(self.status_rtf)
        self.statusBar().addPermanentWidget(self.status_fps)
        self.statusBar().addPermanentWidget(self.status_info)

        self.viewport.body_focused.connect(self._focus_body)

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        load_act = QAction("&Load Model…", self)
        load_act.setShortcut(QKeySequence.Open)
        load_act.triggered.connect(self._load_model_file)
        file_menu.addAction(load_act)
        file_menu.addSeparator()
        save_state_act = QAction("Save &State…", self)
        save_state_act.triggered.connect(self._save_state)
        file_menu.addAction(save_state_act)
        load_state_act = QAction("Load S&tate…", self)
        load_state_act.triggered.connect(self._load_state)
        file_menu.addAction(load_state_act)
        file_menu.addSeparator()
        screenshot_act = QAction("Save &Screenshot…", self)
        screenshot_act.setShortcut(QKeySequence("Ctrl+S"))
        screenshot_act.triggered.connect(self._screenshot)
        file_menu.addAction(screenshot_act)
        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        sim_menu = menubar.addMenu("&Simulation")
        self.play_act = QAction("▶  &Play", self)
        self.play_act.triggered.connect(self._toggle_play)
        sim_menu.addAction(self.play_act)
        step_act = QAction("⏭  S&tep", self)
        step_act.triggered.connect(self._step_once)
        sim_menu.addAction(step_act)
        reset_act = QAction("⏮  &Reset", self)
        reset_act.triggered.connect(self._reset_sim)
        sim_menu.addAction(reset_act)

        help_menu = menubar.addMenu("&Help")
        shortcuts_act = QAction("&Keyboard Shortcuts", self)
        shortcuts_act.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_act)
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _build_toolbar(self):
        tb = QToolBar("Controls")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        self.addToolBar(tb)

        self.btn_play = QPushButton("▶  Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setMinimumWidth(90)
        self.btn_play.clicked.connect(self._toggle_play)
        tb.addWidget(self.btn_play)

        self.btn_step = QPushButton("⏭  Step")
        self.btn_step.clicked.connect(self._step_once)
        tb.addWidget(self.btn_step)

        self.btn_reset = QPushButton("⏮  Reset")
        self.btn_reset.clicked.connect(self._reset_sim)
        tb.addWidget(self.btn_reset)

        tb.addSeparator()

        tb.addWidget(QLabel("  Speed: "))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(
            ["0.1x", "0.25x", "0.5x", "1x", "2x", "4x", "8x", "10x"]
        )
        self.speed_combo.setCurrentIndex(3)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        tb.addWidget(self.speed_combo)

        tb.addSeparator()

        tb.addWidget(QLabel("  Step×: "))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 500)
        self.step_spin.setValue(1)
        self.step_spin.setToolTip("Number of physics steps per Step click")
        tb.addWidget(self.step_spin)

        tb.addSeparator()

        tb.addWidget(QLabel("  Example: "))
        self.example_combo = QComboBox()
        self.example_combo.addItems(EXAMPLES.keys())
        self.example_combo.currentTextChanged.connect(self._load_example)
        tb.addWidget(self.example_combo)

        tb.addSeparator()

        self.btn_load = QPushButton("📂  Load File")
        self.btn_load.clicked.connect(self._load_model_file)
        tb.addWidget(self.btn_load)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

        self.btn_trace = QPushButton("◉  Traces")
        self.btn_trace.setCheckable(True)
        self.btn_trace.setToolTip("Toggle body traces (T)")
        self.btn_trace.clicked.connect(self._toggle_traces)
        tb.addWidget(self.btn_trace)

        self.btn_screenshot = QPushButton("📸  Screenshot")
        self.btn_screenshot.clicked.connect(self._screenshot)
        tb.addWidget(self.btn_screenshot)

        self.btn_fit = QPushButton("🎯  Fit Camera")
        self.btn_fit.clicked.connect(self._fit_camera)
        tb.addWidget(self.btn_fit)

    def _build_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._step_once)
        QShortcut(QKeySequence(Qt.Key_R), self, self._reset_sim)
        QShortcut(QKeySequence(Qt.Key_F), self, self._fit_camera)
        QShortcut(QKeySequence(Qt.Key_T), self, self._toggle_traces_key)
        QShortcut(QKeySequence(Qt.Key_C), self, self._clear_traces)
        QShortcut(QKeySequence("1"), self, lambda: self._set_speed_index(0))
        QShortcut(QKeySequence("2"), self, lambda: self._set_speed_index(1))
        QShortcut(QKeySequence("3"), self, lambda: self._set_speed_index(2))
        QShortcut(QKeySequence("4"), self, lambda: self._set_speed_index(3))
        QShortcut(QKeySequence("5"), self, lambda: self._set_speed_index(4))
        QShortcut(QKeySequence("6"), self, lambda: self._set_speed_index(5))
        QShortcut(QKeySequence("7"), self, lambda: self._set_speed_index(6))
        QShortcut(QKeySequence("8"), self, lambda: self._set_speed_index(7))

    # ────────────────────────────────────────────────────────────
    #  Model loading
    # ────────────────────────────────────────────────────────────

    def _load_example(self, name):
        if name in EXAMPLES:
            self._load_xml_string(EXAMPLES[name])

    def _load_xml_string(self, xml_string: str):
        try:
            model = mujoco.MjModel.from_xml_string(xml_string)
            data = mujoco.MjData(model)
        except Exception as e:
            log.error(f"XML parse error: {e}")
            QMessageBox.critical(self, "Model Error", f"Failed to parse XML:\n{e}")
            self.xml_editor.set_validation(False, str(e)[:80])
            return
        self._current_xml = xml_string
        self._current_file_path = ""
        self.xml_editor.set_xml(xml_string)
        self.xml_editor.set_validation(True, "Loaded")
        log.info("Model loaded from XML string")
        self._set_model(model, data)

    def _load_model_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load MuJoCo Model", "", "MuJoCo XML (*.xml);;All Files (*)"
        )
        if not path:
            return
        self._load_model_path(path)

    def _load_model_path(self, path: str):
        if not os.path.isfile(path):
            log.error(f"File not found: {path}")
            QMessageBox.critical(self, "Load Error", f"File not found: {path}")
            return
        try:
            model = mujoco.MjModel.from_xml_path(path)
            data = mujoco.MjData(model)
        except Exception as e:
            log.error(f"Failed to load model from {path}: {e}")
            QMessageBox.critical(self, "Load Error", f"Failed:\n{e}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                xml_string = f.read()
        except Exception as e:
            log.warning(f"Could not read XML source: {e}")
            xml_string = f"<!-- Loaded from {path} -->"
        self._current_xml = xml_string
        self._current_file_path = path
        self.xml_editor.set_xml(xml_string)
        self.xml_editor.set_validation(True, f"Loaded from {os.path.basename(path)}")
        log.info(f"Model loaded from file: {os.path.basename(path)}")
        self._set_model(model, data)
        self.viewport._toast.show_success(f"Loaded: {os.path.basename(path)}")

    def _apply_xml(self):
        xml_string = self.xml_editor.editor.toPlainText()
        try:
            model = mujoco.MjModel.from_xml_string(xml_string)
            data = mujoco.MjData(model)
        except Exception as e:
            log.error(f"XML apply error: {e}")
            self.xml_editor.set_validation(False, str(e)[:100])
            self.viewport._toast.show_error("XML parse error")
            return
        self._current_xml = xml_string
        self.xml_editor.set_xml(xml_string)
        self.xml_editor.set_validation(True, "Applied")
        log.info("XML changes applied successfully")
        self._set_model(model, data)
        self.viewport._toast.show_success("XML applied successfully")

    # ────────────────────────────────────────────────────────────
    #  Model setup
    # ────────────────────────────────────────────────────────────

    def _set_model(self, model, data):
        self.model, self.data = model, data
        try:
            mujoco.mj_forward(model, data)
        except Exception as e:
            log.error(f"mj_forward failed on model set: {e}")
        self.playing = False
        self.btn_play.setChecked(False)
        self.btn_play.setText("▶  Play")
        self._sim_time_accumulator = 0.0

        self.viewport.set_model(model, data)

        try:
            self.joint_panel.build(model, data)
        except Exception as e:
            log.error(f"Joint panel build error: {e}")

        try:
            self.actuator_panel.build(model, data)
        except Exception as e:
            log.error(f"Actuator panel build error: {e}")

        try:
            self.body_panel.build(model)
        except Exception as e:
            log.error(f"Body panel build error: {e}")

        try:
            self.options_panel.populate_cameras(model)
        except Exception as e:
            log.error(f"Options panel camera populate error: {e}")

        self.status_info.setText(
            f"Bodies: {model.nbody} | Joints: {model.njnt} | "
            f"Actuators: {model.nu} | DoFs: {model.nv}"
        )
        log.info(
            f"Model stats — bodies:{model.nbody} joints:{model.njnt} "
            f"actuators:{model.nu} dofs:{model.nv}"
        )
        self.last_sim_time = data.time
        self.last_real_time = time.time()

    # ────────────────────────────────────────────────────────────
    #  Simulation controls
    # ────────────────────────────────────────────────────────────

    def _toggle_play(self):
        if not self.model:
            return
        self.playing = not self.playing
        self.btn_play.setChecked(self.playing)
        self.btn_play.setText("⏸  Pause" if self.playing else "▶  Play")
        self.viewport._paused = not self.playing
        if self.playing:
            self.last_real_time = time.time()
            self.last_sim_time = self.data.time
            self._sim_time_accumulator = 0.0
            log.info("Simulation playing")
        else:
            log.info("Simulation paused")

    def _step_once(self):
        if self.model and self.data:
            steps = self.step_spin.value()
            try:
                for _ in range(steps):
                    mujoco.mj_step(self.model, self.data)
                self._update_ui()
            except Exception as e:
                log.error(f"Simulation step error: {e}")

    def _reset_sim(self):
        if self.model and self.data:
            try:
                mujoco.mj_resetData(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)
            except Exception as e:
                log.error(f"Reset error: {e}")
            self.playing = False
            self.btn_play.setChecked(False)
            self.btn_play.setText("▶  Play")
            self.viewport._paused = False
            self._sim_time_accumulator = 0.0
            self.last_sim_time = self.data.time
            self.last_real_time = time.time()
            self.viewport.clear_traces()
            self._update_ui()
            self.viewport._toast.show_message("Simulation reset")
            log.info("Simulation reset")

    def _speed_changed(self, index):
        speeds = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0]
        if 0 <= index < len(speeds):
            self.speed_factor = speeds[index]
            log.debug(f"Speed set to {speeds[index]}x")

    def _set_speed_index(self, idx):
        self.speed_combo.setCurrentIndex(idx)

    def _fit_camera(self):
        self.viewport.reset_camera()
        self.options_panel.cam_combo.setCurrentIndex(0)
        self.viewport.render()

    def _focus_body(self, body_id):
        self.viewport.focus_on_body(body_id)

    def _toggle_traces(self):
        self.viewport.toggle_traces()
        self.btn_trace.setChecked(self.viewport._show_traces)
        self.options_panel.trace_cb.setChecked(self.viewport._show_traces)

    def _toggle_traces_key(self):
        self._toggle_traces()

    def _clear_traces(self):
        self.viewport.clear_traces()

    # ────────────────────────────────────────────────────────────
    #  Save / Load
    # ────────────────────────────────────────────────────────────

    def _save_state(self):
        if self.model is None:
            log.warning("Save state requested but no model loaded")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save State", "state.npz", "NumPy Archive (*.npz)"
        )
        if not path:
            return
        try:
            np.savez(
                path,
                qpos=self.data.qpos.copy(),
                qvel=self.data.qvel.copy(),
                ctrl=self.data.ctrl.copy(),
                act=self.data.act.copy(),
                time=np.array([self.data.time]),
            )
            self.viewport._toast.show_success(f"State saved: {os.path.basename(path)}")
            log.info(f"State saved to {os.path.basename(path)}")
        except Exception as e:
            log.error(f"Save state error: {e}")
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_state(self):
        if self.model is None:
            log.warning("Load state requested but no model loaded")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load State", "", "NumPy Archive (*.npz);;All Files (*)"
        )
        if not path:
            return
        try:
            state = np.load(path)
            if len(state["qpos"]) == len(self.data.qpos):
                self.data.qpos[:] = state["qpos"]
            else:
                log.warning(
                    f"qpos size mismatch: expected {len(self.data.qpos)}, "
                    f"got {len(state['qpos'])}"
                )
            if len(state["qvel"]) == len(self.data.qvel):
                self.data.qvel[:] = state["qvel"]
            if "ctrl" in state and len(state["ctrl"]) == len(self.data.ctrl):
                self.data.ctrl[:] = state["ctrl"]
            if "act" in state and len(state["act"]) == len(self.data.act):
                self.data.act[:] = state["act"]
            mujoco.mj_forward(self.model, self.data)
            self.actuator_panel.build(self.model, self.data)
            self._update_ui()
            self.viewport._toast.show_success(f"State loaded: {os.path.basename(path)}")
            log.info(f"State loaded from {os.path.basename(path)}")
        except Exception as e:
            log.error(f"Load state error: {e}")
            QMessageBox.critical(self, "Load State Error", str(e))

    def _screenshot(self):
        if self.viewport._image is None:
            log.warning("Screenshot requested but no rendered image")
            return
        default_name = f"mujoco_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", default_name, "PNG Image (*.png)"
        )
        if path:
            if self.viewport._image.save(path):
                self.viewport._toast.show_success(
                    f"Screenshot saved: {os.path.basename(path)}"
                )
                log.info(f"Screenshot saved: {os.path.basename(path)}")
            else:
                self.viewport._toast.show_error("Failed to save screenshot")
                log.error("Failed to save screenshot image")

    def _show_shortcuts(self):
        dlg = ShortcutsDialog(self)
        dlg.exec()

    def _show_about(self):
        QMessageBox.about(
            self,
            "About MuJoCo Viewer",
            f"<h3>MuJoCo Viewer</h3>"
            f"<p>A professional physics simulation viewer built with "
            f"PySide6 and MuJoCo.</p>"
            f"<p>MuJoCo version: {mujoco.__version__}</p>"
            f"<p><b>Controls:</b> Left-drag = rotate, "
            f"Middle-drag = pan, Right-drag/scroll = zoom, "
            f"Ctrl+Click = select & drag body</p>",
        )

    # ────────────────────────────────────────────────────────────
    #  Main tick
    # ────────────────────────────────────────────────────────────

    def _tick(self):
        if self.model and self.data:
            if self.playing:
                now = time.time()
                real_dt = min(now - self.last_real_time, 0.1)
                self.last_real_time = now
                self._sim_time_accumulator += real_dt * self.speed_factor

                max_steps = 50
                step_count = 0
                timestep = self.model.opt.timestep
                try:
                    while (
                        self._sim_time_accumulator >= timestep
                        and step_count < max_steps
                    ):
                        mujoco.mj_step(self.model, self.data)
                        self._sim_time_accumulator -= timestep
                        step_count += 1
                except Exception as e:
                    log.error(f"Simulation step error during playback: {e}")
                    self.playing = False
                    self.btn_play.setChecked(False)
                    self.btn_play.setText("▶  Play")
                    self.viewport._paused = True

                if step_count >= max_steps:
                    self._sim_time_accumulator = 0.0

                self.viewport.record_trace()

            self._update_ui()
            self._calc_fps()

    def _update_ui(self):
        self.viewport.render()
        try:
            self.joint_panel.refresh()
        except Exception as e:
            log.debug(f"Joint panel refresh error: {e}")
        try:
            self.energy_panel.refresh(self.model, self.data)
        except Exception as e:
            log.debug(f"Energy panel refresh error: {e}")
        try:
            self.contacts_panel.refresh(self.model, self.data)
        except Exception as e:
            log.debug(f"Contacts panel refresh error: {e}")
        self.status_time.setText(f"Time: {self.data.time:.3f}s")

    def _calc_fps(self):
        self.frame_count += 1
        now = time.time()

        sim_dt = self.data.time - self.last_sim_time
        real_dt = now - self.last_real_time
        if real_dt > 0.2:
            self.rtf = sim_dt / real_dt if real_dt > 0 else 0
            self.last_sim_time = self.data.time
            self.last_real_time = now
            self.status_rtf.setText(f"RTF: {self.rtf:.2f}x")

        elapsed = now - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = now
            self.status_fps.setText(f"FPS: {self.fps:.0f}")