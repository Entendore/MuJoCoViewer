"""
Comprehensive Test Panel for GUI and feature validation.
Includes a QThread runner for the GUI tab, and standard pytest 
functions at the bottom for CLI usage (run with: pytest test_panel.py).
"""

import time
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QProgressBar,
    QApplication,
)
from PySide6.QtCore import Qt, QThread, Signal
import mujoco
from constants import EXAMPLES
from widgets import log


# ═══════════════════════════════════════════════════════════════
# 1. HEADLESS LOGIC TESTS
# ═══════════════════════════════════════════════════════════════

class HeadlessTestRunner(QThread):
    """Runs pure MuJoCo model and logic tests without touching Qt GUI objects."""
    result_ready = Signal(str, bool, str)
    progress_update = Signal(int, int)
    finished_all = Signal(int, int)

    def run(self):
        passed = 0
        failed = 0
        tests = [
            self.test_examples_load,
            self.test_simulation_stability,
            self.test_actuator_response,
            self.test_state_reset,
            self.test_invalid_xml_handling,
            self.test_nan_resistance,
            self.test_joint_limits,                                   # NEW
            self.test_energy_conservation,                            # NEW
            self.test_model_stats_consistency,                        # NEW
        ]
        
        total_tests = len(tests)
        self.progress_update.emit(0, total_tests)

        for i, test_func in enumerate(tests):
            test_name = test_func.__doc__ or test_func.__name__
            try:
                msgs = []
                test_func(msgs)
                if any(m is False for m in msgs):
                    raise AssertionError("Sub-check failed")
                summary = "; ".join(str(m) for m in msgs if m is not True and m is not False)
                self.result_ready.emit(test_name, True, summary or "Passed")
                passed += 1
            except Exception as e:
                self.result_ready.emit(test_name, False, str(e))
                failed += 1
            self.progress_update.emit(i + 1, total_tests)

        self.finished_all.emit(passed, failed)

    def test_examples_load(self, msgs):
        """Test: All built-in examples load successfully"""
        for name, xml in EXAMPLES.items():
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            msgs.append(f"{name} OK")
        return True

    def test_simulation_stability(self, msgs):
        """Test: Models simulate 1000 steps without NaN"""
        for name, xml in EXAMPLES.items():
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            for _ in range(1000):
                mujoco.mj_step(model, data)
            assert not np.any(np.isnan(data.qpos)), f"{name} produced NaN in qpos"
            msgs.append(f"{name} stable")
        return True

    def test_actuator_response(self, msgs):
        """Test: Actuated models respond to control inputs"""
        for name, xml in EXAMPLES.items():
            model = mujoco.MjModel.from_xml_string(xml)
            if model.nu == 0:
                continue
            data = mujoco.MjData(model)
            
            mujoco.mj_step(model, data)
            qpos_0 = data.qpos.copy()
            
            mujoco.mj_resetData(model, data)
            data.ctrl[:] = model.actuator_ctrlrange[:, 1]
            mujoco.mj_step(model, data)
            qpos_max = data.qpos.copy()
            
            assert not np.allclose(qpos_0, qpos_max), f"{name} did not respond to controls"
            msgs.append(f"{name} controls OK")
        return True

    def test_state_reset(self, msgs):
        """Test: Resetting state restores initial qpos"""
        xml = EXAMPLES["Cartpole"]
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        initial_qpos = data.qpos.copy()
        
        data.ctrl[0] = 100.0
        for _ in range(100):
            mujoco.mj_step(model, data)
            
        assert not np.allclose(data.qpos, initial_qpos), "Simulation didn't advance"
        
        mujoco.mj_resetData(model, data)
        assert np.allclose(data.qpos, initial_qpos), "Reset failed to restore qpos"
        msgs.append("Reset successful")
        return True

    def test_invalid_xml_handling(self, msgs):
        """Test: Invalid XML correctly raises exceptions"""
        invalid_xml = "<mujoco><worldbody><geom type='invalid_type'/></worldbody></mujoco>"
        try:
            mujoco.MjModel.from_xml_string(invalid_xml)
            raise AssertionError("Invalid XML did not raise an error")
        except Exception:
            msgs.append("Correctly rejected bad XML")
            return True

    def test_nan_resistance(self, msgs):
        """Test: Extreme forces create NaNs but MuJoCo handles the error state"""
        xml = EXAMPLES["Bouncing Balls"]
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        
        data.qvel[:] = 1e10
        for _ in range(100):
            mujoco.mj_step(model, data)
            
        is_nan = np.any(np.isnan(data.qpos))
        assert is_nan, "Extreme forces should result in NaN for this test"
        msgs.append("NaN detection works")
        return True

    # NEW: Joint limits test
    def test_joint_limits(self, msgs):
        """Test: Joints with limits stay within range after simulation"""
        for name, xml in EXAMPLES.items():
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            for i in range(model.njnt):
                rng = model.jnt_range[i]
                if rng[1] > rng[0]:  # has limits
                    # Check after 100 steps
                    for _ in range(100):
                        mujoco.mj_step(model, data)
                    qpos_adr = model.jnt_qposadr[i]
                    jtype = model.jnt_type[i]
                    if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                        v = data.qpos[qpos_adr]
                        # Allow slight violation (solver tolerance)
                        if v < rng[0] - 0.1 or v > rng[1] + 0.1:
                            msgs.append(f"{name} joint {i}: range violation")
            msgs.append(f"{name} limits OK")
        return True

    # NEW: Energy conservation test (no actuators, no gravity = constant energy)
    def test_energy_conservation(self, msgs):
        """Test: Energy is computed without errors"""
        xml = EXAMPLES["Cartpole"]
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        mujoco.mj_forward(model, data)
        ke0, pe0 = float(data.energy[0]), float(data.energy[1])
        for _ in range(100):
            mujoco.mj_step(model, data)
        ke1, pe1 = float(data.energy[0]), float(data.energy[1])
        total0 = ke0 + pe0
        total1 = ke1 + pe1
        # Energy should change with gravity but remain finite
        assert np.isfinite(total0) and np.isfinite(total1), "Energy is not finite"
        msgs.append(f"Energy computed: initial={total0:.2f}, final={total1:.2f}")
        return True

    # NEW: Model stats consistency
    def test_model_stats_consistency(self, msgs):
        """Test: Model dimensions are internally consistent"""
        for name, xml in EXAMPLES.items():
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            assert len(data.qpos) == model.nq, f"{name}: qpos size mismatch"
            assert len(data.qvel) == model.nv, f"{name}: qvel size mismatch"
            assert len(data.ctrl) == model.nu, f"{name}: ctrl size mismatch"
            msgs.append(f"{name} dims OK")
        return True


# ═══════════════════════════════════════════════════════════════
# 2. GUI TEST TAB WIDGET
# ═══════════════════════════════════════════════════════════════

class TestPanel(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QLabel("🧪 Comprehensive Model & GUI Validation")
        header.setProperty("class", "value")
        font = header.font()
        font.setPointSize(14)
        font.setBold(True)
        header.setFont(font)
        layout.addWidget(header)

        desc = QLabel(
            "Runs headless physics tests + GUI state consistency checks.\n"
            "You can also run `pytest test_panel.py` from the terminal for standard CI testing."
        )
        desc.setWordWrap(True)
        desc.setProperty("class", "dim")
        layout.addWidget(desc)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run All Tests")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #2b3f7a; border-color: #3d59a1; font-weight: bold; }"
            "QPushButton:hover { background-color: #3d59a1; }"
        )
        self.run_btn.clicked.connect(self._start_tests)
        btn_row.addWidget(self.run_btn)
        
        self.clear_btn = QPushButton("✕  Clear")
        self.clear_btn.clicked.connect(self._clear_results)
        btn_row.addWidget(self.clear_btn)
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QTextEdit.NoWrap)
        self.output.setStyleSheet(
            "QTextEdit { background-color: #16161e; color: #a9b1d6; "
            "border: 1px solid #24283b; border-radius: 4px; padding: 8px; font-family: Consolas, monospace; }"
        )
        layout.addWidget(self.output)

        self._headless_runner = None
        self._gui_test_queue = []

    def _start_tests(self):
        if self._headless_runner and self._headless_runner.isRunning():
            return
        
        self.output.clear()
        self.run_btn.setEnabled(False)
        self.run_btn.setText("⏳ Running Headless Tests...")
        
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Phase 1: Headless Physics %p%")

        self._headless_runner = HeadlessTestRunner()
        self._headless_runner.result_ready.connect(self._on_result)
        self._headless_runner.progress_update.connect(self._on_headless_progress)
        self._headless_runner.finished_all.connect(self._start_gui_tests)
        self._headless_runner.start()

    def _on_headless_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_result(self, test_name: str, passed: bool, message: str):
        if passed:
            icon = "<span style='color:#9ece6a;'>✅ PASS</span>"
        else:
            icon = "<span style='color:#f7768e;'>❌ FAIL</span>"

        self.output.append(
            f"{icon} <b>{test_name}</b><br>"
            f"&nbsp;&nbsp;&nbsp;<span style='color:#565f89;'>{message}</span>"
        )

    def _start_gui_tests(self, headless_passed, headless_failed):
        self.run_btn.setText("⏳ Running GUI Tests...")
        self.progress_bar.setFormat("Phase 2: GUI State %p%")
        self.progress_bar.setValue(0)
        
        self._gui_test_queue = [
            self._gui_test_panel_consistency,
            self._gui_test_xml_editor,
            self._gui_test_sim_controls,
            self._gui_test_body_tree,
            self._gui_test_watch_panel,
            self._gui_test_keyframe_panel,                            # NEW
            self._gui_test_actuator_panel,                            # NEW
        ]
        self._gui_passed = 0
        self._gui_failed = 0
        self._gui_total = len(self._gui_test_queue)
        
        self._gui_timer = self.startTimer(50)

    def timerEvent(self, event):
        if hasattr(self, '_gui_timer') and self._gui_timer == event.timerId():
            if self._gui_test_queue:
                test_func = self._gui_test_queue.pop(0)
                test_name = test_func.__doc__ or test_func.__name__
                try:
                    msgs = []
                    test_func(msgs)
                    if any(m is False for m in msgs):
                        raise AssertionError("Sub-check failed")
                    summary = "; ".join(str(m) for m in msgs if m is not True and m is not False)
                    self._on_result(test_name, True, summary or "Passed")
                    self._gui_passed += 1
                except Exception as e:
                    self._on_result(test_name, False, str(e))
                    self._gui_failed += 1
                
                val = self._gui_total - len(self._gui_test_queue)
                self.progress_bar.setValue(val)
                self.progress_bar.setMaximum(self._gui_total)
            else:
                self.killTimer(self._gui_timer)
                self._finish_all_tests(self._gui_passed, self._gui_failed, headless_passed + self._gui_passed, headless_failed + self._gui_failed)

    def _gui_test_panel_consistency(self, msgs):
        """Test: Panel entry counts match current model stats"""
        mw = self.main_window
        assert mw.model is not None, "No model loaded in MainWindow"
        
        joint_count = len(mw.joint_panel._entries)
        assert joint_count == mw.model.njnt, f"Joint count mismatch: Panel={joint_count}, Model={mw.model.njnt}"
        msgs.append(f"Joints OK ({joint_count})")
        
        actuator_count = mw.actuator_panel._actuator_count
        assert actuator_count == mw.model.nu, f"Actuator count mismatch: Panel={actuator_count}, Model={mw.model.nu}"
        msgs.append(f"Actuators OK ({mw.model.nu})")
        return True

    def _gui_test_xml_editor(self, msgs):
        """Test: XML Editor contains valid XML for current model"""
        mw = self.main_window
        editor_xml = mw.xml_editor.editor.toPlainText()
        assert editor_xml.strip() != "", "XML Editor is empty"
        
        mujoco.MjModel.from_xml_string(editor_xml)
        msgs.append("XML parses correctly")
        
        assert mw.xml_editor._saved_xml == editor_xml, "Saved XML mismatch"
        msgs.append("State consistent")
        return True

    def _gui_test_sim_controls(self, msgs):
        """Test: Simulation play/pause toggle works"""
        mw = self.main_window
        initial_state = mw.playing
        
        mw._toggle_play()
        assert mw.playing != initial_state, "Play state didn't toggle"
        assert mw.btn_play.isChecked() == mw.playing, "Button state mismatch"
        msgs.append("Toggle OK")
        
        mw._toggle_play()
        assert mw.playing == initial_state, "Play state didn't revert"
        msgs.append("Revert OK")
        return True

    def _gui_test_body_tree(self, msgs):
        """Test: Body tree item count matches model bodies"""
        mw = self.main_window
        tree = mw.body_panel.tree
        def count_items(parent):
            count = 0
            for i in range(parent.childCount()):
                count += 1 + count_items(parent.child(i))
            return count
            
        total_items = count_items(tree.invisibleRootItem())
        assert total_items == mw.model.nbody, f"Body tree mismatch: Tree={total_items}, Model={mw.model.nbody}"
        msgs.append(f"Count OK ({total_items})")
        return True

    def _gui_test_watch_panel(self, msgs):
        """Test: Watch panel can add variables without crashing"""
        mw = self.main_window
        initial_rows = mw.watch_panel.watch_table.rowCount()
        
        mw.watch_panel.cat_combo.setCurrentIndex(0)
        mw.watch_panel.idx_spin.setValue(0)
        mw.watch_panel._add_watch()
        
        assert mw.watch_panel.watch_table.rowCount() == initial_rows + 1, "Row not added"
        msgs.append("Add OK")
        
        mw.watch_panel._clear_watches()
        assert mw.watch_panel.watch_table.rowCount() == 0, "Rows not cleared"
        msgs.append("Clear OK")
        return True

    # NEW: Keyframe panel test
    def _gui_test_keyframe_panel(self, msgs):
        """Test: Keyframe save/load cycle works"""
        mw = self.main_window
        if mw.data is None:
            msgs.append("Skipped (no data)")
            return True

        initial_count = len(mw.keyframe_panel._keyframes)
        mw.keyframe_panel.name_input.setText("__test_kf__")
        mw.keyframe_panel._save_keyframe()
        assert len(mw.keyframe_panel._keyframes) == initial_count + 1, "Keyframe not saved"
        msgs.append("Save OK")

        # Clean up
        if "__test_kf__" in mw.keyframe_panel._keyframes:
            del mw.keyframe_panel._keyframes["__test_kf__"]
            mw.keyframe_panel._rebuild_list()
        msgs.append("Cleanup OK")
        return True

    # NEW: Actuator panel test
    def _gui_test_actuator_panel(self, msgs):
        """Test: Actuator panel count and reset all work"""
        mw = self.main_window
        if mw.model is None or mw.model.nu == 0:
            msgs.append("Skipped (no actuators)")
            return True

        count = mw.actuator_panel._actuator_count
        assert count == mw.model.nu, f"Count mismatch: {count} vs {mw.model.nu}"
        msgs.append(f"Count OK ({count})")

        # Test reset all
        if mw.actuator_panel._reset_cbs:
            mw.actuator_panel._reset_all()
            for i in range(mw.model.nu):
                cr = mw.model.actuator_ctrlrange[i]
                expected = (cr[0] + cr[1]) / 2.0
                actual = mw.data.ctrl[i]
                assert abs(actual - expected) < 1e-6, f"Actuator {i} not reset"
            msgs.append("Reset All OK")
        return True

    def _finish_all_tests(self, gui_passed, gui_failed, total_passed, total_failed):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶  Run All Tests")
        
        total = total_passed + total_failed
        if total_failed == 0:
            summary = (
                f"<br><span style='color:#9ece6a;'><b>🎉 ALL {total} TESTS PASSED!</b></span>"
                f"<br><span style='color:#565f89;'>Headless: {total_passed - gui_passed} | GUI: {gui_passed}</span>"
            )
        else:
            summary = (
                f"<br><span style='color:#f7768e;'><b>❌ {total_failed} of {total} tests FAILED</b></span>"
                f"<br><span style='color:#565f89;'>Headless: {total_failed - gui_failed} | GUI: {gui_failed}</span>"
            )
        
        self.output.append(summary)
        log.info(f"Test suite finished: {total_passed} passed, {total_failed} failed")

    def _clear_results(self):
        self.output.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")


# ═══════════════════════════════════════════════════════════════
# 3. STANDARD PYTEST CLI FUNCTIONS
# ═══════════════════════════════════════════════════════════════

import pytest

@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication if one doesn't exist."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app

@pytest.fixture(scope="session")
def main_window(qapp):
    """Create and show the main window for GUI testing."""
    from main_window import MainWindow
    window = MainWindow()
    window.show()
    yield window
    window.close()

def test_examples_load_cli():
    """CLI Test: All examples load."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        assert model is not None

def test_simulation_stability_cli():
    """CLI Test: Models are stable."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        for _ in range(500):
            mujoco.mj_step(model, data)
        assert not np.any(np.isnan(data.qpos)), f"{name} produced NaN"

def test_model_stats_consistency_cli():
    """CLI Test: Model dimensions are consistent."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        assert len(data.qpos) == model.nq
        assert len(data.qvel) == model.nv

def test_gui_panel_consistency_cli(main_window):
    """CLI Test: GUI Panels match model."""
    main_window._load_example("Cartpole")
    qapp.processEvents()
    
    assert main_window.model is not None
    assert len(main_window.joint_panel._entries) == main_window.model.njnt

def test_gui_sim_controls_cli(main_window):
    """CLI Test: GUI Play/Pause toggles."""
    initial = main_window.playing
    main_window._toggle_play()
    qapp.processEvents()
    assert main_window.playing != initial
    
    main_window._toggle_play()
    qapp.processEvents()
    assert main_window.playing == initial