"""
Standalone pytest suite for MuJoCo Viewer.
Run:  pytest test_mujoco_viewer.py -v
"""

import numpy as np
import mujoco
from constants import EXAMPLES, ACTUATOR_TYPE_NAMES


# ═══════════════════════════════════════════════════════════════
# Pure MuJoCo model / physics tests (no GUI)
# ═══════════════════════════════════════════════════════════════

def test_all_examples_load():
    """Every built-in example XML must compile to a valid MjModel."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        assert model is not None
        assert model.nq > 0


def test_all_examples_step_without_nan():
    """Every example survives 1000 steps without NaN in qpos."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        for _ in range(1000):
            mujoco.mj_step(model, data)
        assert not np.any(np.isnan(data.qpos)), f"{name} → NaN in qpos"


def test_actuator_response():
    """Actuated models change state when given max control vs zero."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        if model.nu == 0:
            continue
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        q0 = data.qpos.copy()

        mujoco.mj_resetData(model, data)
        data.ctrl[:] = model.actuator_ctrlrange[:, 1]
        mujoco.mj_step(model, data)
        q1 = data.qpos.copy()

        assert not np.allclose(q0, q1), f"{name} ignored control input"


def test_state_reset_roundtrip():
    """mj_resetData restores the default qpos."""
    xml = EXAMPLES["Cartpole"]
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    q0 = data.qpos.copy()

    data.ctrl[0] = 100.0
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert not np.allclose(data.qpos, q0), "sim didn't advance"

    mujoco.mj_resetData(model, data)
    assert np.allclose(data.qpos, q0), "reset didn't restore qpos"


def test_invalid_xml_raises():
    """Bad XML must raise, not silently return None."""
    bad = "<mujoco><worldbody><geom type='nope'/></worldbody></mujoco>"
    raised = False
    try:
        mujoco.MjModel.from_xml_string(bad)
    except Exception:
        raised = True
    assert raised, "invalid XML did not raise"


def test_nan_detection():
    """Extreme velocity → NaN, and we can detect it."""
    model = mujoco.MjModel.from_xml_string(EXAMPLES["Bouncing Balls"])
    data = mujoco.MjData(model)
    data.qvel[:] = 1e10
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert np.any(np.isnan(data.qpos)), "extreme vel should produce NaN"


def test_model_dimensions_consistent():
    """Model array sizes match nq, nv, nu."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        assert len(data.qpos) == model.nq, f"{name} qpos len ≠ nq"
        assert len(data.qvel) == model.nv, f"{name} qvel len ≠ nv"
        assert len(data.ctrl) == model.nu, f"{name} ctrl len ≠ nu"


def test_energy_flag_enables_energy():
    """Setting the energy flag makes data.energy finite."""
    model = mujoco.MjModel.from_xml_string(EXAMPLES["Cartpole"])
    data = mujoco.MjData(model)
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    mujoco.mj_forward(model, data)
    assert np.isfinite(data.energy[0]), "KE not finite"
    assert np.isfinite(data.energy[1]), "PE not finite"


def test_actuator_type_names_dict():
    """ACTUATOR_TYPE_NAMES covers the known actuator type integers."""
    for atype_int, name in ACTUATOR_TYPE_NAMES.items():
        assert isinstance(atype_int, int)
        assert isinstance(name, str)
        assert len(name) > 0


def test_model_actuator_type_uses_known_integers():
    """Every actuator_type value in every example is in ACTUATOR_TYPE_NAMES."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        if model.nu == 0:
            continue
        for i in range(model.nu):
            atype = int(model.actuator_type[i])
            assert atype in ACTUATOR_TYPE_NAMES, (
                f"{name} act {i} has unknown type {atype}"
            )


# ═══════════════════════════════════════════════════════════════
# GUI integration tests (requires display / xvfb)
# ═══════════════════════════════════════════════════════════════

try:
    from PySide6.QtWidgets import QApplication
    from main_window import MainWindow
    _has_gui = True
except Exception:
    _has_gui = False

import pytest


@pytest.fixture(scope="session")
def qapp():
    if not _has_gui:
        pytest.skip("PySide6 not available")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(scope="session")
def main_window(qapp):
    win = MainWindow()
    win.show()
    yield win
    win.close()


def test_gui_loads(main_window):
    assert main_window.model is not None


def test_gui_panel_counts(main_window):
    mw = main_window
    assert len(mw.joint_panel._entries) == mw.model.njnt
    assert mw.actuator_panel._actuator_count == mw.model.nu


def test_gui_body_tree_count(main_window):
    tree = main_window.body_panel.tree
    def count(parent):
        n = 0
        for i in range(parent.childCount()):
            n += 1 + count(parent.child(i))
        return n
    assert count(tree.invisibleRootItem()) == main_window.model.nbody


def test_gui_play_toggle(main_window):
    initial = main_window.playing
    main_window._toggle_play()
    assert main_window.playing != initial
    main_window._toggle_play()
    assert main_window.playing == initial


def test_gui_xml_editor_content(main_window):
    xml = main_window.xml_editor.editor.toPlainText()
    assert xml.strip() != ""
    mujoco.MjModel.from_xml_string(xml)  # must parse


def test_gui_keyframe_roundtrip(main_window):
    mw = main_window
    n_before = len(mw.keyframe_panel._keyframes)
    mw.keyframe_panel.name_input.setText("__pytest_kf__")
    mw.keyframe_panel._save_keyframe()
    assert len(mw.keyframe_panel._keyframes) == n_before + 1
    # cleanup
    if "__pytest_kf__" in mw.keyframe_panel._keyframes:
        del mw.keyframe_panel._keyframes["__pytest_kf__"]
        mw.keyframe_panel._rebuild_list()


def test_gui_actuator_reset_all(main_window):
    mw = main_window
    if mw.model.nu == 0:
        pytest.skip("no actuators")
    mw.actuator_panel._reset_all()
    for i in range(mw.model.nu):
        cr = mw.model.actuator_ctrlrange[i]
        expected = (cr[0] + cr[1]) / 2.0
        assert abs(mw.data.ctrl[i] - expected) < 1e-6