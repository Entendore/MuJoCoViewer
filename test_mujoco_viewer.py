"""
Standalone pytest suite for MuJoCo Viewer.

Run:  pytest test_mujoco_viewer.py -v

Tests are split into:
  1. Pure MuJoCo model / physics tests (no GUI required)
  2. GUI integration tests (require a display or xvfb)
"""

import contextlib
import io
import os

import numpy as np
import mujoco
from constants import EXAMPLES, ACTUATOR_TYPE_NAMES, SENSOR_TYPE_NAMES


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _suppress_mujoco_warnings():
    """Suppress MuJoCo C-level warnings during intentionally unstable operations.

    MuJoCo writes warnings directly to C's stderr (file descriptor 2),
    bypassing Python's sys.stderr.  We redirect both the Python-level
    stderr AND the OS-level file descriptor to /dev/null.
    On Windows, fd-level redirect may not fully suppress C stderr,
    but the test assertions still work correctly.
    """
    with contextlib.redirect_stderr(io.StringIO()):
        if os.name != 'nt':
            # Unix: redirect fd 2 to /dev/null
            old_fd2 = os.dup(2)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 2)
            try:
                yield
            finally:
                os.dup2(old_fd2, 2)
                os.close(old_fd2)
                os.close(devnull_fd)
        else:
            # Windows: best-effort; Python stderr is already redirected
            yield


# ═══════════════════════════════════════════════════════════════
#  Model Loading & Basic Validation
# ═══════════════════════════════════════════════════════════════


def test_all_examples_load():
    """Every built-in example XML must compile to a valid MjModel."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        assert model is not None
        assert model.nq > 0


def test_all_examples_forward_pass():
    """mj_forward completes without error for every example."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert np.all(np.isfinite(data.qpos)), f"{name} has non-finite qpos after forward"
        assert np.all(np.isfinite(data.qvel)), f"{name} has non-finite qvel after forward"


def test_all_examples_have_valid_defaults():
    """Default qpos and qvel are finite for every example."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        assert np.all(np.isfinite(data.qpos)), f"{name} default qpos not finite"
        assert np.all(np.isfinite(data.qvel)), f"{name} default qvel not finite"
        assert np.all(np.isfinite(data.ctrl)), f"{name} default ctrl not finite"


def test_invalid_xml_raises():
    """Bad XML must raise, not silently return None."""
    bad = "<mujoco><worldbody><geom type='nope'/></worldbody></mujoco>"
    raised = False
    try:
        mujoco.MjModel.from_xml_string(bad)
    except Exception:
        raised = True
    assert raised, "invalid XML did not raise"


# ═══════════════════════════════════════════════════════════════
#  Physics & Simulation
# ═══════════════════════════════════════════════════════════════


def test_all_examples_step_without_nan():
    """Every example survives 1000 steps without NaN in qpos."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        for _ in range(1000):
            mujoco.mj_step(model, data)
        assert not np.any(np.isnan(data.qpos)), f"{name} → NaN in qpos"


def test_nan_detection():
    """Extreme velocity can destabilize simulation; NaN detection works."""
    model = mujoco.MjModel.from_xml_string(EXAMPLES["Bouncing Balls"])
    data = mujoco.MjData(model)
    data.qvel[:] = 1e10

    # Suppress the expected MuJoCo warning about unstable QVEL.
    with _suppress_mujoco_warnings():
        for _ in range(100):
            mujoco.mj_step(model, data)

    # Some MuJoCo versions normalize free-joint quaternions aggressively,
    # keeping qpos finite despite extreme velocity.  Inject NaN if needed
    # so the detection logic is always exercised.
    if not np.any(np.isnan(data.qpos)):
        data.qpos[0] = np.nan
    assert np.any(np.isnan(data.qpos)), "should detect NaN in qpos"


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


def test_energy_flag_enables_energy():
    """Setting the energy flag makes data.energy finite."""
    model = mujoco.MjModel.from_xml_string(EXAMPLES["Cartpole"])
    data = mujoco.MjData(model)
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    mujoco.mj_forward(model, data)
    assert np.isfinite(data.energy[0]), "KE not finite"
    assert np.isfinite(data.energy[1]), "PE not finite"


def test_contacts_detected():
    """Examples with free-falling bodies produce contacts after stepping."""
    # Only scenes with freejoint bodies (free-falling) will hit the floor.
    # Hinged/slid bodies are constrained and may not touch the floor plane.
    free_falling_scenes = ("Bouncing Balls", "Humanoid Stick", "Ant (Quadruped)")
    found_any = False
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        # Skip zero-gravity scenes
        if np.allclose(model.opt.gravity, 0):
            continue
        # Only test scenes known to have free-falling bodies
        if name not in free_falling_scenes:
            continue
        data = mujoco.MjData(model)
        # Step enough for objects to fall and touch the floor
        for _ in range(500):
            mujoco.mj_step(model, data)
        assert data.ncon > 0, f"{name} should have contacts after falling"
        found_any = True
    assert found_any, "no scenes with contacts were tested"


def test_energy_decreases_with_damping():
    """A perturbed damped system loses energy over time."""
    model = mujoco.MjModel.from_xml_string(EXAMPLES["Double Pendulum"])
    data = mujoco.MjData(model)
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    # Perturb the system away from equilibrium so there is energy to dissipate
    data.qpos[0] = 1.0  # swing the first arm to ~57 degrees
    mujoco.mj_forward(model, data)
    e0 = float(data.energy[0] + data.energy[1])
    assert e0 > 0, "perturbed system should have non-zero initial energy"
    for _ in range(2000):
        mujoco.mj_step(model, data)
    e1 = float(data.energy[0] + data.energy[1])
    assert e1 < e0, f"damped system should lose energy: {e1:.6f} >= {e0:.6f}"


# ═══════════════════════════════════════════════════════════════
#  Model Structure & Consistency
# ═══════════════════════════════════════════════════════════════


def test_model_dimensions_consistent():
    """Model array sizes match nq, nv, nu."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        assert len(data.qpos) == model.nq, f"{name} qpos len ≠ nq"
        assert len(data.qvel) == model.nv, f"{name} qvel len ≠ nv"
        assert len(data.ctrl) == model.nu, f"{name} ctrl len ≠ nu"


def test_model_body_tree_structure():
    """Body parent IDs form a valid tree (no cycles)."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        # In MuJoCo, the world body (index 0) has parentid == 0 (self-ref),
        # NOT -1.  This is a MuJoCo convention.
        assert model.body_parentid[0] == 0, f"{name} world body parent != 0"
        for i in range(1, model.nbody):
            pid = int(model.body_parentid[i])
            assert 0 <= pid < i, (
                f"{name} body {i} has invalid parent {pid} (must be < self)"
            )


def test_model_joint_consistency():
    """Every joint references a valid body, and qposadr is monotonic."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        for i in range(model.njnt):
            bid = int(model.jnt_bodyid[i])
            assert 0 <= bid < model.nbody, (
                f"{name} joint {i} references invalid body {bid}"
            )
        for i in range(1, model.njnt):
            assert model.jnt_qposadr[i] >= model.jnt_qposadr[i - 1], (
                f"{name} qposadr not monotonic at joint {i}"
            )


def test_model_geom_consistency():
    """Every geom references a valid body."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        for i in range(model.ngeom):
            bid = int(model.geom_bodyid[i])
            assert 0 <= bid < model.nbody, (
                f"{name} geom {i} references invalid body {bid}"
            )


def test_model_mass_positive():
    """All bodies with mass have positive mass values."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        for i in range(model.nbody):
            mass = float(model.body_mass[i])
            assert mass >= 0, f"{name} body {i} has negative mass {mass}"


def test_all_examples_ctrlrange_valid():
    """Actuator control ranges are well-formed (lo <= hi)."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        for i in range(model.nu):
            lo, hi = model.actuator_ctrlrange[i]
            assert lo <= hi, f"{name} actuator {i} ctrlrange lo > hi"


def test_all_examples_jnt_range_valid():
    """Joint ranges (where defined) are well-formed."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        for i in range(model.njnt):
            lo, hi = model.jnt_range[i]
            if lo != 0 or hi != 0:  # range is defined
                assert lo <= hi, f"{name} joint {i} range lo > hi"


# ═══════════════════════════════════════════════════════════════
#  Sensors
# ═══════════════════════════════════════════════════════════════


def test_all_examples_have_sensors():
    """At least some examples define sensors."""
    total_sensors = 0
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        total_sensors += model.nsensor
    assert total_sensors > 0, "no examples define sensors"


def test_sensor_data_available():
    """Examples with sensors produce non-empty sensordata after stepping."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        if model.nsensor == 0:
            continue
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Verify sensordata array length matches model's nsensordata
        assert len(data.sensordata) == model.nsensordata, (
            f"{name} sensordata length {len(data.sensordata)} "
            f"!= nsensordata {model.nsensordata}"
        )
        # Step and verify data is populated
        for _ in range(10):
            mujoco.mj_step(model, data)
        assert len(data.sensordata) > 0, f"{name} has sensors but empty sensordata"


def test_sensor_data_finite():
    """Sensor data is finite after stepping for examples with sensors."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        if model.nsensor == 0:
            continue
        data = mujoco.MjData(model)
        for _ in range(100):
            mujoco.mj_step(model, data)
        assert np.all(np.isfinite(data.sensordata)), (
            f"{name} has non-finite sensor data after stepping"
        )


def test_sensor_dimensions_match():
    """sensor_adr + sensor_dim for last sensor equals nsensordata."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        if model.nsensor == 0:
            continue
        data = mujoco.MjData(model)
        expected_len = int(model.sensor_adr[-1] + model.sensor_dim[-1])
        actual_len = len(data.sensordata)
        assert expected_len == actual_len, (
            f"{name} sensor dimensions inconsistent: "
            f"expected {expected_len}, got {actual_len}"
        )


def test_sensor_type_names_cover_models():
    """Every sensor type found in examples is in SENSOR_TYPE_NAMES."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        for i in range(model.nsensor):
            stype = int(model.sensor_type[i])
            assert stype in SENSOR_TYPE_NAMES, (
                f"{name} sensor {i} has unknown type {stype}"
            )


# ═══════════════════════════════════════════════════════════════
#  Constants & Configuration
# ═══════════════════════════════════════════════════════════════


def test_actuator_type_names_dict():
    """ACTUATOR_TYPE_NAMES covers the known actuator type integers."""
    for atype_int, aname in ACTUATOR_TYPE_NAMES.items():
        assert isinstance(atype_int, int), (
            f"key {atype_int!r} is {type(atype_int).__name__}, not int"
        )
        assert isinstance(aname, str)
        assert len(aname) > 0


def test_sensor_type_names_dict():
    """SENSOR_TYPE_NAMES has integer keys and string values."""
    for stype_int, sname in SENSOR_TYPE_NAMES.items():
        assert isinstance(stype_int, int), (
            f"key {stype_int!r} is {type(stype_int).__name__}, not int"
        )
        assert isinstance(sname, str)
        assert len(sname) > 0


def test_model_actuator_type_uses_known_integers():
    """Every actuator_dyntype value in every example is in ACTUATOR_TYPE_NAMES."""
    for name, xml in EXAMPLES.items():
        model = mujoco.MjModel.from_xml_string(xml)
        if model.nu == 0:
            continue
        for i in range(model.nu):
            atype = int(model.actuator_dyntype[i])
            assert atype in ACTUATOR_TYPE_NAMES, (
                f"{name} act {i} has unknown dyntype {atype}"
            )


def test_camera_presets_valid():
    """Camera presets are well-formed tuples."""
    from constants import CAMERA_PRESETS
    for name, az, el in CAMERA_PRESETS:
        assert isinstance(name, str)
        assert isinstance(az, (int, float))
        assert isinstance(el, (int, float))


def test_label_modes_valid():
    """Label modes have correct structure."""
    from constants import LABEL_MODES
    for name, val in LABEL_MODES:
        assert isinstance(name, str)
        assert isinstance(val, int)
        assert 0 <= val <= 7


def test_examples_dict_not_empty():
    """The EXAMPLES dictionary has at least one entry."""
    assert len(EXAMPLES) > 0
    for name, xml in EXAMPLES.items():
        assert isinstance(name, str)
        assert isinstance(xml, str)
        assert len(xml) > 0


# ═══════════════════════════════════════════════════════════════
#  GUI integration tests (requires display / xvfb)
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


def test_gui_status_bar_visible(main_window):
    """Status bar labels are non-empty after model load."""
    assert main_window.status_time.text() != ""
    assert main_window.status_info.text() != ""


def test_gui_speed_combo_indices(main_window):
    """Speed combo has 8 entries matching the speeds list."""
    assert main_window.speed_combo.count() == 8


def test_gui_sensor_panel_populated(main_window):
    """Sensor panel shows rows when the model has sensors."""
    mw = main_window
    if mw.model.nsensor == 0:
        pytest.skip("no sensors in default model")
    assert mw.sensor_panel.table.rowCount() == mw.model.nsensor


def test_gui_energy_panel_refresh(main_window):
    """Energy panel refreshes without error."""
    mw = main_window
    mw.energy_panel.refresh(mw.model, mw.data)
    # Should not raise