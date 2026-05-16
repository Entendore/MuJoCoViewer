"""Shared constants, enums, and the dark theme stylesheet."""

import mujoco

# ── Visualization flags ────────────────────────────────────────
VIS_FLAGS = []
_vis_flag_names = [
    "mjVIS_JOINT", "mjVIS_ACTUATOR", "mjVIS_CONTACTPOINT",
    "mjVIS_CONTACTFORCE", "mjVIS_COM", "mjVIS_BODYFRAME",
    "mjVIS_BODYPOS", "mjVIS_WIREFRAME", "mjVIS_TRANSPARENT",
    "mjVIS_SHADOW", "mjVIS_LIGHT",
]
for name in _vis_flag_names:
    if hasattr(mujoco.mjtVisFlag, name):
        flag = getattr(mujoco.mjtVisFlag, name)
        VIS_FLAGS.append((name[6:].replace('_', ' ').title(), flag))

# ── Joint type display names ───────────────────────────────────
JOINT_TYPE_NAMES = {}
for name in ["mjJNT_FREE", "mjJNT_BALL", "mjJNT_SLIDE", "mjJNT_HINGE"]:
    if hasattr(mujoco.mjtJoint, name):
        JOINT_TYPE_NAMES[getattr(mujoco.mjtJoint, name)] = name[6:].title()

# ── qpos dimension per joint type ─────────────────────────────
QPOS_DIMS = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}

# ── Dark theme stylesheet ─────────────────────────────────────
DARK_STYLE = """
QMainWindow, QWidget { background-color: #1a1b26; color: #a9b1d6; font-family: 'Segoe UI', 'Arial', sans-serif; font-size: 12px; }
QMenuBar { background-color: #1a1b26; border-bottom: 1px solid #24283b; padding: 2px; }
QMenuBar::item { padding: 4px 10px; } QMenuBar::item:selected { background-color: #24283b; border-radius: 4px; }
QMenu { background-color: #1a1b26; border: 1px solid #3d59a1; padding: 4px; } QMenu::item { padding: 5px 24px; } QMenu::item:selected { background-color: #3d59a1; border-radius: 3px; } QMenu::separator { height: 1px; background: #24283b; margin: 4px 8px; }
QToolBar { background-color: #16161e; border-bottom: 1px solid #24283b; spacing: 6px; padding: 3px 6px; }
QToolButton { background-color: #24283b; border: 1px solid #3b4261; border-radius: 4px; padding: 5px 10px; color: #a9b1d6; font-weight: 500; }
QToolButton:hover { background-color: #3d59a1; color: #c0caf5; } QToolButton:pressed { background-color: #2b3f7a; } QToolButton:checked { background-color: #3d59a1; color: #ffffff; }
QPushButton { background-color: #24283b; border: 1px solid #3b4261; border-radius: 4px; padding: 5px 14px; color: #a9b1d6; font-weight: 500; }
QPushButton:hover { background-color: #3d59a1; color: #c0caf5; border-color: #3d59a1; } QPushButton:pressed { background-color: #2b3f7a; } QPushButton:checked { background-color: #3d59a1; color: #ffffff; }
QPushButton:disabled { background-color: #1a1b26; color: #3b4261; border-color: #24283b; }
QSlider::groove:horizontal { height: 6px; background: #24283b; border-radius: 3px; } QSlider::handle:horizontal { width: 16px; height: 16px; background: #7aa2f7; border-radius: 8px; margin: -5px 0; } QSlider::sub-page:horizontal { background: #3d59a1; border-radius: 3px; }
QGroupBox { border: 1px solid #24283b; border-radius: 6px; margin-top: 12px; padding: 12px 8px 8px 8px; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #7aa2f7; }
QTreeWidget { background-color: #1a1b26; border: 1px solid #24283b; alternate-background-color: #1f2035; border-radius: 4px; } QTreeWidget::item { padding: 3px 0; } QTreeWidget::item:selected { background-color: #3d59a1; }
QHeaderView::section { background-color: #16161e; border: none; border-bottom: 1px solid #24283b; padding: 4px; font-weight: bold; color: #565f89; }
QScrollArea { border: none; background: transparent; } QTabWidget::pane { border: 1px solid #24283b; border-radius: 4px; top: -1px; }
QTabBar::tab { background-color: #16161e; border: 1px solid #24283b; padding: 7px 16px; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }
QTabBar::tab:selected { background-color: #24283b; border-bottom-color: #24283b; color: #7aa2f7; }
QLabel { color: #a9b1d6; background: transparent; } QLabel[class="value"] { color: #7aa2f7; font-family: 'Consolas', 'Courier New', monospace; } QLabel[class="dim"] { color: #565f89; font-size: 11px; } QLabel[class="error"] { color: #f7768e; } QLabel[class="success"] { color: #9ece6a; }
QStatusBar { background-color: #16161e; border-top: 1px solid #24283b; color: #565f89; font-size: 11px; } QStatusBar QLabel { color: #565f89; }
QComboBox { background-color: #24283b; border: 1px solid #3b4261; border-radius: 4px; padding: 4px 8px; color: #a9b1d6; min-width: 80px; } QComboBox QAbstractItemView { background-color: #1a1b26; border: 1px solid #3d59a1; selection-background-color: #3d59a1; }
QCheckBox { spacing: 8px; background: transparent; } QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #3b4261; border-radius: 4px; background-color: #24283b; } QCheckBox::indicator:checked { background-color: #7aa2f7; border-color: #7aa2f7; }
QSplitter::handle { background-color: #24283b; width: 2px; }
QTableWidget { background-color: #1a1b26; border: 1px solid #24283b; alternate-background-color: #1f2035; border-radius: 4px; gridline-color: #24283b; }
QPlainTextEdit { background-color: #16161e; border: 1px solid #24283b; color: #a9b1d6; font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; border-radius: 4px; padding: 6px; selection-background-color: #3d59a1; }
QProgressBar { border: 1px solid #24283b; border-radius: 4px; background-color: #16161e; text-align: center; color: #a9b1d6; } QProgressBar::chunk { background-color: #3d59a1; border-radius: 3px; }
QSpinBox { background-color: #24283b; border: 1px solid #3b4261; border-radius: 4px; padding: 3px 6px; color: #a9b1d6; }
QDialog { background-color: #1a1b26; }
"""