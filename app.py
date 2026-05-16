#!/usr/bin/env python3
"""
MuJoCo PySide6 Viewer — Entry Point
Run this file to start the application.
Requires: pip install mujoco PySide6 numpy
"""

import sys
from PySide6.QtWidgets import QApplication
from constants import DARK_STYLE
from main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())