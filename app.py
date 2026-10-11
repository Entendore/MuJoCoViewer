#!/usr/bin/env python3
"""
MuJoCo PySide6 Viewer — Entry Point

Run this file to start the application.

Requirements:
    pip install mujoco PySide6 numpy

Usage:
    python app.py                # Launch GUI with default scene
    python app.py model.xml      # Launch GUI loading a specific model
"""

import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from constants import DARK_STYLE
from main_window import MainWindow


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)
    app.setApplicationName("MuJoCo Viewer")
    app.setApplicationVersion("1.3.0")
    app.setOrganizationName("MuJoCoViewer")

    window = MainWindow()

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            window._load_model_path(os.path.abspath(path))
        else:
            print(f"Warning: file not found: {path}")

    window.show()

    screen = app.primaryScreen()
    if screen:
        size = screen.availableSize()
        if size.width() >= 1920 and size.height() >= 1080:
            window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()