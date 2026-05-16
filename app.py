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
    # NEW: Enable high-DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        QApplication.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)
    
    # NEW: Set application metadata
    app.setApplicationName("MuJoCo Viewer")
    app.setApplicationVersion("1.1.0")
    app.setOrganizationName("MuJoCoViewer")
    
    window = MainWindow()
    window.show()
    
    screen = app.primaryScreen()
    if screen:
        size = screen.availableSize()
        if size.width() >= 1920 and size.height() >= 1080:
            window.showMaximized()
    
    sys.exit(app.exec())