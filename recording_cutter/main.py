#!/usr/bin/env python3
"""Entry point for the standalone Recording Cutter tool. See cutter_window.py
for what it does. Run: python3 recording_cutter/main.py"""
import sys

from PyQt5.QtWidgets import QApplication

from cutter_window import RecordingCutterWindow


def main():
    app = QApplication(sys.argv)
    window = RecordingCutterWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
