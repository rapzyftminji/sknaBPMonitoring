import os
import sys

# All program modules live in src/. Put it on the import path so the flat
# absolute imports inside them (e.g. "from main_window import main",
# "from model import ...") resolve whether the app is launched from the repo
# root, an IDE, or elsewhere. Keep launching via "python main.py".
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    import torch  # noqa: F401
except Exception:
    pass

from main_window import main

if __name__ == "__main__":
    main()
