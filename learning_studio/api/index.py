"""Vercel ASGI entrypoint when learning_studio is the project root."""

import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType

project_root = Path(__file__).resolve().parents[1]
package = ModuleType("learning_studio")
package.__path__ = [str(project_root)]
sys.modules.setdefault("learning_studio", package)
app = import_module("learning_studio.main").app

__all__ = ["app"]
