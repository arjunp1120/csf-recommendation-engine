from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class ShadowModelState:
    model: Any | None = None
    lock: Lock = Lock()
