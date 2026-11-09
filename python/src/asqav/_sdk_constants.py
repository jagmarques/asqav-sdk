"""Values the Python and TypeScript SDKs share, read from sdk_constants.json."""

from __future__ import annotations

import json
from importlib.resources import files

SDK_CONSTANTS: dict[str, str] = json.loads(
    files("asqav").joinpath("sdk_constants.json").read_text(encoding="utf-8")
)
