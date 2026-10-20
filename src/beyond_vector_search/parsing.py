"""parsing.py — pull a JSON value out of a model reply that may wear a code fence."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json(payload: str) -> Any:
    text = payload.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)
