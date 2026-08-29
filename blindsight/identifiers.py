"""Contract identifiers: the shape every `/v1` id must match, never a storage path."""

from __future__ import annotations

import re

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
MIN_LENGTH = 8
MAX_LENGTH = 64


def is_valid_identifier(value: str) -> bool:
    return MIN_LENGTH <= len(value) <= MAX_LENGTH and bool(IDENTIFIER_PATTERN.match(value))
