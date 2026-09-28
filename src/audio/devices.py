"""Parse sounddevice indices or name/host-API selectors without loading audio."""

from __future__ import annotations


def parse_output_device(value: str) -> int | str:
    value = value.strip()
    if not value:
        raise ValueError("output device name cannot be empty")
    try:
        index = int(value)
    except ValueError:
        return value
    if index < 0:
        raise ValueError("output device index must be non-negative")
    return index
