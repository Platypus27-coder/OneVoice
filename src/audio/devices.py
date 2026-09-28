"""Parse sounddevice indices or name/host-API selectors without loading audio."""

from __future__ import annotations


def parse_output_device(value: str) -> int | str:
    return _parse_device(value, "output")


def parse_input_device(value: str) -> int | str:
    return _parse_device(value, "input")


def _parse_device(value: str, kind: str) -> int | str:
    value = value.strip()
    if not value:
        raise ValueError(f"{kind} device name cannot be empty")
    try:
        index = int(value)
    except ValueError:
        return value
    if index < 0:
        raise ValueError(f"{kind} device index must be non-negative")
    return index
