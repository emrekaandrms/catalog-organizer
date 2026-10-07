"""A minimal GLB writer: just enough glTF for a metal body plus stones.

No dependency on a glTF library on purpose -- the output is a handful of accessors, and the
viewer is the only reader. Node and scene `extras` carry the data the stone shader needs
(each stone's frame, the facet planes of its cut) so the browser recomputes none of it.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

FLOAT, UINT = 5126, 5125
ARRAY_BUFFER, ELEMENT_BUFFER = 34962, 34963


class Glb:
    def __init__(self) -> None:
        self.bin = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add(self, arr: np.ndarray, *, target: int, kind: str, bounds: bool = False) -> int:
        data = arr.tobytes()
        offset = len(self.bin)
        self.bin += data
        self.bin += b"\0" * (-len(self.bin) % 4)
        self.views.append({"buffer": 0, "byteOffset": offset,
                           "byteLength": len(data), "target": target})
        acc = {"bufferView": len(self.views) - 1,
               "componentType": FLOAT if arr.dtype == np.float32 else UINT,
               "count": int(len(arr)), "type": kind}
        if bounds:
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def write(self, path: Path, doc: dict) -> int:
        doc["accessors"] = self.accessors
        doc["bufferViews"] = self.views
        doc["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(doc, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        body = bytes(self.bin)
        total = 12 + 8 + len(js) + 8 + len(body)
        with Path(path).open("wb") as fh:
            fh.write(struct.pack("<4sII", b"glTF", 2, total))
            fh.write(struct.pack("<I4s", len(js), b"JSON") + js)
            fh.write(struct.pack("<I4s", len(body), b"BIN\0") + body)
        return total


def rounded(a, digits: int = 6):
    """JSON-friendly rounded list of an array."""
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def read_json_chunk(path: Path) -> dict:
    """The JSON chunk of a GLB, for tests and for the gallery."""
    raw = Path(path).read_bytes()
    magic, version, _length = struct.unpack("<4sII", raw[:12])
    if magic != b"glTF" or version != 2:
        raise ValueError(f"not a GLB 2.0 file: {path}")
    json_len, _kind = struct.unpack("<I4s", raw[12:20])
    return json.loads(raw[20:20 + json_len])
