# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build specialized logical geometries without altering the 256/64 fast source."""

from pathlib import Path
from tempfile import TemporaryDirectory

from .geometry import OctaveGeometry
from .hip import build_library, cache_root

ROOT = Path(__file__).resolve().parent


def build_geometry_library(head_dim, rope_dim, build_dir=None, arch="gfx1201"):
    OctaveGeometry(head_dim, rope_dim)
    build_dir = cache_root() / "geometry" if build_dir is None else Path(build_dir)
    folder = build_dir / f"d{head_dim}_r{rope_dim}"
    source = folder / "kernel_src"
    source.mkdir(parents=True, exist_ok=True)
    text = (
        f"#define OCTAVE_LOGICAL_HEAD {head_dim}\n"
        f"#define OCTAVE_LOGICAL_ROPE {rope_dim}\n"
        + (ROOT / "native/geometry/octave_rdna4.hip").read_text()
    )
    path = source / "octave_rdna4.hip"
    if not path.exists() or path.read_text() != text:
        with TemporaryDirectory(dir=folder) as temp:
            tmp = Path(temp) / path.name
            tmp.write_text(text)
            tmp.replace(path)
    return build_library(source, folder / "build", arch)
