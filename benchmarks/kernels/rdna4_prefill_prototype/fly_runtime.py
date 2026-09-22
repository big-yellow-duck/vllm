# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile and launch offline FlyDSL prototypes on first use."""

from threading import Lock
from weakref import WeakKeyDictionary

import flydsl.compiler as flyc

_COMPILE_LOCK = Lock()
_COMPILED: WeakKeyDictionary = WeakKeyDictionary()


def run_compiled(executable, *args) -> None:
    with _COMPILE_LOCK:
        compiled = _COMPILED.get(executable)
        if compiled is None:
            compiled = flyc.compile(executable, *args)
            _COMPILED[executable] = compiled
    compiled(*args)
