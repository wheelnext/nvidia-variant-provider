# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"A Variant Provider Plugin for the `nvidia` namespace"

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

__version__ = "0.0.2"


def _load_vendored_packaging() -> None:
    """
    Load a vendored `archspec` library.

    Returns:
        module (ModuleType): The loaded module.
    """
    name = "packaging"

    spec = importlib.util.spec_from_file_location(
        name=name,
        location=Path(__file__).parent / "vendor/packaging/src/packaging/__init__.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("The submodule `packaging` is missing.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


_load_vendored_packaging()
