# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from packaging.version import Version

from nvidia_variant_provider import plugin

if TYPE_CHECKING:
    from collections.abc import Iterable


def _clear_plugin_caches() -> None:
    """Clear caches on cached classmethods to avoid cross-test contamination."""

    cls = plugin.NvidiaVariantPlugin
    cached_method_names: Iterable[str] = (
        "_cuda_environment",
        "generate_all_umd_values",
        "generate_all_sm_values",
        "umd_version",
        "get_sm_architectures",
        "get_supported_configs",
        "get_all_configs",
    )

    for name in cached_method_names:
        method = getattr(cls, name, None)
        func = getattr(method, "__func__", None)
        clear = getattr(func, "cache_clear", None)
        if callable(clear):
            clear()


def _reset_env() -> None:
    os.environ.pop("NV_VARIANT_PROVIDER_FORCE_CUDA_DRIVER_VERSION", None)
    os.environ.pop("NV_VARIANT_PROVIDER_FORCE_SM_ARCH", None)


def setup_function() -> None:
    """Per-test setup: reset env and clear caches."""
    _reset_env()
    _clear_plugin_caches()


def test_static_metadata() -> None:
    assert plugin.NvidiaVariantPlugin.namespace == "nvidia"
    assert plugin.NvidiaVariantPlugin.is_build_plugin is False


def test_generate_all_umd_values_basic() -> None:
    values = plugin.NvidiaVariantPlugin.generate_all_umd_values()
    # 5 majors (15..11) * (21 minors + None) = 110
    assert len(values) == 110
    assert values[0] == "15.20"
    assert values[-1] == "11"


def test_generate_all_sm_values_basic() -> None:
    values = plugin.NvidiaVariantPlugin.generate_all_sm_values()
    # majors 12..5 (8) * minors 9..0 (10) * 2 (real+virtual) = 160
    assert len(values) == 160
    assert values[0] == "129_real"
    assert values[80] == "129_virtual"  # first virtual entry
    assert "120_real" in values
    assert values[-1] == "50_virtual"


def test_umd_version_env_override() -> None:
    os.environ["NV_VARIANT_PROVIDER_FORCE_CUDA_DRIVER_VERSION"] = "12.7"
    _clear_plugin_caches()
    ver = plugin.NvidiaVariantPlugin.umd_version()

    assert ver == Version("12.7")


def test_get_supported_configs_with_env_overrides() -> None:
    os.environ["NV_VARIANT_PROVIDER_FORCE_CUDA_DRIVER_VERSION"] = "12.7"
    os.environ["NV_VARIANT_PROVIDER_FORCE_SM_ARCH"] = "12.5"
    _clear_plugin_caches()

    configs = plugin.NvidiaVariantPlugin.get_supported_configs()
    assert isinstance(configs, list)
    assert len(configs) >= 2

    by_name = {c.name: c for c in configs}

    lower = by_name.get("cuda_version_lower_bound")
    assert lower is not None
    assert lower.values[0] == "12.7"
    assert "11.20" in lower.values
    assert lower.values[-1] == "11"

    upper = by_name.get("cuda_version_upper_bound")
    assert upper is not None
    assert upper.values[0] == "15.20"
    # Upper bound excludes the local version 12.7
    assert "12.7" not in upper.values
    assert upper.values[-1] == "12.8"

    sm = by_name.get("sm_arch")
    assert sm is not None
    assert sm.multi_value is True
    assert sm.values[0] == "125_real"
    assert sm.values[-1] == "120_virtual"


def test_get_all_configs_shapes() -> None:
    configs = plugin.NvidiaVariantPlugin.get_all_configs()
    names = {c.name for c in configs}
    assert {"cuda_version_lower_bound", "cuda_version_upper_bound", "sm_arch"}.issubset(
        names
    )
    for cfg in configs:
        assert isinstance(cfg.values, list)
        assert cfg.values, f"Config {cfg.name} should not be empty"
