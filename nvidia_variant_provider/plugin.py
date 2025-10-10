# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import cache
from typing import Protocol
from typing import runtime_checkable

from packaging.version import Version

from nvidia_variant_provider.detect_cuda import CudaEnvironment

FeatureIndex = int
PropertyIndex = int

VariantNamespace = str
VariantFeatureName = str
VariantFeatureValue = str


@runtime_checkable
class VariantPropertyType(Protocol):
    """A protocol for variant properties"""

    @property
    def namespace(self) -> VariantNamespace:
        """Namespace (from plugin)"""
        raise NotImplementedError

    @property
    def feature(self) -> VariantFeatureName:
        """Feature name (within the namespace)"""
        raise NotImplementedError

    @property
    def value(self) -> VariantFeatureValue:
        """Feature value"""
        raise NotImplementedError


@dataclass(frozen=True)
class VariantFeatureConfig:
    name: str

    # Acceptable values in priority order
    values: list[str]
    multi_value: bool = False


class NvidiaVariantFeatureKey:
    # DRIVER = "driver"
    CUDA = "cuda_version"
    SM = "sm_arch"


class NvidiaVariantPlugin:
    namespace = "nvidia"
    is_build_plugin = False

    UMD_MAJOR_RANGE = range(11, 16)
    UMD_MINOR_RANGE = range(21)

    @classmethod
    @cache
    def _cuda_environment(cls) -> CudaEnvironment | None:
        """Lookup the system to determine the driver / GPU state."""
        return CudaEnvironment.from_system()

    @classmethod
    @cache
    def generate_all_umd_values(cls) -> list[str]:
        return [
            f"{major}.{minor}" if minor is not None else f"{major}"
            for major in reversed(cls.UMD_MAJOR_RANGE)
            for minor in [*reversed(cls.UMD_MINOR_RANGE), None]
        ]

    @classmethod
    @cache
    def generate_all_sm_values(cls) -> list[str]:
        """Generate all possible SM values based on the supported range."""
        return [
            f"{major}{minor}_real"
            for major in range(12, 4, -1)
            for minor in range(9, -1, -1)
        ] + [
            f"{major}{minor}_virtual"
            for major in range(12, 4, -1)
            for minor in range(9, -1, -1)
        ]

    # @property
    # def kmd_version(cls) -> Version | None:
    #     if (
    #       driver_ver := os.environ.get("NV_VARIANT_PROVIDER_FORCE_KMD_DRIVER_VERSION"
    #     ):
    #         return Version(driver_ver)

    #     if (cuda_env := cls._cuda_environment()) is None:
    #         return None

    #     return (
    #         Version(cuda_env.system_driver_versions)
    #         if cuda_env.system_driver_versions
    #         else None
    #     )

    @classmethod
    @cache
    def umd_version(cls) -> Version | None:
        if driver_ver := os.environ.get(
            "NV_VARIANT_PROVIDER_FORCE_CUDA_DRIVER_VERSION"
        ):
            return Version(driver_ver)

        if (cuda_env := cls._cuda_environment()) is None:
            return None

        umd_version = (
            Version(cuda_env.cuda_driver_version)
            if cuda_env.cuda_driver_version
            else None
        )

        if umd_version is None:
            warnings.warn(
                "No UMD version found. NVIDIA platform is not supported or "
                "properly installed on this machine.",
                UserWarning,
                stacklevel=1,
            )
            return None

        min_umd_version = Version(
            f"{min(cls.UMD_MAJOR_RANGE)}.{min(cls.UMD_MINOR_RANGE)}"
        )
        max_umd_version = Version(
            f"{max(cls.UMD_MAJOR_RANGE)}.{max(cls.UMD_MINOR_RANGE)}"
        )

        if min_umd_version > umd_version or umd_version > max_umd_version:
            warnings.warn(
                f"The UMD version `{umd_version}` is outside the supported range: "
                f"`>={min_umd_version},<={max_umd_version}`.\n"
                "NVIDIA platform detection has been deactivated.",
                UserWarning,
                stacklevel=1,
            )
            return None

        return umd_version

    @classmethod
    @cache
    def get_sm_architectures(cls) -> tuple[int, int] | None:
        """Get the SM architectures from the CUDA environment."""

        if sm_arch := os.environ.get("NV_VARIANT_PROVIDER_FORCE_SM_ARCH"):
            return tuple(map(int, sm_arch.split(".", maxsplit=1)))  # type: ignore[return-value]

        if (cuda_env := cls._cuda_environment()) is None:
            return None

        return sorted(cuda_env.architectures, reverse=True)[0]

    @classmethod
    @cache
    def get_supported_configs(cls) -> list[VariantFeatureConfig]:
        """Filter and sort the properties based on the plugin's logic."""

        if (umd_version := cls.umd_version()) is None:
            return []

        keyconfigs: list[VariantFeatureConfig] = []

        # ============= User-Mode Driver (UMD) / LIBCUDA Driver Version ============= #

        # Priority 1 - UMD Lower Bound: `>= {major}.{minor}` or `>= {major}`
        # Example Output with Local UMD Version: 12.7
        # ['12.7', ..., '12.0', '12', '11.20', ..., '11.0', '11']
        umd_values_low = [
            f"{major}.{minor}" if minor is not None else f"{major}"
            for major in range(umd_version.major, min(cls.UMD_MAJOR_RANGE) - 1, -1)
            for minor in (
                [*range(max(cls.UMD_MINOR_RANGE), -1, -1), None]
                if major < umd_version.major
                else [*range(umd_version.minor, -1, -1), None]
            )
        ]
        if umd_values_low:
            keyconfigs.append(
                VariantFeatureConfig(
                    name=f"{NvidiaVariantFeatureKey.CUDA}_lower_bound",
                    values=umd_values_low,
                )
            )

        # Priority 2 - UMD Higher Bound (excluded): `< {major}.{minor}` or `< {major}`
        # Example Output with Local UMD Version: 12.7
        # ['15.20', ..., '15.0', '15', ...,  '13.0', '13', ..., '12.9', '12.8']
        umd_values_low = [
            f"{major}.{minor}" if minor is not None else f"{major}"
            for major in range(max(cls.UMD_MAJOR_RANGE), umd_version.major - 1, -1)
            for minor in (
                [*range(max(cls.UMD_MINOR_RANGE), -1, -1), None]
                if major > umd_version.major
                else range(max(cls.UMD_MINOR_RANGE), umd_version.minor, -1)
            )
        ]
        if umd_values_low:
            keyconfigs.append(
                VariantFeatureConfig(
                    name=f"{NvidiaVariantFeatureKey.CUDA}_upper_bound",
                    values=umd_values_low,
                )
            )

        # Priority 3 - NVIDIA SM Arch Compatibility - based on CMAKE Flags
        if sm_arch := cls.get_sm_architectures():
            # For now we only use the highest SM architecture on the system
            sm_major, sm_minor = sm_arch

            if Version("5.0") > Version(f"{sm_major}.{sm_minor}"):
                warnings.warn(
                    "The SM version is lower than the lowest SM architecture "
                    f"supported by the NVIDIA Variant Plugin: {sm_major}.{sm_minor}. "
                    "NVIDIA platform detection may not work as expected.",
                    UserWarning,
                    stacklevel=1,
                )

            else:
                sm_cmake_flags = [
                    *[f"{sm_major}{minor}_real" for minor in range(sm_minor, -1, -1)],
                    f"{sm_major}0_virtual",
                ]

                if sm_cmake_flags:
                    keyconfigs.append(
                        VariantFeatureConfig(
                            name=NvidiaVariantFeatureKey.SM,
                            values=sm_cmake_flags,
                            multi_value=True,
                        )
                    )

        return keyconfigs

    @classmethod
    @cache
    def get_all_configs(cls) -> list[VariantFeatureConfig]:
        """Not used - transparently returns all known properties as configs."""

        all_umd_values = cls.generate_all_umd_values()

        return [
            VariantFeatureConfig(
                name=f"{NvidiaVariantFeatureKey.CUDA}_lower_bound",
                values=all_umd_values,
            ),
            VariantFeatureConfig(
                name=f"{NvidiaVariantFeatureKey.CUDA}_upper_bound",
                values=all_umd_values,
            ),
            VariantFeatureConfig(
                name=NvidiaVariantFeatureKey.SM,
                values=cls.generate_all_sm_values(),
                multi_value=True,
            ),
        ]


if __name__ == "__main__":
    import os

    os.environ["NV_VARIANT_PROVIDER_FORCE_CUDA_DRIVER_VERSION"] = "12.7"
    os.environ["NV_VARIANT_PROVIDER_FORCE_SM_ARCH"] = "12.5"

    # print(NvidiaVariantPlugin.get_supported_configs(None))
    # print(NvidiaVariantPlugin.get_all_configs(None))
