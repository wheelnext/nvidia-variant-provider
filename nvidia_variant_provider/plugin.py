from __future__ import annotations

import os
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from functools import cached_property
from typing import Protocol
from typing import runtime_checkable

from packaging.specifiers import SpecifierSet
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
    @abstractmethod
    def namespace(self) -> VariantNamespace:
        """Namespace (from plugin)"""
        raise NotImplementedError

    @property
    @abstractmethod
    def feature(self) -> VariantFeatureName:
        """Feature name (within the namespace)"""
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self) -> VariantFeatureValue:
        """Feature value"""
        raise NotImplementedError


@dataclass(frozen=True)
class VariantFeatureConfig:
    name: str

    # Acceptable values in priority order
    values: list[str]


class NvidiaVariantFeatureKey(StrEnum):
    DRIVER = "driver"
    CUDA = "cuda_version"
    SM = "sm_arch"


class InputPreservingSpecifierSet(SpecifierSet):
    original_value: str

    def __init__(
        self,
        specifiers: str = "",
        prereleases: bool | None = None,
    ) -> None:
        self.original_value = specifiers
        super().__init__(specifiers=specifiers, prereleases=prereleases)


class NvidiaVariantPlugin:
    namespace = "nvidia"
    dynamic = False

    UMD_LOWEST_VERSION = Version("11.0")
    UMD_HIGHEST_VERSION = Version("15.20")

    @cached_property
    def _cuda_environment(self) -> CudaEnvironment | None:
        """Lookup the system to determine the driver / GPU state."""

        return CudaEnvironment.from_system()

    @cache  # noqa: B019
    def generate_all_umd_values(self) -> list[str]:
        return [
            f"{major}.{minor}" if minor is not None else f"{major}"
            for major in range(
                self.UMD_HIGHEST_VERSION.major, self.UMD_LOWEST_VERSION.major - 1, -1
            )
            for minor in [*range(self.UMD_HIGHEST_VERSION.minor, -1, -1), None]
        ]

    # @property
    # def kmd_version(self) -> Version | None:
    #     if driver_ver := os.environ.get("NV_VARIANT_PROVIDER_FORCE_KMD_DRIVER_VERSION"):
    #         return Version(driver_ver)

    #     if (cuda_env := self._cuda_environment) is None:
    #         return None

    #     return (
    #         Version(cuda_env.system_driver_versions)
    #         if cuda_env.system_driver_versions
    #         else None
    #     )

    @property
    def umd_version(self) -> Version | None:
        if driver_ver := os.environ.get(
            "NV_VARIANT_PROVIDER_FORCE_CUDA_DRIVER_VERSION"
        ):
            return Version(driver_ver)

        if (cuda_env := self._cuda_environment) is None:
            return None

        return (
            Version(cuda_env.cuda_driver_version)
            if cuda_env.cuda_driver_version
            else None
        )

    def get_supported_configs(
        self, known_properties: frozenset[VariantPropertyType] | None
    ) -> list[VariantFeatureConfig]:
        """Filter and sort the properties based on the plugin's logic."""

        keyconfigs: list[VariantFeatureConfig] = []

        # ============= User-Mode Driver (UMD) / LIBCUDA Driver Version ============= #

        # Priority 1 - UMD Lower Bound: `>= {major}.{minor}` or `>= {major}`
        # Example Output with Local UMD Version: 12.7
        # ['12.7', ..., '12.0', '12', '11.20', ..., '11.0', '11']

        if self.umd_version is not None:
            if self.umd_version >= self.UMD_LOWEST_VERSION:
                umd_values_low = [
                    f"{major}.{minor}" if minor is not None else f"{major}"
                    for major in range(
                        self.umd_version.major, self.UMD_LOWEST_VERSION.major - 1, -1
                    )
                    for minor in (
                        [*range(self.UMD_HIGHEST_VERSION.minor, -1, -1), None]
                        if major < self.umd_version.major
                        else [*range(self.umd_version.minor, -1, -1), None]
                    )
                ]
                if umd_values_low:
                    keyconfigs.append(
                        VariantFeatureConfig(
                            name=f"{NvidiaVariantFeatureKey.CUDA}_lower_bound",
                            values=umd_values_low,
                        )
                    )
            else:
                warnings.warn(
                    f"The UMD version {self.umd_version} is lower than the "
                    f"minimum supported version {self.UMD_LOWEST_VERSION}. "
                    "No lower bound will be set for CUDA versions. Unexpected behavior "
                    "may occur.",
                    UserWarning,
                    stacklevel=1,
                )

        # Priority 2 - UMD Higher Bound (excluded): `< {major}.{minor}` or `< {major}`
        # Example Output with Local UMD Version: 12.7
        # ['15.20', ..., '15.0', '15', ...,  '13.0', '13', ..., '12.9', '12.8']

        if self.umd_version is not None:
            if self.umd_version < self.UMD_HIGHEST_VERSION:
                umd_values_low = [
                    f"{major}.{minor}" if minor is not None else f"{major}"
                    for major in range(
                        self.UMD_HIGHEST_VERSION.major, self.umd_version.major - 1, -1
                    )
                    for minor in (
                        [*range(self.UMD_HIGHEST_VERSION.minor, -1, -1), None]
                        if major > self.umd_version.major
                        else range(
                            self.UMD_HIGHEST_VERSION.minor,
                            self.umd_version.minor,
                            -1,
                        )
                    )
                ]
                if umd_values_low:
                    keyconfigs.append(
                        VariantFeatureConfig(
                            name=f"{NvidiaVariantFeatureKey.CUDA}_upper_bound",
                            values=umd_values_low,
                        )
                    )
            else:
                warnings.warn(
                    f"The UMD version {self.umd_version} is higher than the "
                    f"maximum supported version {self.UMD_HIGHEST_VERSION}. "
                    "No upper bound will be set for CUDA versions. Unexpected behavior "
                    "may occur.",
                    UserWarning,
                    stacklevel=1,
                )

        return keyconfigs

    def get_all_configs(
        self, known_properties: list[VariantPropertyType] | None
    ) -> list[VariantFeatureConfig]:
        """Not used - transparently returns all known properties as configs."""

        all_umd_values = self.generate_all_umd_values()

        return [
            VariantFeatureConfig(
                f"{NvidiaVariantFeatureKey.CUDA}_lower_bound", all_umd_values
            ),
            VariantFeatureConfig(
                f"{NvidiaVariantFeatureKey.CUDA}_upper_bound", all_umd_values
            ),
        ]

    def validate_property(self, variant_property: VariantPropertyType) -> bool:
        assert variant_property.namespace == self.namespace

        if variant_property.feature not in [
            f"{NvidiaVariantFeatureKey.CUDA}_lower_bound",
            f"{NvidiaVariantFeatureKey.CUDA}_upper_bound",
        ]:
            return False

        return variant_property.value in self.generate_all_umd_values()
