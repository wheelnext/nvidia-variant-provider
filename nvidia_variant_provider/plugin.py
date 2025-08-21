from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import cache
from functools import cached_property
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


class NvidiaVariantFeatureKey:
    # DRIVER = "driver"
    CUDA = "cuda_version"
    SM = "sm_arch"


class NvidiaVariantPlugin:
    namespace = "nvidia"
    dynamic = False

    UMD_MAJOR_RANGE = range(11, 16)
    UMD_MINOR_RANGE = range(21)

    @cached_property
    def _cuda_environment(self) -> CudaEnvironment | None:
        """Lookup the system to determine the driver / GPU state."""
        return CudaEnvironment.from_system()

    @cache  # noqa: B019
    def generate_all_umd_values(self) -> list[str]:
        return [
            f"{major}.{minor}" if minor is not None else f"{major}"
            for major in reversed(self.UMD_MAJOR_RANGE)
            for minor in [*reversed(self.UMD_MINOR_RANGE), None]
        ]

    @cache  # noqa: B019
    def generate_all_sm_values(self) -> list[str]:
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
            f"{min(self.UMD_MAJOR_RANGE)}.{min(self.UMD_MINOR_RANGE)}"
        )
        max_umd_version = Version(
            f"{max(self.UMD_MAJOR_RANGE)}.{max(self.UMD_MINOR_RANGE)}"
        )

        if min_umd_version > umd_version or umd_version > max_umd_version:
            warnings.warn(
                f"The UMD version `{self.umd_version}` is outside the supported range: "
                f"`>={min_umd_version},<={max_umd_version}`.\n"
                "NVIDIA platform detection has been deactivated.",
                UserWarning,
                stacklevel=1,
            )
            return None

        return umd_version

    @property
    def get_sm_architectures(self) -> tuple[int, int] | None:
        """Get the SM architectures from the CUDA environment."""

        if sm_arch := os.environ.get("NV_VARIANT_PROVIDER_FORCE_SM_ARCH"):
            return tuple(map(int, sm_arch.split(".", maxsplit=1)))  # pyright: ignore[reportReturnType]

        if self._cuda_environment is None:
            return None

        return sorted(self._cuda_environment.architectures, reverse=True)[0]

    def get_supported_configs(
        self, known_properties: frozenset[VariantPropertyType] | None
    ) -> list[VariantFeatureConfig]:
        """Filter and sort the properties based on the plugin's logic."""

        if self.umd_version is None:
            return []

        keyconfigs: list[VariantFeatureConfig] = []

        # ============= User-Mode Driver (UMD) / LIBCUDA Driver Version ============= #

        # Priority 1 - UMD Lower Bound: `>= {major}.{minor}` or `>= {major}`
        # Example Output with Local UMD Version: 12.7
        # ['12.7', ..., '12.0', '12', '11.20', ..., '11.0', '11']
        umd_values_low = [
            f"{major}.{minor}" if minor is not None else f"{major}"
            for major in range(
                self.umd_version.major, min(self.UMD_MAJOR_RANGE) - 1, -1
            )
            for minor in (
                [*range(max(self.UMD_MINOR_RANGE), -1, -1), None]
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

        # Priority 2 - UMD Higher Bound (excluded): `< {major}.{minor}` or `< {major}`
        # Example Output with Local UMD Version: 12.7
        # ['15.20', ..., '15.0', '15', ...,  '13.0', '13', ..., '12.9', '12.8']
        umd_values_low = [
            f"{major}.{minor}" if minor is not None else f"{major}"
            for major in range(
                max(self.UMD_MAJOR_RANGE), self.umd_version.major - 1, -1
            )
            for minor in (
                [*range(max(self.UMD_MINOR_RANGE), -1, -1), None]
                if major > self.umd_version.major
                else range(max(self.UMD_MINOR_RANGE), self.umd_version.minor, -1)
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
        if sm_arch := self.get_sm_architectures:
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
                        )
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
            VariantFeatureConfig(
                NvidiaVariantFeatureKey.SM, self.generate_all_sm_values()
            ),
        ]

    def validate_property(self, variant_property: VariantPropertyType) -> bool:
        assert isinstance(variant_property, VariantPropertyType)
        assert variant_property.namespace == self.namespace

        if variant_property.feature in [
            f"{NvidiaVariantFeatureKey.CUDA}_lower_bound",
            f"{NvidiaVariantFeatureKey.CUDA}_upper_bound",
        ]:
            return variant_property.value in self.generate_all_umd_values()

        if variant_property.feature == NvidiaVariantFeatureKey.SM:
            return variant_property.value in self.generate_all_sm_values()

        warnings.warn(
            "Unknown variant feature received: "
            f"`nvidia :: {variant_property.feature}`.",
            UserWarning,
            stacklevel=1,
        )
        return False


if __name__ == "__main__":
    import os

    os.environ["NV_VARIANT_PROVIDER_FORCE_CUDA_DRIVER_VERSION"] = "12.7"
    os.environ["NV_VARIANT_PROVIDER_FORCE_SM_ARCH"] = "12.5"
    plugin = NvidiaVariantPlugin()

    # print(plugin.get_supported_configs(None))  # noqa: T201
    # print(plugin.get_all_configs(None))  # noqa: T201

    # May fail if `variantlib` is not installed: not an actual dependency
    # from variantlib.api import VariantProperty

    # print(  # noqa: T201
    #     plugin.validate_property(
    #         VariantProperty(
    #             namespace="nvidia", feature="cuda_version_lower_bound", value="12.0"
    #         )
    #     )
    # )

    # print(  # noqa: T201
    #     plugin.validate_property(
    #         VariantProperty(
    #             namespace="nvidia", feature="cuda_version_upper_bound", value="12.8"
    #         )
    #     )
    # )

    print(  # noqa: T201
        plugin.validate_property(
            VariantProperty(namespace="nvidia", feature="sm_arch", value="70_real")
        )
    )
