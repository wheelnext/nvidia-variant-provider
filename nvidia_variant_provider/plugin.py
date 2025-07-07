from __future__ import annotations

import os
import warnings
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from packaging.specifiers import InvalidSpecifier
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from packaging.version import parse as parse_version

from nvidia_variant_provider.detect_cuda import NVIDIA_GPU_ARCHITECTURE
from nvidia_variant_provider.detect_cuda import CudaEnvironment
from nvidia_variant_provider.version_sort import sort_specifier_sets

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


DRIVER_KEY = "driver"
CUDA_KEY = "cuda_version"
SM_ARCH_KEY = "sm_arch"
CTK_KEY = "ctk"

LATEST_CUDA_MINOR_VERSIONS = {11: 8, 12: 9}


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
    dynamic = True

    @cached_property
    def _cuda_environment(self) -> CudaEnvironment | None:
        """Lookup the system to determine the driver / GPU state."""

        return CudaEnvironment.from_system()

    @property
    def umd_driver(self) -> str | None:
        if driver_ver := os.environ.get("NV_VARIANT_PROVIDER_FORCE_UMD_DRIVER_VERSION"):
            return driver_ver

        if (cuda_env := self._cuda_environment) is None:
            return None

        return cuda_env.system_driver_versions

    @property
    def cuda_driver(self) -> str | None:
        if driver_ver := os.environ.get(
            "NV_VARIANT_PROVIDER_FORCE_CUDA_DRIVER_VERSION"
        ):
            return driver_ver

        if (cuda_env := self._cuda_environment) is None:
            return None

        return cuda_env.cuda_driver_version

    @property
    def architectures(self) -> list[str] | None:
        if architectures := os.environ.get(
            "NV_VARIANT_PROVIDER_FORCE_GPU_ARCHITECTURES"
        ):
            return architectures.split(",")

        if (cuda_env := self._cuda_environment) is None:
            return None

        return cuda_env.architectures

    def get_supported_configs(
        self, known_properties: frozenset[VariantPropertyType] | None
    ) -> list[VariantFeatureConfig]:
        """Filter and sort the properties based on the plugin's logic."""

        # 1.A - Validation: Validate input types.
        assert isinstance(known_properties, frozenset)
        assert all(isinstance(vprop, VariantPropertyType) for vprop in known_properties)

        if not known_properties:
            # Nothing In => Nothing Out
            return []

        # ============================================================================ #
        # A. Preprocessing: Sort the properties by feature, and value.
        # ============================================================================ #

        prop_values: dict[VariantFeatureName, list[VariantFeatureValue]] = defaultdict(
            list
        )
        for vprop in known_properties:
            prop_values[vprop.feature].append(vprop.value)

        # ============================================================================ #
        # B. Validation: Ensure all properties belong to the proper namespace.
        # ============================================================================ #

        issues_found: list[str] = [
            f"Property `{vprop}` does not belong to namespace {self.namespace}"
            for vprop in known_properties
            if vprop.namespace != self.namespace
        ]
        if issues_found:
            raise ValueError(
                f"Non compatible properties found in variant plugin "
                f"`{self.namespace}`:"
                + ("\n- " + ("\n- ".join(issues_found)) if issues_found else "")
            )

        # ============================================================================ #
        # C. Processing: Filter and sort supported variant property values
        # ============================================================================ #

        keyconfigs: list[VariantFeatureConfig] = []

        # Priority 1: User-Mode Driver (UMD) Version

        if (umd_ver := self.umd_driver) is not None:
            keyconfigs.append(VariantFeatureConfig(name=DRIVER_KEY, values=[umd_ver]))

        # Priority 2: CUDA Driver Version

        if (cuda_ver := self.cuda_driver) is not None:
            keyconfigs.append(VariantFeatureConfig(name=CUDA_KEY, values=[cuda_ver]))

        # Priority 3: SM Architectures

        if architectures := prop_values.get(SM_ARCH_KEY, []):

            def sort_key(item: str) -> tuple[int, int]:
                num_str, type_str = item.split("_")
                num = int(num_str)
                # real = 0, virtual = 1 to prioritize 'real' first
                type_priority = 0 if type_str == "real" else 1
                return (-num, type_priority)

            keyconfigs.append(
                VariantFeatureConfig(
                    name=SM_ARCH_KEY, values=sorted(architectures, key=sort_key)
                )
            )

        # Priority 4: CTK
        # - All versions supported by the plugin - purely sort by version
        # - Used exclusively for dependency resolution

        if ctks := prop_values.get(CTK_KEY, []):
            keyconfigs.append(
                VariantFeatureConfig(
                    name=CTK_KEY,
                    values=sorted(ctks, key=parse_version, reverse=True),
                )
            )

        return keyconfigs

    def get_all_configs(
        self, known_properties: list[VariantPropertyType] | None
    ) -> list[VariantFeatureConfig]:
        """Get all supported configurations, including those not in the known
        properties."""
        if known_properties is None:
            return []

        prop_values: dict[VariantFeatureName, list[VariantFeatureValue]] = defaultdict(
            list
        )
        for vprop in known_properties:
            prop_values[vprop.feature].append(vprop.value)

        return [
            VariantFeatureConfig(name, values) for name, values in prop_values.items()
        ]
