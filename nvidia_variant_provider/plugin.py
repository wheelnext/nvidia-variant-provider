from __future__ import annotations

import os
import warnings
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from packaging.specifiers import InvalidSpecifier
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from packaging.version import parse as parse_version

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


class NvidiaVariantFeatureKey(StrEnum):
    DRIVER = "driver"
    CUDA = "cuda_version"
    SM = "sm_arch"
    CTK = "ctk"


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
    def kmd_version(self) -> Version | None:
        if driver_ver := os.environ.get("NV_VARIANT_PROVIDER_FORCE_KMD_DRIVER_VERSION"):
            return Version(driver_ver)

        if (cuda_env := self._cuda_environment) is None:
            return None

        return (
            Version(cuda_env.system_driver_versions)
            if cuda_env.system_driver_versions
            else None
        )

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

        # Priority 1: Kernel-Mode Driver (KMD) Version

        key = NvidiaVariantFeatureKey.DRIVER
        if (kmd_version := self.kmd_version) is not None:
            vprops_specset: list[InputPreservingSpecifierSet] = []
            for vprop in prop_values.get(key, []):
                try:
                    vprops_specset.append(InputPreservingSpecifierSet(vprop))
                except InvalidSpecifier:  # noqa: PERF203
                    warnings.warn(
                        f"The variant property `{self.namespace} :: {key} :: {vprop}` "
                        f"is not a valid `SpecifierSet`. Will be ignored.",
                        UserWarning,
                        stacklevel=1,
                    )

            vprops_specset = sort_specifier_sets(vprops_specset)
            vprops_specset.reverse()  # most generic to most specific, forward first

            kmd_valid_values = [
                specset.original_value
                for specset in vprops_specset
                if kmd_version in specset
            ]

            if kmd_valid_values:
                keyconfigs.append(
                    VariantFeatureConfig(
                        name=key,
                        values=kmd_valid_values,
                    )
                )

        # Priority 2: User-Mode Driver (UMD) / LIBCUDA Driver Version

        key = NvidiaVariantFeatureKey.CUDA
        if (umd_version := self.umd_version) is not None:
            vprops_specset: list[InputPreservingSpecifierSet] = []
            for vprop in prop_values.get(key, []):
                try:
                    vprops_specset.append(InputPreservingSpecifierSet(vprop))
                except InvalidSpecifier:  # noqa: PERF203
                    warnings.warn(
                        f"The variant property `{self.namespace} :: {key} :: {vprop}` "
                        f"is not a valid `SpecifierSet`. Will be ignored.",
                        UserWarning,
                        stacklevel=1,
                    )

            vprops_specset = sort_specifier_sets(vprops_specset)
            vprops_specset.reverse()  # most generic to most specific, forward first

            umd_valid_values = [
                specset.original_value
                for specset in vprops_specset
                if umd_version in specset
            ]

            if umd_valid_values:
                keyconfigs.append(
                    VariantFeatureConfig(
                        name=key,
                        values=umd_valid_values,
                    )
                )

        # Priority 3: SM Architectures
        # - All versions are supported by the plugin - purely sort:
        #    - Bigger numbers first
        #    - Real architectures first, then virtual architectures

        if architectures := prop_values.get(NvidiaVariantFeatureKey.SM, []):

            def sort_key(item: str) -> tuple[int, int]:
                num_str, type_str = item.split("_")
                num = int(num_str)
                # real = 0, virtual = 1 to prioritize 'real' first
                type_priority = 0 if type_str == "real" else 1
                return (-num, type_priority)

            if architectures:
                keyconfigs.append(
                    VariantFeatureConfig(
                        name=NvidiaVariantFeatureKey.SM,
                        values=sorted(architectures, key=sort_key),
                    )
                )

        # Priority 4: CTK
        # - All versions are supported by the plugin - purely sort by version
        # - Used exclusively for dependency resolution

        if ctks := prop_values.get(NvidiaVariantFeatureKey.CTK, []):
            keyconfigs.append(
                VariantFeatureConfig(
                    name=NvidiaVariantFeatureKey.CTK,
                    values=sorted(ctks, key=parse_version, reverse=True),
                )
            )

        return keyconfigs

    def get_all_configs(
        self, known_properties: list[VariantPropertyType] | None
    ) -> list[VariantFeatureConfig]:
        """Not used - transparently returns all known properties as configs."""

        if known_properties is None:
            return []

        prop_values: dict[VariantFeatureName, list[VariantFeatureValue]] = defaultdict(
            list
        )
        for vprop in known_properties:
            prop_values[vprop.feature].append(vprop.value)

        return [
            VariantFeatureConfig(name, list(set(values)))
            for name, values in prop_values.items()
        ]

    def validate_property(self, variant_property: VariantPropertyType) -> bool:
        assert variant_property.namespace == self.namespace
        return True
