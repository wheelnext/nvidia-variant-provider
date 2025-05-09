from __future__ import annotations

import os
from functools import cached_property

from packaging import version

from nvidia_variant_provider.detect_cuda import CudaEnvironment
from nvidia_variant_provider.detect_cuda import get_cuda_environment
from variantlib.models.provider import VariantFeatureConfig

LATEST_CUDA_MINOR_VERSIONS = {11: 8, 12: 8}


DRIVER_KEY = "cuda"


class NvidiaVariantPlugin:
    namespace = "nvidia"

    def _get_supposer_driver(self) -> list[str]:
        """Lookup the system to decide what `nvidia :: drivers` is locally supported.
        Returns a list of strings in order of priority."""

        if self.cuda_driver_version is None:
            return None

        driver_version = version.parse(self.cuda_driver_version)

        # Descending list (i.e. priority) => 12.4, 12.3, 12.2, ..., 12.0, 12
        return [
            f"{driver_version.major}.{minor}"
            for minor in range(driver_version.minor, -1, -1)
        ] + [str(driver_version.major)]

    def get_supported_configs(self) -> list[VariantFeatureConfig]:
        keyconfigs = []

        # Top Priority
        if (values := self._get_supposer_driver()) is not None:
            keyconfigs.append(VariantFeatureConfig(name=DRIVER_KEY, values=values))

        return keyconfigs

    def get_all_configs(self) -> list[VariantFeatureConfig]:
        return [
            VariantFeatureConfig(
                name=DRIVER_KEY,
                values=(
                    [
                        f"{major}.{minor}"
                        for major in LATEST_CUDA_MINOR_VERSIONS
                        for minor in range(LATEST_CUDA_MINOR_VERSIONS[major] + 1)
                    ]
                    + [str(major) for major in LATEST_CUDA_MINOR_VERSIONS]
                ),
            )
        ]

    @cached_property
    def cuda_environment(self) -> CudaEnvironment | None:
        return get_cuda_environment()

    @property
    def cuda_driver_version(self) -> str | None:
        if driver_ver := os.environ.get("NV_PROVIDER_FORCE_DRIVER_VERSION"):
            return driver_ver

        return self.cuda_environment.version if self.cuda_environment else None
