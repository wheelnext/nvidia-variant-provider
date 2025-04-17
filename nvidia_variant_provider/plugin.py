from __future__ import annotations

import os
from functools import cached_property

from packaging import version

from nvidia_variant_provider.detect_cuda import CudaEnvironment
from nvidia_variant_provider.detect_cuda import get_cuda_version
from variantlib.models.provider import VariantFeatureConfig


class NvidiaVariantPlugin:
    namespace = "nvidia"

    def _get_supposer_driver(self) -> list[str]:
        """Lookup the system to decide what `nvidia :: drivers` is locally supported.
        Returns a list of strings in order of priority."""

        if self.cuda_environment is None:
            return None

        driver_version = version.parse(self.cuda_environment.version)

        # Descending list (i.e. priority) => 12.4, 12.3, 12.2, ..., 12.0
        # Backward compatibility only for now
        # TODO: Add forward compability when it makes sense.
        return [
            f"{driver_version.major}.{minor}"
            for minor in range(driver_version.minor, stop=-1, step=-1)
        ]

    def get_supported_configs(self) -> list[VariantFeatureConfig]:
        keyconfigs = []

        # Top Priority
        if (values := self._get_supposer_driver()) is not None:
            keyconfigs.append(VariantFeatureConfig(name="driver", values=values))

        return keyconfigs

    def get_all_configs(self) -> list[VariantFeatureConfig]:
        return [
            VariantFeatureConfig(
                name="driver",
                values=(
                    [f"11.{minor}" for minor in range(1, 9)]
                    + [f"12.{minor}" for minor in range(1, 9)]
                ),
            )
        ]

    @cached_property
    def cuda_environment(self) -> CudaEnvironment | None:
        if driver_ver := os.environ.get("NV_PROVIDER_FORCE_DRIVER_VERSION") is not None:
            return driver_ver

        return get_cuda_version()
