from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property

from nvidia_variant_provider.detect_cuda import NVIDIA_GPU_ARCHITECTURE
from nvidia_variant_provider.detect_cuda import CudaEnvironment


@dataclass(frozen=True)
class VariantFeatureConfig:
    name: str

    # Acceptable values in priority order
    values: list[str]


DRIVER_KEY = "driver"
CUDA_KEY = "cuda"
GPU_FAMILY_KEY = "gpu_family"
SM_ARCH_KEY = "sm_arch"

LATEST_CUDA_MINOR_VERSIONS = {11: 8, 12: 9}


class NvidiaVariantPlugin:
    namespace = "nvidia"

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
    def gpu_families(self) -> list[str] | None:
        if gpu_families := os.environ.get("NV_VARIANT_PROVIDER_FORCE_GPU_FAMILIES"):
            return gpu_families.split(",")

        if (cuda_env := self._cuda_environment) is None:
            return None

        return cuda_env.gpu_families

    @property
    def architectures(self) -> list[str] | None:
        if architectures := os.environ.get(
            "NV_VARIANT_PROVIDER_FORCE_GPU_ARCHITECTURES"
        ):
            return architectures.split(",")

        if (cuda_env := self._cuda_environment) is None:
            return None

        return cuda_env.architectures

    def get_supported_configs(self) -> list[VariantFeatureConfig]:
        keyconfigs = []

        # Priority 1: User-Mode Driver (UMD) Version
        if (umd_ver := self.umd_driver) is not None:
            keyconfigs.append(VariantFeatureConfig(name=DRIVER_KEY, values=[umd_ver]))

        # Priority 2: CUDA Driver Version
        if (cuda_ver := self.cuda_driver) is not None:
            keyconfigs.append(VariantFeatureConfig(name=CUDA_KEY, values=[cuda_ver]))

        # Priority 3: SM Architectures
        if (architectures := self.architectures) is not None:
            keyconfigs.append(
                VariantFeatureConfig(name=SM_ARCH_KEY, values=architectures)
            )

        # Priority 4: GPU Families
        if (gpu_families := self.gpu_families) is not None:
            keyconfigs.append(
                VariantFeatureConfig(
                    name=GPU_FAMILY_KEY, values=[s.lower() for s in gpu_families]
                )
            )

        return keyconfigs

    def get_all_configs(self) -> list[VariantFeatureConfig]:
        # https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
        # The lowest supported UMD Driver for CUDA version 11.0 is 450.36.06
        return [
            VariantFeatureConfig(  # generates 450k values - temporary idea
                name=DRIVER_KEY,
                values=[
                    f"{major}.{minor:02d}.{patch:02d}"
                    for major in range(450, 600)
                    for minor in range(200)
                    for patch in range(15)
                ],
            ),
            VariantFeatureConfig(
                name=CUDA_KEY,
                values=(
                    [
                        f"{major}.{minor}"
                        for major in LATEST_CUDA_MINOR_VERSIONS
                        for minor in range(LATEST_CUDA_MINOR_VERSIONS[major] + 1)
                    ]
                    + [str(major) for major in LATEST_CUDA_MINOR_VERSIONS]
                ),
            ),
            VariantFeatureConfig(
                name=SM_ARCH_KEY, values=[f"sm_{major}" for major in range(50, 110)]
            ),
            VariantFeatureConfig(
                name=GPU_FAMILY_KEY,
                values=[s.lower() for s in NVIDIA_GPU_ARCHITECTURE.values()],
            ),
        ]


if __name__ == "__main__":
    plugin = NvidiaVariantPlugin()
    print(plugin.get_supported_configs())  # noqa: T201
    # print(plugin.get_all_configs())
