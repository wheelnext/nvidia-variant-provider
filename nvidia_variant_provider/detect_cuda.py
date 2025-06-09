# Copyright (C) 2012 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause
"""Detect CUDA version."""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass

import pynvml

NVIDIA_GPU_ARCHITECTURE = {
    pynvml.NVML_DEVICE_ARCH_KEPLER: "Kepler",  # example: GeForce GTX 680, GeForce GTX 780, Tesla K80
    pynvml.NVML_DEVICE_ARCH_MAXWELL: "Maxwell",  # example: GeForce GTX 750 Ti, GeForce GTX 980, Tesla M40
    pynvml.NVML_DEVICE_ARCH_PASCAL: "Pascal",  # example: GeForce GTX 1080 Ti, GeForce GTX 1060, Tesla P100
    pynvml.NVML_DEVICE_ARCH_VOLTA: "Volta",  # example: Tesla V100, Titan V
    pynvml.NVML_DEVICE_ARCH_TURING: "Turing",  # example: GeForce RTX 2080 Ti, GeForce GTX 1660 Ti, Tesla T4
    pynvml.NVML_DEVICE_ARCH_AMPERE: "Ampere",  # example: GeForce RTX 3080, GeForce RTX 3060, A100
    pynvml.NVML_DEVICE_ARCH_ADA: "Ada",  # example: GeForce RTX 4090, GeForce RTX 4080, L40
    pynvml.NVML_DEVICE_ARCH_HOPPER: "Hopper",  # example: H100, H800
    pynvml.NVML_DEVICE_ARCH_BLACKWELL: "Blackwell",  # example: B100
    pynvml.NVML_DEVICE_ARCH_T23X: "Tegra",  # Tegra SKUs
    pynvml.NVML_DEVICE_ARCH_UNKNOWN: "Unknown",  # Error
}


@dataclass(frozen=True, order=True)
class CudaEnvironment:
    system_driver_versions: str | None
    cuda_driver_version: str | None
    gpu_families: list[str]
    architectures: list[str]

    @classmethod
    @functools.lru_cache(maxsize=1)
    def from_system(cls) -> CudaEnvironment | None:
        """Get the CUDA environment from the system."""
        try:
            pynvml.nvmlInit()

            # UMD / KMD Version
            system_driver_version = pynvml.nvmlSystemGetDriverVersion()

            # CUDA Driver
            cuda_driver_version = None
            try:
                cuda_driver_version = str(pynvml.nvmlSystemGetCudaDriverVersion_v2())
            except pynvml.NVMLError:
                # Fallback to the v1 API if the v2 API is not available
                with contextlib.suppress(pynvml.NVMLError):
                    cuda_driver_version = str(pynvml.nvmlSystemGetCudaDriverVersion())

            if cuda_driver_version is not None:
                with contextlib.suppress(ValueError, TypeError):
                    cuda_driver_version = (
                        f"{int(cuda_driver_version) // 1000}."
                        f"{(int(cuda_driver_version) % 1000) // 10}"
                    )

            gpu_families: set[str] = set()
            architectures: set[str] = set()

            for device_id in range(pynvml.nvmlDeviceGetCount()):
                device = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                with contextlib.suppress(pynvml.NVMLError, KeyError):
                    arch = NVIDIA_GPU_ARCHITECTURE[
                        pynvml.nvmlDeviceGetArchitecture(device=device)
                    ]
                    gpu_families.add(arch)

                with contextlib.suppress(pynvml.NVMLError):
                    cc = pynvml.nvmlDeviceGetCudaComputeCapability(handle=device)
                    architectures.add(f"sm_{''.join([str(i) for i in cc])}")

            return cls(
                system_driver_versions=system_driver_version,
                cuda_driver_version=cuda_driver_version,
                gpu_families=list(gpu_families),
                architectures=list(architectures),
            )

        except pynvml.NVMLError:
            return None

        finally:
            with contextlib.suppress(pynvml.NVMLError):
                # Shutdown NVML to release resources
                pynvml.nvmlShutdown()


if __name__ == "__main__":
    print(f"{CudaEnvironment.from_system()=}")  # noqa: T201
