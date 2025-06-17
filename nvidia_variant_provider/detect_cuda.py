# Copyright (C) 2012 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause
"""Detect CUDA version."""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass

import nvidia_variant_provider.pynvml as _pynvml

NVIDIA_GPU_ARCHITECTURE = {
    _pynvml.NVML_DEVICE_ARCH_KEPLER: "Kepler",  # example: Tesla K80
    _pynvml.NVML_DEVICE_ARCH_MAXWELL: "Maxwell",  # example: Tesla M40
    _pynvml.NVML_DEVICE_ARCH_PASCAL: "Pascal",  # example: Tesla P100
    _pynvml.NVML_DEVICE_ARCH_VOLTA: "Volta",  # example: Tesla V100, Titan V
    _pynvml.NVML_DEVICE_ARCH_TURING: "Turing",  # example: GeForce RTX 2080 Ti, Tesla T4
    _pynvml.NVML_DEVICE_ARCH_AMPERE: "Ampere",  # example: GeForce RTX 3080, A100
    _pynvml.NVML_DEVICE_ARCH_ADA: "Ada",  # example: GeForce RTX 4090, L40
    _pynvml.NVML_DEVICE_ARCH_HOPPER: "Hopper",  # example: H100, H800
    _pynvml.NVML_DEVICE_ARCH_BLACKWELL: "Blackwell",  # example: B100
    _pynvml.NVML_DEVICE_ARCH_T23X: "Tegra",  # Tegra SKUs
    _pynvml.NVML_DEVICE_ARCH_UNKNOWN: "Unknown",  # Error
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
            _pynvml.nvmlInit()

            # UMD / KMD Version
            system_driver_version = _pynvml.nvmlSystemGetDriverVersion()

            # CUDA Driver
            cuda_driver_version = None
            try:
                cuda_driver_version = str(_pynvml.nvmlSystemGetCudaDriverVersion_v2())
            except _pynvml.NVMLError:
                # Fallback to the v1 API if the v2 API is not available
                with contextlib.suppress(_pynvml.NVMLError):
                    cuda_driver_version = str(_pynvml.nvmlSystemGetCudaDriverVersion())

            if cuda_driver_version is not None:
                with contextlib.suppress(ValueError, TypeError):
                    cuda_driver_version = (
                        f"{int(cuda_driver_version) // 1000}."
                        f"{(int(cuda_driver_version) % 1000) // 10}"
                    )

            gpu_families: set[str] = set()
            architectures: set[str] = set()

            for device_id in range(_pynvml.nvmlDeviceGetCount()):
                device = _pynvml.nvmlDeviceGetHandleByIndex(device_id)
                with contextlib.suppress(_pynvml.NVMLError, KeyError):
                    arch = NVIDIA_GPU_ARCHITECTURE[
                        _pynvml.nvmlDeviceGetArchitecture(device=device)
                    ]
                    gpu_families.add(arch)

                with contextlib.suppress(_pynvml.NVMLError):
                    cc = _pynvml.nvmlDeviceGetCudaComputeCapability(handle=device)
                    architectures.add(f"sm_{''.join([str(i) for i in cc])}")

            return cls(
                system_driver_versions=system_driver_version,
                cuda_driver_version=cuda_driver_version,
                gpu_families=list(gpu_families),
                architectures=list(architectures),
            )

        except _pynvml.NVMLError:
            return None

        finally:
            with contextlib.suppress(_pynvml.NVMLError):
                # Shutdown NVML to release resources
                _pynvml.nvmlShutdown()


if __name__ == "__main__":
    print(f"{CudaEnvironment.from_system()=}")  # noqa: T201
