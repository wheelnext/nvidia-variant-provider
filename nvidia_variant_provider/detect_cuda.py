"""Detect CUDA version."""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass

import nvidia_variant_provider.pynvml as _pynvml


@dataclass(frozen=True, order=True)
class CudaEnvironment:
    system_driver_versions: str | None
    cuda_driver_version: str | None
    architectures: list[tuple[int, int]]

    @classmethod
    @functools.lru_cache(maxsize=1)
    def from_system(cls) -> CudaEnvironment | None:
        """Get the CUDA environment from the system."""
        try:
            _pynvml.nvmlInit()

            # KMD Version: (e.g. 525.85.12)
            system_driver_version = _pynvml.nvmlSystemGetDriverVersion()

            # UMD / LIBCUDA Driver Version: (e.g. 12.5.1)
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

            architectures: set[tuple[int, int]] = set()

            for device_id in range(_pynvml.nvmlDeviceGetCount()):
                device = _pynvml.nvmlDeviceGetHandleByIndex(device_id)
                with contextlib.suppress(_pynvml.NVMLError):
                    architectures.add(
                        _pynvml.nvmlDeviceGetCudaComputeCapability(handle=device)
                    )

            return cls(
                system_driver_versions=system_driver_version,
                cuda_driver_version=cuda_driver_version,
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
