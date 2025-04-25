# Copyright (C) 2012 Anaconda, Inc
# SPDX-License-Identifier: BSD-3-Clause
"""Detect CUDA version."""

from __future__ import annotations

import ctypes
import functools
import itertools
import multiprocessing
import os
import platform
from contextlib import suppress
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class CudaEnvironment:
    version: str
    architectures: list[str]


@functools.cache
def get_cuda_environment() -> CudaEnvironment | None:
    # Do not inherit file descriptors and handles from the parent process.
    # The `fork` start method should be considered unsafe as it can lead to
    # crashes of the subprocess. The `spawn` start method is preferred.
    context = multiprocessing.get_context("spawn")
    queue = context.SimpleQueue()
    # Spawn a subprocess to detect the CUDA version
    detector = context.Process(
        target=_cuda_detector_target,
        args=(queue,),
        name="CUDA driver version detector",
        daemon=True,
    )
    try:
        detector.start()
        detector.join(timeout=60.0)
    finally:
        # Always cleanup the subprocess
        detector.kill()  # requires Python 3.7+

    if queue.empty():
        return None

    result: str | None = queue.get()

    if result is None:
        return result

    driver_version, architectures = result.split(";")
    return CudaEnvironment(driver_version, architectures.split(","))


def _cuda_detector_target(queue: multiprocessing.SimpleQueue):
    """
    Attempt to detect the version of CUDA present in the operating system in a
    subprocess.

    On Windows and Linux, the CUDA library is installed by the NVIDIA
    driver package, and is typically found in the standard library path,
    rather than with the CUDA SDK (which is optional for running CUDA apps).

    On macOS, the CUDA library is only installed with the CUDA SDK, and
    might not be in the library path.

    Returns: version string with CUDA version first, then a set of unique SM's for the
             GPUs present in the system
             (e.g., '12.4;8.6,9.0') or None if CUDA is not found.
             The result is put in the queue rather than a return value.
    """
    # Platform-specific libcuda location
    system = platform.system()
    if system == "Darwin":
        lib_filenames = [
            "libcuda.1.dylib",  # check library path first
            "libcuda.dylib",
            "/usr/local/cuda/lib/libcuda.1.dylib",
            "/usr/local/cuda/lib/libcuda.dylib",
        ]
    elif system == "Linux":
        lib_filenames = [
            "libcuda.so",  # check library path first
            "/usr/lib64/nvidia/libcuda.so",  # RHEL/Centos/Fedora
            "/usr/lib/x86_64-linux-gnu/libcuda.so",  # Ubuntu
            "/usr/lib/wsl/lib/libcuda.so",  # WSL
        ]
        # Also add libraries with version suffix `.1`
        lib_filenames = list(
            itertools.chain.from_iterable((f"{lib}.1", lib) for lib in lib_filenames)
        )
    elif system == "Windows":
        bits = platform.architecture()[0].replace("bit", "")  # e.g. "64" or "32"
        lib_filenames = [f"nvcuda{bits}.dll", "nvcuda.dll"]
    else:
        queue.put(None)  # CUDA not available for other operating systems
        return

    # Open library
    dll = ctypes.windll if system == "Windows" else ctypes.cdll
    for lib_filename in lib_filenames:
        with suppress(Exception):
            libcuda = dll.LoadLibrary(lib_filename)
            break
    else:
        queue.put(None)
        return

    # Empty `CUDA_VISIBLE_DEVICES` can cause `cuInit()` returns `CUDA_ERROR_NO_DEVICE`
    # Invalid `CUDA_VISIBLE_DEVICES` can cause `cuInit()` returns `CUDA_ERROR_INVALID_DEVICE`  # noqa: E501
    # Unset this environment variable to avoid these errors
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    # Get CUDA version
    try:
        cuInit = libcuda.cuInit
        flags = ctypes.c_uint(0)
        ret = cuInit(flags)
        if ret != 0:
            queue.put(None)
            return

        cuDriverGetVersion = libcuda.cuDriverGetVersion
        version_int = ctypes.c_int(0)
        ret = cuDriverGetVersion(ctypes.byref(version_int))
        if ret != 0:
            queue.put(None)
            return

        # Convert version integer to version string
        value = version_int.value
        version_value = f"{value // 1000}.{(value % 1000) // 10}"

        count = ctypes.c_int(0)
        libcuda.cuDeviceGetCount(ctypes.pointer(count))

        architectures = set()
        for device in range(count.value):
            major = ctypes.c_int(0)
            minor = ctypes.c_int(0)
            libcuda.cuDeviceComputeCapability(
                ctypes.pointer(major), ctypes.pointer(minor), device
            )
            architectures.add(f"{major.value}.{minor.value}")
        queue.put(f"{version_value};{','.join(architectures)}")
    except Exception:  # noqa: BLE001
        queue.put(None)
        return


if __name__ == "__main__":
    print(get_cuda_version())  # noqa: T201
