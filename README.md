# NVIDIA Variant Provider Plugin

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)
[![Wheel Variant](https://img.shields.io/badge/Wheel_Variant_Plugin-NVIDIA-green)](https://wheelnext.dev)

A variant provider plugin for the Wheel Variant upcoming proposed standard that enables automatic detection and selection of NVIDIA GPU-optimized Python packages.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Building Variant Wheels](#building-variant-wheels)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

The NVIDIA Variant Provider Plugin is part of the work conducted under [WheelNext](https://wheelnext.dev) initiative to "Re-invent the Wheel" and Python package distribution for scientific computing and hardware-accelerated computing. 

This package provides the logic to automatically detects NVIDIA GPU capabilities and CUDA environments to select the most optimized package variants for your system.

### Key Benefits

- **Automatic GPU Detection**: Detects CUDA driver versions, NVIDIA GPU compute capabilities
- **Seamless Integration**: Works transparently with pip, uv, and other Python package installers (when they will support Wheel Variants)
- **Backward Compatible**: Maintains compatibility with existing Python packaging infrastructure

### The Problem

Current Python wheels for GPU-accelerated packages like PyTorch can be non trivial to install. Users must manually:
- Check their CUDA version
- Navigate to special package indexes
- Deal with compatibility matrices

### The Solution

With Wheel Variants and this plugin, users can simply run:

```bash
[uv] pip install torch
```

The plugin automatically:

- Detects your GPU (e.g., RTX 4090 with compute capability 8.9)
- Identifies your CUDA driver version
- Downloads the right version of `torch` (or other requested package) compatible with your machine.

## Features

### Detected Hardware Properties

1. **User-Mode Driver (UMD) Version**
   - System NVIDIA driver version (e.g., "12.9")
   - Follows CUDA version compatibility rules

2. **GPU Architecture (Compute Capability)**
   - Determine the compute capability available on the system.
   - Resolve with compute capability compatibility in mind.
   - Follows the `CMAKE` flag standard for clarity: `{major}{minor}_[real|virtual]`

## Installation

This package is automatically installed when necessary, it is **not necessary** to download and install this package. However, if you still wish to do so, here is how:

### Install from PyPI (when available)

```bash
pip install nvidia-variant-provider
```

### Install from Source

```bash
pip install "nvidia-variant-provider @ git+https://github.com/wheelnext/nvidia-variant-provider.git"
```

## How It Works

### Variant Property Format

Variant Properties emitted follow a three-tuple structure

```
namespace :: feature :: value
```

Examples:
- `nvidia :: cuda_version_lower_bound :: 12.6` - Means: `CUDA version >= 12.6`
- `nvidia :: cuda_version_upper_bound :: 13`   - Means: `CUDA version < 13`
- `nvidia :: sm_arch :: 90_real`               - Means compatible with CMAKE flag `90_real`

### Detection Process

1. **Initialization**: The plugin uses NVIDIA Management Library (NVML) to query system information
2. **Hardware Detection**: Identifies all NVIDIA GPUs and their capabilities, read the NVIDIA User-Mode Driver version.
3. **Property Generation**: Creates variant properties in the format `nvidia :: feature :: value`
4. **Priority Ordering**: Returns features in order of importance for package selection

## Usage

### For End Users

Once installed, the plugin works automatically with variant-aware package installers:

```bash
# Automatic variant selection
pip install torch  # Automatically selects GPU-optimized variant

# Force specific variant (if needed)
pip install "torch#cu129"
pip install "torch==2.8.0#cu129"

# Disable variant selection
pip install --no-variant torch
```

### DEBUG - Overwrite Detection

This plugin includes 2 environments variable to overwrite the detection mechanism and force the resolution to anything you wish.

This can be used to either work around a known problem with this plugin, debug the installer toolchain without an actual NVIDIA GPU or any other purpose.

**Disclaimer:•• Using this feature may lead you to a non functional installation.

```bash
export NV_VARIANT_PROVIDER_FORCE_CUDA_DRIVER_VERSION = "12.9"  # replace with the value you want to try
export NV_VARIANT_PROVIDER_FORCE_SM_ARCH = "9.0"               # replace with the value you want to try
```

**Example Usage:**

```bash
# Test with specific CUDA version
export NV_VARIANT_PROVIDER_FORCE_CUDA_DRIVER_VERSION=12.8
pip install torch

# Test with specific architecture
export NV_VARIANT_PROVIDER_FORCE_SM_ARCH=10.0
pip install torch
```

## Configuring Your Project

Add variant configuration to your `pyproject.toml`:

```toml
[variant.default-priorities]
namespace = ["nvidia"]

[variant.providers.nvidia]
requires = ["nvidia-variant-provider>=0.0.1,<1.0.0"]
enable-if = "platform_system == 'Linux' or platform_system == 'Windows'"
plugin-api = "nvidia_variant_provider.plugin:NvidiaVariantPlugin"
```

## API Reference

### NvidiaVariantPlugin

The main plugin class that implements hardware detection.

```python
from nvidia_variant_provider.plugin import NvidiaVariantPlugin

plugin = NvidiaVariantPlugin()
assert plugin.namespace == "nvidia"
```

#### Methods

##### `get_supported_configs() -> list[VariantFeatureConfig]`

Returns the list of supported variant configurations based on the current system.

```python
configs = plugin.get_supported_configs()
# Returns:
# [
#   VariantFeatureConfig(name='cuda_version_lower_bound', values=['12.8', '12.7', ...]),
#   VariantFeatureConfig(name='cuda_version_upper_bound', values=[..., '13.0', '12.9']),
#   VariantFeatureConfig(name='sm_arch', values=['95_real', ..., '90_real', '90_virtual']),
# ]
```

##### `get_all_configs() -> list[VariantFeatureConfig]`

Returns all possible variant configurations (for validation/testing).

### Example 3: Project Configuration

Complete `pyproject.toml` example:

```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-cuda-package"
version = "1.0.0"
dependencies = [
    "numpy",
    "nvidia-cuda-runtime-cu12 ; 'nvidia::cuda::12' in variant_properties",
    "nvidia-cuda-runtime-cu11 ; 'nvidia::cuda::11' in variant_properties",
]

[variant.providers.nvidia]
requires = ["nvidia-variant-provider>=0.0.1,<1.0.0"]
enable-if = "platform_system == 'Linux' or platform_system == 'Windows'"
plugin-api = "nvidia_variant_provider.plugin:NvidiaVariantPlugin"

[variant.default-priorities]
namespace = ["nvidia"]
```

## Troubleshooting

### Common Issues

#### 1. NVML Initialization Failed

**Problem**: Error message about NVML
```
NVMLError: Initialization error
```

**Solution**: 
- Ensure NVIDIA drivers are properly installed
- Check if `nvidia-smi` command works
- May need to install NVIDIA CUDA Driver

#### 2. No Variants Found

**Problem**: Falls back to generic wheel despite having GPU

**Solution**: Check detection output:
```python
from nvidia_variant_provider.detect_cuda import CudaEnvironment
print(CudaEnvironment.from_system())
```

#### 3. Wrong Variant Selected

**Problem**: Incorrect variant being chosen

**Solution**: Check variant priorities in package's `pyproject.toml` and use explicit selection:
```bash
pip install package-name#cu126
```

### Reporting Issues

Please report issues on our [GitHub Issues](https://github.com/wheelnext/nvidia-variant-provider/issues) page with:
- System information (OS, Python version)
- GPU information (`nvidia-smi` output)
- Complete error messages
- Steps to reproduce

## License

This project is licensed under the Apache 2 License - see the [LICENSE](LICENSE) file for details.

## Related Projects

- [variantlib](https://github.com/wheelnext/variantlib) - Core library for variant support
- [WheelNext](https://wheelnext.dev) - The broader initiative for next-generation Python packaging
- [PEP XXX](https://wheelnext.dev/proposals/pepxxx_wheel_variant_support/) - The Wheel Variants proposal

---

For more information about the WheelNext initiative and Wheel Variants, visit [wheelnext.dev](https://wheelnext.dev).