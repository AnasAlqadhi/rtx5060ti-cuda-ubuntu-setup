# CUDA and PyTorch Configuration Guide
## NVIDIA RTX 5060 Ti on Ubuntu 22.04

This document provides a complete installation and configuration guide for setting up CUDA and PyTorch on a system equipped with an NVIDIA GeForce RTX 5060 Ti GPU running Ubuntu 22.04 LTS. It is intended for users performing a first-time setup on this hardware configuration.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Driver Verification](#2-driver-verification)
3. [Environment Variable Verification](#3-environment-variable-verification)
4. [PyTorch Installation](#4-pytorch-installation)
5. [Installation Verification](#5-installation-verification)
6. [Persistence Configuration](#6-persistence-configuration)
7. [Troubleshooting Reference](#7-troubleshooting-reference)
8. [Technical Notes](#8-technical-notes)

---

## 1. System Requirements

| Component | Required Version |
|-----------|-----------------|
| Operating System | Ubuntu 22.04 LTS |
| NVIDIA Driver | 570 or newer |
| CUDA Toolkit | 12.8 |
| PyTorch | Nightly build (cu128) |
| Python | 3.10 |

### Architecture Notice

The RTX 5060 Ti is based on the **NVIDIA Blackwell architecture (compute capability sm_120)**, released in early 2026. As of this writing, stable PyTorch releases do not yet include compiled kernels for this architecture. The **nightly build** of PyTorch is required. Using the stable release will result in CUDA being reported as unavailable even when all other components are correctly installed.

---

## 2. Driver Verification

Before proceeding with any software installation, verify that the NVIDIA driver is correctly installed and that the GPU is detected by the system.

### 2.1 Check GPU Detection

```bash
nvidia-smi
```

The output should list the RTX 5060 Ti with a driver version of 570 or higher. Example of a successful output:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01   Driver Version: 590.48.01   CUDA Version: 13.1                  |
|=========================================================================================|
|   0  NVIDIA GeForce RTX 5060 Ti     Off | 00000000:01:00.0  On |                  N/A  |
+-----------------------------------------------------------------------------------------+
```

If this command fails or no GPU is listed, the driver is not correctly installed. Driver installation must be completed before continuing. Drivers older than version 570 do not support the Blackwell architecture and will not detect the RTX 5060 Ti.

### 2.2 Verify Kernel Modules

Confirm that the required NVIDIA kernel modules are loaded:

```bash
lsmod | grep nvidia
```

The output must include `nvidia`, `nvidia_uvm`, and `nvidia_modeset`. If any are absent, load them manually:

```bash
sudo modprobe nvidia_uvm
sudo modprobe nvidia_modeset
```

To ensure these modules load automatically on every system boot, proceed to [Section 6](#6-persistence-configuration).

---

## 3. Environment Variable Verification

This step is critical and is the most common source of failure on systems that have been previously configured for multi-GPU environments or have followed third-party setup guides.

### 3.1 Check the CUDA_VISIBLE_DEVICES Variable

```bash
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
```

If the output is `CUDA_VISIBLE_DEVICES=` (empty string) or `CUDA_VISIBLE_DEVICES=-1`, this variable is actively hiding all GPU devices from CUDA-dependent applications, including PyTorch. Note that `nvidia-smi` does not respect this variable and will continue to show the GPU correctly, which makes this issue particularly difficult to identify without explicitly checking.

### 3.2 Locate and Remove the Variable

Search all shell configuration files for the source of this assignment:

```bash
grep -r "CUDA_VISIBLE_DEVICES" ~/.bashrc ~/.profile ~/.bash_profile ~/.config/ 2>/dev/null
```

Remove all occurrences automatically:

```bash
sed -i '/CUDA_VISIBLE_DEVICES/d' ~/.bashrc
```

Confirm the variable has been removed:

```bash
grep "CUDA_VISIBLE_DEVICES" ~/.bashrc
```

This command should produce no output.

### 3.3 Apply the Change to the Current Session

```bash
unset CUDA_VISIBLE_DEVICES
source ~/.bashrc
```

---

## 4. PyTorch Installation

### 4.1 Remove Existing PyTorch Installation

Any previously installed version of PyTorch must be removed before proceeding. Failure to do so may result in pip skipping the installation entirely if it detects an existing package, regardless of whether the CUDA version is compatible.

```bash
pip3 uninstall torch torchvision torchaudio -y
```

### 4.2 Install PyTorch Nightly with CUDA 12.8

```bash
pip3 install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128 \
  --force-reinstall \
  --no-cache-dir
```

**Explanation of flags:**

| Flag | Purpose |
|------|---------|
| `--pre` | Enables pre-release and nightly builds, which include Blackwell (sm_120) kernel support |
| `--force-reinstall` | Forces pip to replace any existing installation even if the package name matches |
| `--no-cache-dir` | Prevents pip from using a locally cached version that may be outdated or incompatible |

**Note:** This installation downloads approximately 2-3 GB of packages. Allow sufficient time for the download to complete before proceeding.

---

## 5. Installation Verification

Run the following command to confirm CUDA is available and the GPU is accessible through PyTorch:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**Expected output:**

```
True
NVIDIA GeForce RTX 5060 Ti
```

A result of `True` followed by the GPU name confirms that the installation is complete and functioning correctly. Open a new terminal session and repeat this test to confirm the configuration persists across sessions.

---

## 6. Persistence Configuration

To ensure NVIDIA kernel modules are loaded automatically on every system boot, add them to the modules load configuration:

```bash
echo -e "nvidia\nnvidia_uvm\nnvidia_modeset" | sudo tee /etc/modules-load.d/nvidia.conf
```

Verify the file was written correctly:

```bash
cat /etc/modules-load.d/nvidia.conf
```

Expected output:
```
nvidia
nvidia_uvm
nvidia_modeset
```

---

## 7. Troubleshooting Reference

| Observed Symptom | Probable Cause | Resolution |
|------------------|---------------|------------|
| `nvidia-smi: command not found` | NVIDIA driver is not installed | Install NVIDIA driver version 570 or newer |
| `nvidia-smi` shows GPU but `torch.cuda.is_available()` returns `False` | `CUDA_VISIBLE_DEVICES` set to empty string in shell configuration | Follow Section 3 |
| `cudaGetDeviceCount returned 100` | `nvidia_uvm` kernel module is not loaded | Run `sudo modprobe nvidia_uvm` |
| pip prints "Requirement already satisfied" and skips installation | A cached or existing PyTorch version was detected | Re-run Section 4 with `--force-reinstall --no-cache-dir` |
| PyTorch CUDA works in one terminal but fails in a new session | `CUDA_VISIBLE_DEVICES` is still present in a shell configuration file | Re-examine `.bashrc`, `.profile`, and any conda activation scripts |
| `RuntimeError: No CUDA GPUs are available` | One or more of the above causes | Follow all sections in order |

---

## 8. Technical Notes

This section documents the specific issues encountered during the development of this guide and explains the reasoning behind each step.

### 8.1 Incorrect PyTorch CUDA Version (cu124 vs cu128)

The system had PyTorch 2.6 built against CUDA 12.4 already installed. Running `pip install torch --index-url .../cu128` produced no effect because pip determined the package was already satisfied and skipped the download. The incompatible cu124 build remained in place silently. This is why `--force-reinstall` is a required flag and not optional.

### 8.2 Stable PyTorch Does Not Support Blackwell

As of March 2026, the stable PyTorch release does not include compiled GPU kernels for compute capability sm_120 (Blackwell). Installing the stable cu128 build results in a version that is aware of CUDA 12.8 but cannot execute operations on the RTX 5060 Ti. The nightly build must be used until Blackwell support is incorporated into a stable release.

### 8.3 CUDA_VISIBLE_DEVICES Set to Empty String in Shell Configuration

The variable `export CUDA_VISIBLE_DEVICES=""` had been added to `.bashrc` on two separate occasions, likely copied from online guides describing multi-GPU or containerized environments. An empty string assignment is not equivalent to leaving the variable unset. When set to an empty string, the CUDA runtime interprets this as an instruction to expose zero devices to any application. This caused `torch.cuda.is_available()` to return `False` consistently, regardless of driver or PyTorch version. Because `nvidia-smi` does not use this variable, the GPU appeared healthy in all system-level checks, making the root cause non-obvious. The variable must be explicitly unset and all assignments removed from shell configuration files.

---

## References

- [PyTorch - Get Started](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA Driver Downloads](https://www.nvidia.com/en-us/drivers/)
- [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/)

---

## License

This document is released under the MIT License. You are free to use, adapt, and distribute it with attribution.
