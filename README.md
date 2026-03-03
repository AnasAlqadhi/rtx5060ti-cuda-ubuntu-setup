# CUDA and PyTorch Setup Guide
## NVIDIA RTX 5060 Ti — Ubuntu 22.04 LTS — Complete From-Scratch Installation

This guide covers the complete installation process for a system with no prior NVIDIA, CUDA, or PyTorch configuration. It is written specifically for the RTX 5060 Ti (Blackwell architecture) on Ubuntu 22.04 LTS and documents real-world issues encountered during setup that are not covered by standard NVIDIA documentation.

> **Source:** This guide was developed through direct troubleshooting of a working RTX 5060 Ti system on Ubuntu 22.04. All steps have been verified on hardware.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Pre-Installation Checks](#2-pre-installation-checks)
3. [NVIDIA Driver Installation](#3-nvidia-driver-installation)
4. [CUDA Toolkit 12.8 Installation](#4-cuda-toolkit-128-installation)
5. [Environment Variable Configuration](#5-environment-variable-configuration)
6. [PyTorch Installation](#6-pytorch-installation)
7. [Final Verification](#7-final-verification)
8. [Persistence Configuration](#8-persistence-configuration)
9. [Troubleshooting Reference](#9-troubleshooting-reference)
10. [Technical Notes](#10-technical-notes)
11. [References](#11-references)

---

## 1. System Requirements

| Component | Required |
|-----------|----------|
| Operating System | Ubuntu 22.04 LTS (64-bit) |
| GPU | NVIDIA RTX 5060 Ti (or any Blackwell GPU) |
| NVIDIA Driver | 570 or newer (590 recommended) |
| CUDA Toolkit | 12.8 |
| PyTorch | Nightly build (cu128) |
| Python | 3.10 |
| RAM | 8 GB minimum |

### Important Architecture Notice

The RTX 5060 Ti is based on the **NVIDIA Blackwell architecture (sm_120)**. This architecture requires:
- Driver version **570 or newer** — older drivers will not detect the GPU at all
- CUDA **12.8 or newer**
- **PyTorch nightly builds** — stable PyTorch does not yet include Blackwell GPU kernels as of early 2026

---

## 2. Pre-Installation Checks

### 2.1 Verify GPU is Detected by the System

```bash
lspci | grep -i nvidia
```

You should see your GPU listed. Example output:
```
01:00.0 VGA compatible controller: NVIDIA Corporation Device 2d04 (rev a1)
```

If nothing appears, there is a hardware or BIOS issue that must be resolved before continuing.

### 2.2 Check GCC is Installed

CUDA requires GCC for kernel module compilation.

```bash
gcc --version
```

If not installed:

```bash
sudo apt update
sudo apt install -y build-essential gcc linux-headers-$(uname -r)
```

### 2.3 Remove Conflicting Previous Installations

If this is a completely fresh system, skip this step. If NVIDIA or CUDA packages have been previously installed, clean them first:

```bash
sudo apt purge nvidia* cuda* -y
sudo apt autoremove -y
sudo apt autoclean
sudo rm -rf /usr/local/cuda*
```

Reboot after cleaning:

```bash
sudo reboot
```

---

## 3. NVIDIA Driver Installation

### 3.1 Check Available Drivers

```bash
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers devices
```

This will show recommended drivers for your GPU. For the RTX 5060 Ti, look for entries showing `nvidia-driver-590-open` or newer.

### 3.2 Install the Driver

Install the recommended open driver (required for Blackwell GPUs):

```bash
sudo apt install -y nvidia-driver-590-open
```

> **Why the open driver?** NVIDIA's Blackwell architecture (RTX 50 series) requires the open kernel module. The proprietary (non-open) driver variant does not support sm_120 correctly on Linux.

### 3.3 Reboot

```bash
sudo reboot
```

### 3.4 Verify Driver Installation

After rebooting, confirm the driver is loaded and the GPU is visible:

```bash
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01   Driver Version: 590.48.01   CUDA Version: 13.1                  |
|=========================================================================================|
|   0  NVIDIA GeForce RTX 5060 Ti     Off | 00000000:01:00.0  On |                  N/A  |
+-----------------------------------------------------------------------------------------+
```

If `nvidia-smi` fails at this point, the driver installation was unsuccessful. Do not proceed until this works.

---

## 4. CUDA Toolkit 12.8 Installation

These commands follow the official NVIDIA installation method using the network repository.
Official reference: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

### 4.1 Add the NVIDIA CUDA Repository

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
```

### 4.2 Install CUDA Toolkit 12.8

```bash
sudo apt-get install -y cuda-toolkit-12-8
```

This installs the CUDA compiler (`nvcc`), libraries, and development tools.

### 4.3 Set Environment Variables

Add the CUDA paths to your shell configuration:

```bash
echo 'export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
source ~/.bashrc
```

### 4.4 Verify CUDA Installation

```bash
nvcc --version
```

Expected output:
```
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 12.8, V12.8.x
```

---

## 5. Environment Variable Configuration

This step must not be skipped. An incorrectly set environment variable is the single most common cause of PyTorch failing to detect the GPU even after a correct installation.

### 5.1 Check for CUDA_VISIBLE_DEVICES

```bash
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
```

**If the output is `CUDA_VISIBLE_DEVICES=` (empty string) or `CUDA_VISIBLE_DEVICES=-1`, your GPU will be invisible to PyTorch.**

This variable, when set to an empty string, instructs the CUDA runtime to expose zero devices to all applications. `nvidia-smi` does not use this variable and will still show the GPU correctly, which makes this issue very difficult to diagnose without explicitly checking.

### 5.2 Search for the Variable in All Config Files

```bash
grep -r "CUDA_VISIBLE_DEVICES" ~/.bashrc ~/.profile ~/.bash_profile ~/.config/ 2>/dev/null
```

### 5.3 Remove All Occurrences

```bash
sed -i '/CUDA_VISIBLE_DEVICES/d' ~/.bashrc
```

Confirm it is gone:

```bash
grep "CUDA_VISIBLE_DEVICES" ~/.bashrc
# This should produce no output
```

### 5.4 Apply to Current Terminal Session

```bash
unset CUDA_VISIBLE_DEVICES
source ~/.bashrc
```

---

## 6. PyTorch Installation

### 6.1 Remove Any Existing PyTorch

```bash
pip3 uninstall torch torchvision torchaudio -y
```

### 6.2 Install PyTorch Nightly with CUDA 12.8

```bash
pip3 install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128 \
  --force-reinstall \
  --no-cache-dir
```

**Flag reference:**

| Flag | Reason |
|------|--------|
| `--pre` | Required to access nightly builds which include Blackwell (sm_120) kernel support |
| `--force-reinstall` | Forces pip to replace any existing version even if it considers requirements satisfied |
| `--no-cache-dir` | Prevents pip from silently using a cached incompatible version |

> **Note:** This download is approximately 2-3 GB. Allow several minutes for completion depending on your connection speed.

> **Why not the stable release?** The stable PyTorch cu128 build does not include compiled GPU kernels for sm_120 (Blackwell). Installing it results in a PyTorch that is aware of CUDA 12.8 but cannot execute on the RTX 5060 Ti. The nightly build must be used until official Blackwell support is added to the stable release.

---

## 7. Final Verification

### 7.1 Test CUDA Availability in PyTorch

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**Expected output:**
```
True
NVIDIA GeForce RTX 5060 Ti
```

### 7.2 Test in a New Terminal

Close the current terminal, open a new one, and run the same command again. This confirms the configuration persists across sessions and is not only working due to temporary in-session changes.

### 7.3 Extended Verification

For a more thorough check:

```bash
python3 -c "
import torch
print('PyTorch version :', torch.__version__)
print('CUDA version     :', torch.version.cuda)
print('CUDA available   :', torch.cuda.is_available())
print('Device count     :', torch.cuda.device_count())
print('Device name      :', torch.cuda.get_device_name(0))
print('Device memory    :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')
"
```

---

## 8. Persistence Configuration

Ensure NVIDIA kernel modules load automatically on every system boot:

```bash
echo -e "nvidia\nnvidia_uvm\nnvidia_modeset" | sudo tee /etc/modules-load.d/nvidia.conf
```

Verify:

```bash
cat /etc/modules-load.d/nvidia.conf
```

Expected:
```
nvidia
nvidia_uvm
nvidia_modeset
```

---

## 9. Troubleshooting Reference

| Symptom | Probable Cause | Resolution |
|---------|---------------|------------|
| `lspci` shows no NVIDIA device | Hardware or BIOS issue | Check PCIe slot, re-seat GPU, verify BIOS settings |
| `nvidia-smi` command not found | Driver not installed | Complete Section 3 |
| `nvidia-smi` works but `torch.cuda.is_available()` returns `False` | `CUDA_VISIBLE_DEVICES` set to empty string | Complete Section 5 |
| `cudaGetDeviceCount returned 100` | `nvidia_uvm` module not loaded | `sudo modprobe nvidia_uvm` |
| `nvcc` command not found | CUDA paths not set in `.bashrc` | Complete Section 4.3 |
| pip prints "Requirement already satisfied" and skips | Existing PyTorch version detected by pip | Re-run Section 6.2 — the `--force-reinstall` flag is mandatory |
| PyTorch CUDA works in one terminal, fails in new sessions | `CUDA_VISIBLE_DEVICES` still present in shell config | Re-run Section 5 in full |
| `RuntimeError: No CUDA GPUs are available` | One or more of the above | Follow all sections in order from the beginning |
| Driver installation fails or creates display issues | Conflicting previous driver packages | Complete Section 2.3 (cleanup) before reinstalling |

---

## 10. Technical Notes

This section documents the specific issues encountered during the real-world setup that produced this guide.

### 10.1 Driver Version Requirement for Blackwell

The RTX 5060 Ti uses compute capability sm_120, which is only supported from driver version 570 onward. Systems with older drivers will have `nvidia-smi` either fail entirely or show "no devices found", and `lspci` will show an unrecognised device ID. The open kernel module variant is required — the proprietary variant does not correctly support Blackwell on Linux.

### 10.2 PyTorch cu124 Silently Remaining Installed

The system initially had PyTorch 2.6 built against CUDA 12.4 installed. Running:
```
pip install torch --index-url .../cu128
```
produced the message "Requirement already satisfied" and did nothing. The cu124 build remained. This is why `--force-reinstall` and `--no-cache-dir` are required flags in this guide and not optional suggestions.

### 10.3 Stable PyTorch cu128 Does Not Support Blackwell

Even after correctly installing the cu128 variant of stable PyTorch, the RTX 5060 Ti was not detected. This is because stable PyTorch does not yet compile GPU kernels for sm_120. The nightly build includes these kernels. This will change once NVIDIA Blackwell support is officially incorporated into a stable PyTorch release.

### 10.4 CUDA_VISIBLE_DEVICES Set to Empty String

The variable `export CUDA_VISIBLE_DEVICES=""` was found twice in `.bashrc`, likely added when following online guides for multi-GPU or containerised environments. An empty string is not the same as an unset variable. When set to an empty string, the CUDA runtime treats it as an explicit instruction to make zero devices visible to any application. Because `nvidia-smi` does not consult this variable, the GPU appeared fully operational in system-level checks while being completely invisible to PyTorch. This is an easy mistake to introduce and a difficult one to find without explicitly checking the variable.

---

## 11. References

- [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
- [NVIDIA CUDA Installation Guide for Linux 12.8](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html)
- [PyTorch — Get Started (Nightly)](https://pytorch.org/get-started/locally/)
- [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [NVIDIA Driver Downloads](https://www.nvidia.com/en-us/drivers/)

---

## License

This document is released under the MIT License. You are free to use, adapt, and distribute it with attribution.
