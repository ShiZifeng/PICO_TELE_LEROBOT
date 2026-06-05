#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Check the operating system
OS_NAME=$(uname -s)
OS_VERSION=""

if [[ "$OS_NAME" == "Linux" ]]; then
    if command -v lsb_release &>/dev/null; then
        OS_VERSION=$(lsb_release -rs)
    elif [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS_VERSION=$VERSION_ID
    fi
    if [[ "$OS_VERSION" != "22.04" ]]; then
        echo "Warning: This script has only been tested on Ubuntu 22.04"
        echo "Your system is running Ubuntu $OS_VERSION."
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation cancelled."
            exit 1
        fi
    fi
else
    echo "Unsupported operating system: $OS_NAME"
    exit 1
fi

echo "Operating system check passed: $OS_NAME $OS_VERSION"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Python is not installed on this system."
    exit 1
fi
PYTHON_MAJOR_MINOR=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$(printf '%s\n' "3.10" "$PYTHON_MAJOR_MINOR" | sort -V | head -n1)" != "3.10" ]]; then
    echo "Error: LeRobot requires Python >= 3.10, but current Python is $PYTHON_MAJOR_MINOR."
    echo "Please use setup_conda.sh to create/install in a Python 3.10+ environment."
    exit 1
fi

    # Install the required packages
    rm -rf dependencies
    mkdir dependencies
    cd dependencies

    git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git
    cd XRoboToolkit-PC-Service-Pybind
    bash setup_ubuntu.sh

    cd ..
    git clone https://github.com/zhigenzhao/R5.git
    cd R5
    git checkout dev/python_pkg
    cd py/ARX_R5_python/
    pip install .

    cd ../../../..

    LEROBOT_PATH="${LEROBOT_PATH:-$SCRIPT_DIR/third_party/lerobot-v0.3.3}"
    if [[ -d "$LEROBOT_PATH" ]]; then
        echo "[INFO] Installing LeRobot editable from: $LEROBOT_PATH"
        pip install -e "$LEROBOT_PATH" || { echo "Failed to install lerobot from $LEROBOT_PATH"; exit 1; }
    else
        echo "[WARN] LeRobot path not found: $LEROBOT_PATH"
        echo "[WARN] Set LEROBOT_PATH=/path/to/lerobot-v0.3.3 and rerun if XRoboToolkit needs lerobot."
    fi

    pip install -e . || { echo "Failed to install xrobotoolkit_teleop with pip"; exit 1; }


    echo -e "\n"
    echo -e "[INFO] xrobotoolkit_teleop is installed in conda environment '$ENV_NAME'.\n"
    echo -e "\n"
