# XRoboToolkit Teleop 在 Ubuntu 20.04 + PICO 上的安装

本文记录在 **Ubuntu 20.04** 上尝试安装并运行 `XRoboToolkit-Teleop-Sample-Python`，并连接 **PICO / XRoboToolkit** 进行遥操作验证的完整过程。

> 说明：官方主要支持 Ubuntu 22.04 / 24.04。Ubuntu 20.04 可以尝试，但需要额外处理 PC Service 的 glibc 兼容问题。

---

## 1. 环境信息

测试环境：

```text
OS: Ubuntu 20.04
Python: 3.10.20
Conda env: xrteleop
Project path: /media/shizifeng/projects21/XRoboToolkit-Teleop-Sample-Python
PICO app: XRoboToolkit-PICO-1.1.1.apk
PC Service: XRoboToolkit-PC-Service_1.0.0.0_ubuntu20.04_amd64.deb
```

需要的核心组件：

```text
1. XRoboToolkit-Teleop-Sample-Python
2. XRoboToolkit-PC-Service-Pybind
3. XRoboToolkit-PC-Service
4. XRoboToolkit-PICO APK
5. PICO 头显 + USB 调试 + 同局域网连接
```

---

## 2. 创建 Conda 环境

Ubuntu 20.04 默认 Python 通常是 3.8，但该项目需要 Python 3.10+，因此不要直接用系统 Python。

```bash
conda create -n xrteleop python=3.10 -y
conda activate xrteleop
```

安装基础工具：

```bash
python -m pip install -U pip setuptools wheel uv
```

可选：安装一些编译和图形依赖：

```bash
sudo apt update
sudo apt install -y \
    git build-essential cmake pkg-config \
    libgl1-mesa-dev libglfw3 libglfw3-dev \
    libglew-dev libosmesa6-dev \
    libx11-dev libxrandr-dev libxinerama-dev \
    libxcursor-dev libxi-dev \
    adb unzip wget
```

---

## 3. 克隆 Python 示例库

```bash
cd /media/shizifeng/projects21

git clone https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python.git
cd XRoboToolkit-Teleop-Sample-Python
```

---

## 4. 安装 Python Teleop Sample

在 Ubuntu 20.04 上运行官方安装脚本时，会提示系统版本未正式测试：

```bash
bash setup_conda.sh --install
```

提示：

```text
Warning: This script has only been tested on Ubuntu 22.04 and 24.04
Your system is running Ubuntu 20.04.
Do you want to continue anyway? (y/N): y
```

输入：

```text
y
```

安装过程会自动处理：

```text
1. XRoboToolkit-PC-Service-Pybind
2. XRoboToolkit SDK Python binding: xrobotoolkit_sdk
3. ARX R5 Python SDK
4. xrobotoolkit_teleop 主库
5. mujoco / placo / meshcat / torch / opencv 等依赖
```

成功时会看到类似：

```text
[INFO] xrobotoolkit_teleop is installed in conda environment 'xrteleop'.
```

---

## 5. 安装过程中遇到的问题与处理
### 5.1 jedi / IPython 版本冲突

运行 MuJoCo demo 时可能报：

```text
AttributeError: module 'jedi' has no attribute 'settings'
```

报错链路大致为：

```text
meshcat -> IPython -> jedi
```

修复方式是降级 `jedi`：

```bash
conda activate xrteleop
uv pip install "jedi==0.19.2" --reinstall
```

验证：

```bash
python - <<'PY'
import jedi
import IPython
print("jedi:", jedi.__version__)
print("IPython:", IPython.__version__)
print("has jedi.settings:", hasattr(jedi, "settings"))
PY
```

期望输出：

```text
has jedi.settings: True
```

---

## 6. 验证 Python 仿真能否运行

先进入环境和项目目录：

```bash
conda activate xrteleop
cd /media/shizifeng/projects21/XRoboToolkit-Teleop-Sample-Python
```

测试基础依赖：

```bash
python - <<'PY'
import mujoco
import placo
import cv2
import torch
print("basic deps ok")
PY
```

测试 XR SDK：

```bash
python - <<'PY'
import xrobotoolkit_sdk
print("xrobotoolkit_sdk ok")
PY
```

运行 MuJoCo 示例：

```bash
# 双腕 UR5e
python scripts/simulation/teleop_dual_ur5e_mujoco.py

# 双腕 A1X (Galaxea)
python scripts/simulation/teleop_dual_a1x_mujoco.py

# 单腕 Flexiv Rizon4s
python scripts/simulation/teleop_flexiv_rizon4s_mujoco.py

# SO-101 单腕 (6-DOF 桌面臂 + 平行夹爪)
python scripts/simulation/teleop_so101_mujoco.py

# SO-101 双腕 (左右各一个，间距 30cm)
python scripts/simulation/teleop_dual_so101_mujoco.py

# 仅 Placo 可视化 (无物理仿真)
python scripts/simulation/teleop_x7s_placo.py
python scripts/simulation/teleop_unitree_g1_placo.py
```

如果窗口能启动，说明 Python sample 和仿真部分已经可用。

### 6.1 SO-101 仿真说明

SO-101 是基于 STS3215 舵机的 6-DOF 桌面机械臂，使用 MuJoCo 物理仿真 + Placo IK 求解。

**单腕操控：**
| 输入 | 功能 |
|------|------|
| 右手握键 (right_grip) | 激活/解除手臂跟踪 |
| 右手扳机 (right_trigger) | 夹爪开合 (0=全开，1=全闭) |
| 右手柄移动 | 末端位姿跟踪 |

**双腕操控：**
| 输入 | 功能 |
|------|------|
| 左手握键 (left_grip) | 激活/解除左臂跟踪 |
| 左手扳机 (left_trigger) | 左夹爪开合 |
| 左手柄移动 | 左臂末端位姿跟踪 |
| 右手握键 (right_grip) | 激活/解除右臂跟踪 |
| 右手扳机 (right_trigger) | 右夹爪开合 |
| 右手柄移动 | 右臂末端位姿跟踪 |

**自定义参数：**
```bash
# 调整运动缩放
python scripts/simulation/teleop_so101_mujoco.py --scale-factor 2.0

# 关闭 Placo 可视化窗口
python scripts/simulation/teleop_so101_mujoco.py --no-visualize-placo

# 双腕同样支持
python scripts/simulation/teleop_dual_so101_mujoco.py --scale-factor 2.0 --no-visualize-placo
```

**SO-101 资产文件：**
```
assets/so101/
├── so101_new_calib.urdf    # 单腕 URDF (new calibration)
├── so101_new_calib.xml     # 单腕 MJCF
├── dual_so101.urdf         # 双腕 URDF
├── dual_so101.xml          # 双腕 MJCF
├── scene_teleop.xml        # 单腕场景 (mocap 目标 + 地面)
├── scene_dual_teleop.xml   # 双腕场景
└── assets/                 # STL 网格文件 (13 个)
```

> 标定方法说明：new calibration 的虚拟零点在各关节行程的中间位置；old calibration 的虚拟零点在手臂完全水平伸展的位置。当前仿真使用 new calibration。

---

## 7. 安装 XRoboToolkit PC Service

### 7.1 Ubuntu 22.04 版 deb 在 20.04 上的问题

如果安装官方 Ubuntu 22.04 版 PC Service，在 Ubuntu 20.04 上启动可能报：

```text
./RoboticsServiceProcess: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found
./RoboticsServiceProcess: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found
```

原因：Ubuntu 20.04 的 glibc 版本低于 22.04，不能直接运行按高版本 glibc 编译的二进制程序。

不要强行升级系统 glibc，否则可能破坏 Ubuntu 20.04、ROS Noetic、显卡驱动和 conda 环境。

---

### 7.2 使用 Ubuntu 20.04 构建版 PC Service
```
cd ~/Downloads
wget -O XRoboToolkit_PC_Service_ubuntu20.zip https://files.catbox.moe/08evux.zip
unzip XRoboToolkit_PC_Service_ubuntu20.zip
sudo dpkg -i XRoboToolkit-PC-Service_1.0.0.0_ubuntu20.04_amd64.deb
sudo apt --fix-broken install

```

检查目录：

```bash
ls /opt/apps/roboticsservice/
```

启动服务：

```bash
/opt/apps/roboticsservice/runService.sh
```

如果不再报 `GLIBC_2.34 not found`，说明 PC Service 已经能在 Ubuntu 20.04 上运行。

---

## 8. PICO 开启开发者模式
默认打开
在 PICO 头显中：

```text
设置 Settings
→ 连续点击软件版本 7～12 次
```

出现开发者菜单后：

```text
Developer / 开发者
→ USB Debug / USB 调试
→ 打开
```

---

## 9. 安装 PICO 端 APK

确认 APK 文件存在：

```bash
ls ~/Downloads/XRoboToolkit-PICO-1.1.1.apk
```

USB 连接 PICO 和电脑。

查看设备：

```bash
adb devices
```

如果显示：

```text
unauthorized
```

戴上 PICO，在头显里点击允许 USB 调试。

正常应该显示：

```text
xxxxxxxx    device
```

安装 APK：

```bash
cd ~/Downloads
adb install -r -g XRoboToolkit-PICO-1.1.1.apk
```

安装后，在 PICO 应用库中找到：

```text
XRoboToolkit
```

可能位于：

```text
未知来源 / Unknown Sources
```

---

## 10. PICO 和电脑连接方式

XRoboToolkit 通信需要 PICO 和电脑在同一局域网。

推荐方式：

```text
PICO 和电脑连接同一个路由器 Wi-Fi
```

如果在校园网或实验室网络下搜不到设备，优先尝试手机热点，因为校园网可能开启 AP isolation，导致同一 Wi-Fi 下设备互相不可见。

电脑查询 IP：

```bash
hostname -I
```

例如输出：

```text
192.168.43.123
```

如果 PICO 端 XRoboToolkit 需要填写 PC IP，就填这个地址。

---

## 11. 最终运行顺序

### 终端 1：启动 PC Service

```bash
/opt/apps/roboticsservice/runService.sh
```

保持该终端不要关闭。

---

### PICO 端：打开 XRoboToolkit

在 PICO 中打开：

```text
XRoboToolkit
```

如果需要手动连接电脑，填入电脑 IP。

---

### 终端 2：运行 Python 遥操作 Demo

```bash
conda activate xrteleop
cd /media/shizifeng/projects21/XRoboToolkit-Teleop-Sample-Python

# 双腕 UR5e MuJoCo 仿真
python scripts/simulation/teleop_dual_ur5e_mujoco.py

# SO-101 单腕 MuJoCo 仿真 (6-DOF 桌面臂 + 平行夹爪)
python scripts/simulation/teleop_so101_mujoco.py

# SO-101 双腕 MuJoCo 仿真 (左右间距 30cm)
python scripts/simulation/teleop_dual_so101_mujoco.py
```

如果 PICO 已连接，移动手柄、按下 grip / trigger 后，MuJoCo 中的机械臂应能响应。

**SO-101 仿真操控说明：**

| 输入 | 功能 |
|------|------|
| 右/左手握键 (grip) | 激活/解除手臂跟踪 |
| 右/左手扳机 (trigger) | 夹爪开合（0=全开，1=全闭） |
| 手柄移动 | 末端位姿跟踪 |

自定义参数：
```bash
# 调整运动缩放
python scripts/simulation/teleop_so101_mujoco.py --scale-factor 2.0

# 关闭 Placo 可视化
python scripts/simulation/teleop_so101_mujoco.py --no-visualize-placo
```

---

# PICO_TELE_LEROBOT
