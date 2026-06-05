# XRoboToolkit-Teleop-Sample-Python

基于 PICO 的遥操作演示程序，支持 MuJoCo 仿真和实物机器人。

## 概述

本项目提供了一套基于 XR（VR/AR）输入设备的机器人控制框架，同时支持实物机器人和 MuJoCo 仿真环境。用户可通过 XR 手柄的自然手部动作来操控机械臂。

## 安装

1. 下载并安装 [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)。运行以下演示前请先启动已安装的程序。

2. **克隆仓库：**
   ```bash
   git clone https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python.git
   cd XRoboToolkit-Teleop-Sample-Python
   ```

3. **安装**
   > 注意：安装脚本目前仅在 Ubuntu 22.04 上测试通过。

   推荐使用 Conda 环境并通过自带脚本安装：
   ```bash
   bash setup_conda.sh --conda <可选环境名>
   conda activate <环境名>
   bash setup_conda.sh --install
   ```

   如需安装到系统 Python：
   ```bash
   bash setup.sh
   ```

## 使用方法

请使用以下命令运行示例脚本。详细说明请参阅 [`teleop_details.md`](teleop_details.md)。

### 运行 MuJoCo 仿真

在 MuJoCo 中运行 UR5e 双臂遥操作：

```bash
python scripts/simulation/teleop_dual_ur5e_mujoco.py
```

此脚本使用 UR5e 模型初始化 [`MujocoTeleopController`](xrobotoolkit_teleop/simulation/mujoco_teleop_controller.py) 并启动遥操作循环。

其他仿真脚本：

```bash
# 双腕 A1X (Galaxea)
python scripts/simulation/teleop_dual_a1x_mujoco.py

# 单腕 Flexiv Rizon4s
python scripts/simulation/teleop_flexiv_rizon4s_mujoco.py

# SO-101 单腕 (6-DOF 桌面臂)
python scripts/simulation/teleop_so101_mujoco.py

# SO-101 双腕
python scripts/simulation/teleop_dual_so101_mujoco.py
```

### 运行 Placo 可视化

```bash
# ARX X7S
python scripts/simulation/teleop_x7s_placo.py

# Unitree G1 双腕
python scripts/simulation/teleop_unitree_g1_placo.py

# Flexiv Rizon4s
python scripts/simulation/teleop_flexiv_rizon4s_placo.py
```

### 运行灵巧手仿真

```bash
# Shadow Hand (MuJoCo)
python scripts/simulation/teleop_shadow_hand_mujoco.py

# Inspire Hand (Placo 可视化)
python scripts/simulation/teleop_inspire_hand_placo.py
```

### 运行实物硬件 Demo

#### 双腕 UR5e + Dynamixel 云台

```bash
# 正常运行
python scripts/hardware/teleop_dual_ur5e_hardware.py

# 复位机械臂到初始位置并初始化夹爪
python scripts/hardware/teleop_dual_ur5e_hardware.py --reset

# 遥操作时可视化 IK 求解结果
python scripts/hardware/teleop_dual_ur5e_hardware.py --visualize_placo
```

#### ARX R5 双腕

```bash
python scripts/hardware/teleop_dual_arx_r5_hardware.py
```

#### Galaxea R1 Lite 人形机器人

```bash
python scripts/hardware/teleop_r1lite_hardware.py
```

控制器通过 ROS 与机器人硬件通信。

#### SO-101 远程遥操作（PC + Jetson 分布式）

PC 端启动（XR + IK + 录制）：

```bash
python scripts/hardware/teleop_so101_remote_pc.py \
  --listen-ip 0.0.0.0 \
  --repo-id local/so101_dual_xr_teleop \
  --mode dual \
  --scale-factor 1.0 \
  --xr-frame simulation \
  --mjpeg-port 8080
```

启动后面对机器人正前方，按下手柄 `A` 键确认 XR yaw 对齐，然后按住 grip 开始遥操作。

详细说明请参阅 [`REMOTE_SO101_RECORD.md`](REMOTE_SO101_RECORD.md)。

## 数据采集

### 采集遥操作数据

框架在运行硬件 Demo 时自动记录遥操作数据。采集内容包括：

- **机器人关节状态** 和末端执行器位姿
- **多视角相机流**
- **XR 手柄输入数据**
- **跨数据流的同步时间戳**

#### 开始数据采集

1. **运行任意硬件遥操作脚本：**
   ```bash
   python scripts/hardware/teleop_dual_arx_r5_hardware.py
   ```

2. **按下 X 键** 开始/停止录制
   - 第一次按下：开始录制
   - 第二次按下：停止录制并保存到磁盘

3. **紧急丢弃：** 按下右摇杆可丢弃当前录制

#### 数据存储

采集的数据以 `.pkl` 文件保存到 `logs/` 目录，带有时间戳：

```
logs/
├── <机器人名>/
│   └── teleop_log_YYYYMMDD_HHMMSS_<会话ID>.pkl
└── <其他机器人>/
    ├── teleop_log_YYYYMMDD_HHMMSS_<会话ID>.pkl
    └── teleop_log_YYYYMMDD_HHMMSS_<会话ID>.pkl
```

### 验证采集数据

使用分析脚本检查数据完整性：

```bash
python scripts/misc/test_data_log_analysis.py logs/<机器人名>/teleop_log_YYYYMMDD_HHMMSS_1.pkl
```

脚本功能：
- 列出可用数据字段及其类型
- 验证机器人状态和相机图像是否正确保存
- 展示样本条目和数据统计
- 统计总记录条目数

### 转换为 LeRobot 数据集

如需训练模仿学习模型，可将采集数据转换为 [LeRobot](https://github.com/huggingface/lerobot) 格式。

PC 端录制直接输出 LeRobot 格式（`datasets/` 目录）。对于旧的 `.pkl` 文件，使用转换脚本：

```bash
python scripts/misc/convert_pkl_to_lerobot.py \
  --pkl-path logs/arx_r5/teleop_log_20240604_133700_1.pkl \
  --repo-id local/arx_r5_teleop \
  --output-dir datasets/arx_r5_teleop
```

外部参考：[ARX 双腕数据转换器](https://github.com/zhigenzhao/openpi/blob/dev/finetuning/examples/arx_r5/arx_dual/convert_dual_arm_data_to_lerobot.py)

转换为 LeRobot 格式后可实现：
- 标准化的机器学习数据集格式
- 接入 LeRobot 训练管线
- 支持多种模仿学习算法
- 便于数据共享和复现

## 遥操作指南

### 跟踪模式

遥操作系统支持多种跟踪模式来控制机器人末端执行器：

#### 1. 手柄跟踪（默认）
- **描述**：使用 VR/AR 手柄位姿控制机器人末端执行器
- **适用场景**：精确操作任务的首选方式
- **配置**：设置 `pose_source` 为 `"left_controller"` 或 `"right_controller"`
- **跟踪**：完整 6DOF 位姿（位置 + 姿态）或 3DOF 仅位置

#### 2. 手部跟踪
- **描述**：使用 XR 相机的手部姿态估计
- **适用场景**：自然手势控制

#### 3. 头部跟踪
- **描述**：使用头显位姿控制特定机器人部件
- **适用场景**：人形机器人的头部/颈部控制或相机朝向控制

#### 4. 运动跟踪器
- **描述**：使用额外的运动跟踪设备控制辅助机器人连杆
- **适用场景**：多点控制（如在控制末端执行器的同时控制肘部位置）
- **配置**：添加 `motion_tracker` 配置，指定设备序列号和目标连杆
- **注意**：不推荐用于 UR5e 等 6DOF 臂；更适合冗余臂

### 手柄按键功能

#### 握把按钮 (Grip)
- **左手 Grip** (`left_grip`)：激活左臂遥操作
- **右手 Grip** (`right_grip`)：激活右臂遥操作
- **功能**：按住启用手臂控制，松开停用

#### 扳机按钮 (Trigger)
- **左手扳机** (`left_trigger`)：控制左手爪/夹爪
- **右手扳机** (`right_trigger`)：控制右手爪/夹爪
- **功能**：模拟量控制（0.0 = 全开，1.0 = 全闭）

#### 系统按钮
- **A 键**：SO-101 远程遥操作中确认 XR yaw 对齐
- **X 键**：切换数据录制开/关
  - 按一次：开始录制
  - 再按一次：停止录制并保存

#### 摇杆
- **左摇杆**：移动机器人的线速度指令
- **右摇杆**：移动机器人的角速度指令
- **右摇杆按下**：停止录制（丢弃当前数据）

## 依赖

XR Robotics 依赖：
- [`xrobotookit_sdk`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind)：XRoboToolkit PC Service SDK 的 Python 绑定，MIT License

机器人仿真与求解器：
- [`mujoco`](https://github.com/google-deepmind/mujoco)：机器人仿真，Apache 2.0 License
- [`placo`](https://github.com/rhoban/placo)：逆运动学求解，MIT License

硬件控制：
- [`dynamixel_sdk`](https://github.com/ROBOTIS-GIT/DynamixelSDK.git)：Dynamixel 舵机控制，Apache 2.0 License
- [`ur_rtde`](https://gitlab.com/sdurobotics/ur_rtde)：UR 机器人控制与数据接收接口，MIT License
- [`ARX R5 SDK`](https://github.com/zhigenzhao/R5/tree/dev/python_pkg)：ARX R5 机械臂控制接口

## 许可证

本项目基于 MIT License 开源 — 详见 [LICENSE](LICENSE) 文件。
