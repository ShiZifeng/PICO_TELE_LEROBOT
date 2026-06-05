# SO-101 远程录制操作指南

Jetson 负责硬件控制 + 相机采集，PC 负责 XR 遥操作 + IK + LeRobot 录制。

## 网络配置

| 设备 | IP | 角色 |
|------|-----|------|
| PC | 192.168.50.101 | XR + IK + 录制 + MJPEG 推流 |
| Jetson | 192.168.50.47 | 机械臂控制 + 相机采集 |
| PICO | 同一局域网 | XR 手柄追踪 |

## 端口分配

| 端口 | 方向 | 协议 | 内容 |
|------|------|------|------|
| 5570 | Jetson → PC | ZMQ PUSH/PULL | 观测数据（关节+相机） |
| 5580 | PC → Jetson | ZMQ PUB/SUB | IK 关节目标 |
| 5571 | PC → Jetson | ZMQ PUB/SUB | 控制命令 (stop) |
| 8080 | PC → PICO | HTTP | MJPEG 相机画面 |

---

## 1. PC 端同步工程到 Jetson
默认已完成

```bash
conda activate xvla_lerobot
cd /media/shizifeng/projects21/lerobot-v0.3.3

rsync -av \
  --exclude='.git' --exclude='outputs' --exclude='__pycache__' --exclude='*.pyc' \
  /media/shizifeng/projects21/lerobot-v0.3.3/ \
  wheeltec@192.168.50.47:~/szf_lerobot/lerobot-v0.3.3/

# 同时同步 XRoboToolkit 工程的 Jetson 脚本
rsync -av \
  /media/shizifeng/projects21/XRoboToolkit-Teleop-Sample-Python/scripts/hardware/teleop_so101_remote_jetson.py \
  wheeltec@192.168.50.47:~/szf_lerobot/scripts/

# Jetson 上重新安装
ssh wheeltec@192.168.50.47 "cd ~/szf_lerobot/lerobot-v0.3.3 && conda activate szf_lerobot && pip install -e ."
```

## 2. Jetson 端确认相机

```bash
ssh wheeltec@192.168.50.47
conda activate szf_lerobot
cd ~/szf_lerobot/lerobot-v0.3.3

python -m lerobot.find_cameras
```

确认输出类似：
```
opencv__dev_video10 ✅
opencv__dev_video8  ✅
```

## 3. Jetson 端启动硬件发送

```bash
conda activate szf_lerobot
cd ~/szf_lerobot

python scripts/teleop_so101_remote_jetson.py \
  --pc-ip 192.168.50.101 \
  --ports '{"left": "/dev/left_follower_mobile", "right": "/dev/right_follower_mobile"}' \
  --cameras '{
    "left_arm": {"type": "opencv", "index": 10, "width": 640, "height": 480, "fps": 30},
    "right_arm": {"type": "opencv", "index": 8, "width": 640, "height": 480, "fps": 30},
    "head": {"type": "opencv", "index": 0, "width": 640, "height": 480, "fps": 30}
  }' \
  --fps 30 \
  --control-fps 60 \
  --jpeg-quality 80
```

看到 `Ready. Waiting for joint targets from PC...` 即启动成功。

> **安全提示**：第一次建议加 `--max-relative-target 3.0` 限制舵机每步最大移动量。

## 4. PC 端启动 PC Service

```bash
# 终端 1：启动 PC Service（保持运行）
/opt/apps/roboticsservice/runService.sh
```

## 5. PICO 端连接

1. PICO 开机 → 打开 XRoboToolkit app
2. 确认 PICO 和 PC 在同一局域网
3. 如需要，填入 PC IP `192.168.50.101`

## 6. PC 端启动 XR 遥操作 + 录制

```bash
# 终端 2
conda activate xrteleop
cd /media/shizifeng/projects21/XRoboToolkit-Teleop-Sample-Python

# 双臂模式
python scripts/hardware/teleop_so101_remote_pc.py \
  --listen-ip 0.0.0.0 \
  --repo-id local/so101_dual_xr_teleop \
  --mode dual \
  --scale-factor 1.0 \
  --xr-frame simulation \
  --mjpeg-port 8080

# 单臂模式
python scripts/hardware/teleop_so101_remote_pc.py \
  --listen-ip 0.0.0.0 \
  --repo-id local/so101_xr_teleop \
  --mode single \
  --scale-factor 1.0 \
  --xr-frame simulation
```

如果感觉手柄前后/左右方向不对，先不要连机械臂调 IK，单独跑 XR 轴向调试：

```bash
python scripts/hardware/debug_xr_axis_mapping.py --controller right
```

按提示先把手柄放在中立位置并回车，然后分别只做一个方向的移动：

| 物理移动 | 期望输出 |
|----------|----------|
| 手柄向机器人前方 | `simulation -> +X forward` |
| 手柄向机器人左侧 | `simulation -> +Y left` |
| 手柄向上 | `simulation -> +Z up` |

如果要在实际遥操作中看映射，加：

```bash
--debug-xr-delta
```

如果单独 XR 轴向调试是对的，但实机仍感觉方向乱，先用 position-only IK 排除姿态耦合：

```bash
--ik-control-mode position --debug-xr-delta
```

如果 full pose IK 抖动明显，可以增量测试 `position_wrist`：用 4DOF 解 4 个标量约束，追踪 `wrist_link` 的 `xyz`，并用手柄俯仰控制 `gripper_frame_link` 本地 Z 轴相对水平面的仰角；手柄摇杆横向控制最后一个 `wrist_roll` 的旋转速度，且腕旋摇杆不需要按住 grip。

```bash
--ik-control-mode position_wrist
```

如果摇杆控制腕旋方向反了，加：

```bash
--wrist-roll-scale 1.0
```

如果机械臂伸到最远端仍然抖动，先保留默认 `--max-wrist-reach-m 0.285`；它会限制 `shoulder_link -> wrist_link` 的目标半径，避免 4DOF IK 追不可达目标。若活动范围太小，可以小幅调大，例如：

```bash
--max-wrist-reach-m 0.300
```

`simulation` 预设和 MuJoCo/Placo 仿真流程使用完全相同的坐标对齐；`pico` 当前只是 `simulation` 的兼容别名。SO-101 场景中机器人前方是世界 `+X`。

看到 `Dataset created` 和 `MJPEG streaming: http://...` 即启动成功。

## 7. PICO 查看相机画面

```
PICO 浏览器 → http://192.168.50.101:8080
```

三个相机窗口并排显示。可在浏览器中调整窗口大小。

## 8. 操控与录制

启动 PC 端脚本后，先面向机器人前方，按一次手柄 **A** 键完成 XR 朝向确认；确认前 grip 不会驱动机械臂。

| 操作 | 方式 |
|------|------|
| 确认 XR 朝向 | 面向机器人前方，按 **A** |
| 激活左/右臂跟踪 | 按住 左/右手 grip 键 |
| 夹爪开合 | 左/右手 扳机 |
| 移动机械臂 | 移动对应手柄 |
| **开始录制** | 按 **Space** |
| **停止录制（保存）** | 再按 **Space** |
| **丢弃当前录制** | 按 **R** |
| **退出** | 按 **Q** 或 **Esc** |

PC 预览窗口状态栏：
```
PREVIEW | frames:0 | saved:3 | fps:29.8 | space start/stop, r discard, q/Esc quit
REC     | frames:452 | saved:3 | fps:30.1 | ...
```

## 9. 录制输出

数据保存在 PC 端：

```
datasets/so101_dual_xr_teleop/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── ...
└── meta/
    └── info.json
```

### 查看录制数据

```bash
conda activate xvla_lerobot
cd /media/shizifeng/projects21/lerobot-v0.3.3

# 查看数据集信息
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/so101_dual_xr_teleop', root='datasets')
print(f'Episodes: {ds.num_episodes}')
print(f'Frames: {ds.num_frames}')
print(f'Features: {list(ds.features.keys())}')
"

# 转换为 EE 空间并可视化
python -m lerobot.scripts.convert_dataset_to_ee \
  --src datasets/so101_dual_xr_teleop \
  --dst datasets/so101_dual_xr_teleop_ee \
  --urdf assets/so101/dual_so101.urdf

python -m lerobot.scripts.visualize_ee_trajectory \
  --dataset-root datasets/so101_dual_xr_teleop_ee \
  --output-dir outputs/ee_trajectory
```

## 10. 关机顺序

```
1. PC 预览窗口按 Q 退出
2. Jetson 端 Ctrl+C 停止
3. PICO 退出 XRoboToolkit app
4. PC Service 可保持运行
```

## 11. 常见问题

### Jetson 连不上 PC

在 Jetson 上测试端口：
```bash
nc -vz 192.168.50.101 5570
nc -vz 192.168.50.101 5580
```

如果连不上，检查：
- PC 防火墙：`sudo ufw status`
- 网络是否在同一子网
- PC IP 是否正确：`hostname -I`

### 数据集已存在

加 `--resume` 继续追加 episode，或换新 `--repo-id`：
```bash
--repo-id local/so101_dual_xr_teleop_v2
```

### 相机没画面

在 Jetson 上单独测试相机：
```bash
python -m lerobot.scripts.check_cameras_one_by_one
```

### 机械臂不动

检查 Jetson 日志是否收到 IK 目标：
```
# Jetson 端应该有类似输出：
# Received target, moving to ...
```

如果没有，检查 PC 端 PICO 是否已连接，grip 键是否按住。

### 机械臂抖动

```bash
# Jetson 端降低 max_relative_target
--max-relative-target 2.0

# PC 端增大 joint regularization（修改源码中 1e-3 → 1e-2）
# 或降低 scale_factor
--scale-factor 0.5
```
