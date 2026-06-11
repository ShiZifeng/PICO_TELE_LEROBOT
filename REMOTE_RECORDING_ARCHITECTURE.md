# XRoboToolkit 远程录制架构说明

## 1. 总体架构

```
┌── PICO VR 头显 ──────────────────────────────────────────┐
│  XRoboToolkit APK                                         │
│  - 手柄 6DoF 位姿追踪 (left/right controller)             │
│  - 按钮: grip, trigger, A/B/X/Y, joystick                │
│  - H.264 视频接收 (Remote Vision → ZEDMINI → Listen)     │
└──────┬───────────────────────────────────────────────────┘
       │ XR SDK (TCP 60061)
       ▼
┌── PC (192.168.50.75) ─────────────────────────────────────┐
│                                                            │
│  ┌─ IK 线程 (60Hz) ────────────────────────────────────┐  │
│  │  XrClient 读手柄位姿 → XRIKController.step()        │  │
│  │  → Placo IK 求解 → 关节角度(度) → ZMQ PUB(5580)    │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─ State 接收线程 (60Hz) ────────────────────────────┐  │
│  │  ZMQ PULL(5572) ← Jetson 高频关节状态               │  │
│  │  → update_robot_state() → 更新 Placo q 向量         │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─ 录制主循环 (15Hz) ────────────────────────────────┐  │
│  │  ZMQ PULL(5570) ← Jetson 观测帧                     │  │
│  │  → decode_frame() → add_frame() → LeRobotDataset    │  │
│  │  → MJPEG(8080) / H.264(12345) 推流 → PICO          │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─ H.264 TCP Streamer ───────────────────────────────┐  │
│  │  控制端口(13579) ← PICO CameraRequest               │  │
│  │  视频端口(12345) → PICO H.264 视频流                │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─ MJPEG Streamer ───────────────────────────────────┐  │
│  │  HTTP(8080) → PICO 浏览器 相机预览                  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  按键控制: Space/X=录制 R=丢弃 Q/Esc=退出 Y=左臂归位 B=右臂归位 │
└──────┬───────────────────────────────────────────────────┘
       │ ZMQ TCP
       ▼
┌── Jetson (192.168.50.47) ──────────────────────────────────┐
│                                                            │
│  ┌─ 控制线程 (60Hz) ───────────────────────────────────┐  │
│  │  ZMQ SUB(5580) ← PC IK 关节目标                     │  │
│  │  → robot.send_action() → SO101 Follower 电机        │  │
│  │  → robot.get_observation() → 关节状态               │  │
│  │  → ZMQ PUSH(5572) → PC state 回传 (60Hz)            │  │
│  │  → 每 4 tick: Queue.put() 通知录制主循环            │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─ 录制主循环 (15Hz, Queue 同步) ────────────────────┐  │
│  │  Queue.get() 等待控制线程信号 (每 4 tick)            │  │
│  │  → 采样 state + action → 读相机 → JPEG 编码         │  │
│  │  → 构建 frame → ZMQ PUSH(5570) → PC                 │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─ 硬件 ─────────────────────────────────────────────┐  │
│  │  SO101 Follower 双臂 (left/right)                   │  │
│  │  OpenCV 相机 ×3 (left_arm/dev10, right_arm/dev8,    │  │
│  │                    head/dev0)                        │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ZMQ SUB(5571) ← PC 控制命令 (stop)                       │
└────────────────────────────────────────────────────────────┘
```

## 2. 端口分配

| 端口 | 方向 | 协议 | 频率 | 内容 |
|------|------|------|------|------|
| 5570 | Jetson → PC | ZMQ PUSH/PULL | 15Hz | 录制帧 (obs+action+相机JPEG) |
| 5572 | Jetson → PC | ZMQ PUSH/PULL | 60Hz | 关节状态回传 (IK 用) |
| 5580 | PC → Jetson | ZMQ PUB/SUB | 60Hz | IK 关节目标角度 |
| 5571 | PC → Jetson | ZMQ PUB/SUB | 按需 | 控制命令 (stop) |
| 8080 | PC → PICO | HTTP MJPEG | 实时 | 相机预览 (浏览器) |
| 12345 | PC → PICO | TCP H.264 | 实时 | 视频推流 (APK Remote Vision) |
| 13579 | PICO → PC | TCP | 按需 | H.264 控制通道 |

## 3. Jetson 端详解

文件: `scripts/hardware/teleop_so101_remote_jetson.py`

### 3.1 双线程架构

```
控制线程 (60Hz, next_tick 相位锚定):
  while not stop:
      drain ZMQ SUB(5580) → 最新 PC IK 目标
      robot.send_action(action)          # 控制电机
      obs = robot.get_observation()      # 读关节状态
      → 更新 _control_state (共享内存)
      → ZMQ PUSH(5572) 高频回传 state
      → 每 N 个 tick: Queue.put(True)   # 通知录制主循环
      next_tick += 1/60
      sleep(remaining)

录制主循环 (15Hz, Queue.get 阻塞等待):
  while not stop:
      Queue.get(timeout=0.5)             # 等待控制线程信号
      → 从 _control_state 采样 obs + action
      → read_cameras() → JPEG 编码
      → 构建 frame: {observation.state, action, observation.images.*}
      → ZMQ PUSH(5570) → PC
      → 每 2s 补发 setup 消息(防止 ZMQ 丢包)
```

### 3.2 同步机制

使用 `queue.Queue(maxsize=1)` 实现控制线程到录制循环的同步：
- 控制线程每 N 个 tick 调用 `put_nowait(True)`（`N = control_fps / fps`，如 60/15 = 4）
- 录制循环 `get(timeout=0.5)` 阻塞等待，唤醒后采样当前最新状态
- `maxsize=1` 确保不堆积信号（录制慢时不积压）
- 比 `threading.Condition` 更可靠（跨平台兼容）
- 控制线程完全不阻塞（put_nowait 非阻塞）

### 3.3 帧格式

```python
frame = {
    "observation.state": np.array([...], dtype=float32),  # shape (12,) 双臂12关节
    "action":           np.array([...], dtype=float32),  # shape (12,) PC IK目标
    "observation.images.left_arm":  {"__remote_image_encoding__": "jpg", "data": bytes},
    "observation.images.right_arm": {"__remote_image_encoding__": "jpg", "data": bytes},
    "observation.images.head":      {"__remote_image_encoding__": "jpg", "data": bytes},
}
```

### 3.3 Features 定义

与 lerobot 标准格式完全对齐（聚合格式）:

```python
features = {
    "action":              {"dtype": "float32", "shape": (12,), "names": ["left_shoulder_pan.pos", ...]},
    "observation.state":   {"dtype": "float32", "shape": (12,), "names": [...]},
    "observation.images.*": {"dtype": "video",   "shape": (h, w, 3)},
}
# 另有 DEFAULT_FEATURES: timestamp, frame_index, episode_index, index, task_index
```

## 4. PC 端详解

文件: `scripts/hardware/teleop_so101_remote_pc.py`

### 4.1 XRIKController

核心类，替代 lerobot 的 Leader 臂，将 XR 手柄位姿转换为关节目标。

**输入**: XR 手柄 6DoF 位姿 (XrClient SDK)
**输出**: 电机角度目标 dict (`{left_shoulder_pan: 45.0, ...}`)

**控制模式**:
- `pose`: 全位姿 IK（6DoF 手柄 → 6DoF 末端执行器）
- `position`: 位置 IK（3DoF 手柄位置 → 3DoF 末端位置）
- `position_wrist`: 腕部 IK（3DoF 手柄位置 + 手柄俯仰 → wrist_link xyz + wrist pitch elevation + joystick wrist_roll）

**step() 主流程**:

```
1. 读取 Y/B 按钮 → 触发归位
2. 计算 homing_overrides（归位臂的插值目标）
3. update_kinematics()（从最新观测状态）
4. XR 确认检查（A 键校准朝向）
5. 处理每臂:
   - 归位臂: hold effector（当前位姿），skip XR 更新
   - grip 按下: 激活 effector task，更新 XR 位姿目标
   - grip 松开: hold effector，deactivate
6. 夹爪控制（跳过归位臂）
7. 腕部摇杆控制
8. IK 求解（Placo KinematicsSolver）
9. 组装目标: _ik_to_motor_dict() + homing_overrides
10. 低通滤波 + 步进限幅 + 关节限位裁剪
11. 安全检查（max_action_delta）
12. 返回目标 dict
```

### 4.2 归位功能

每个臂独立归位，互不影响:

| 按钮 | 效果 |
|------|------|
| Y | 左臂归位 |
| B | 右臂归位 |
| Y+B 同时 | 双臂同时归位 |

**归位姿态** (电机角度):

| 关节 | 左臂 | 右臂 |
|------|------|------|
| shoulder_pan | 0° | 4° |
| shoulder_lift | -80° | -83° |
| elbow_flex | 84° | 86° |
| wrist_flex | 85° | 89° |
| wrist_roll | -57° | -53° |
| gripper | 45° | 45° |

**实现**:
- 2 秒 smoothstep 曲线 (t²·(3-2t))
- 归位臂从 IK solver 中隔离（hold effector + active=False + _restore_inactive_arm_joints）
- 归位插值完全在 IK 外部，不影响另一臂的 grip 跟随
- 归位期间 IK 失败不触发安全暂停

### 4.3 安全机制

| 机制 | 说明 |
|------|------|
| XR 朝向确认 | 按 A 键校准 headset yaw，确认前不驱动机械臂 |
| max_action_delta_deg (60°) | 目标与观测偏差过大 → 安全暂停 |
| max_target_step_deg (3°) | 每 tick 目标变化限幅 |
| target_filter_alpha (0.05) | 一阶低通滤波平滑 |
| max_wrist_reach_m (0.285m) | 腕部工作空间半径限制 |
| 关节限位裁剪 | 目标裁剪到 URDF 定义的 [lower, upper] |
| IK 连续失败 (3次) | 自动安全暂停 |

### 4.4 录制流程

```
PC 录制主循环 (obs_socket.recv_pyobj 阻塞):
  收到 "setup"  → LeRobotDataset.create() / resume
  收到 "frame"  → decode_frame(JPEG解码)
                → 检查录制状态 (Space/X 切换, R 丢弃)
                → dataset.add_frame(frame)
                → 推流 MJPEG / H.264
                → cv2 预览窗口
  收到 "done"   → 保存最后 episode → 退出
```

## 5. 数据流全景

```
PICO手柄位姿                    Jetson硬件状态
     │                               │
     ▼                               │
PC: XrClient                   SO101 Follower 双臂
     │                               │
     ▼                               ▼
PC: XRIKController.step()      robot.get_observation()
     │                               │
     ├─ IK求解 (Placo)               ├─ joint positions (deg)
     │                               │
     ▼                               ▼
PC: ZMQ PUB(5580) ────joint targets────→ Jetson: ZMQ SUB(5580)
     60Hz                                       │
                                               ▼
                                        robot.send_action()
                                               │
                                               ▼
                                        robot.get_observation()
                                               │
                                    ┌──────────┴──────────┐
                                    │                      │
                                    ▼                      ▼
                            ZMQ PUSH(5572)         Queue.put()→主循环采样
                            state回传(60Hz)             15Hz
                                    │                      │
                                    ▼                      ▼
                            PC: IK state更新      构建frame(obs+action+相机)
                                                          │
                                                    JPEG编码
                                                          │
                                                          ▼
                                                  ZMQ PUSH(5570)
                                                          │
                                                          ▼
                                                  PC: 录制主循环
                                                  decode → add_frame
                                                          │
                                                          ▼
                                                  LeRobotDataset
                                                  (parquet + mp4)
```

## 6. 与 lerobot 标准流程的差异

| 维度 | lerobot 标准 | XRoboToolkit |
|------|-------------|-------------|
| 遥操作来源 | Leader 臂机械映射 | PICO VR + Placo IK |
| IK 位置 | 无 (直接关节映射) | PC 端 (Placo solver) |
| 控制与录制 | 单线程串行 (15Hz) | 双线程解耦 (60Hz控制 + 15Hz录制) |
| 线程同步 | busy_wait | Queue 信号 |
| 数据集格式 | action + observation.state (聚合) | ✅ 完全对齐 |
| 视频推流 | 无 | MJPEG(8080) + H.264(12345) |
| 安全机制 | max_relative_target | 多层安全链 (见 4.3) |
| 归位 | 无 | Y/B 单键独立归位 |

## 7. 关键设计决策

1. **Queue 替代 Condition**: `threading.Condition` 在 Jetson ARM Linux 上存在兼容性问题，`queue.Queue(maxsize=1)` 更可靠。

2. **归位在 IK 外部**: 归位插值不走 Placo solver，作为最终覆盖应用到输出。避免归位臂污染另一臂的 IK 求解。

3. **Setup 无条件补发**: 每 2 秒重新发送 setup 消息到 PC，防止 ZMQ NOBLOCK 在连接建立前丢包导致 PC 永久无法创建 dataset。

4. **State 与录制分离**: 60Hz 状态回传 (5572) 专用于 IK，15Hz 录制帧 (5570) 专用于数据集。IK 不受录制帧率影响。

5. **数据集聚合格式**: action/observation.state 使用 lerobot 标准聚合格式 (12,) 而非逐关节独立列，保证下游训练流程兼容。
