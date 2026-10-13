# Parking-lite Demo

轻量级停车场环视验证栈：**4 路鱼眼相机 → AVM / BEV**。

基于 [ModernGL](https://github.com/moderngl/moderngl)（offscreen / llvmpipe）+ OpenCV `cv2.fisheye`（Kannala–Brandt）做仿真成像与 IPM 拼接。适合本地快速验证相机 `K`/`D`、外参与环视一致性，无需完整游戏引擎或物理仿真。

**状态：Phase 5 完成** — 一键 demo、拼图打包、短程 ego-motion BEV 序列。

**另含：** 可配置 **运动学车辆控制** + 轮速脉冲 / 车速反馈（`vehicle/`、`configs/vehicle.yaml`）。

**坐标系约定：** 车辆 **FLU** — forward-x、left-y、up-z。配置中的相机 `K`/`D` 为仿真 **ground-truth**（OpenCV fisheye，`D=[k1,k2,k3,k4]`）。

**许可证：** [MIT](LICENSE)（Copyright (c) 2026）。

---

## 功能特性

- **程序性停车场场景** + 动态自车网格（lot / ego）
- **四路鱼眼**（front / rear / left / right），YAML 配置 `K`、`D`、外参
- **OpenCV Kannala–Brandt** 真鱼眼成像（宽 FOV pinhole → 原 KB LUT → GPU remap，与 OpenCV 逐像素核对）
- **AVM / BEV 拼接**：地面 IPM + feather 融合，自车车身 ROI 遮罩
- **God view**：世界固定针孔俯视概览
- **Phase 4 校验**：内参一致性、地标对齐、接缝 MAD、外参扰动敏感度
- **一键 demo**：拼图 collage + 短程运动 GIF
- **运动学自行车模型** 车辆控制（方向盘 / 加速度 / 挡位 + 轮脉冲反馈）

---

## 架构

```
configs/default.yaml  ──►  World.from_config
                              │
                     set_ego_pose(x,y,z,yaw)   # FLU, yaw about up-z
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       render_cameras()  render_god_view()  stitch_avm(images)
              │               │               │
         4× fisheye      pinhole orbit    IPM + feather BEV
         (KB remap)      (world-fixed)    (same K/D/extrinsics)
```

| 层 | 职责 |
|----|------|
| `scene/world.py` | 配置加载、ego SE3、render / stitch 入口 |
| `scene/lot.py` + `ego_vehicle.py` | 程序性车位 + 自车网格 |
| `sensors/cameras.py` | `CameraRig`、FLU↔GL、每相机 K/D |
| `sensors/moderngl_backend.py` | Offscreen GL：lot/ego VAO、宽 FOV pinhole |
| `sensors/fisheye_opencv.py` | KB 逆投影 LUT → `cv2.remap` |
| `avm/stitcher.py` | 地面 IPM LUT + feather → BEV |
| `avm/validate.py` | 地标 / 接缝 / 敏感度（Phase 4） |
| `vehicle/` | 运动学自行车控制 + 轮脉冲（YAML 驱动） |

---

## 快速开始

在全新机器上（需 Python 3.10+，建议 3.11–3.13）：

```bash
git clone <YOUR_REPO_URL> parking-lite-demo
cd parking-lite-demo

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python scripts/run_demo.py
```

输出写入 `output/demo/`：

| 产物 | 说明 |
|------|------|
| `collage.png` | God + BEV + 四路鱼眼拼图 |
| `god_view.png` | 世界固定针孔概览 |
| `{front,rear,left,right}.png` | 环视鱼眼（OpenCV K/D） |
| `bev.png` / `bev_overlay.png` | AVM 拼接（+ 车位 CAD 叠加） |
| `validation_summary.txt` | 覆盖率等 smoke 指标 |
| `motion/bev_frame_XX.png` | 短程 ego-motion BEV 帧 |
| `motion/bev_motion.gif` | 过道运动 BEV 动画 |

常用参数：`--no-motion`、`--no-validate`、`--config PATH`、`--out DIR`。

> 更完整的安装、改 K/D、AVM 校验与排错见 **[docs/指南.md](docs/指南.md)**。

---

## 配置

| 文件 | 用途 |
|------|------|
| `configs/default.yaml` | 场景、车辆尺寸、四相机 K/D/外参、BEV 范围、god_view、render |
| `configs/vehicle.yaml` | 运动学控制参数（轴距、转向比、轮半径、脉冲等） |
| `configs/j01.yaml` | **本车 J01**：Kannala–Brandt 内参 + VFF 平均外参，后轴原点 |
| `configs/j01_vehicle.yaml` | J01 运动学（轴距 2.8 m、转向比 15.3、轮半径 0.334 m） |

本车参数入口：

```bash
python scripts/run_demo.py --config configs/j01.yaml --out output/j01

# 环绕自车看 J01 数模（无编辑器窗口；出 collage + GIF）
python scripts/view_scene.py --config configs/j01.yaml --out output/j01_view

# 可拖动的实时窗口（本机 GNOME 在 :1）
./scripts/run_view_window.sh
# 方向键开车，左键绕转，右键平移，滚轮缩放
```

`default.yaml` 中的 `vehicle_config:` 指向控制配置；场景里的 `vehicle:`（长宽高等）由 `World` 使用，两者职责分离。

关键片段示例（相机）：

```yaml
cameras:
  front:
    translation_m: [2.35, 0.0, 1.05]
    rotation_deg: [0.0, -28.0, 0.0]   # [roll, pitch, yaw]，FLU 固有 XYZ
    K: [[420.0, 0.0, 640.0],
        [0.0, 420.0, 480.0],
        [0.0, 0.0, 1.0]]
    D: [-0.045, 0.018, -0.006, 0.001]  # OpenCV fisheye [k1,k2,k3,k4]
```

修改 `K`/`D` 或外参后重新跑 `run_demo.py` / `run_phase2_fisheye.py` / `run_phase3_avm.py` 即可生效；无需改代码。

`j01.yaml` 的 `vehicle.origin: rear_axle` 与量产 `extrinsic.txt` 一致：相机 `translation_m` 相对后轴中心，车身网格 / BEV ROI 用 `rear_overhang_m` 前移，不要再把后轴外参当成几何中心。

---

## 车辆控制 API

运动学自行车模型，**无动力学 / 无物理引擎**。与渲染解耦；demo 通过 `World.set_ego_pose` 同步位姿。

```python
from vehicle import VehicleController

ctrl = VehicleController("configs/vehicle.yaml")
ctrl.reset(x=0, y=0, yaw=0, speed=0)
ctrl.set_controls(steer_wheel_deg=200, accel_mps2=1.0, gear="D")
state = ctrl.step()          # 或 step(dt=0.02)
# state.x, state.y, state.yaw, state.speed_mps, state.speed_signal_mps
# state.wheel_pulse_count, state.wheel_speed_mps  # fl,fr,rl,rr
ctrl.apply_to_world(world)   # 可选：写入 world.set_ego_pose(...)
```

| 量 | 单位 / 约定 |
|----|-------------|
| Pose `x,y` | 米，世界 **FLU** |
| `yaw` | 弧度，绕 +z |
| `speed_mps` | **有符号** 车体纵向 vx — D ≥ 0，R ≤ 0 |
| `speed_signal_mps` | 发布车速 = `abs(speed_mps)` |
| 方向盘 / 路轮转角 | 度；+路轮 = 左转（+y） |
| 轮脉冲 | 累计 ABS 风格 tick |

Demo：

```bash
python scripts/run_vehicle_control_demo.py
```

产物在 `output/vehicle_control/`（`trajectory.csv`、轨迹图、若干 god_view）。

---

## Phase 脚本一览

| 脚本 | 阶段 | 说明 |
|------|------|------|
| `scripts/run_phase0_smoke.py` | 0 | 配置 / World / CameraRig / 鱼眼 smoke |
| `scripts/run_phase1_scene.py` | 1 | Lot + ego、god view、位姿变化 |
| `scripts/run_phase2_fisheye.py` | 2 | 真鱼眼（K/D）+ collage + D 对比 |
| `scripts/run_phase3_avm.py` | 3 | IPM 拼接 → BEV |
| `scripts/run_phase4_validate.py` | 4 | 对齐 / 接缝 / 敏感度报告 |
| `scripts/run_demo.py` | 5 | **一键 demo**（推荐入口） |
| `scripts/run_vehicle_control_demo.py` | — | 运动学控制 + 位姿同步 |
| `render_fisheye_demo.py` | legacy | 旧 CLI；优先用 `run_demo.py` |

World API 速览：

```python
from scene.world import World
world = World.from_config("configs/default.yaml")
world.set_ego_pose(x, y, z, yaw)          # yaw 弧度，绕 up-z
images = world.render_cameras()           # dict[str, HxWx3 uint8]
overview = world.render_god_view()
bev = world.stitch_avm(images)
```

---

## 坐标系

| 坐标系 | 轴向 | 用途 |
|--------|------|------|
| **Vehicle / world FLU** | +x 前，+y 左，+z 上 | Ego 位姿、相机外参、BEV 范围 |
| **OpenCV optical** | +X 右，+Y 下，+Z 前 | 鱼眼投影 / IPM 射线 |
| **GL scene** | Y-up，+Z 前 | 程序性网格（边界处转换） |

**BEV 图像布局：** 上 = +x 前，左 = +y 左（俯视 −z）。自车车身 ROI 涂黑。

**相机外参：** `translation_m` + `rotation_deg [roll, pitch, yaw]`，车辆 FLU（固有 XYZ）。Pitch 朝向地面。

---

## 目录结构

```
parking-lite-demo/
  configs/default.yaml         # 场景 + 四相机 K/D/外参 + BEV
  configs/vehicle.yaml         # 运动学控制参数
  scene/                       # World、lot、ego、mesh helpers
  sensors/                     # CameraRig、ModernGL、OpenCV fisheye
  avm/                         # stitcher + validate
  vehicle/                     # VehicleController API
  assets/                      # 预留 lot/、vehicle/ 网格（见 assets/README.md）
  scripts/                     # Phase 0–5 + vehicle demo
  docs/指南.md                 # 中文完整使用说明
  requirements.txt
  LICENSE                      # MIT
```

---

## 已知限制

- 程序性 lot / ego 网格在 **GL Y-up / +Z forward**；位姿与相机在边界 FLU↔GL 转换。
- 仅 ego 跟随 `set_ego_pose`；静态停车与车位线不动。
- 鱼眼使用 YAML **`K`/`D` 作为仿真 GT**；可选 legacy equidistant GPU 路径。
- **单宽 pinhole → KB remap**：θ ≥ ~90° / 落在 frustum 外的采样为黑，非全球面。
- BEV IPM 假设 **平坦地面 z=0**；高出地面的物体（柱子、车身、抬高 marker）会剪切/错投。
- Feather 软化接缝；Phase 4 报告 overlap MAD（光照残差可能 WARN）。
- Validator **检测**外参误差，**无在线外参求解 / 精化**。
- Phase 5 运动 demo 为离线逐帧（无 rolling shutter）。
- 车辆控制为 **纯运动学**（自行车模型）；无轮胎动力学、侧滑；左右轮速暂相等（无 Ackermann 差速）。

---

## 后续方向（Phase 5 范围外）

- Cube-face / 多 frustum 捕获以覆盖全球面鱼眼边角
- 在线外参恢复 / 精化求解器
- 连续 ego-motion 下的多帧 BEV 一致性指标
- glTF/OBJ 资源替换 lot / ego

---

## License

MIT License — 见 [LICENSE](LICENSE)。Copyright (c) 2026。

## H264 四路输入

`World.render_camera_grid()` 在 GPU 直接生成 `front|left / right|rear` 的 2×2 拼图，只读回一次。原 `render_cameras()` 字典接口保留；KB LUT、K/D、FOV 和外参继续使用现有配置。GPU 按 OpenCV INTER_LINEAR 的 1/32 像素权重与整数舍入采样，invalid 区域为黑。

用交互窗口控制板端视频，先按板端工具说明启用视频输入，再启动：

```bash
rtk proxy bash scripts/run_view_window.sh --stream-adb
```

内部采样默认 4×、发送默认 10 fps / 总码率 12M。ADB 自动选择唯一已连接设备；多设备时加 `--serial 5e9350b6`。不加 `--stream-adb` 只打开本机窗口，窗口和独立发送脚本都默认使用 4× 采样，可用 `--capture-scale 1` 切回原采样方式。

窗口驾驶更新相机进程的车辆状态，相机独立按 10 fps 采样和编码；每帧只生成一次全分辨率四路画面，窗口读取同一批图像的最新预览。预览带来源序号和采样位姿，可与对应编码帧核对。停车、隐藏预览时也持续发送；关闭窗口停止发送。上下方向键前进 / 倒车，左右键转向，空格制动，R 回到初始位置。鼠标拖动与缩放只改变主机观察视角。该模式只需启动窗口；同时运行独立发送进程会争用同一个接收连接。

窗口 `Board SEND` 行显示成功发送的累计 AU 数及最近约一秒的实际 fps / Mbps，同时列出目标帧率 / 码率。它统计主机 TCP 发送完成，不代表板端解码或屏幕刷新帧率。`check_window_stream.py` 放慢预览请求，核对来源帧并验证视频仍为 10 fps；吞吐检查排除硬件编码器冷启动，应关闭另一个发送窗口后运行。

联动模式的概览也按相机节拍更新；未变化的大图缓存为显示 surface，减少重复渲染、缩放和像素复制给视频发送造成的额外开销。ADB 建立转发时的端口号输出已隔离，避免污染相机进程的二进制通信；无板端自检包含这个启动场景。

`--capture-scale 1|2|4` 默认为 4，只提高内部针孔捕获尺寸，输出 K/D、FOV 和四路 1280×960 保持一致。4× 在 GPU 上生成 5120×3840 中间图，直接映射到最终鱼眼，仍只读回最终拼图。中间缓冲按需创建，主机概览仍按原尺寸渲染。当前主机四路出图 P95：1× 16.6 ms、2× 23.9 ms、4× 27.2 ms，4× 的轮廓和车位线更清楚。早期窗口驱动采样在桌面负载下运动时降至约 9.49 fps，现采样 / 编码与预览读写分离；最终实机 25 秒驾驶约 9.99 fps，持续链路约 10.00 fps，板端接收 / 解码 / 发布一致。保持总码率 12M。`--capture-scale 1` 为原采样方式。

```bash
rtk proxy .venv/bin/python scripts/check_gpu_fisheye.py --capture-scale 4 --frames 8
rtk proxy .venv/bin/python scripts/check_window_stream.py --capture-scale 4
```

`rtk proxy .venv/bin/python scripts/check_window_stream.py` 可独立检查 8 个指定车辆位姿，核对预览与源图像像素、完整 H264 帧及解码后的场景，无需板端。

下面的独立发送命令使用脚本自身的自动运动，与不带 `--stream-adb` 的窗口各自运行：

```bash
rtk proxy .venv/bin/python scripts/check_gpu_fisheye.py
rtk proxy .venv/bin/python scripts/stream_h264_adb.py \
  --serial 5e9350b6 --scale 1 --fps 10 --seconds 60
```

发送前会预热 GL shader 和 LUT，再给实时帧赋 PTS。流通过 Android 字节转发进入 QNX 解码器，保留两端 AIS/QCarCam 输入；板端启动与恢复见 [ais_camera_tools 说明](/home/huang/fawvw_apa_env/src/tools/ais_camera_tools/stream/README.md)。

持续查看板端时使用 `--seconds 0 --loop-motion`：车辆在场内平滑往返，最大速度 0.4 m/s、周期约 63 秒，四路画面持续变化。可先用 `--loop-motion --seconds 2 --file output/loop_smoke.plit` 检查真实仿真出图和编码。

当前全分辨率同条件出图均值约 14.5 ms（CPU 参考约 37.7 ms），120 秒实流发送、解码和发布 1198 帧。测试范围与资源记录见 [验证报告](/home/huang/work/ais-video-converge-20261012/validation_report.md)。

延迟诊断可用 `--sequence-pattern --frame-log output/frames.jsonl`：四路输入使用相同的逐帧颜色编码，记录生成与发送的主机单调时间。`scripts/check_stream_latency.py` 检查 36 个主画面点与四个俯视区，并通过 Android uptime 与主机 RPC 时间边界校准时钟。512 个颜色在 10 fps 下每 51.2 秒重复，适用于本链路的短延迟诊断。

在发送脚本持续运行期间，另一终端执行：

```bash
rtk proxy .venv/bin/python scripts/check_stream_latency.py \
  --source-log output/frames.jsonl --out output/screen_clock --samples 3
rtk proxy adb -s 5e9350b6 shell su 0 screenrecord \
  --bugreport --time-limit 60 --size 2560x1440 --bit-rate 8000000 \
  /data/local/tmp/avm_latency.mp4
rtk proxy adb -s 5e9350b6 pull /data/local/tmp/avm_latency.mp4 output/avm_latency.mp4
rtk proxy .venv/bin/python scripts/check_stream_latency.py \
  --source-log output/frames.jsonl --video output/avm_latency.mp4 \
  --clock-calibration output/screen_clock/results.json --out output/video_latency
rtk proxy .venv/bin/python scripts/check_stream_latency.py --self-test
```

单张截图只给出画面新鲜度的时间区间。连续录像使用 Android 11 `screenrecord` 内嵌的 Winscope 逐帧 elapsed 时间戳，给出首次观察与显示切换的 P95 区间；终点是 SurfaceFlinger 虚拟显示，不包含物理屏幕扫描。结果不会把录像相对 PTS 当成主机时间。

较长录像可在结束后再采一次时钟，给 `--clock-calibration` 同时传入录像前后的两个 `results.json`，保留漂移与 RPC 的不确定区间。当前 600 秒真实流发送 / 解码 / 发布 5989 帧；60 秒显示录像有 5 处未录到的源序号，严格检查返回非零，尚未计作无跳帧验收。记录见 [2026-10-13 报告](/home/huang/work/ais-video-latency-20261013/validation_report.md)。

板端启动工具现默认采用无 B 帧码流的 `DisplayPictureBuffer=1`，到达至发布中位耗时约 130→29 ms；关闭逐帧日志后，60 秒录像首次观察 P95 为 210–238 ms，仍有 3 个源序号未录到。原厂 AIS / QCarCam 与算法链保留，最新边界与证据见 [后续优化报告](/home/huang/work/ais-video-default-20261013/validation_report.md)。
