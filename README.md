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
- **OpenCV Kannala–Brandt** 真鱼眼成像（宽 FOV pinhole → LUT → `cv2.remap`）
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
