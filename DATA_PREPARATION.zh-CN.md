# 数据准备

本文记录已使用的数据、对应工具仓库，以及下载和源格式转换示例。人体数据准备完成后，应为 Holosoma 可直接读取的输入，并指定对应的 `data_format`；PHP G1 成品数据保持原格式，用于流程对照。以下命令从仓库根目录执行，`/path/to/` 替换为自己的路径。

## 使用的数据

已有流程使用以下六类数据：PARC、NUS、HOLD 提供障碍技能，LAFAN1 提供移动，Holosoma 和 PHP G1 提供流程参考。另已补充 CMU Mocap 和 PRISM 跑酷候选子集，进度见下文。EgoHTR 不纳入本流程。

| 数据来源 | 格式 | 使用规模 | 已完成内容 |
|---|---|---|---|
| **LAFAN1** | BVH，Holosoma `lafan` | 全部 12 条 walk＋4 条 run，共 115,831 帧 | 16/16 条已完成官方 T2 重定向，共 115,831 帧（30 Hz）；本轮未构库或 match。原 6 条保留历史构库及 100 组移动合成 |
| **Holosoma 攀爬示例** | 53 关节 NPY，`mocap`，含配对障碍 | `mocap_climb_seq_0–4`，共 5 组 | 早期 T2 重定向、构库及 10 条原始/镜像轨迹合成 |
| **PHP 官方 G1 数据** | 机器人 qpos NPZ、BIN、tracking NPZ 和地形 | 19 个匹配前动作库、296 对训练数据，另有 1 条构库前示例 | 源动作准备、数据核验及逐库生成，用于流程对照 |
| **HOLD** | BVH＋逐帧箱体，转换后为 `hold` | I1–I16，21,816 源帧 | 16 条带箱体 T2 重定向全部成功，共 5,469 帧、30 Hz |
| **PARC** | PKL 动作＋高度场，转换后为 `parc` | 77 条核心动作、73 条配对版本 | 带地形 T2 重定向 65 条成功，共 9,773 帧；8 条求解不可行 |
| **NUS 障碍交互动捕** | BVH 转关节位置 NPZ，Holosoma `nus` | 13 条、16,521 帧（120 Hz） | 格式转换与无障碍 T2 重定向，共 4,136 帧（30 Hz） |

PARC、NUS、HOLD 当前完成到重定向，尚未构库或进行 motion matching；NUS 为无障碍结果。早期 Holosoma 结果作为历史流程参考。

## PARC

来源：[官方数据](https://huggingface.co/datasets/mxucg/PARC/tree/e8f119532d2f03955031ce629d08243403443e58)，固定版本 `e8f119532d2f03955031ce629d08243403443e58`；[读取器与人体骨架](https://github.com/mshoe/PARC/tree/846ed98bd526c1dd93ddc1701b601036d553cd2f) 固定版本 `846ed98bd526c1dd93ddc1701b601036d553cd2f`。

已提取 77 条核心动作及 73 条对应的官方动作—地形配对版本，覆盖攀爬、跳隙、平台、楼梯和撑越。使用 `initial_aug` 中的完整配对版本；`initial` 仅有动作，不能与其他版本的地形混用。4 条 dive/double kong 暂无同名配对地形。

PARC PKL 不是 LAFAN 或 SMPL。准备后的 Holosoma 格式为 `parc`：

- 按上游 `humanoid.xml` 对根位姿和 14 个局部关节四元数做 FK，保留米制 Z-up 坐标；脚尖取脚部几何的前端底面点。
- `motion.npz` 保存 `global_joint_positions: (T,17,3)`、`root_quaternion: (T,4)`（wxyz）、`height` 和 `fps=30`。源模型站立高度为 1.611461 m。
- 高度场按 `min_point + [i,j] * dx` 定位单元中心；相同高度的相邻单元合并为长方体，保持顶面和边界，向地下延伸为实体。每条目录提供 `multi_boxes.obj`、`multi_boxes.urdf`、`box_assets.xml` 与 T2 场景 XML。
- 重定向时人体与地形统一按 `1.4 / 1.611461` 缩放，不对动作单独归零高度。

## HOLD

来源：[HOLD 固定版本](https://github.com/rai-opensource/hold/tree/f80b04d384d4ec8922c88deada878c88597a3cf9)，许可为 CC BY 4.0。选择原始尺度的 `I1–I16`：低箱 6 条、中箱 6 条、高箱 4 条，共 21,816 帧、约 23.45 MB，不含镜像或 G1 成品。

BVH 已由上游处理，并含人体、估计足底骨骼及箱体骨骼。箱体位姿来自采集，尺寸来自上游推定的角点范围，不能视为精确实测尺寸。使用每个骨骼自己的平移通道和欧拉旋转顺序做 FK，保留米制 Z-up 世界坐标；以 `Hips` 源朝向转换到机器人前/左/上轴。

按文件实际 `Frame Time=0.008342` 读取时间，通过局部平移插值和旋转 SLERP 转为 30 Hz，共 5,469 帧，未做平滑或额外姿态修正。参考高度由骨架头端至估计足底计算，约 1.66–1.67 m，人体与箱体共用缩放比例。

准备后的格式为独立 `hold`：每条目录的 `motion.npz` 包含 16 关节位置、根朝向、参考高度和逐帧 `object_poses`（wxyz＋xyz），另配箱体网格、URDF 与含箱体自由关节的 T2 场景。原生重定向结果保留机器人和箱体的 45 维 qpos；回放副本拆为机器人 `qpos: (T,38)` 与箱体 `object_qpos: (T,7)`（xyz＋wxyz），不改变轨迹。

## LAFAN1

数据及读取代码来自 [Ubisoft LAFAN1 仓库](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/tree/94084601bacdf9cc3764b5c73daaeccae6035fac)，固定版本为 `94084601bacdf9cc3764b5c73daaeccae6035fac`。
数据采用 [CC BY-NC-ND 4.0 许可](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/94084601bacdf9cc3764b5c73daaeccae6035fac/license.txt)。
格式转换使用 [Holosoma](https://github.com/amazon-far/holosoma/tree/4a0bf04cde5930af5ed59da797568d32be972525) 的关节位置提取入口，固定版本为 `4a0bf04cde5930af5ed59da797568d32be972525`。

从[官方 ZIP](https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/94084601bacdf9cc3764b5c73daaeccae6035fac/lafan1/lafan1.zip)下载数据，提取全部 walk/run 序列，不按采集人筛选：

```bash
mkdir -p /path/to/lafan1/bvh
curl -L \
  https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/94084601bacdf9cc3764b5c73daaeccae6035fac/lafan1/lafan1.zip \
  -o /path/to/lafan1/lafan1.zip
unzip /path/to/lafan1/lafan1.zip \
  'walk*.bvh' 'run*.bvh' \
  -d /path/to/lafan1/bvh
```

在 Holosoma 重定向环境中，使用官方 LAFAN1 读取代码和 Holosoma 的 `extract_global_positions` 入口，将 BVH 转为世界坐标关节位置：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git /path/to/lafan1-code
GIT_LFS_SKIP_SMUDGE=1 git -C /path/to/lafan1-code checkout \
  94084601bacdf9cc3764b5c73daaeccae6035fac
PYTHONPATH=/path/to/lafan1-code \
python -m holosoma_retargeting.data_utils.extract_global_positions \
  --input-dir /path/to/lafan1/bvh \
  --output-dir /path/to/lafan1/positions
```

每条 BVH 输出一个完整的 `(T,22,3)` 米制 NPY，Holosoma 输入格式为 `lafan`。

## NUS

来源为 [NUS Motion Capture Database](https://mocap.cs.sfu.ca/nusmocap.html)，允许科研使用。已取得 13 条 BVH、对应 C3D 和三个主体的 VSK：跑动上下台 2 条、跳跃上下台 1 条、跨障碍 2 条、speed vault 4 条、下台滚翻 2 条、跪姿滚翻和 monkey vault 各 1 条。

下载示例：

```bash
mkdir -p /path/to/nus/bvh
curl -fL https://mocap.cs.sfu.ca/nusmocap/0017_RunningOnBench001.bvh \
  -o /path/to/nus/bvh/0017_RunningOnBench001.bvh
```

BVH 为英寸、Y-up、120 Hz；配对 VSK 的腿长与 BVH 偏移之比为 25.4 mm，C3D 声明 120 Hz。以下复用固定版本 LAFAN1 仓库的通用 BVH 读取/FK，用 `[x,y,z]→[x,-z,y]` 保持右手系，并每四帧取一帧。根朝向同时由 BVH 转到机器人前/左/上坐标轴，主体身高采用 NUS 公布值。

```bash
PYTHONPATH=/path/to/lafan1-code python - <<'PY'
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from lafan1 import extract, utils
from holosoma_retargeting.config_types.data_type import NUS_DEMO_JOINTS

heights = {"0012": 1.74, "0015": 1.71, "0017": 1.65}
world = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
robot_axes = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
output = Path("/path/to/nus/positions")
output.mkdir(parents=True, exist_ok=True)
for source in Path("/path/to/nus/bvh").glob("*.bvh"):
    anim = extract.read_bvh(str(source))
    _, positions = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    positions = positions[:, [anim.bones.index(n) for n in NUS_DEMO_JOINTS]]
    positions = ((positions * 0.0254) @ world.T)[::4]
    root = world @ Rotation.from_quat(anim.quats[:, 0], scalar_first=True).as_matrix() @ robot_axes
    root = Rotation.from_matrix(root).as_quat(scalar_first=True)[::4]
    np.savez_compressed(output / f"{source.stem}.npz",
        global_joint_positions=positions,
        height=heights[source.name.split("_")[0]], root_quaternion=root, fps=30)
PY
```

输出为 `(T,25,3)` 关节位置 NPZ，接口为 `nus`，不属于 `lafan` 或 53 关节 `mocap`。这些文件没有配对障碍几何；当前输入用于无障碍动作重定向。

## Holosoma 攀爬示例

数据来自固定版本 Holosoma 的 [climb 示例目录](https://github.com/amazon-far/holosoma/tree/4a0bf04cde5930af5ed59da797568d32be972525/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb)，包含五组人体关节位置和配对障碍。
该版本由父仓库的子模块提交引用固定为 `4a0bf04cde5930af5ed59da797568d32be972525`，通过以下命令获取：

```bash
git submodule update --init --recursive --checkout
```

五组示例位于 `thirdparty/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb/`，名称为 `mocap_climb_seq_0` 至 `mocap_climb_seq_4`。
输入已是 Holosoma 可读取的 53 关节位置 NPY，格式为 `mocap`；保留人体数据与配对障碍的原目录结构。

## PHP 官方 G1 数据

官方 [motion-matching-assets-v1](https://github.com/amazon-far/php_parkour/releases/tag/motion-matching-assets-v1) 公开了完整的 **19 个 G1 匹配前动作库**（BIN，60 FPS），以及 32 个 OBJ、32 个 JSON 地形文件。

[五组训练数据发布包](https://github.com/amazon-far/php_parkour/releases/tag/training-motion-examples-v1) 为匹配后已合成的 296 对动作/地形数据。
下载工具为 PHP 仓库中的 `scripts/download_assets.py`，下载文件按仓库的 `release-assets.json` 清单校验：

```bash
python scripts/download_assets.py databases
for component in motions-locomotion motions-low-step motions-high-step \
  motions-low-climb-76 motions-high-climb-76; do
  python scripts/download_assets.py "$component"
done
```

这批数据已是机器人 BIN、tracking NPZ 及配对地形，可直接用于 G1 流程。已完成数据核验及逐库生成，作为 T2 数据流程的对照。

此外，官方源码附带一条[构库前 G1 动作示例](https://github.com/amazon-far/php_parkour/blob/c7c9ce3fc75a8754a7e403c58de3b6933254d01b/motion_matching/motion_matching/assets/robot_motions/climb_76_high_speed_up.npz)：190 帧、30 FPS，`qpos: (190,36)`，PHP `[wxyz, xyz, joint29]` 布局。

```bash
mkdir -p /path/to/g1/source_motions
curl -fL \
  https://raw.githubusercontent.com/amazon-far/php_parkour/c7c9ce3fc75a8754a7e403c58de3b6933254d01b/motion_matching/motion_matching/assets/robot_motions/climb_76_high_speed_up.npz \
  -o /path/to/g1/source_motions/climb_76_high_speed_up.npz
```

该文件可直接作为 `motion_matching.generate_database` 的输入，配对地形为动作库发布包中的 `terrain/climb_76_high_speed/box.obj`。
用于统一播放器时，按 `np.concatenate((qpos[:, 4:7], qpos[:, :4], qpos[:, 7:]), axis=1)` 转为 `[xyz, wxyz, joint29]`，保留原 `fps`；已完成的可视化副本放在 `g1/retargeted/`。

## CMU Mocap 候选子集

来源：[CMU 官方目录](https://mocap.cs.cmu.edu/subjects.php)。已下载 `127_21–26`、`128_09–11`、`82_02–04`，共 12 条、7,470 帧（120 Hz），包括跨越、飞扑接滚翻、低姿滚过与下台落地；每个 subject 保留对应 ASF 骨架。

文件为 ASF＋AMC 骨架动捕，不是 Holosoma 的 `mocap`、`lafan` 或 SMPL 输入；已转换为独立 `cmu` 输入，12/12 条完成无障碍 T2 重定向，共 1,871 帧（30 Hz）；未构库或 match。ASF 长度采用官方换算 `0.0254/0.45`，Y-up→Z-up 保持手性并保留源根朝向。单条下载示例：

```bash
mkdir -p /path/to/cmu/127
curl -fL https://mocap.cs.cmu.edu/subjects/127/127.asf -o /path/to/cmu/127/127.asf
curl -fL https://mocap.cs.cmu.edu/subjects/127/127_23.amc -o /path/to/cmu/127/127_23.amc
```

## PRISM 候选子集

来源：[PRISM 官方仓库](https://github.com/RyosukeHori/PRISM/tree/ef9dd31c759b94b8cb7598c6a21a255256e7c817)。通过仓库中的申请表获取下载地址，仅用于非商业学术研究。

已选 `subj001/take014、take021、take022` 和 `subj006/take015、take021、take023`：前跳、上下箱、台阶各 2 条，共 65,063 帧（100 Hz），约 246.2 MB。从 ZIP 提取对应 `PRISM/<subject>/<take>.pkl`，保留完整 take；上下箱和台阶文件内含配对物体网格。

格式为 SMPL 参数 PKL，含压力/接触数据；不是 SMPL-X，也不能直接使用 Holosoma `mocap` 或 `lafan`。已完成 30 Hz 姿态参数采样（19,520 帧）及配对网格提取。世界坐标 Z-up，平移为 `trans + root_offset`；完整人体关节恢复需已授权的男性 SMPL 模型，目前尚未重定向。
