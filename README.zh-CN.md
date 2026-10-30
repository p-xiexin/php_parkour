# T2 适配说明

本文说明已完成的 T2 适配、模型配置、Holosoma 重定向、构库与合成接口。以下命令从仓库根目录执行，`/path/to/` 替换为自己的路径。

## 当前范围

当前适配覆盖以下离线数据流程：

```text
Holosoma 人体输入 → T2 重定向 → PHP 构库与镜像
                 → motion matching → T2 tracking NPZ
```

- T2 31 关节 MJCF 和配套网格随源码及安装包分发；标准模型导出 48 个身体。
- 保留 T2 的关节顺序、关节轴、静态绑定旋转和真实骨盆/双脚映射。
- Holosoma 求解和 PHP 构库、检索、惯性化沿用上游算法；仅加入必要的模型、格式和接口兼容。

## 数据准备

当前使用 LAFAN1、Holosoma、PHP G1、PARC、NUS 和 HOLD。数据来源、使用规模、下载地址，以及各数据集到 Holosoma 输入格式的转换步骤，见[数据准备文档](DATA_PREPARATION.zh-CN.md)。主流程假定人体数据已准备为 Holosoma 可直接读取的输入，从人体到 T2 的重定向开始。

## 环境与模型

需要 Python 3.10 或更高版本。在独立 Python 环境中，从仓库根目录安装动作匹配包：

```bash
python -m pip install -e ./motion_matching
```

T2 模型位于 `motion_matching/motion_matching/assets/robots/booster_t2/`，
包含当前数据流程使用的拳手版 MJCF、36 个网格及来源版权/Apache 2.0 许可证，随源码、sdist 和 wheel 分发。
默认自动加载包内模型；`--model` 或 `PHP_T2_MJCF` 可用于选择其他兼容模型。

模型沿用本流程已有配置，24 个无名可视化调试球不参与碰撞。
网格通过相对路径加载，安装包不依赖外部模型目录。

适配从 MJCF 读取 31 个关节的顺序、轴和静态绑定旋转；骨盆使用
`waist_yaw_link`，双脚使用左右 `ankle_roll_link`。标准模型导出 48 个身体，
额外的 `left_hand_contact_link`、`right_hand_contact_link` 辅助体不进入 tracking 输出。
匹配骨架由 Simulation、自由根身体和关节身体组成，共 33 个骨骼。

Holosoma 固定为 `4a0bf04cde5930af5ed59da797568d32be972525`，该版本已记录在父仓库的子模块提交引用中。
以下命令按记录的提交检出子模块，再应用 T2 兼容补丁一次：

```bash
git submodule update --init --recursive --checkout
git -C thirdparty/holosoma apply ../holosoma_t2.patch
```

## Holosoma 重定向

使用准备好的人体输入，并按其格式指定 `data_format`。以下以已完成的 `lafan` 流程为例：输入为世界坐标关节位置 NPY，任务为 `robot_only`，人体尺度使用 `1.4 / 1.7`。

```python
from motion_matching.utils import T2_MJCF
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.data_type import DATA_FORMAT_CONSTANTS, MotionDataConfig

robot = RobotConfig(robot_type="t2", robot_urdf_file=T2_MJCF)
motion = MotionDataConfig(data_format="lafan", robot_type="t2")
DATA_FORMAT_CONSTANTS["lafan"]["default_scale_factor"] = 1.4 / 1.7
```

上游重定向入口为 `holosoma_retargeting/examples/robot_retarget.py`。输入目录由 `data_path` 指定，`task_name` 对应序列文件名（不含扩展名）。
重定向后输出包含 `qpos` 和 `fps` 的 NPZ；T2 机器人部分为 `(T,38)`，采用 `[root_xyz, root_quat_wxyz, joint31]` 排列。含动态物体的原生结果会在末尾追加物体位姿，需与机器人部分分开保存。

NUS 使用准备好的 30 Hz 关节位置 NPZ，格式为 `nus`，按输入中的主体身高缩放，并以 `root_quaternion` 的首帧初始化根朝向。无障碍重定向示例：

```bash
python -m holosoma_retargeting.examples.robot_retarget \
    --task-type robot_only --robot t2 --data-format nus \
    --data-path /path/to/nus/positions --task-name 0017_RunningOnBench001 \
    --save-dir /path/to/nus/retargeted \
    --robot-config.robot-type t2 \
    --robot-config.robot-urdf-file "$PWD/motion_matching/motion_matching/assets/robots/booster_t2/mjcf/t2_31dof_fc.xml" \
    --task-config.ground-range -10 10 --retargeter.foot-sticking-tolerance 0.02
```

当前 NUS 重定向仅包含地面，不包含障碍接触约束。CMU 使用相同的 `robot_only` 命令，将格式改为 `cmu`、输入目录改为 `holosoma_cmu`、序列名改为 `127_21` 等；输入为已转换的关节位置 NPZ，保留 ASF/AMC 根朝向。

PARC 使用独立的 `parc` 格式和 `climbing` 任务。每条输入目录包含 `motion.npz` 与配对地形资产；人体和地形按同一比例缩放，保留源坐标及根朝向。

```bash
python -m holosoma_retargeting.examples.robot_retarget \
    --task-type climbing --robot t2 --data-format parc \
    --data-path /path/to/parc/holosoma_parc --task-name beyond_speed_vault_001_aug001_dm \
    --save-dir /path/to/parc/retargeted \
    --robot-config.robot-type t2 \
    --robot-config.robot-urdf-file "$PWD/motion_matching/motion_matching/assets/robots/booster_t2/mjcf/t2_31dof_fc.xml" \
    --retargeter.foot-sticking-tolerance 0.02
```


HOLD 使用相同的 `climbing` 命令，将 `--data-format` 改为 `hold`、输入目录改为准备好的 `holosoma_hold`、`--task-name` 改为 `I1` 等序列名。箱体逐帧位姿参与重定向；原生结果额外包含末尾 7 维箱体位姿。

## 数据格式

| 阶段 | 数据布局或内容 | 帧率 |
|---|---|---|
| Holosoma 重定向 NPZ | `qpos: (T,38)`，`[root_xyz, root_quat_wxyz, joint31]`，另有 `fps` | 读取文件中的 `fps` |
| PHP 内部 qpos | `(T,38)`，`[root_quat_wxyz, root_xyz, joint31]` | 由调用方提供 |
| PHP 数据库 BIN | Simulation 骨骼、机器人骨架、速度、接触状态和原始/镜像范围 | 60 Hz |
| T2 tracking NPZ | 31 个关节与 48 个身体的状态及名称，合成时可附带 `vel_cmd` | 默认合成导出 50 Hz |

`RobotSpec.from_mujoco_qpos()` 与 `to_mujoco_qpos()` 负责两种 qpos 布局互转。
`RobotSpec.encode_qpos_to_bones()` 与 `decode_bones_to_qpos()` 完成关节和匹配骨架互转；解码时移除静态绑定旋转，并按实际关节轴恢复角度。`mirror_qpos()` 根据左右关节与轴方向生成镜像。

## 构库与镜像

在安装了 `motion_matching` 包的环境中执行：

```bash
python -m motion_matching.generate_t2_database \
    --motion-file /path/to/retargeted.npz --layout mujoco \
    --output /path/to/database_t2_locomotion.bin
```

多条源动作可重复传入 `--motion-file`。如果输入已经是 PHP 布局，使用
`--layout php`；省略 `--layout` 时默认采用 PHP 布局。

输出为 BIN 和同名 JSON。每条源动作对应相邻的原始、镜像范围；JSON 记录模型/关节/身体名称、
源文件、源帧率及范围。镜像依据 T2 实际关节轴与左右对应关系完成。
构库的 cubic 重采样、Simulation 骨骼平滑、速度和接触计算沿用上游 PHP 规则。

## 移动合成接口

`MotionMatchingSystem` 通过 `robot_spec`、`registry` 和 `database_dir` 接入 T2。
例如，加载刚构建的移动库：

```python
from motion_matching.robots import get_robot_spec
from motion_matching.run import MotionMatchingSystem

spec = get_robot_spec("t2")
registry = {
    "locomotion": {
        "bin": "database_t2_locomotion.bin",
        "features": "features_t2_locomotion.pkl",
        "search_ranges": None,
    }
}
system = MotionMatchingSystem(
    robot_spec=spec, registry=registry, database_dir="/path/to/bank"
)
```

特征使用 T2 的骨盆和双脚索引；检索、停止、惯性化和 anchor 逻辑沿用官方实现。
保存时向 `save_motion_npz()` 传入同一 `robot_spec`，输出 T2 tracking 格式：
`joint_pos`、`joint_vel`、`body_pos_w`、`body_quat_w`、`body_lin_vel_w`、
`body_ang_vel_w`、`fps`，以及 `joint_names`、`body_names`、`robot_name` 和 `root_body_name`。

`generate_segment()` 可生成移动片段，保存时应将方向指令与动作重采样到相同帧率：

```python
from motion_matching.io import save_motion_npz
from motion_matching.run import generate_segment
from motion_matching.utils.data_utils import resample_vel_cmd

qpos, commands = generate_segment(
    system, speed=0.8, turn_angle_degrees=0.0, duration_s=5.0, vel_cmd_idx=1
)
save_motion_npz(
    qpos, "t2_motion.npz", source_fps=60, target_fps=50,
    vel_cmd=resample_vel_cmd(commands, required_fps=50, original_fps=60),
    robot_spec=spec,
)
```

## 可视化

使用同一入口查看重定向、构库和 motion matching 结果：

```bash
python -m motion_matching.motion_visualizer \
    --motion-file /path/to/retargeted /path/to/databases /path/to/matched_motions
```

`--motion-file` 支持多个 NPZ、BIN 或目录。默认自动识别机器人，可用 `--robot t2` 或 `--robot g1` 筛选。

打开终端显示的浏览器地址，通过 `Robot`、`Stage`、`Seq` 选择机器人、阶段和序列。`Retargeted` 为 Holosoma 重定向 NPZ，`Database` 为 PHP 构库 BIN，`Motion matching` 为合成训练 NPZ。

## 输出与支持范围

重定向 NPZ、BIN 中的原始/镜像范围，以及合成后的 tracking NPZ 是不同阶段的输出。

上游 `motion_matching.run` 命令行、`play_database` 及训练入口仍使用 G1 配置；
T2 移动合成使用上述 Python 接口。

适配不增加额外的时序平滑、足底接触或重心修正；上游算法自带的平滑与约束保持。

## 必要兼容修改

- `motion_matching/motion_matching/robots/`：读取 T2 原生模型、关节轴、静态绑定旋转和身体映射。
- `generate_t2_database.py` 与匹配/导出接口：支持 31 关节骨架、T2 镜像和 tracking 格式。
- `thirdparty/holosoma_t2.patch`：Holosoma 固定为 `4a0bf04cde5930af5ed59da797568d32be972525`；补丁注册 T2 模型及 LAFAN、SMPL-X、NUS、CMU、PARC、HOLD 关节映射；NUS/PARC/HOLD 使用源根朝向初始化，PARC/HOLD 保留动作与配对场景的共同坐标。其余兼容项包括 qpos 关节限位地址、`robot_only` 对象位姿赋值和 MuJoCo Jacobian 参数；优化器与约束保持上游实现。
