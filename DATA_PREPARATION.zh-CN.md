# 数据准备

本文记录已使用的数据、对应工具仓库，以及下载和源格式转换示例。人体数据准备完成后，应为 Holosoma 可直接读取的输入，并指定对应的 `data_format`；PHP G1 成品数据保持原格式，用于流程对照。以下命令从仓库根目录执行，`/path/to/` 替换为自己的路径。

## 使用的数据

| 数据来源 | 格式 | 使用规模 | 已完成内容 |
|---|---|---|---|
| **LAFAN1** | BVH，Holosoma `lafan` | 4 条走路＋2 条跑步，共 41,783 帧 | T2 重定向、原始/镜像构库、100 组移动数据合成 |
| **Holosoma 攀爬示例** | 53 关节 NPY，`mocap`，含配对障碍 | `mocap_climb_seq_0–4`，共 5 组 | 早期 T2 重定向、构库及 10 条原始/镜像轨迹合成 |
| **PHP 官方 G1 数据** | 机器人 BIN、tracking NPZ 和地形 | 19 个动作库、296 对训练数据 | 数据核验及逐库生成，用于流程对照 |

## LAFAN1

数据及读取代码来自 [Ubisoft LAFAN1 仓库](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/tree/94084601bacdf9cc3764b5c73daaeccae6035fac)，固定版本为 `94084601bacdf9cc3764b5c73daaeccae6035fac`。
数据采用 [CC BY-NC-ND 4.0 许可](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/94084601bacdf9cc3764b5c73daaeccae6035fac/license.txt)。
格式转换使用 [Holosoma](https://github.com/amazon-far/holosoma/tree/4a0bf04cde5930af5ed59da797568d32be972525) 的关节位置提取入口，固定版本为 `4a0bf04cde5930af5ed59da797568d32be972525`。

从[官方 ZIP](https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/94084601bacdf9cc3764b5c73daaeccae6035fac/lafan1/lafan1.zip)下载数据，只解压已使用的六条序列：

```bash
mkdir -p /path/to/lafan1/bvh
curl -L \
  https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/94084601bacdf9cc3764b5c73daaeccae6035fac/lafan1/lafan1.zip \
  -o /path/to/lafan1/lafan1.zip
unzip /path/to/lafan1/lafan1.zip \
  walk1_subject1.bvh walk2_subject3.bvh walk3_subject2.bvh \
  walk4_subject1.bvh run1_subject2.bvh run2_subject1.bvh \
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

## Holosoma 攀爬示例

数据来自固定版本 Holosoma 的 [climb 示例目录](https://github.com/amazon-far/holosoma/tree/4a0bf04cde5930af5ed59da797568d32be972525/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb)，包含五组人体关节位置和配对障碍。
该版本由父仓库的子模块提交引用固定为 `4a0bf04cde5930af5ed59da797568d32be972525`，通过以下命令获取：

```bash
git submodule update --init --recursive --checkout
```

五组示例位于 `thirdparty/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb/`，名称为 `mocap_climb_seq_0` 至 `mocap_climb_seq_4`。
输入已是 Holosoma 可读取的 53 关节位置 NPY，格式为 `mocap`；保留人体数据与配对障碍的原目录结构。

## PHP 官方 G1 数据

数据来自 PHP 官方的 [G1 动作库发布包](https://github.com/amazon-far/php_parkour/releases/tag/motion-matching-assets-v1)和[五组训练数据发布包](https://github.com/amazon-far/php_parkour/releases/tag/training-motion-examples-v1)。
下载工具为 PHP 仓库中的 `scripts/download_assets.py`，下载文件按仓库的 `release-assets.json` 清单校验：

```bash
python scripts/download_assets.py databases
for component in motions-locomotion motions-low-step motions-high-step \
  motions-low-climb-76 motions-high-climb-76; do
  python scripts/download_assets.py "$component"
done
```

这批数据已是机器人 BIN、tracking NPZ 及配对地形，可直接用于 G1 流程。已完成数据核验及逐库生成，作为 T2 数据流程的对照。
