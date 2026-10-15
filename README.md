# GuidedGS

面向工业实景的三维高斯建模，结合 **PSS 渐进监督调度、IABG 重要性感知预算增长、VARP 可见性感知冗余剪枝**。

代码与配置保存在本仓库；数据和 Python 环境从 [ModelScope](https://modelscope.cn/datasets/JingrunDu/converter_value) 下载。

## 文件结构

```text
GuidedGS/
├── train.py                 # 核心训练程序
├── arguments/               # 参数定义
├── scene/                   # 相机、点云与高斯模型
├── gaussian_renderer/       # 渲染接口
├── utils/                   # 损失、评分与辅助工具
├── submodules/              # CUDA 扩展源码
├── lpipsPyTorch/            # 感知质量评估
├── configs/
│   ├── paper.json           # 正式参数与 C0–C7 组合消融
│   ├── sweeps.json          # 七类参数扫描
│   └── splits/              # 固定训练 / 测试名单
├── scripts/
│   ├── run.py               # 检查、训练、渲染、评估入口
│   ├── sweep.py             # 参数扫描入口
│   └── evaluate.py          # 统一评估实现
├── tools/                   # 下载、校验与材料恢复
└── result/                  # 已有实验指标摘要
```

运行生成的 `data/`、`envs/`、`downloads/` 和 `outputs/` 不纳入 Git。

## 复现流程

### 1. 准备代码、数据与环境

需要 **Linux x86_64、glibc ≥ 2.28、支持 CUDA 12.1 的 NVIDIA GPU 驱动**；材料准备使用 Python 3.10+，建议预留至少 80 GB 磁盘。私有仓库需先配置 GitHub 访问权限。

```bash
git clone https://github.com/JingrunDu/GuidedGS.git
cd GuidedGS

python3 -m venv envs/download
envs/download/bin/python -m pip install modelscope-hub==0.4.6
envs/download/bin/python tools/prepare_materials.py \
  --dataset subset_6650 --environment industrialgs_py310_cu121_v1
```

工具自动下载、校验并恢复：

| 材料 | 本地目录 | 内容 |
|---|---|---|
| 数据 | `data/perspective/` | 6,650 张图像、SAM3 掩码、相机、初始点云 |
| 环境 | `envs/guidedgs/` | Python 3.10 / PyTorch 2.3.0 / CUDA 12.1 |

无需重新预处理。若 ModelScope 要求认证，在准备命令末尾添加 `--token-file "$HOME/.modelscope/read_token"`，令牌文件放在仓库外。已有相同材料时，可跳过下载并设置下方对应路径。

### 2. 检查与训练

以下命令均在仓库根目录执行：

```bash
export PYTHON="$PWD/envs/guidedgs/bin/python"
export DATA="$PWD/data/perspective"
export MODEL="$PWD/outputs/C7"

"$PYTHON" scripts/run.py check --data "$DATA"
"$PYTHON" scripts/run.py train --data "$DATA" --output "$MODEL"
```

默认运行完整方法 **C7**：训练 **665,000 步**，固定 **5,985 张训练 / 665 张测试**，seed=0，使用硬二值掩码。参数见 [paper.json](configs/paper.json)。添加 `--dry-run` 可预览训练命令；输出目录须为空。

### 3. 渲染与评估

```bash
# 渲染六个测试视角及其对应实拍图
"$PYTHON" scripts/run.py render --data "$DATA" --output "$MODEL" --views 6

# 在全部测试视角上计算 PSNR、SSIM、LPIPS、L1
"$PYTHON" scripts/run.py evaluate --data "$DATA" --output "$MODEL"
```

LPIPS 首次评估需联网下载 VGG 和感知距离权重；离线使用须预先准备 PyTorch 权重缓存。

| 输出位置（相对 `$MODEL`） | 内容 |
|---|---|
| `point_cloud/iteration_665000/point_cloud.ply` | 训练模型 |
| `visualization/render/`、`visualization/gt/` | 渲染图、对应实拍图 |
| `metrics.json`、`metrics_per_view.json` | 汇总指标、逐视角指标 |
| `train.log`、`protocol.json`、`runtime.json` | 日志、运行配置、进程耗时 |

日志中的 `Training time` 为 GPU 事件累计时间；`runtime.json` 为包含初始化与保存的进程墙钟耗时。

## 消融与参数扫描

### 模块组合

| 配置 | 启用模块 | 配置 | 启用模块 |
|---|---|---|---|
| C0 | 无 | C4 | PSS + IABG |
| C1 | PSS | C5 | PSS + VARP |
| C2 | IABG | C6 | IABG + VARP |
| C3 | VARP | C7 | PSS + IABG + VARP |

```bash
"$PYTHON" scripts/run.py train --data "$DATA" --output outputs/C1 --experiment C1
```

### 单模块参数扫描

[sweeps.json](configs/sweeps.json) 定义七类扫描；每组只启用对应模块，其余参数保持一致。

| 扫描组 | 参数 |
|---|---|
| `pss_sigma` | PSS 初始模糊强度 |
| `iabg_growth`、`iabg_decay` | IABG 增长比例、衰减 |
| `varp_budget`、`varp_importance` | VARP 剪枝比例、重要性阈值 |
| `varp_opacity`、`varp_visibility` | VARP 透明度阈值、可见性倍率 |

```bash
"$PYTHON" scripts/sweep.py --group pss_sigma --data "$DATA" \
  --output-root outputs/sweeps --dry-run
```

去掉 `--dry-run` 开始训练；`--group all` 顺序运行全部扫描，共 **25 个独立配置**。共用配置复用已完成模型；渲染和评估时将 `--output` 指向对应模型目录。

## 结果与验证

已有指标见 [实验结果摘要](result/README.md)。发布版本已完成数据校验、恢复环境下的子集 30 步训练及两视角渲染验证；未重新运行完整 665,000 步训练或完整 LPIPS 评估。

许可证：[LICENSE](LICENSE) · [原始许可证](LICENSE_ORIGINAL.md) · [第三方说明](THIRD_PARTY_NOTICES.md)。
