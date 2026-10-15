# GuidedGS

面向工业实景的三维高斯建模。默认使用 C7：PSS 渐进监督调度、IABG 重要性感知预算增长、VARP 可见性感知冗余剪枝。

## 1. 准备代码、数据与环境

需要 Linux x86_64（glibc ≥ 2.28）、支持 CUDA 12.1 的 NVIDIA 驱动及 GPU。准备材料需 Python 3.10 及以上，建议预留至少 80 GB 磁盘。仓库为私有，需要 GitHub 访问权限。

```bash
git clone https://github.com/JingrunDu/GuidedGS.git
cd GuidedGS
python3 -m venv envs/download
envs/download/bin/python -m pip install modelscope-hub==0.4.6
envs/download/bin/python tools/prepare_materials.py \
  --dataset subset_6650 --environment industrialgs_py310_cu121_v1 \
  --token-file "$HOME/.modelscope/read_token"
```

令牌放在仓库外的私有文件中；公开访问可省略 `--token-file`。工具完成下载校验、数据解压、环境合并恢复和离线依赖安装，得到 `data/perspective/` 与 `envs/guidedgs/`。环境为 Python 3.10、PyTorch 2.3.0 + CUDA 12.1，复用已有通用环境版本，不重复上传环境。

数据来自 [ModelScope：JingrunDu/converter_value](https://modelscope.cn/datasets/JingrunDu/converter_value)，包含图像、人物掩码、相机和初始点云，无需重新运行 SAM3。其他项目已经恢复过相同材料时，可直接复用数据目录和 Python 环境。

## 2. 检查与训练

以下命令均在仓库根目录执行；复用材料时修改 `PYTHON` 和 `DATA`。

```bash
export PYTHON="$PWD/envs/guidedgs/bin/python"
export DATA="$PWD/data/perspective"
export MODEL="$PWD/outputs/C7"

"$PYTHON" scripts/run.py check --data "$DATA"
"$PYTHON" scripts/run.py train --data "$DATA" --output "$MODEL" --dry-run
"$PYTHON" scripts/run.py train --data "$DATA" --output "$MODEL"
```

默认训练 665,000 步，使用固定的 5,985 张训练图和 665 张测试图，seed=0，人物区域采用硬二值掩码。协议与模块参数集中在 `configs/paper.json`，划分位于 `configs/splits/`。`--experiment C0` 至 `C7` 可选择模块消融；默认 C7 启用全部三个模块。请使用空输出目录，避免覆盖已有成果。

输出包括模型 PLY、`cfg_args`、loss CSV、训练日志和进程耗时。`Training time` 是原训练程序的 GPU 事件累计时间；`runtime.json` 记录包含初始化与保存的训练进程墙钟耗时，两者不是同一指标。

## 3. 消融与参数扫描

组合消融使用 `--experiment C0` 至 `C7`。七类单模块扫描集中在 `configs/sweeps.json`：

| 扫描组 | 参数取值 |
|---|---|
| `pss_sigma` | 0、1、1.5、2、2.5、3、3.5、5 |
| `iabg_growth` | 0.010、0.013、0.015、0.017、0.020、0.030 |
| `iabg_decay` | 0、0.5、1 |
| `varp_budget` | 0、0.25、0.50、0.75 |
| `varp_importance` | 1、2、3 |
| `varp_opacity` | 0.02、0.05、0.10 |
| `varp_visibility` | 1、1.5、2 |

每组只启用对应模块，其余参数沿用正式配置。关闭 PSS 或 VARP 的取值复用 C0，各组默认取值复用 C1/C2/C3；七类扫描共 25 个独立配置。

```bash
"$PYTHON" scripts/sweep.py --group pss_sigma --data "$DATA" \
  --output-root outputs/sweeps --dry-run
```

去掉 `--dry-run` 开始顺序训练；`--group all` 运行全部扫描。同一个输出根目录可复用已成功完成的共享配置，未完成的非空目录会报错。单个配置也可通过 `scripts/run.py train --experiment pss_sigma_1.5` 启动，渲染与评估方式相同。

## 4. 渲染与评估

```bash
"$PYTHON" scripts/run.py render --data "$DATA" --output "$MODEL" --views 6
"$PYTHON" scripts/run.py evaluate --data "$DATA" --output "$MODEL"
```

渲染图位于 `$MODEL/visualization/render/`，对应实拍图位于 `gt/`，视角记录位于 `index.json`。默认均匀选择六个测试视角。评估覆盖全部 665 个测试视角，结果为 `metrics.json` 和 `metrics_per_view.json`，采用相同硬掩码与渲染参数 `mult=0.6`。LPIPS 首次评估需要下载 VGG 和感知距离权重，离线使用须预先准备 PyTorch 权重缓存。

## 内容与验证

`train.py`、核心模块及 CUDA 扩展源码保留；论文草稿、历史队列、外部基线完整副本、模型和日志不进入代码仓库。历史主线实现保留兼容参数，但推荐通过上述入口选择正式 C7 或模块消融。

历史测试摘要见 [result](result/README.md)。数据与环境由 ModelScope 单独维护，不包含在 GitHub 中。

本次发布已验证完整数据与划分、恢复环境的依赖导入、隔离子集的 30 步训练及两个视角渲染；未重新执行完整 665,000 步训练。短程验证缩小了数据、初始点云、分辨率和增密日程，不用于评价最终质量。完整 LPIPS 评估仍需准备预训练权重。

许可证与版权声明见 `LICENSE`、`LICENSE_ORIGINAL.md` 和 [第三方说明](THIRD_PARTY_NOTICES.md)。
