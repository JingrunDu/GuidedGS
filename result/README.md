# 实验摘要

以下为既有 C7 实验记录，不表示本次整理重新运行了完整训练：

| 项目 | 记录值 |
|---|---:|
| 训练 / 测试图像 | 5,985 / 665 |
| 训练迭代 | 665,000 |
| PSNR | 23.0046 |
| SSIM | 0.7549 |
| LPIPS | 0.3709 |
| 最终高斯数量 | 1,520,590 |

质量指标在 665 个测试视角上按硬二值人物掩码协议评估，seed=0、mult=0.6。来源为原研究目录中 `experiments/exp_6650/unified_eval_results/paper_final_hardmask_combinations/C7_M2_M7_M9.json`。

本仓库不包含训练模型与完整研究档案。数据、环境从 ModelScope 获取；新的训练和渲染结果写入本地 `outputs/`。
