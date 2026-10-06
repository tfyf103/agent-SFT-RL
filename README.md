# 银行多轮服务 Agent：SFT / RL 与可靠性实验

以客户工单受理和员工工单处理为任务，研究 **材料缺失、工具超时和规程变化时的恢复决策**。
提供可执行环境、训练数据、真实CPU学习实验、语言模型训练入口和面试 walkthrough。

> **研究原型；不属于北京银行官方项目。无真实银行账户、内部制度或生产接口。**
> 原聊天平台源码未接入本仓库。本仓库是后训练实验平台，不声称完成银行生产迁移。

## 当前证据状态

| 层级 | 当前事实 | 能说明什么 |
|---|---|---|
| 合成环境 + 脚本对照 | 已在本地与GitHub Actions CPU运行400个留出任务 | 工具契约、状态验证、恢复机制可执行 |
| 1,416参数NumPy小策略 | 已实际完成3种子SFT和组相对REINFORCE | 结构化状态下恢复轨迹的数据价值；非LLM |
| 135M模型CPU SFT smoke | 已通过：2步SFT后加载checkpoint，再恢复训练1步 | 仅验证参数更新、保存、恢复，不证明银行任务效果 |
| Transformers多轮GRPO | 参考实现与单元测试 | GPU/业务质量实验尚未执行 |
| Slime适配与官方τ外测 | 固定上游源码、复现入口 | 未实测的部分不得作为训练成果 |

[已核验云端运行与日志](https://github.com/tfyf103/agent-SFT-RL/actions/runs/37505944936)
· [135M验证记录](results/cloud/verification.json)
· [实验协议](docs/EXPERIMENTS.md)
· [完整CPU实验卡](results/toy_policy/EXPERIMENT_CARD.md)
· [面试逐步讲解](docs/INTERVIEW_WALKTHROUGH.md)

该云端作业在安装Torch的环境下 **54项测试全部通过**，并实际训练
HuggingFaceTB/SmolLM2-135M-Instruct。参数变化、checkpoint重载和恢复至第3步均已检查。
这是8条短单轮样本上的训练链路验证；未进行银行任务评测或语言模型GRPO训练。

## 已测结果：不要混淆模型类型

以下是**35个可观测特征、8个离散工具动作的1,416参数MLP**。参数填充与阈值比较由确定性
适配器完成。它不具备自然语言理解，成绩不代表LLM或官方τ-bench表现。

| CPU小策略 | Test授权任务成功率 | Stress授权任务成功率 |
|---|---:|---:|
| 正常轨迹 SFT | 55.89% ± 5.63个百分点 | 0.64% ± 1.10个百分点 |
| 恢复轨迹 SFT | 100.00% | 100.00% |
| 恢复 SFT + 组相对 REINFORCE | 100.00% | 100.00% |

三种子17/29/43；±为种子间样本标准差。Test有164个授权任务、Stress有157个；
未授权任务单独报告。环境小且已饱和，**没有观察到RL增加greedy成功率的证据**。
详见完整日志、配置、checkpoint和局限，不能把100%写为真实银行可靠率。

## 5分钟运行

Python 3.10+，无需GPU、模型下载或API密钥：
```bash
python -m pip install -e ".[dev,toy]"
python -m pytest -q
python -m scripts.prepare_data --train 500 --eval 200
python -m scripts.demo --role customer --mode recovery
python -m scripts.demo --role employee --mode recovery
python -m scripts.evaluate --tasks data/generated/test.jsonl --output artifacts/scripted/test
python scripts/run_toy_experiment.py --output artifacts/toy_policy
```

如果默认PowerShell编码影响中文显示，可使用支持UTF-8的终端；JSONL文件固定UTF-8。

## 语言模型路径

独立模型服务可通过普通chat-completions接口接入，输出单个JSON工具动作：
```bash
python -m scripts.evaluate --tasks data/generated/test.jsonl --output artifacts/model-baseline \
  --base-url http://127.0.0.1:8000/v1 --model YOUR_LOCAL_MODEL
```
可选密钥放环境变量 BANK_AGENT_API_KEY，不写进Git。原模型/SFT/RL分别启动同一配置服务、
使用同一任务和解码参数评测，调用失败仍计入分母。

单设备训练参考入口（需自行准备合适算力及依赖）：
```bash
python -m pip install -r requirements-train.txt
python scripts/train_sft.py --model /models/instruct-model --offline \
  --data data/generated/sft_recovery.jsonl --output outputs/sft --lora --device cuda
python scripts/train_grpo.py --model /models/instruct-model --offline \
  --adapter outputs/sft/epoch-001/model --tasks data/generated/train.jsonl \
  --output outputs/grpo --device cuda --groups 100 --group-size 4
```
目录参数需匹配实际checkpoint；资源取决于模型、上下文和优化器。
reference为固定SFT初始策略；不宣称此命令已经在A800验证。
[Slime与官方τ外测的固定版本说明](docs/UPSTREAM_REPRODUCTION.md)。

## 代码导航

- bank_agent/env.py：身份、工具、规程、幂等、隐藏故障、终态判定。
- bank_agent/data.py：先划分后生成、模拟数据、轨迹验证。
- bank_agent/policies.py：可解释脚本控制与示范生成。
- bank_agent/toy_policy.py：真实CPU小策略学习。
- bank_agent/training.py：助手mask、多轮rollout、GRPO损失和checkpoint。
- bank_agent/evaluate.py：逐任务轨迹、不同分母与置信区间。
- integrations/：可选上游框架适配；未运行部分见复现说明。
- docs/：架构、实验、复现、面试和简历。

## 结果来源

已执行结果保留在 results/，原始大体积轨迹可从其压缩证据包或Actions artifacts查看；
生成训练数据用固定脚本重建，manifest中的任务SHA-256来自规范化JSON，应在各平台一致；
原始文件字节hash与任务内容hash是不同的校验。135M证据包保留数据、配置及验证日志，
模型/optimizer checkpoint仅在该次作业中完成保存重载，未作为下载产物保留。
公开上游成绩仅是外部参考，未填入任何本项目实测表。

[架构](docs/ARCHITECTURE.md) · [简历措辞](docs/RESUME.md) · [来源与许可](NOTICE.md)
