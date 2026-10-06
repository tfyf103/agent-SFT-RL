# 可审查的实验记录

本目录保存实际执行的证据，不包含从上游论文借用的成绩。

- [小策略实验卡](toy_policy/EXPERIMENT_CARD.md)：模型规模、数据、预算、种子、结果与局限。
- [本地完整原始记录](raw_evidence.tar.gz)：82个文件，包含逐任务结果、工具轨迹、训练日志和小策略权重；[SHA-256清单](evidence_manifest.json)。
- [云端运行记录](cloud/run.json)：提交、作业、测试与artifact摘要；[运行链接](https://github.com/tfyf103/agent-SFT-RL/actions/runs/37505944936)。
- [135M SFT核验](cloud/verification.json)：真实模型3步更新与恢复训练，仅验证训练通路。
- [云端三种子复跑汇总](cloud/toy_overall.json)：与本地表格一致的主要指标。

在仓库根目录执行 `tar -xzf results/raw_evidence.tar.gz` 可恢复原始记录的目录结构。先核对压缩包SHA-256，再核对清单中的文件。解压会写入 results/ 下的原始证据路径。

云端Actions artifacts保留30天。长期证据是本目录中的本地原始压缩包、云端验证JSON、日志摘录与可复现源码。135M模型及优化器checkpoint只在云端运行期间验证，未上传；模型hash是重载后的权重状态摘要。

数据清单的split哈希使用排序后的规范JSON计算，与操作系统换行无关。发布时必须保持中文UTF-8完整；测试会核对四个split的固定内容哈希。
