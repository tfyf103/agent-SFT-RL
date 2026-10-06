# 实验协议与结果解释

## 研究假设

H1：在相同优化预算下，含故障恢复的示范比只含正常路径的示范更有助于异常任务。
H2：对SFT策略继续使用可执行终态奖励优化，可能进一步改善任务完成。
H1在小型结构化策略上得到支持；H2在当前greedy评测中没有增益证据。
不能外推至大语言模型、真实银行或官方τ基准。

## 固定数据

train500、valid200、test200、stress200，种子由data.py固定。
先分split再生成身份、实体和表述，stress保留组合故障。共享业务状态机，不是跨业务泛化。
知识在运行时允许检索；保留新版本是测试条件更新，不是让模型背诵静态规则。
生成结果见 data/generated/manifest.json，仓库保留 results/data_manifest.json。
两者的split SHA-256针对规范化任务JSON，与文件换行符无关；任务哈希不一致时必须
核对源码版本、任务数、中文编码与生成内容，不能用Windows/Linux换行差异解释。
源码字节hash和整份manifest文件hash另受字节格式及运行元信息影响。

CPU小策略使用确定性公开特征，忽略自然语言表述，因此不能把它的test成绩称语言泛化。

## 脚本控制：环境验证，不是算法SOTA

| 控制策略 | Test授权成功率 | Stress授权成功率 |
|---|---:|---:|
| 遇异常直接结束的open_loop | 20.12% | 0.00% |
| 不补充材料 | 60.37% | 65.61% |
| 不查询提交后状态 | 60.98% | 0.00% |
| 完整恢复参考控制器 | 100.00% | 100.00% |

这组对照用于说明恢复步骤的作用和验证环境，不是强LLM基线。
参考控制器编码了有限流程规则，100%主要反映环境可解和实现一致性。
拒绝样本不能抬高授权任务成功率；工具失败、超时和截断均留在分母。
原始逐任务结果与轨迹在证据压缩包和可重复Actions任务中保留。

## CPU学习策略

具体参数、预算、3种子均值/标准差、每组指标和原始日志：
[实验卡](../results/toy_policy/EXPERIMENT_CARD.md)。

两组SFT均400×128=51200次动作采样，但唯一任务/动作数不同。
结论是“增加恢复覆盖的数据配方有效”，不能独立归因于一种新的优化算法；
若研究样本效率，需要进一步做同唯一任务数、相同故障比例等控制。

RL不是GRPO名称的替身：此处实现group-relative REINFORCE与精确离散KL，无PPO clip。
真实LLM参考实现另有old logprob/ratio/clip路径。
RL非零优势组很少，SFT已饱和；所有种子和零增益结果均保留。

## 语言模型正式实验需要补什么

固定同一个模型revision、环境、生成参数、用户模拟器、最大轮数和评测任务：
A 原模型；B 正常数据SFT；C 恢复数据SFT；D C+GRPO。
比较正常任务、单故障、未见组合、新规程；主要看授权成功率，
并列报告误拒、虚假完成、越权尝试、重复写入、token、调用次数和延迟。
至少3个训练种子，训练/验证选checkpoint，最终才测test。
正式主实验尚无GPU执行证据，不生成虚构表格或预设“提升目标”充当成绩。

## 已执行的云端CPU SFT验证

[GitHub Actions运行](https://github.com/tfyf103/agent-SFT-RL/actions/runs/37505944936)的llm-training-smoke作业已成功，
完整Torch环境下**54项测试通过**。机器为GitHub托管Ubuntu CPU runner，不是ChatGPT GPU。
验证数据与指纹见[verification.json](../results/cloud/verification.json)。

| 项目 | 实际记录 |
|---|---|
| 模型 | HuggingFaceTB/SmolLM2-135M-Instruct |
| 模型revision | `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| 训练 | CPU、float32、全参数SFT；8条短单轮JSON动作样本 |
| 更新 | 先2个optimizer steps；重载checkpoint及训练状态，再执行1步 |
| 检查 | 原模型、step 2、step 3重载后的完整state_dict SHA-256均不同；恢复步数断言为3 |
| 软件 | Torch 2.6.0+cpu、Transformers 4.51.3、PEFT 0.15.2 |

累计监督token从36变为54。训练日志中的NLL是训练过程累计值，不能当作独立评测损失，
也不作为银行业务改善的证据。此次未执行语言模型GRPO、银行任务评测或GPU吞吐测试。

checkpoint重载和继续更新已验证；没有额外比较“连续3步”与“2步后恢复1步”的权重完全等价。
Actions证据包保留6个数据/配置/日志文件；未上传模型和optimizer checkpoint。
源码入口为scripts/cpu_llm_smoke.py，未来运行可能解析到新的模型revision，应以各次记录为准。

## 官方外部评测

固定tau commit与模型版本；banking_knowledge全部作为外部测试。
必须报告检索配置、agent/user模型、任务ID、重复次数和版本。
官方任务修改会影响分数，不能拿历史论文数字填写当前代码的结果。
pass^4是同一任务四次均成功，不等于总体成功率的四次方，也不等于pass@4。

## 复现与审核

1. 跑pytest：验证权限/幂等/泄漏与数值梯度。
2. 固定脚本生成数据，核对manifest。
3. 运行脚本基线与三种子小策略，比较逐任务结果而非只看平均分。
4. 从失败trace解释一项负结果。
5. 环境或数据变化后，必须新运行并更新hash，不混用旧结果。
