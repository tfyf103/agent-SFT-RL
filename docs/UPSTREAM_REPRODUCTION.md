# 上游复现与 GPU 训练说明

本仓库实现了独立合成银行工单环境与 Slime rollout 适配器。**本次没有运行 GPU SFT/GRPO，没有执行官方 τ-Knowledge 模型评测，也没有验证 A800 训练兼容性。** 本地规则策略结果不是大模型训练成绩。

## 固定上游

| 组件 | 固定版本 | 许可 |
|---|---|---|
| Slime | 8c17b676cb57af1d17ee4402e91e9209af84b60b | Apache-2.0 |
| τ-bench / τ-Knowledge | 5bfa7e37b36656b37dc6d022156be6563c1007f3 | MIT |
| Qwen3-4B-Instruct-2507 | HF revision cdbee75f17c01a7cc42f958dc650907174af0554 | Apache-2.0 |

机器可读版本见 [upstream.lock.json](../upstream.lock.json)。容器 digest、Megatron commit、SGLang 版本尚未锁定，必须在实际服务器记录。代码 SHA 不能替代完整训练环境锁。

已通过 GitHub 插件检查：

- [Slime 自定义 rollout 契约](https://github.com/THUDM/slime/blob/8c17b676cb57af1d17ee4402e91e9209af84b60b/docs/en/get_started/customization.md)。
- [Slime 原生 SFT rollout](https://github.com/THUDM/slime/blob/8c17b676cb57af1d17ee4402e91e9209af84b60b/slime/rollout/sft_rollout.py)。
- [Slime 多轮 token/logprob 示例](https://github.com/THUDM/slime/blob/8c17b676cb57af1d17ee4402e91e9209af84b60b/examples/search-r1/generate_with_search.py)。
- [Slime 原始 τ-bench 示例](https://github.com/THUDM/slime/blob/8c17b676cb57af1d17ee4402e91e9209af84b60b/examples/tau-bench/README.md)。
- [银行环境与任务加载](https://github.com/sierra-research/tau2-bench/blob/5bfa7e37b36656b37dc6d022156be6563c1007f3/src/tau2/domains/banking_knowledge/environment.py)。
- [官方评测语义](https://github.com/sierra-research/tau2-bench/blob/5bfa7e37b36656b37dc6d022156be6563c1007f3/docs/evaluation.md)。
- [模型 revision 与模板来源](https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507)。

本仓库没有复制上游完整模块，适配器和启动脚本依据接口独立编写。上游依赖保留其原有许可证。

## 迁移中已识别的问题

Slime 的 examples/tau-bench 使用 τ¹ 的 tau_bench API，不是现在 tau2 银行域。该示例把训练 split 写死，并包含 pkill 进程清理命令，不能直接拷贝到共享服务器。

当前官方银行环境忽略 task_split_name 参数，没有独立 train/test 划分，且禁止 solo mode。本仓库合成任务用于训练与消融，官方银行任务全部留作外部评测；两者的任务、工具、语言与分数分别报告。

官方 τ 的 actions 通常用于重放生成目标数据库状态，并不强制执行同样的工具顺序；只有 reward_basis 含 ACTION 时才匹配动作。本仓库状态验证器与官方奖励不完全等价。

## 适配器契约

[integrations/slime_bank.py](../integrations/slime_bank.py) 的输入形如：

```json
{"prompt":"task_id","metadata":{"task":{"task_id":"...", "split":"train", "...":"完整任务"}}}
```

- generate(..., evaluation=False) 原地更新 Sample，保留 index、group_index、rollout_id 和原 metadata。
- 训练只接受 train；evaluation=True 才接受 valid/dev/validation/test/stress。
- 模型仅看到 BankEnv 消息，不看到隐藏故障、授权真值和目标状态。
- 请求使用 SGLang input_ids；生成 token IDs/logprobs 直接来自 output_token_logprobs，不重新分词生成文本。
- assistant mask=1，环境观测和补入的结构 token mask=0；后者的 logprob 占位为0。
- response_length 包含响应区的全部 token，包括被屏蔽的观测。
- 仅支持固定 Qwen3-4B-Instruct-2507 非思考 ChatML；不支持 partial rollout 或 score centering。
- 每轮最多256个生成 token，整段响应区最多8192 token；截断输出不执行半截 JSON。
- 奖励为最终状态判定的0/1，合法任务直接拒绝没有额外奖励。
- bank_trace 与 bank_result 放入 Sample.metadata，实际训练仍需导出采样记录检查。

假采样器的 CPU 合同测试不能覆盖真实 SGLang、Megatron、GPU通信、OOM或收敛。

## 离线服务器准备

联网机器下载固定模型和代码，再整体转入独立容器：

```bash
git clone https://github.com/THUDM/slime.git /workspace/slime
git -C /workspace/slime checkout 8c17b676cb57af1d17ee4402e91e9209af84b60b
# In the compatible container, install this exact checkout without replacing dependencies:
python -m pip install -e /workspace/slime --no-deps
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --local-dir /workspace/models/Qwen3-4B-Instruct-2507
```

Slime、Megatron、SGLang、PyTorch/CUDA 必须使用兼容版本。官方推荐配套镜像，但这里没有验证 A800 镜像；latest 不能作为锁。记录镜像 digest、nvidia-smi、pip freeze，各组件 commit，先跑前向、checkpoint转换、一轮训练。

转换初始权重，路径依实际镜像调整：

```bash
cd /workspace/slime
source scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /workspace/models/Qwen3-4B-Instruct-2507 \
  --save /workspace/checkpoints/base_mcore
```

只在自己的隔离容器启动 Ray；不要用其他项目的共享地址：

```bash
ray start --head --num-gpus 4 --dashboard-host=127.0.0.1 --dashboard-port=8265
```

本仓库脚本不会停止 Ray 或其他进程。

## SFT → GRPO

1. 生成数据并验证划分互斥，先对原模型做固定测试。
2. SFT：SFT_DATA 填实际 messages 训练文件；行内 metadata.split 或 metadata.task.split 必须为 train。

```bash
export SLIME_DIR=/workspace/slime
export MEGATRON_DIR=/root/Megatron-LM
export HF_MODEL=/workspace/models/Qwen3-4B-Instruct-2507
export MCORE_INIT=/workspace/checkpoints/base_mcore
export RAY_DASHBOARD_ADDRESS=http://127.0.0.1:8265
export SFT_DATA=/workspace/agent-SFT-RL/data/generated/sft_recovery.jsonl
export OUTPUT_DIR=/workspace/outputs/bank_sft_seed42
bash scripts/run_slime_sft.sh
```

3. 将实际 SFT checkpoint 转回 HF，iter_xxx 必须改为实际产物：

```bash
PYTHONPATH=/root/Megatron-LM python /workspace/slime/tools/convert_torch_dist_to_hf.py \
  --input-dir /actual/sft/checkpoint/iter_xxx \
  --output-dir /workspace/models/bank_sft \
  --origin-hf-dir /workspace/models/Qwen3-4B-Instruct-2507
```

先评测导出模型，检查转换正确；再将这个 HF checkpoint 按前面方法转换为新的 Megatron 初始化目录。

4. GRPO 的 HF_MODEL 与 MCORE_INIT 必须来自同一个 SFT checkpoint，文件名均依本仓库实际数据产物填写：

```bash
export HF_MODEL=/workspace/models/bank_sft
export MCORE_INIT=/workspace/checkpoints/bank_sft_mcore
export TRAIN_DATA=/workspace/agent-SFT-RL/data/generated/rl_train.jsonl
export DEV_DATA=/workspace/agent-SFT-RL/data/generated/rl_valid.jsonl
export OUTPUT_DIR=/workspace/outputs/bank_grpo_smoke42
NUM_ROLLOUT=1 bash scripts/run_slime_grpo.sh
```

单轮检查 token/mask、loss、奖励、KL、权重更新。通过后换新 OUTPUT_DIR 运行20轮或更多；开发集选模型，测试集不参与选 checkpoint。启动参数是起点，没有经过调优或实际显存验证：4提示×4采样=16轨迹/global batch，TP2、四卡colocate、lr=1e-6、KL系数0.001、clip范围[0.8,1.28]。同组全零先检查模型输出、数据和SFT，不删除失败测试任务。没有启用动态采样过滤，以免困难任务全部被跳过。

## 官方外部评测

单独 Python >=3.12,<3.14 环境：

```bash
git clone https://github.com/sierra-research/tau2-bench.git /workspace/tau2-bench
git -C /workspace/tau2-bench checkout 5bfa7e37b36656b37dc6d022156be6563c1007f3
cd /workspace/tau2-bench
uv sync --frozen --extra knowledge
```

回本仓库，准备能输出标准工具调用的模型服务与固定用户模拟器：

```bash
export TAU_DIR=/workspace/tau2-bench
export AGENT_LLM=openai/local-agent
export USER_LLM=openai/local-user
export AGENT_LLM_ARGS='{"api_base":"http://127.0.0.1:8000/v1","temperature":0}'
export USER_LLM_ARGS='{"api_base":"http://127.0.0.1:8001/v1","temperature":0}'
NUM_TASKS=2 TRIALS=1 bash scripts/eval_tau_external.sh
```

认证由环境变量配置，勿提交真实密钥。小样本通过后 unset NUM_TASKS，默认四次试验跑完整任务。bm25 只免除在线embedding和shell sandbox，agent/user模型仍需运行。

官方工具schema与本项目JSON action协议不同。服务需开启相容的function-call parser，先检查工具调用输出；此脚本只是独立外部评测入口，不能承诺合成任务训练会改善外部泛化。结果存到上游 data/simulations；保留所有失败、异常、实际分母与运行配置。

## 成绩归属

官方 GPT-5.2 在 banking_knowledge 的 pass¹=32.22%、pass⁴=18.56%，条件是 alltools、4 trials、GPT-5.2用户模拟器、v1.0.1重评。[公开原始JSON](https://github.com/sierra-research/tau2-bench/blob/5bfa7e37b36656b37dc6d022156be6563c1007f3/web/leaderboard/public/submissions/gpt-5-2_sierra_2026-02-26/submission.json)。

这是引用值，不是本项目成绩。官方<1.0.1和>=1.0.1银行分数不兼容；这里默认bm25也不同于alltools。τ的pass^k指k次都成功的可靠性，不是k次中至少一次成功的pass@k。

模型训练成绩必须来自真实checkpoint、任务级结果和日志。建议比较原模型、正常轨迹SFT、恢复轨迹SFT、SFT+GRPO，做等token预算的数据消融、奖励消融和多种子重复。没有这些实测时，只能写“实现训练与评测管线”，不能写“RL已提升X%”。
