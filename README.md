# ���ж��ַ��� Agent��SFT / RL ��ɿ���ʵ��

�Կͻ�����������Ա����������Ϊ�����о� **����ȱʧ�����߳�ʱ�͹�̱仯ʱ�Ļָ�����**��
�ṩ��ִ�л�����ѵ�����ݡ���ʵCPUѧϰʵ�顢����ģ��ѵ����ں����� walkthrough��

> **�о�ԭ�ͣ������ڱ������йٷ���Ŀ������ʵ�����˻����ڲ��ƶȻ������ӿڡ�**
> ԭ����ƽ̨Դ��δ���뱾�ֿ⡣���ֿ��Ǻ�ѵ��ʵ��ƽ̨�������������������Ǩ�ơ�

## ��ǰ֤��״̬

| �㼶 | ��ǰ��ʵ | ��˵��ʲô |
|---|---|---|
| �ϳɻ��� + �ű����� | ���ڱ���CPUʵ������400���������� | ������Լ��״̬��֤���ָ����ƿ�ִ�� |
| 1,416����NumPyС���� | ��ʵ�����3����SFT�������REINFORCE | �ṹ��״̬�»ָ��켣�����ݼ�ֵ����LLM |
| 135Mģ��CPU SFT smoke | GitHub Actionsִ�У�״̬�Ը���ҵ��verification.jsonΪ׼ | ����֤�������¡����桢�ָ�����֤����������Ч�� |
| Transformers����GRPO | �ο�ʵ���뵥Ԫ���� | GPU/ҵ������ʵ����δִ�� |
| Slime������ٷ������ | �̶�����Դ�롢������� | δʵ��Ĳ��ֲ�����Ϊѵ���ɹ� |

[�ƶ���������־](https://github.com/tfyf103/agent-SFT-RL/actions)
�� [ʵ��Э��](docs/EXPERIMENTS.md)
�� [����CPUʵ�鿨](results/toy_policy/EXPERIMENT_CARD.md)
�� [�����𲽽���](docs/INTERVIEW_WALKTHROUGH.md)

## �Ѳ�������Ҫ����ģ������

������**35���ɹ۲�������8����ɢ���߶�����1,416����MLP**�������������ֵ�Ƚ���ȷ����
��������ɡ������߱���Ȼ�������⣬�ɼ�������LLM��ٷ���-bench���֡�

| CPUС���� | Test��Ȩ����ɹ��� | Stress��Ȩ����ɹ��� |
|---|---:|---:|
| �����켣 SFT | 55.89% �� 5.63���ٷֵ� | 0.64% �� 1.10���ٷֵ� |
| �ָ��켣 SFT | 100.00% | 100.00% |
| �ָ� SFT + ����� REINFORCE | 100.00% | 100.00% |

������17/29/43����Ϊ���Ӽ�������׼�Test��164����Ȩ����Stress��157����
δ��Ȩ���񵥶����档����С���ѱ��ͣ�**û�й۲쵽RL����greedy�ɹ��ʵ�֤��**��
���������־�����á�checkpoint�;��ޣ����ܰ�100%дΪ��ʵ���пɿ��ʡ�

## 5��������

Python 3.10+������GPU��ģ�����ػ�API��Կ��
```bash
python -m pip install -e ".[dev,toy]"
python -m pytest -q
python -m scripts.prepare_data --train 500 --eval 200
python -m scripts.demo --role customer --mode recovery
python -m scripts.demo --role employee --mode recovery
python -m scripts.evaluate --tasks data/generated/test.jsonl --output artifacts/scripted/test
python scripts/run_toy_experiment.py --output artifacts/toy_policy
```

���Ĭ��PowerShell����Ӱ��������ʾ����ʹ��֧��UTF-8���նˣ�JSONL�ļ��̶�UTF-8��

## ����ģ��·��

����ģ�ͷ����ͨ����ͨchat-completions�ӿڽ��룬�������JSON���߶�����
```bash
python -m scripts.evaluate --tasks data/generated/test.jsonl --output artifacts/model-baseline \
  --base-url http://127.0.0.1:8000/v1 --model YOUR_LOCAL_MODEL
```
��ѡ��Կ�Ż������� BANK_AGENT_API_KEY����д��Git��ԭģ��/SFT/RL�ֱ�����ͬһ���÷���
ʹ��ͬһ����ͽ���������⣬����ʧ���Լ����ĸ��

���豸ѵ���ο���ڣ�������׼��������������������
```bash
python -m pip install -r requirements-train.txt
python scripts/train_sft.py --model /models/instruct-model --offline \
  --data data/generated/sft_recovery.jsonl --output outputs/sft --lora --device cuda
python scripts/train_grpo.py --model /models/instruct-model --offline \
  --adapter outputs/sft/epoch-001/model --tasks data/generated/train.jsonl \
  --output outputs/grpo --device cuda --groups 100 --group-size 4
```
Ŀ¼������ƥ��ʵ��checkpoint����Դȡ����ģ�͡������ĺ��Ż�����
referenceΪ�̶�SFT��ʼ���ԣ������ƴ������Ѿ���A800��֤��
[Slime��ٷ������Ĺ̶��汾˵��](docs/UPSTREAM_REPRODUCTION.md)��

## ���뵼��

- bank_agent/env.py�����ݡ����ߡ���̡��ݵȡ����ع��ϡ���̬�ж���
- bank_agent/data.py���Ȼ��ֺ����ɡ�ģ�����ݡ��켣��֤��
- bank_agent/policies.py���ɽ��ͽű�������ʾ�����ɡ�
- bank_agent/toy_policy.py����ʵCPUС����ѧϰ��
- bank_agent/training.py������mask������rollout��GRPO��ʧ��checkpoint��
- bank_agent/evaluate.py��������켣����ͬ��ĸ���������䡣
- integrations/����ѡ���ο�����䣻δ���в��ּ�����˵����
- docs/���ܹ���ʵ�顢���֡����Ժͼ�����

## �����Դ

��ִ�н�������� results/��ԭʼ������켣�ɴ���ѹ��֤�ݰ���Actions artifacts�鿴��
����ѵ�������ù̶��ű��ؽ���manifestУ��SHA-256��
�������γɼ������ⲿ�ο���δ�����κα���Ŀʵ�����

[�ܹ�](docs/ARCHITECTURE.md) �� [�������](docs/RESUME.md) �� [��Դ������](NOTICE.md)
