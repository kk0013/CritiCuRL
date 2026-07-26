# O²-CritiCuRL: Offline-Online Curriculum RL for Multimodal Reasoning

Official implementation of **O²-CritiCuRL**, a curriculum reinforcement learning framework that introduces critical-step awareness through an iterative **offline–online** paradigm: the offline stage performs multi-rollout analysis over step-annotated trajectories to identify critical reasoning steps, and the online stage applies progressive step-level GRPO training on the distilled data.

📄 **Paper**: [Offline-Online Curriculum RL for Multimodal Reasoning](https://)

This codebase is built on [verl](https://github.com/volcengine/verl). 

## Quick start

### 1. Installation

Follow verl's [installation guide](https://verl.readthedocs.io/en/latest/start/install.html) with the vLLM rollout backend, then install this repo:

```bash
conda create -n criticurl python=3.10 -y && conda activate criticurl
git clone https://github.com/kk0013/CritiCuRL.git && cd CritiCuRL
pip install -e .
pip install -r requirements.txt
```

Experiments in the paper are run on 8× A800 GPUs with Qwen2.5-VL as the backbone.

### 2. Data preparation

We use the [Mulberry](https://github.com/HITsz-TMG/Mulberry) dataset: a randomly sampled subset for the SFT cold start, and the remaining data for the offline–online RL stages (organized into difficulty stages for curriculum learning; multiple-choice questions are converted to open-ended QA).

For the RL stages, each question is expanded into multiple **step-truncated variants** stored as rows of a parquet file following verl's `RLHFDataset` schema (`prompt`, `images`, `data_source`, `reward_model.ground_truth`, ...), where `extra_info` must additionally contain:

| Field | Meaning |
|---|---|
| `sample_id` | ID of the original question (shared by all its variants) |
| `judging_step` | `0` = baseline variant; `k` = reference reasoning truncated before step *k* |
| `step_num` | Total number of reference reasoning steps |

### 3. SFT cold start

Initialize the policy with verl's SFT trainer on the Mulberry SFT subset (paper setting: 3 epochs, batch size 32, learning rate 2e-6).

### 4. Configure the step-level reward

The online-stage reward (`verl/utils/reward_score/mulberry_with_steps.py`) extracts reference keywords with an external judge model (Qwen3-Max in the paper) through an OpenAI-compatible API. Fill in the credentials at the top of the file:

```python
QWEN_API_KEY  = "your-api-key"
QWEN_BASE_URL = "your-base-url"
QWEN_MODEL    = "qwen3-max"
```

Also set `working_dir` in `verl/trainer/runtime_env.yaml` to your local repo path when running multi-node Ray.

### 5. Offline–online RL training

```bash
python -m verl.trainer.main_grpo_all \
    data.train_files=/path/to/train.parquet \
    data.val_files=/path/to/test.parquet \
    data.rollout_batch_size=64 \
    data.train_batch_size=1024 \
    data.key_json_path=/path/to/key.json \
    actor_rollout_ref.model.path=/path/to/qwen2.5-vl-sft-ckpt \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.rollout_n=8 \
    actor_rollout_ref.rollout.update_n=8 \
    custom_reward_function.path=verl/utils/reward_score/mulberry.py \
    custom_reward_function.name=compute_score \
    trainer.top_k_key_steps=2 \
    trainer.rollout_data_dir=/path/to/rollout_dump \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
```

Each epoch first runs the **offline rollout** (multi-rollout scoring of all step-truncated variants, critical-step selection into `key.json`, diagnostics in `<rollout_data_dir>/diff_metrics.jsonl`), then runs the **online update** (criticality-ordered curriculum GRPO on the selected variants). Repeat the training over the difficulty stages of the curriculum, advancing to the next stage once the reward stabilizes.

Key method-specific options in `verl/trainer/config/grpo_trainer.yaml`:

| Option | Meaning |
|---|---|
| `data.rollout_batch_size` | Batch size of the offline rollout stage (packed by whole `sample_id` groups) |
| `data.train_batch_size` | Batch size of the online update stage |
| `data.key_json_path` | Where the critical-step `key.json` is written/read |
| `actor_rollout_ref.rollout.rollout_n` | Samples per variant in the offline stage (score-distribution estimation) |
| `actor_rollout_ref.rollout.update_n` | GRPO group size in the online stage |
| `trainer.top_k_key_steps` | Number of critical steps kept per question |
| `trainer.rollout_data_dir` | Required; stores offline rollout diagnostics |
| `custom_reward_function.path` | Offline-stage reward; the online stage automatically loads the sibling `*_with_steps.py` |

### 6. Evaluation

We evaluate on MathVista, MathVision, MathVerse, ChartQA, LogicVista, ScienceQA, MMMU, MMStar and MME<sub>_sum</sub> (see the paper for details).

## Citation

If you find this work useful, please cite:

```bibtex
@article{deng2026criticurl,
  title  = {Offline-Online Curriculum RL for Multimodal Reasoning},
  author = {Deng, Wendi and Du, Hang and Nan, Guoshun and Tian, Haokun and Yu, Jiaqi and Cao, Xinlei and Li, Jiale and Chen, Jingfeng and Deng, Ling and Li, Ting and Yang, Hao and Liu, Jun and Jiang, Xudong and Leng, Sicong},
  year   = {2026},
  url    = {}
}
```

## Acknowledgement

This codebase is built on [verl](https://github.com/volcengine/verl) (Apache-2.0). We thank the verl team for the excellent RL infrastructure. We also thank the authors of [Mulberry](https://github.com/HITsz-TMG/Mulberry) for the dataset and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) for the backbone model.
