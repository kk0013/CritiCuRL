# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import math
import json
import os
import uuid
import random
from collections import defaultdict
from copy import deepcopy
import copy
from tensordict import TensorDict
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type, List

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem, collate_fn
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.dataset.sampler import AbstractCurriculumSampler
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager
# from verl.trainer.ppo.filter_key_steps import filter_and_segment
from verl.trainer.ppo.packFullGroupsBatchSampler import PackFullGroupsBatchSampler

WorkerType = Type[Worker]


# class Role(Enum):
#     """
#     To create more roles dynamically, you can subclass Role and add new members
#     """

#     Actor = 0
#     Rollout = 1
#     ActorRollout = 2
#     Critic = 3
#     RefPolicy = 4
#     RewardModel = 5
#     ActorRolloutRef = 6


# @dataclass
# class ResourcePoolManager:
#     """
#     Define a resource pool specification. Resource pool will be initialized first.
#     """

#     resource_pool_spec: dict[str, list[int]]
#     mapping: dict[Role, str]
#     resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

#     def create_resource_pool(self):
#         for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
#             # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
#             # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
#             # For Megatron backend, we recommend using max_colocate_count>1
#             # that can utilize different WorkerGroup for differnt models
#             resource_pool = RayResourcePool(
#                 process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
#             )
#             self.resource_pool_dict[resource_pool_name] = resource_pool

#         self._check_resource_available()

#     def get_resource_pool(self, role: Role) -> RayResourcePool:
#         """Get the resource pool of the worker_cls"""
#         return self.resource_pool_dict[self.mapping[role]]

#     def get_n_gpus(self) -> int:
#         """Get the number of gpus in this cluster."""
#         return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

#     def _check_resource_available(self):
#         """Check if the resource pool can be satisfied in this ray cluster."""
#         node_available_resources = ray.state.available_resources_per_node()
#         node_available_gpus = {
#             node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
#             for node, node_info in node_available_resources.items()
#         }

#         # check total required gpus can be satisfied
#         total_available_gpus = sum(node_available_gpus.values())
#         total_required_gpus = sum(
#             [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
#         )
#         if total_available_gpus < total_required_gpus:
#             raise ValueError(
#                 f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
#             )

#         # check each resource pool can be satisfied, O(#resource_pools * #nodes)
#         for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
#             num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
#             for node, available_gpus in node_available_gpus.items():
#                 if available_gpus >= num_gpus:
#                     node_available_gpus[node] -= num_gpus
#                     num_nodes -= 1
#                     if num_nodes == 0:
#                         break
#             if num_nodes > 0:
#                 raise ValueError(
#                     f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
#                     + "cannot be satisfied in this ray cluster"
#                 )

def deep_copy_dataproto(dp: DataProto, *, make_contiguous: bool = False, make_consolidated: bool = False) -> DataProto:
    """
    彻底拷贝一个 DataProto：
    - TensorDict 做深拷贝（新存储，不共享叶子 tensor）
    - non_tensor_batch 做深拷贝（数值 ndarray 用 .copy()；object 数组用 deepcopy）
    - meta_info 用 deepcopy
    """
    # 1) 深拷贝 TensorDict
    td_copy = None
    if dp.batch is not None:
        # tensordict.clone() 会克隆叶子张量，得到独立存储
        td_copy = dp.batch.clone()
        if make_contiguous:
            td_copy = td_copy.contiguous()
        if make_consolidated and hasattr(td_copy, "consolidate"):
            td_copy = td_copy.consolidate()

    # 2) 深拷贝 non_tensor_batch
    non_tensor_copy = {}
    for k, v in dp.non_tensor_batch.items():
        if isinstance(v, np.ndarray):
            if v.dtype == object:
                # object 类型必须用 deepcopy，否则元素对象仍共享
                non_tensor_copy[k] = copy.deepcopy(v)
            else:
                non_tensor_copy[k] = v.copy()
        else:
            # 理论上 DataProto 里应全是 ndarray，这里兜底
            non_tensor_copy[k] = copy.deepcopy(v)

    # 3) 深拷贝 meta_info
    meta_copy = copy.deepcopy(dp.meta_info)

    # 4) 组装新的 DataProto
    return DataProto(batch=td_copy, non_tensor_batch=non_tensor_copy, meta_info=meta_copy)


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data

class IndexInjectingDataset(Dataset):
    """包装原始 dataset，在 __getitem__ 时把数据集索引塞进 non_tensor 字段。"""
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # 兼容 DataProtoItem / dict 两种返回形态（多数 RLHF 数据集返回 DataProtoItem）
        if isinstance(item, DataProtoItem):
            # non_tensor_batch 在 collate_fn 里会变成 np.array(dtype=object)
            # 这里放 python int，后续会被收集成数组
            item.non_tensor_batch = dict(item.non_tensor_batch)
            item.non_tensor_batch["dataset_idx"] = idx
            return item
        elif isinstance(item, dict):
            # 如果你的 dataset 返回 dict（键里有 batch/non_tensor_batch 等）
            item = dict(item)
            if "non_tensor_batch" in item and isinstance(item["non_tensor_batch"], dict):
                item["non_tensor_batch"] = dict(item["non_tensor_batch"])
                item["non_tensor_batch"]["dataset_idx"] = idx
            else:
                # 简单场景：直接塞一个键；后续在 from_single_dict 时会进 non-tensor
                item["dataset_idx"] = idx
            return item
        else:
            raise TypeError(f"Unsupported dataset item type: {type(item)}")


class RayPPOTrainer:
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        step_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during offline rollout.
            val_reward_fn: Function for computing rewards during validation.
            step_reward_fn: Function for computing rewards during online update.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        
        # if getattr(self.config.data, "key_json_path", None):
        #     os.environ["KEY_JSON_PATH"] = self.config.data.key_json_path

        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.step_reward_fn = step_reward_fn
        self.key_step_buffer = []
        self.min_update_samples = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        self.max_buffer_size = self.min_update_samples * 4
        # self._current_original_batch: Optional[DataProto] = None
        self._id2dsidx_current: dict[str, int] = {}
        self._collate_fn_for_replay = collate_fn
        self.epoch_replay_idx_buffer: List[int] = []  # 新增：用于整个 epoch 收集筛选出的数据集 id
        self._rollout_counter = 0
        self._update_counter = 0

        self.last_val_metrics = None

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GPG,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self._key_json_path = (
            getattr(self.config.data, "key_json_path", None)
            or getattr(self.config.trainer, "key_json_path", None)
        )
        # 写入环境变量，供 mulberry_with_steps.py 读取
        if self._key_json_path:
            os.environ["KEY_JSON_PATH"] = str(self._key_json_path)
    
    def _set_epoch_for_dataloaders(self, epoch: int):
        """DDP 风格：每个 epoch 开始时驱动 sampler 的随机性（洗牌）。"""
        # 训练集优先使用 batch_sampler（你用的是 PackFullGroupsBatchSampler）
        bs = getattr(self.train_dataloader, "batch_sampler", None)
        if hasattr(bs, "set_epoch"):
            bs.set_epoch(epoch)
        else:
            # 兜底：如果不是 batch_sampler，试试普通 sampler
            s = getattr(self.train_dataloader, "sampler", None)
            if hasattr(s, "set_epoch"):
                s.set_epoch(epoch)

        # 验证集如启用 shuffle，同样设置一下（大多数时候没必要）
        if getattr(self.config.data, "validation_shuffle", False):
            vs = getattr(self.val_dataloader, "sampler", None)
            if hasattr(vs, "set_epoch"):
                vs.set_epoch(epoch)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert (
                config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None
                or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None
            ), (
                "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, "
                "due to no role-playing support"
            )

        # 若启用 in-reward KL, 则必须启用 ref policy
        if config.algorithm.use_kl_in_reward and not self.use_reference_policy:
            raise ValueError("When enabling in-reward KL, reference policy must be enabled.")

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        训练集用PackFullGroupsBatchSampler, 保证一个batch内有若干个完整sample_id组
        (组内不拆分, 贪心装箱到batch_size上限)
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        # 1) 构建数据集；训练集包一层 IndexInjectingDataset 注入 dataset_idx
        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = IndexInjectingDataset(train_dataset), val_dataset

        # 2) collate_fn
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
            collate_fn = default_collate_fn
        # 回放重建要用
        self._collate_fn_for_replay = collate_fn
        

        # if train_sampler is None:
        #     train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        
        # 3) curriculum / num_workers 约束
        num_workers = self.config.data["dataloader_num_workers"]
        if isinstance(train_sampler, AbstractCurriculumSampler):
            assert num_workers == 0, (
                "If using curriculum, num_workers must be 0 to prevent data caching. "
                "If the dataloader caches data before the batch is done the "
                "curriculum sampler won't have the opportunity to reorder it. "
            )

        # 4) 训练集：使用分组批采样器 PackFullGroupsBatchSampler
        train_bsz_for_rollout = int(self.config.data.get("rollout_batch_size", self.config.data.train_batch_size))
        grouped_shuffle = bool(self.config.data.get("grouped_sampler_shuffle", True))
        grouped_seed = int(self.config.data.get("seed", 42))
        allow_oversized_group = bool(self.config.data.get("allow_oversized_group", True))
        grouped_verbose = bool(self.config.data.get("grouped_sampler_verbose", True))

        # 定义好的采样器
        grouped_batch_sampler = PackFullGroupsBatchSampler(
            dataset=self.train_dataset,
            batch_size=train_bsz_for_rollout,
            shuffle=grouped_shuffle,
            seed=grouped_seed,
            allow_oversized_group=allow_oversized_group,
            verbose=grouped_verbose,
            divisor=8,
        )
        print(f"[dataloader] rollout_batch_size = {train_bsz_for_rollout}, train_batch_size = {self.config.data.train_batch_size}")

        # 5) StatefulDataloader (训练)
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_sampler=grouped_batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

        # self.train_dataloader = StatefulDataLoader(
        #     dataset=self.train_dataset,
        #     batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
        #     num_workers=num_workers,
        #     drop_last=True,
        #     collate_fn=collate_fn,
        #     sampler=train_sampler,
        # )

        # 6) 验证集：保持原先逻辑
        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        # 7) 基本检查与训练步数
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        # total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        # if self.config.trainer.total_training_steps is not None:
        #     total_training_steps = self.config.trainer.total_training_steps

        # self.total_training_steps = total_training_steps
        # print(f"Total training steps: {self.total_training_steps}")

        # try:
        #     OmegaConf.set_struct(self.config, True)
        #     with open_dict(self.config):
        #         if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
        #             self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
        #         if OmegaConf.select(self.config, "critic.optim"):
        #             self.config.critic.optim.total_training_steps = total_training_steps
        # except Exception as e:
        #     print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

        # 修改为按照epoch控制训练，不使用total_training_steps
        self.total_training_steps = None
        print(
            f"[Epoch-based training] total_epochs={self.config.trainer.total_epochs}, "
            f"rollout_batches_per_epoch={len(self.train_dataloader)}"
        )

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"update_{self._update_counter:06d}.jsonl")
        self._update_counter += 1

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # 在计算验证 reward 之前，把 key_json_path 注入到 extra_info
            self._inject_key_json_path_into_extra_info(test_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _compress_records_for_diff(self, records):
        """
        将同一 sample 内“judging_step 相同”的多条记录压缩为一条（metrics 取均值），
        仅用于计算 diff，不改变原始 records 的写盘/展示。
        排序规则仍保持：jd==0 最前，其余按 jd 降序，None 最后。
        """
        from collections import defaultdict
        import numpy as np
        def _rank_key(js):
            if js == 0: return (0, 0)
            if js is None: return (2, 0)
            return (1, -js)

        buckets = defaultdict(list)
        for r in records:
            buckets[r.get("judging_step", None)].append(r)

        uniq = []
        for js, lst in sorted(buckets.items(), key=lambda kv: _rank_key(kv[0])):
            # 聚合 metrics：逐 key 取均值（忽略 NaN）
            keys = list((lst[0].get("metrics") or {}).keys())
            agg = {}
            for k in keys:
                vals = []
                for r in lst:
                    v = (r.get("metrics") or {}).get(k, float("nan"))
                    try:
                        v = float(v)
                    except Exception:
                        v = float("nan")
                    if math.isfinite(v):
                        vals.append(v)
                agg[k] = float(np.mean(vals)) if vals else float("nan")

            # 复制第一条作为模板，仅替换 metrics（保留 input/uid/rollout_generations 等原字段以便调试）
            merged = dict(lst[0])
            merged["metrics"] = agg
            uniq.append(merged)
        return uniq

    def _select_key_steps_by_diff(self, records, k=2):
        """
        在既有顺序的records上,比较相邻两条记录的std/mean之差;
        若 std差值 + mean差值为同组内前k大, 则把“第二条记录”标记为关键步骤。
        返回：满足条件的记录列表
        """
        key_steps = []
        # eps = 1e-12
        if not records or len(records) < 2:
            return key_steps, []

        def _to_float(x):
            try:
                v = float(x)
                return v if math.isfinite(v) else float("nan")
            except Exception:
                return float("nan")
        
        diffs = []
        for i in range(1, len(records)):
            prev_m = records[i-1].get("metrics", {}) or {}
            curr_m = records[i].get("metrics", {}) or {}
            
            prev_std, curr_std = _to_float(prev_m.get("std")), _to_float(curr_m.get("std"))
            prev_mean, curr_mean = _to_float(prev_m.get("mean")), _to_float(curr_m.get("mean"))
            
            # std_diff = abs(curr_std - prev_std)
            # mean_diff = abs(curr_mean - prev_mean)
            # total_diff = std_diff + mean_diff
            # if prev_mean !=1:
            #     diff_metric = abs(curr_mean-prev_mean) / (1-prev_mean)
            # else:
            #     diff_metric = 0.0

            eps = 1e-6
            tol = 1e-12
            
            prev_mean = float(prev_mean)
            curr_mean = float(curr_mean)
            if curr_mean >= 1-tol:
                curr_mean = prev_mean
                diff_metric = 0
            
            else:
                prev_mean = min(max(prev_mean, eps), 1.0-eps)
                curr_mean = min(max(curr_mean, eps), 1.0-eps)

                if prev_mean <= 0.01:   # 避免极小时波动过大
                    prev_mean = 0.01

                ratio_1 = prev_mean / curr_mean
                ratio_2 = (1-prev_mean) / (1-curr_mean)

                if prev_mean + eps < curr_mean:
                    # 出错，因为我们的假设都是prev_mean一大于curr_mean
                    diff_metric = 0
                    print("[Warning] ===============> 出现 prev_mean<curr_mean 的情况！")
                else:
                    diff_metric = prev_mean * (math.log(ratio_1)) + (1-prev_mean) * (math.log(ratio_2))
            
            js = records[i].get("judging_step", None)
            if js is not None:
                diffs.append((diff_metric, js))
        
        # 取前k大的 total_diff 对应的 judging_step
        topk = sorted(diffs, key=lambda x: x[0], reverse=True)[:k]  # 按 total_diff 排序
        seen = set()
        key_steps_ordered = []
        for _, js in topk:
            try:
                js = int(js)
            except Exception:
                continue
            if js not in seen:
                key_steps_ordered.append(js)
                seen.add(js)
        
        return key_steps_ordered, diffs

    def _desired_judging_steps(self, key_steps, include_baseline=False):
        # 与你 filter_key_steps 中保持一致：k 步前截断 -> 需要 jd = k
        out = []
        for k in key_steps or []:
            try:
                k = int(k)
            except Exception:
                continue
            if k >= 1:
                out.append(k)
        if include_baseline:
            out = [0] + out
        return out

    def _rebuild_dataproto_from_dataset(self, indices: list[int]) -> "DataProto":
        assert hasattr(self, "train_dataset"), "Need self.train_dataset to fetch raw samples by index"
        items = [self.train_dataset[i] for i in indices]

        # 用训练阶段的 collate_fn 重建
        batch_dict = self._collate_fn_for_replay(items) if hasattr(self, "_collate_fn_for_replay") else collate_fn(items)

        # 统一转成 DataProto
        return batch_dict if isinstance(batch_dict, DataProto) else DataProto.from_single_dict(batch_dict)

    def _filter_batch_index_replay(
        self,
        batch: DataProto,
        key_map: dict[str, list[int]],
        include_baseline: bool = False,
        keep_others_without_keys: bool = False,
    ) -> list[int]:
        """
        在处理过的 batch 上选 key-steps，但最终返回“从数据集重取”的原始 DataProto。
        """
        # 1) 基于 extra_info 计算 keep_indices（基本照你原来的逻辑）
         # === 计算 keep_indices：对每个 sample_id 逐一处理 ===
        extra_infos = batch.non_tensor_batch.get("extra_info", None)
        if extra_infos is None:
            raise ValueError("extra_info 缺失，无法依据 sample_id/judging_step 过滤")
        if hasattr(extra_infos, "tolist"):
            extra_infos = extra_infos.tolist()

        # 先按 sample_id 分组，并记录每条的 jd
        from collections import defaultdict
        sid2idxs: dict[str, list[int]] = defaultdict(list)
        jd_by_idx: dict[int, Optional[int]] = {}
        sid_order: list[str] = []   # 记录sample_id首次出现的顺序，用来保持跨样本的稳定顺序

        def _to_int_or_none(x):
            try:
                if x is None:
                    return None
                # 字符串也尝试转 int
                return int(x)
            except Exception:
                return None

        for i, ex in enumerate(extra_infos):
            if not isinstance(ex, dict):
                continue
            sid = str(ex.get("sample_id", None))
            jd  = _to_int_or_none(ex.get("judging_step", None))
            if sid not in sid2idxs:
                sid_order.append(sid)
            sid2idxs[sid].append(i)
            jd_by_idx[i] = jd

        keep_indices: list[int] = []
        for sid in sid_order:
            idxs = sid2idxs[sid]
            if sid in key_map:
                # 该样本命中 key_map：只保留需要的 jd
                wanted = self._desired_judging_steps(key_map[sid], include_baseline=include_baseline)  # list, 保序
                wanted_set = set(wanted)  # 仅用于 O(1) 过滤
                cand = [i for i in idxs if jd_by_idx.get(i) in wanted_set]
                order = {jd: r for r, jd in enumerate(wanted)}  # diff 降序 → 次序 0,1,2,...
                cand.sort(key=lambda i: order.get(jd_by_idx[i], 10**9))  # 按 diff 顺序

                keep_indices.extend(cand)
            else:
                # 不在 key_map：只保留基线 jd==0；否则取最小 jd；再不行取第一条
                # base = [i for i in idxs if jd_by_idx.get(i) == 0]
                # if base:
                #     keep_indices.append(base[0])
                # else:
                #     idx_jd_pairs = [(i, jd_by_idx[i]) for i in idxs if jd_by_idx.get(i) is not None]
                #     if idx_jd_pairs:
                #         i_min = min(idx_jd_pairs, key=lambda p: p[1])[0]
                #         keep_indices.append(i_min)
                #     else:
                #         keep_indices.append(idxs[0])
                if keep_others_without_keys:
                    pass
                continue

        # 保留顺序并去重
        keep_indices = list(dict.fromkeys(keep_indices))

        # # 为了保持与原 batch 顺序一致，按索引升序
        # keep_indices = sorted(set(keep_indices))

        # 2) 用 keep_indices 取得对应的 original_id（注意：original_id 在 rollout 时已 repeat 对齐）
        batch_original_ids = batch.non_tensor_batch.get("original_id", None)
        if batch_original_ids is None:
            raise ValueError("batch 缺失 original_id；确保在 rollout 开头添加 original_id")
        if hasattr(batch_original_ids, "tolist"):
            batch_original_ids = batch_original_ids.tolist()
        selected_original_ids = [batch_original_ids[i] for i in keep_indices]

        # 3) original_id -> dataset_idx（来自 rollout 开头建好的小映射）
        id2ds = getattr(self, "_id2dsidx_current", None)
        assert isinstance(id2ds, dict) and len(id2ds) > 0, "id -> dataset_idx 映射缺失；请在 _rollout_to_filter 开头建立并填充"

        missing = [oid for oid in selected_original_ids if oid not in id2ds]
        assert not missing, f"以下 original_id 不在当前 step 映射中（可能在 repeat/union 前未写入或被清空）：{missing[:5]}"

        selected_ds_indices = [int(id2ds[oid]) for oid in selected_original_ids]
        assert all(0 <= i < len(self.train_dataset) for i in selected_ds_indices), "dataset_idx 越界"

        selected_ds_indices = list(dict.fromkeys(selected_ds_indices))
        return selected_ds_indices

    def _dump_key_map_json(self, key_map: dict, json_path: str) -> str:
        """
        将当前计算得到的 key_map 合并到指定的 json_path 文件中。
        格式：
        {
            "0": {
                "key_steps": [0, 3, 5],
                "key_map": {"1": null, "2": null}
            }
        }
        给每条数据填入key_steps列表。
        """
        if json_path is None:
            parent_dir = os.path.dirname(self.config.data.train_files)
            json_path = os.path.join(parent_dir, "key.json")
        
        # 如果已有则读取旧数据
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                key_json = json.load(f)
        else:
            key_json = {}
        
        # 更新逻辑
        for sid, ks in key_map.items():
            # 强制成 int list，避免下游因 str/int 混用导致再加工
            ordered_int_list = [int(x) for x in ks if x is not None]
            if sid not in key_json:
                key_json[sid] = {"key_steps": [], "key_map": {}}
            key_json[sid]["key_steps"] = ordered_int_list  # 不排序，保持 diff 降序
            
        # 保存
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(key_json, f, ensure_ascii=False, indent=2)
        
        return json_path

    def _call_with_tp_padding(self, dp: DataProto, chunks: int, call_fn):
        """
        将 DataProto pad 到能被 chunks 整除，调用 call_fn(dp_padded) 后再 unpad 结果并返回。
        - dp: 原始 DataProto
        - chunks: 并行分片数(通常等于 TP size)
        - call_fn: 接受 DataProto, 返回 DataProto 的函数
        """
        if chunks is None or chunks <= 1:
            return call_fn(dp)  # 无需 pad

        dp_padded, pad_size = pad_dataproto_to_divisor(dp, chunks)
        out_padded = call_fn(dp_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)
        return out

    def _ensure_list(self, a):
        if a is None:
            return []
        if hasattr(a, "tolist"):
            return a.tolist()
        return list(a)

    # def _accumulate_key_steps(self, filtered_batch: DataProto) -> Optional[DataProto]:
    #     """
    #     用“数据集下标”而不是 DataProto 来积累。
    #     当累计样本数 >= train_batch_size 时，从数据集重建一个 DataProto 返回。
    #     避免 DataProto.concat 引起的 non_tensor 变长维度不一致问题。
    #     """
    #     # 1) 取出这批对应的数据集下标（在 _filter_batch_index_replay 里已经写入）
    #     ds_idx_arr = filtered_batch.non_tensor_batch.get("replay_from_dataset_idx", None)
    #     if ds_idx_arr is None:
    #         # 极端兜底：如果没有 replay 索引（不应该发生），退回到旧的 concat 逻辑（见方案 B）
    #         # 为了不让你卡住，这里直接调用“安全拼接版”
    #         return self._accumulate_key_steps_fallback_concat(filtered_batch)

    #     ds_indices = self._ensure_list(ds_idx_arr)
    #     ds_indices = [int(x) for x in ds_indices]

    #     # 2) 初始化索引缓冲
    #     if not hasattr(self, "_replay_idx_buffer"):
    #         self._replay_idx_buffer: list[int] = []

    #     # 3) 追加到缓冲
    #     self._replay_idx_buffer.extend(ds_indices)

    #     # 4) 是否够一个 update？
    #     bsz = int(self.config.data.train_batch_size)
    #     mini_bsz = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
    #     min_required = max(bsz, mini_bsz)

    #     total = len(self._replay_idx_buffer)
    #     if total < min_required:
    #         print(f"[accumulate] collecting dataset indices: {total}/{min_required}")
    #         return None

    #     # 5) 取前 bsz 个索引，重建 DataProto；其余留下次用
    #     selected = self._replay_idx_buffer[:bsz]
    #     remain   = self._replay_idx_buffer[bsz:]
    #     self._replay_idx_buffer = remain

    #     rebuilt = self._rebuild_dataproto_from_dataset(selected)
    #     # 把“这批的来源索引”带上，方便排查
    #     rebuilt.non_tensor_batch["replay_from_dataset_idx"] = np.asarray(selected, dtype=np.int64)

    #     # === 在这里把 data_source 追加 "_with_steps" 后缀 ===
    #     src = rebuilt.non_tensor_batch.get("data_source", None)
    #     if src is not None:
    #         if isinstance(src, np.ndarray):
    #             src_list = src.tolist()
    #         elif isinstance(src, (list, tuple)):
    #             src_list = list(src)
    #         else:
    #             # 标量（很少见），复制到 batch 大小
    #             src_list = [src] * len(rebuilt)

    #         def add_suffix(s):
    #             s = "" if s is None else str(s)
    #             return s if s.endswith("_with_steps") else (s + "_with_steps")

    #         rebuilt.non_tensor_batch["data_source"] = np.array(
    #             [add_suffix(s) for s in src_list], dtype=object
    #         )

    #     return rebuilt
    

    def _merge_batches(self, batch_list: list[DataProto]) -> DataProto:
        """合并多个DataProto"""
        if len(batch_list) == 1:
            return batch_list[0]
        
        # 合并所有tensor字段
        merged_batch = {}
        merged_non_tensor = {}
        
        for key in batch_list[0].batch.keys():
            values = [batch.batch[key] for batch in batch_list]
            merged_batch[key] = torch.cat(values, dim=0)
        
        for key in batch_list[0].non_tensor_batch.keys():
            values = [batch.non_tensor_batch[key] for batch in batch_list]
            if isinstance(values[0], np.ndarray):
                merged_non_tensor[key] = np.concatenate(values, axis=0)
            elif isinstance(values[0], list):
                merged_non_tensor[key] = [item for sublist in values for item in sublist]
        
        return DataProto(
            batch=merged_batch,
            non_tensor_batch=merged_non_tensor,
            meta_info=batch_list[0].meta_info
        )

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )
        
        # —— 新增：把 rollout 采样器的 divisor 对齐到真实并行需求 —— 
        try:
            req_div = self._get_required_divisor()
            bs = getattr(self.train_dataloader, "batch_sampler", None)
            if hasattr(bs, "set_divisor"):
                bs.set_divisor(req_div)
            elif hasattr(bs, "divisor"):
                bs.divisor = req_div
            # 基本校验：生成阶段 batch_size 需能被 req_div 整除
            gen_bsz = int(self.config.data.get("rollout_batch_size", self.config.data.train_batch_size))
            # assert gen_bsz % req_div == 0, (
            #     f"gen_batch_size ({gen_bsz}) must be divisible by required divisor ({req_div})"
            # )
            if gen_bsz % req_div != 0:
                new_gen = gen_bsz - (gen_bsz % req_div)
                print(f"[warn] gen_batch_size {gen_bsz} not divisible by {req_div}; using {new_gen} instead.")
                if hasattr(bs, "set_batch_size"): bs.set_batch_size(new_gen)
        except Exception as e:
            print(f"[warn] failed to sync sampler divisor: {e}")

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _log_seqlen_unbalance_no_reorder(self, batch: DataProto, metrics: dict, logging_prefix="global_seqlen"):
        """只计算并记录 global_seqlen/* 指标，不改变 batch 顺序。
        按“当前顺序的连续切分”来模拟各 DP rank 分到的样本。"""
        attention_mask = batch.batch["attention_mask"]
        bsz = attention_mask.shape[0]
        seqlen_list = attention_mask.view(bsz, -1).sum(-1).tolist()

        world_size = self.actor_rollout_wg.world_size
        # 用“连续分段”的方式模拟各 rank 的分配（不改变顺序）
        base = bsz // world_size
        extra = bsz % world_size
        start = 0
        partitions = []
        for r in range(world_size):
            size = base + (1 if r < extra else 0)
            partitions.append(list(range(start, start + size)))
            start += size

        stats = log_seqlen_unbalance(
            seqlen_list=seqlen_list,
            partitions=partitions,
            prefix=logging_prefix,
        )
        metrics.update(stats)
    
    def _append_jsonl_lines(self, path:str, records:list[dict]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False)+"\n")

    def _inject_key_json_path_into_extra_info(self, batch: DataProto):
        """把 config 里的 key_json_path 注入到 batch.non_tensor_batch['extra_info'] 的每条记录里。"""
        if not self._key_json_path:
            return
        extra_infos = batch.non_tensor_batch.get("extra_info", None)
        if extra_infos is None:
            return

        # 转成列表以便原地修改
        if hasattr(extra_infos, "tolist"):
            extra_infos = extra_infos.tolist()

        if isinstance(extra_infos, list):
            changed = False
            for ex in extra_infos:
                if isinstance(ex, dict) and "key_json_path" not in ex:
                    ex["key_json_path"] = self._key_json_path
                    changed = True
            if changed:
                import numpy as np
                batch.non_tensor_batch["extra_info"] = np.array(extra_infos, dtype=object)
        elif isinstance(extra_infos, dict):
            if "key_json_path" not in extra_infos:
                extra_infos["key_json_path"] = self._key_json_path
                batch.non_tensor_batch["extra_info"] = extra_infos


    def _get_required_divisor(self) -> int:
        sizes = [getattr(self, "actor_rollout_wg", None) and self.actor_rollout_wg.world_size,
                getattr(self, "critic_wg", None) and self.critic_wg.world_size if self.use_critic else None,
                getattr(self, "ref_policy_wg", None) and self.ref_policy_wg.world_size if (self.use_reference_policy and not self.ref_in_actor) else None,
                getattr(self, "rm_wg", None) and self.rm_wg.world_size if self.use_rm else None]
        sizes = [int(s) for s in sizes if s and s > 1]
        if not sizes:
            return 1
        from math import lcm
        d = sizes[0]
        for s in sizes[1:]:
            d = lcm(d, s)
        return d

    def _rollout_to_filter(self, batch: DataProto, timing_raw: dict) -> list[int]:
        """
        对一个batch内的数据只用actor rollout采样,不更新模型;
        根据采样结果,选取关键步骤(key steps);
        根据关键步骤过滤并重新划分batch,将关键步骤及关键字符串输出在json中;
        返回划分后的batch和key_json_path。
        """
        do_profile = (
            self.config.trainer.profile_steps is not None
            and (self.global_steps in self.config.trainer.profile_steps)
        )
        if do_profile:
            self.actor_rollout_wg.start_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

        batch_size = len(batch)
        # 清理之前的缓存
        self._id2dsidx_current.clear()
        # 为原始数据添加唯一标识并落到 non_tensor_batch 里
        original_ids = [f"orig_{i}_{uuid.uuid4().hex[:8]}" for i in range(batch_size)]
        batch.non_tensor_batch["original_id"] = np.array(original_ids, dtype=object)
        # 立刻抓取dataset_idx，建立 original_id -> dataset_idx 的映射
        ds_idx_arr = batch.non_tensor_batch.get("dataset_idx", None)
        assert ds_idx_arr is not None, "dataset_idx is required; wrap your dataset with IndexInjectingDataset"
        
        if hasattr(ds_idx_arr, "tolist"):
            ds_idx_arr = ds_idx_arr.tolist()
        assert len(original_ids)==len(ds_idx_arr), "len(original_ids) != len(dataset_idx)"
        # 只保存当前 step 这个 batch 的映射
        self._id2dsidx_current.update({oid: int(ds_idx) for oid, ds_idx in zip(original_ids, ds_idx_arr)})

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        if "agent_name" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("agent_name")
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.rollout_n, interleave=True)

        # generate a batch
        with marked_timer("gen", timing_raw, color="red"):
            if not self.async_rollout_mode:
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            else:
                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
            timing_raw.update(gen_batch_output.meta_info["timing"])
            gen_batch_output.meta_info.pop("timing", None)

        # 在repeat时建立ID映射关系
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
        )
        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.rollout_n, interleave=True)
        batch = batch.union(gen_batch_output)

        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)
        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        # TODO: Decouple the DP balancing and mini-batching.
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics={})

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        with marked_timer("reward", timing_raw, color="yellow"):
            # —— 新增：把 key_json_path 写进每条 extra_info —— 
            self._inject_key_json_path_into_extra_info(batch)
            # compute reward model score
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)

            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            else:
                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

            # 写回batch，供后续scores使用
            batch.batch["token_level_scores"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

        key_map = {}
        # Log rollout generations if enabled
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        assert rollout_data_dir is not None, "rollout_data_dir must be set for rollout!"
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            print(batch.batch.keys())
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            
            # 新增：假如uid字段
            uids = [str(u) for u in batch.non_tensor_batch["uid"].tolist()]
            # 有些情况下 reward_extra_infos_dict 可能为 None/空，先兜底成普通 dict
            rextras = dict(reward_extra_infos_dict) if reward_extra_infos_dict else {}

            # 抽取extra_info中的sample_id/judging_step/step_num
            extra_infos = batch.non_tensor_batch.get("extra_info", None)
            if extra_infos is not None and hasattr(extra_infos, "tolist"):
                extra_infos = extra_infos.tolist()
            if extra_infos is None:
                extra_infos = [None] * len(uids)

            # 第一层聚合：按uid聚合多次采样（每个uid有n次采样）
            by_uid = defaultdict(lambda: {
                "input": None,
                "samples": [],
                "sample_id": None,
                "judging_step": None,
                "step_num": None
            })
            
            extra_keys = [k for k in rextras.keys() if k != "uid"]

            for idx, (uid, inp, out, sc, exi) in enumerate(zip(uids, inputs, outputs, scores, extra_infos)):
                if by_uid[uid]["input"] is None:
                    by_uid[uid]["input"] = inp
                
                # 从extra_info取sample_id/judging_step/step_num
                if isinstance(exi, dict):
                    by_uid[uid]["sample_id"] = exi.get("sample_id", by_uid[uid]["sample_id"])
                    by_uid[uid]["judging_step"] = exi.get("judging_step", by_uid[uid]["judging_step"])
                    by_uid[uid]["step_num"] = exi.get("step_num", by_uid[uid]["step_num"])

                sample = {
                    "output": out,
                    "score": sc,
                    "step": self.global_steps,  # 当前 step，一批样本一致
                }
                # 将其它逐样本额外信息并入 sample（长度需与当前批一致）
                for k in extra_keys:
                    try:
                        vlist = rextras[k]
                        sample[k] = vlist[idx]
                    except Exception:
                        # 静默跳过不对齐/无此字段的情况
                        pass

                by_uid[uid]["samples"].append(sample)
            
            # 通过score分布判断步骤是否关键
            metrics_by_uid = {}
            for uid, obj in by_uid.items():
                sc_list = [s["score"] for s in obj["samples"]]
                arr = np.asarray(sc_list, dtype=float)
                # 极差&标准差
                rng = float(np.max(arr)-np.min(arr))
                std = float(np.std(arr, ddof=0))
                # 双端覆盖率
                tau0, tau1 = 0.1, 0.9
                n = max(len(arr), 1)
                p_low = float(np.mean(arr <= tau0))
                p_high = float(np.mean(arr >= tau1))
                # 均值
                mean = float(np.mean(arr))

                metrics_by_uid[uid] = {
                    "range": rng,
                    "std": std,
                    "mean": mean,
                    "p_low": p_low,
                    "p_high": p_high
                }
            
            # 第二层聚合：按sample_id汇总，把同一个问题的不同judging_step数据放到一起
            by_sample = {}
            for uid, rec in by_uid.items():
                sid = rec.get("sample_id", None)
                jstep = rec.get("judging_step", None)
                snum = rec.get("step_num", None)

                # 若没有 sample_id，也可以回退为把 uid 自己作为一个独立样本输出
                key = sid if sid is not None else uid
                if key not in by_sample:
                    by_sample[key] = {
                        "sample_id": sid if sid is not None else uid,
                        "step_num": snum,
                        "records": []
                    }

                by_sample[key]["records"].append({
                    "judging_step": jstep,
                    "uid": uid,  # 保留 uid 以便定位
                    "input": rec["input"],
                    "rollout_generations": rec["samples"],
                    "metrics": metrics_by_uid.get(uid, {}),
                })

            # 对每个sample_id内对数据按judging_step排序，为了按顺序判断关键步骤
            for key, pack in by_sample.items():
                pack["records"].sort(
                    key=lambda r: (
                        0 if r["judging_step"] == 0 else 1, # 标记0最小
                        -r["judging_step"] if r["judging_step"] is not None else float("-inf") # 其余倒序
                    )
                )

            # 选取关键步骤并写入文件
            os.makedirs(rollout_data_dir, exist_ok=True)
            filename = os.path.join(rollout_data_dir, f"rollout_{self._rollout_counter:06d}.jsonl")
            self._rollout_counter += 1

            # with open(filename, "w", encoding="utf-8") as f:
            #     for _, pack in by_sample.items():
            #         # 得到关键步骤编号列表
            #         records_for_diff = self._compress_records_for_diff(pack["records"])
            #         key_steps, diffs = self._select_key_steps_by_diff(
            #             records_for_diff, k=self.config.trainer.get("top_k_key_steps", 2)
            #         )

            #         sid = str(pack["sample_id"])
            #         key_map[sid] = key_steps
                    
            #         # 全部记录也保留（已按你之前的规则排好序：0 最前，其余倒序）
            #         out_obj = {
            #             "sample_id": pack["sample_id"],
            #             "step_num": pack["step_num"],
            #             "key_steps": key_steps,          # 关键步骤编号
            #             "records": pack["records"],      # 全部步骤记录
            #             "diffs": [{"diff": d, "judging_step": js} for d, js in diffs]
            #         }
            #         f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            
            diff_jsonl_path = os.path.join(rollout_data_dir, "diff_metrics.jsonl")
            diff_records = []
            for _, pack in by_sample.items():
                records_for_diff = self._compress_records_for_diff(pack["records"])
                key_steps, diffs = self._select_key_steps_by_diff(records_for_diff, k=self.config.trainer.get("top_k_key_steps", 2))
                
                sid = str(pack["sample_id"])
                key_map[sid] = key_steps

                for d, js in diffs:
                    if d is None or not math.isfinite(float(d)):
                        continue
                    diff_records.append({
                        "sample_id": sid,
                        "judging_step": (int(js) if js is not None else None),
                        "diff": float(d),
                        "global_step": int(self.global_steps),
                    })
            if diff_records:
                self._append_jsonl_lines(diff_jsonl_path, diff_records)
                print(f"[rollout] appened {len(diff_records)} diff rows -> {diff_jsonl_path}")

            # 仅保存一个很小的 JSON 备查
            # rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            # assert rollout_data_dir is not None
            # parent_dir = os.path.dirname(self.config.data.train_files)
            # key_json_path = os.path.join(parent_dir, "key.json")
            key_json = self._dump_key_map_json(key_map, self._key_json_path)

        # === 直接在内存中过滤当前 batch，得到“分段后的新 batch”===
        filtered_dsidx = self._filter_batch_index_replay(
            batch=batch,
            key_map=key_map,
            include_baseline=False,
            keep_others_without_keys=False,
        )

        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

        # return filtered_dsidx, key_json
        return filtered_dsidx

    
    def _update_from_new_batch(self, batch: DataProto, timing_raw: dict, epoch: int, logger):
        """
        使用处理后的batch跑一次GRPO更新模型
        """

        metrics = {}
        reward_extra_infos_dict: dict[str, list] = {}   # 提前定义，避免后续报错
        # is_last_step = self.global_steps >= self.total_training_steps

        # 清除之前的reward计算结果
        if "token_level_scores" in batch.batch:
            del batch.batch["token_level_scores"]

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        if "agent_name" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("agent_name")
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.update_n, interleave=True)


        with marked_timer("step", timing_raw):
            # generate a batch
            with marked_timer("gen", timing_raw, color="red"):
                if not self.async_rollout_mode:
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                else:
                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                timing_raw.update(gen_batch_output.meta_info["timing"])
                gen_batch_output.meta_info.pop("timing", None)
            
            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                with marked_timer("gen_max", timing_raw, color="purple"):
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                    batch = batch.union(gen_baseline_output)
                    reward_baseline_tensor = self.step_reward_fn(batch)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    batch.batch["reward_baselines"] = reward_baseline_tensor

                    del gen_baseline_batch, gen_baseline_output

            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )
            # repeat to align with repeated responses in rollout
            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.update_n, interleave=True)
            batch = batch.union(gen_batch_output)

            if "response_mask" not in batch.batch:
                batch.batch["response_mask"] = compute_response_mask(batch)
            # Balance the number of valid tokens across DP ranks.
            # NOTE: This usually changes the order of data in the `batch`,
            # which won't affect the advantage calculation (since it's based on uid),
            # but might affect the loss calculation (due to the change of mini-batching).
            # TODO: Decouple the DP balancing and mini-batching.
            # if self.config.trainer.balance_batch:
            #     self._log_seqlen_unbalance_no_reorder(batch, metrics=metrics)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with marked_timer("reward", timing_raw, color="yellow"):
                # —— 新增：把 key_json_path 写进每条 extra_info —— 
                self._inject_key_json_path_into_extra_info(batch)
                # compute reward model score
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                else:
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.step_reward_fn)
            
            # # 过滤 keyword为NONE的样本
            # def _filter_dp_rows(dp: DataProto, keep_idx: list[int]) -> DataProto | None:
            #     if len(keep_idx) == 0:
            #         return None
            #     new_batch = {k: v.index_select(0, torch.as_tensor(keep_idx, device=v.device)) for k, v in dp.batch.items()}
            #     new_non_tensor = {}
            #     for k, v in dp.non_tensor_batch.items():
            #         if isinstance(v, np.ndarray):
            #             new_non_tensor[k] = v[keep_idx]
            #         elif isinstance(v, list):
            #             new_non_tensor[k] = [v[i] for i in keep_idx]
            #         else:
            #             new_non_tensor[k] = v
            #     return DataProto(batch=new_batch, non_tensor_batch=new_non_tensor, meta_info=dict(dp.meta_info))
            # if isinstance(reward_extra_infos_dict, dict):
            #     if "keyword_is_none" in reward_extra_infos_dict:
            #         kw_none = reward_extra_infos_dict["keyword_is_none"]
            #     else:
            #         kw_none = [False] * len(batch)
            # if kw_none is not None:
            #     if hasattr(kw_none, "tolist"):
            #         kw_none = kw_none.tolist()
            #     kw_none = [bool(x) for x in kw_none]
            #     assert len(kw_none) == len(batch), f"keyword_is_none 长度({len(kw_none)})与 batch({len(batch)}) 不一致"

            #     keep_idx = [i for i, f in enumerate(kw_none) if not f]
            #     drop_cnt = len(kw_none) - len(keep_idx)
            #     if drop_cnt > 0:
            #         print(f"===== [drop] 去掉 {drop_cnt}/{len(kw_none)} 条关键词为NONE的样本 =====")
            #         new_batch = _filter_dp_rows(batch, keep_idx)
            #         if new_batch is None:
            #             # 整批为空：跳过本次 update（不推进 global_steps）
            #             return
            #         batch = new_batch
            #         # 同步裁剪 reward_tensor & reward_extra_infos_dict
            #         if reward_tensor is not None:
            #             reward_tensor = reward_tensor.index_select(0, torch.as_tensor(keep_idx, device=reward_tensor.device))
            #         if isinstance(reward_extra_infos_dict, dict):
            #             for k, v in list(reward_extra_infos_dict.items()):
            #                 try:
            #                     if isinstance(v, np.ndarray):
            #                         reward_extra_infos_dict[k] = v[keep_idx]
            #                     elif isinstance(v, list):
            #                         reward_extra_infos_dict[k] = [v[i] for i in keep_idx]
            #                 except Exception:
            #                     pass

            # recompute old_log_probs
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

                if "rollout_log_probs" in batch.batch.keys():
                    # TODO: we may want to add diff of probs too.
                    rollout_old_log_probs = batch.batch["rollout_log_probs"]
                    actor_old_log_probs = batch.batch["old_log_probs"]
                    attention_mask = batch.batch["attention_mask"]
                    responses = batch.batch["responses"]
                    response_length = responses.size(1)
                    response_mask = attention_mask[:, -response_length:]

                    rollout_probs = torch.exp(rollout_old_log_probs)
                    actor_probs = torch.exp(actor_old_log_probs)
                    rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                    rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                    rollout_probs_diff_max = torch.max(rollout_probs_diff)
                    rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                    rollout_probs_diff_std = torch.std(rollout_probs_diff)
                    metrics.update(
                        {
                            "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                            "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                            "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                        }
                    )

            if self.use_reference_policy:
                # compute reference log_prob
                with marked_timer("ref", timing_raw, color="olive"):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)
                    

            # critic values
            if self.use_critic:
                with marked_timer("values", timing_raw, color="cyan"):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)
            
            # advantage
            with marked_timer("adv", timing_raw, color="brown"):
                # we combine with rule-based rm
                reward_extra_infos_dict: dict[str, list]
                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                batch.batch["token_level_scores"] = reward_tensor

                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                # compute rewards. apply_kl_penalty if available
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # compute advantages, executed on the driver process

                norm_adv_by_std_in_grpo = self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True
                )  # GRPO adv normalization factor

                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.update_n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )

            # update critic
            if self.use_critic:
                with marked_timer("update_critic", timing_raw, color="pink"):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)
            
            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor
                with marked_timer("update_actor", timing_raw, color="red"):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

            # Log rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                    print(batch.batch.keys())
                    inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                    outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                    scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                    # self._dump_generations(
                    #     inputs=inputs,
                    #     outputs=outputs,
                    #     scores=scores,
                    #     reward_extra_infos_dict=reward_extra_infos_dict,
                    #     dump_path=rollout_data_dir,
                    # )
            
            # validate
            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (self.global_steps % self.config.trainer.test_freq == 0)
            ):
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
                    print("=======> 进行Validate阶段")
                metrics.update(val_metrics)
                self.last_val_metrics = val_metrics
        
        steps_duration = timing_raw["step"]
        self.max_steps_duration = max(self.max_steps_duration, steps_duration)
        
        self.global_steps += 1
        # logging
        # training metrics
        metrics.update(
            {
                "training/global_step": self.global_steps,
                "training/epoch": epoch,
            }
        )
        # collect metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
            self.train_dataloader.sampler.update(batch=batch)
        
        logger.log(data=metrics, step=self.global_steps)

    def fit(self):
        """
        训练总循环
        对于每个batch:
            1) 仅rollout(多次采样、算reward、选关键步骤并过滤) -> 得到new_batch与key_map json
            2) 用new_batch执行一次 GRPO 更新(old/ref logprob、KL、advantage、update)
        然后继续下一个batch,直到达到total_training_steps
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._rollout_log_step = 0
        self._load_checkpoint()
        
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # # add tqdm
        # rollout_progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Rollout Progress")

        self.max_steps_duration = 0
        last_val_metrics = None

        # 训练主循环
        for epoch in range(self.config.trainer.total_epochs):
            print(f"=========== 开始进行第{epoch+1}/{self.config.trainer.total_epochs}个epoch ===========")

            # 每个 epoch 一个 rollout 进度条
            rollout_progress_bar = tqdm(
                total=len(self.train_dataloader),
                desc=f"Rollout Progress (epoch {epoch+1}/{self.config.trainer.total_epochs})"
            )

            # —— 新增：显式控制本 epoch 的洗牌种子 —— 
            self._set_epoch_for_dataloaders(epoch)
            
            self.epoch_replay_idx_buffer = []   # 每个epoch开始时清空重放索引缓存

            for batch_dict in self.train_dataloader:
                # if self.global_steps >= self.total_training_steps:
                #     if self.last_val_metrics is not None:
                #         pprint(f"Final validation metrics: {self.last_val_metrics}")
                #     rollout_progress_bar.close()
                #     return
                
                # 组装原始batch对象
                batch = DataProto.from_single_dict(batch_dict)

                # 打印每条数据的 sample_id 和 batch 索引
                extra_infos = batch.non_tensor_batch.get("extra_info", None)
                if extra_infos is not None and hasattr(extra_infos, "tolist"):
                    extra_infos = extra_infos.tolist()
                if extra_infos is None:
                    extra_infos = [None] * len(batch)
                # print("==> [Rollout] 当前batch内容:")
                # for i, ex in enumerate(extra_infos):
                #     sid = ex.get("sample_id", None) if isinstance(ex, dict) else None
                #     print(f"  batch_idx={i}, sample_id={sid}")

                # 第一阶段：只做rollout，选关键步骤并过滤
                timing_raw_rollout = {}
                # filtered_dsidx, key_json = self._rollout_to_filter(batch, timing_raw_rollout)
                filtered_dsidx = self._rollout_to_filter(batch, timing_raw_rollout)
                
                rollout_progress_bar.update(1)
                
                if filtered_dsidx is not None:
                    self.epoch_replay_idx_buffer.extend([int(x) for x in filtered_dsidx])
                # # 把key_json路径记录到日志里，便于追踪
                # print(f"[rollout] key_map json saced at: {key_json}")
                # logger.log({"rollout/key_json_written": 1.0}, step=self.global_steps)

            # 第二阶段：按 sample_id 分组 + 轮转取样分批 update
            ds_indices = self.epoch_replay_idx_buffer
            bsz = int(self.config.data.get("train_batch_size", 0))
            if bsz <= 0:
                raise ValueError(f"Invalid train_batch_size: {bsz}")

            def _to_int_or_none(x):
                try:
                    return int(x) if x is not None else None
                except (TypeError, ValueError):
                    return None

            def _extract_extra_info(item, dsidx):
                """返回 (sample_id:str, judging_step:int|None)"""
                # 兼容 DataProtoItem / dict
                if hasattr(item, "non_tensor_batch"):
                    ntb = item.non_tensor_batch
                    if not isinstance(ntb, dict):
                        raise TypeError(f"dataset item at idx {dsidx} has non_tensor_batch type {type(ntb)}, expect dict")
                elif isinstance(item, dict):
                    ntb = item.get("non_tensor_batch", item)
                    if not isinstance(ntb, dict):
                        raise TypeError(f"dataset item at idx {dsidx} (dict) has invalid non_tensor container: {type(ntb)}")
                else:
                    raise TypeError(f"dataset item at idx {dsidx} type {type(item)} not supported")

                if "extra_info" not in ntb:
                    raise KeyError(f"dataset item at idx {dsidx} has no 'extra_info' field")

                extra_info = ntb["extra_info"]
                if hasattr(extra_info, "tolist"):
                    extra_info = extra_info.tolist()

                if isinstance(extra_info, dict):
                    sid = extra_info.get("sample_id")
                    js  = _to_int_or_none(extra_info.get("judging_step"))
                elif isinstance(extra_info, list):
                    if not extra_info:
                        raise ValueError(f"dataset item at idx {dsidx} has empty extra_info list")
                    head = extra_info[0]
                    if not isinstance(head, dict):
                        raise TypeError(f"dataset item at idx {dsidx} extra_info[0] type {type(head)}, expect dict")
                    sid = head.get("sample_id")
                    js  = _to_int_or_none(head.get("judging_step"))
                else:
                    raise TypeError(f"dataset item at idx {dsidx} extra_info type {type(extra_info)} not supported")

                if sid is None:
                    raise ValueError(f"dataset item at idx {dsidx} has no 'sample_id' in extra_info")
                return str(sid), js

            # 1) 先扫描所有样本，构建 {sample_id: [(js, dsidx), ...]}
            sid2items: dict[str, list[tuple[Optional[int], int]]] = defaultdict(list)
            for dsidx in ds_indices:
                item = self.train_dataset[dsidx]
                sid, js = _extract_extra_info(item, dsidx)
                sid2items[sid].append((js, dsidx))

            # 2) 组内排序：js==0 最前；其余按 js 降序；js 为 None 的放最后（保持稳定）
            # —— 新增：读取 key.json 的 diff 降序步骤序列 —— 
            key_json = {}
            if self._key_json_path and os.path.exists(self._key_json_path):
                with open(self._key_json_path, "r", encoding="utf-8") as f:
                    key_json = json.load(f)

            def _rank_by_diff_order(sid, js):
                ks = None
                if key_json and sid in key_json:
                    ks = key_json[sid].get("key_steps", None)
                if not ks:
                    return (1, 0)  # 无键时丢到后面（也可保持原位）
                # ks 已按 diff 降序，构造 order 映射：第一大→0，第二大→1...
                order = {int(v): r for r, v in enumerate(ks if isinstance(ks, list) else [])}
                # 仅考虑 jd>=1 的关键步骤；若你想允许 jd==0 进入轮转，可在这里插入逻辑
                if js in order:
                    return (0, order[js])  # 先选择关键步骤，按 diff 序
                return (1, 0)  # 非关键步骤在后

            for sid in sid2items:
                sid2items[sid].sort(key=lambda p: _rank_by_diff_order(sid, p[0]))
                # 再去重：同一 judging_step 只保留第一次出现（即保留 diff 序中最靠前的那条）
                seen_jd = set()
                dedup = []
                for js, dsidx in sid2items[sid]:
                    if js in seen_jd:
                        continue
                    seen_jd.add(js)
                    dedup.append((js, dsidx))
                sid2items[sid] = dedup

            # 3) 分“位置层”构造批次：第 pos 层只用各组第 pos 条，严格不跨层
            sids_in_order = list(sid2items.keys())  # 需要固定顺序的话可以排序
            max_len = max(len(lst) for lst in sid2items.values()) if sid2items else 0

            # 统计总批次数方便进度条
            total_batches = 0
            for pos in range(max_len):
                count_at_pos = sum(1 for sid in sids_in_order if pos < len(sid2items[sid]))
                total_batches += (count_at_pos + bsz - 1) // bsz if count_at_pos else 0
            update_progress_bar = tqdm(total=total_batches, desc=f"Update Progress (epoch {epoch+1}/{self.config.trainer.total_epochs})")

            do_profile = (
                self.config.trainer.profile_steps is not None
                and (self.global_steps in self.config.trainer.profile_steps)
            )

            for pos in range(max_len):
                # 收集“这一位置”的所有样本（每组最多 1 个）
                pos_indices = []
                for sid in sids_in_order:
                    lst = sid2items[sid]
                    if pos < len(lst):
                        _, dsidx = lst[pos]
                        pos_indices.append(dsidx)

                divisor = self._get_required_divisor()
                # assert self.config.data.train_batch_size % divisor == 0, \
                #     f"train_batch_size must be a multiple of {divisor}"

                # 把这一位置的样本按 bsz 切成若干个 batch；注意：不从“下一位置”补齐
                start = 0
                while start < len(pos_indices):
                    filtered = pos_indices[start : start + bsz]
                    start += bsz

                    # —— 关键：对齐到 divisor（不超过 bsz，不跨 pos）——
                    if divisor > 1:
                        rem = len(filtered) % divisor
                        if rem != 0:
                            need = min(divisor - rem, max(0, bsz - len(filtered)))
                            if need > 0 and filtered:          # 有元素才好 pad
                                filtered = filtered + [filtered[-1]] * need

                    do_profile = (
                        self.config.trainer.profile_steps is not None
                        and (self.global_steps in self.config.trainer.profile_steps)
                    )
                    if do_profile:
                        self.actor_rollout_wg.start_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.start_profile()
                        if self.use_critic:
                            self.critic_wg.start_profile()
                        if self.use_rm:
                            self.rm_wg.start_profile()

                    update_batch = self._rebuild_dataproto_from_dataset(filtered)

                    # 打印本 step 用到的每个样本的 sample_id 和 judging_step
                    extra_infos = update_batch.non_tensor_batch.get("extra_info", None)
                    if extra_infos is not None and hasattr(extra_infos, "tolist"):
                        extra_infos = extra_infos.tolist()
                    if extra_infos is None:
                        extra_infos = [None] * len(update_batch)
                    # print(f"==> [Update step {self.global_steps}] 当前batch内容:")
                    # for i, ex in enumerate(extra_infos):
                    #     sid = ex.get("sample_id", None) if isinstance(ex, dict) else None
                    #     js = ex.get("judging_step", None) if isinstance(ex, dict) else None
                    #     print(f"  batch_idx={i}, sample_id={sid}, judging_step={js}")

                    timing_raw_update = {}
                    self._update_from_new_batch(
                        batch=update_batch,
                        timing_raw=timing_raw_update,
                        epoch=epoch,
                        logger=logger,
                    )
                    update_progress_bar.update(1)
                        
                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()

                    # is_last_step = self.global_steps >= self.total_training_steps
                    # if is_last_step:
                    #     if self.last_val_metrics is not None:
                    #         pprint(f"Final validation metrics: {self.last_val_metrics}")
                    #     update_progress_bar.close()
                    #     return

                    step_duration = timing_raw_update.get("step", None)
                    if step_duration is not None:
                        self.max_steps_duration = max(self.max_steps_duration, step_duration)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw_update, color="green"):
                            self._save_checkpoint()

            update_progress_bar.close()
            rollout_progress_bar.close()
