# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
import numpy as np
from numbers import Number

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


def _to_float_scalar(x):
    """将各种可能的标量类型安全地转换为 Python float。"""
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise TypeError(f"Reward tensor must be scalar, got shape {tuple(x.shape)}")
        return float(x.detach().to(torch.float32).item())
    if isinstance(x, np.ndarray):
        if x.size != 1:
            raise TypeError(f"Reward ndarray must be scalar, got shape {x.shape}")
        return float(x.astype(np.float32).item())
    if isinstance(x, Number):
        return float(x)
    # 字符串数字也尽量兜底一次
    try:
        return float(x)
    except Exception as e:
        raise TypeError(f"Reward must be convertible to a scalar float, got type {type(x)}") from e


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            # attention_mask 里前半段是 prompt，后半段是 response
            attention_mask = data_item.batch["attention_mask"]
            # 可能是 tensor，需要转换为 Python int
            valid_prompt_length = int(attention_mask[:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(attention_mask[prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            # === 兼容多返回形式：标量 / (score, extra...) / dict ===
            res = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score_dict_for_print = None  # 用于打印
            if isinstance(res, dict):
                # 约定：dict 必须包含 "score"
                reward_val = res.get("score")
                if reward_val is None:
                    raise KeyError("compute_score returned dict without 'score' key.")
                reward_val = _to_float_scalar(reward_val)
                score_dict_for_print = res
                # 将 dict 的所有键都累计进 extra info（包括原始 score 备查）
                for key, value in res.items():
                    reward_extra_info[key].append(value)
            elif isinstance(res, tuple):
                # 取第一个作为分数，其余作为 extra
                if len(res) == 0:
                    raise TypeError("compute_score returned empty tuple.")
                reward_val = _to_float_scalar(res[0])
                # 将剩余部分打包存到 "extra" 下，避免结构不一致
                extra_payload = res[1] if len(res) == 2 else res[1:]
                reward_extra_info["extra"].append(extra_payload)
            else:
                # 纯标量或可转标量
                reward_val = _to_float_scalar(res)

            # ===== 写入 reward 到最后一个有效 response 位置 =====
            if valid_response_length <= 0:
                # 防御式兜底：没有有效 response，写到 0 位置
                valid_index = 0
            else:
                valid_index = valid_response_length - 1

            reward_tensor[i, valid_index] = reward_val

            # ===== 受控打印若干样本用于排障 =====
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if score_dict_for_print is not None:
                    for key, value in score_dict_for_print.items():
                        print(f"[{key}]", value)
                else:
                    # 非 dict 情况的简单打印
                    print("[score]", reward_val)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
