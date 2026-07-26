import json
import random
from typing import List, Iterator, Optional, Dict, Any, Tuple
from torch.utils.data import Sampler

# --------- 抽取工具函数 ----------
_CANDIDATE_SID_KEYS = ("sample_id", "id", "qid", "question_id")
_CANDIDATE_JD_KEYS  = ("judging_step", "jd", "turn_id", "step", "stage")
_CANDIDATE_STEPNUM_KEYS = ("step_num", "round", "turn")

def _maybe_json_load(x):
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return x if isinstance(x, (dict, list)) else None

def _pick_first(d: dict, keys: Tuple[str, ...]):
    for k in keys:
        if k in d:
            return d[k], k
    return None, None

def _extract_meta(item: Any) -> Tuple[str, Optional[int], Optional[int]]:
    """
    返回: (sample_id_str, judging_step_int_or_None, step_num_int_or_None)
    适配以下几种常见结构：
      - item 为 DataProtoItem: 从 item.non_tensor_batch / item.non_tensor_batch['extra_info'] 提取
      - item 为 dict: 从 item['non_tensor_batch'] / item['extra_info'] 提取
    """
    # 统一拿到一个可能的 non_tensor_batch 与 extra_info
    nt = None
    ei = None

    # 1) DataProtoItem 风格
    if hasattr(item, "non_tensor_batch"):
        nt = item.non_tensor_batch or {}
        ei = nt.get("extra_info", None)
        ei = _maybe_json_load(ei)
    # 2) dict 风格
    elif isinstance(item, dict):
        nt = item.get("non_tensor_batch", None)
        if isinstance(nt, dict):
            ei = _maybe_json_load(nt.get("extra_info", None))
        # 顶层 extra_info 也兜底试一下（你的 parquet 就是这个字段）
        if ei is None and "extra_info" in item:
            ei = _maybe_json_load(item["extra_info"])
    else:
        raise TypeError(f"Unsupported dataset item type: {type(item)}")

    sid = jd = stepnum = None
    sid_key = jd_key = stepnum_key = None

    # 优先 extra_info
    if isinstance(ei, dict):
        sid, sid_key = _pick_first(ei, _CANDIDATE_SID_KEYS)
        jd,  jd_key  = _pick_first(ei, _CANDIDATE_JD_KEYS)
        stepnum, stepnum_key = _pick_first(ei, _CANDIDATE_STEPNUM_KEYS)

    # 其次 non_tensor_batch 顶层
    if (sid is None or jd is None or stepnum is None) and isinstance(nt, dict):
        if sid is None:
            sid, sid_key = _pick_first(nt, _CANDIDATE_SID_KEYS)
        if jd is None:
            jd, jd_key = _pick_first(nt, _CANDIDATE_JD_KEYS)
        if stepnum is None:
            stepnum, stepnum_key = _pick_first(nt, _CANDIDATE_STEPNUM_KEYS)

    # 再次：若 item 是 dict，也试试顶层（极少数数据会这么放）
    if isinstance(item, dict) and (sid is None or jd is None or stepnum is None):
        if sid is None:
            sid, sid_key = _pick_first(item, _CANDIDATE_SID_KEYS)
        if jd is None:
            jd, jd_key = _pick_first(item, _CANDIDATE_JD_KEYS)
        if stepnum is None:
            stepnum, stepnum_key = _pick_first(item, _CANDIDATE_STEPNUM_KEYS)

    if sid is None:
        # 打印一个更友好的报错信息，帮助定位
        msg_parts = []
        if isinstance(nt, dict):
            msg_parts.append(f"non_tensor_batch.keys={list(nt.keys())[:8]}")
        if isinstance(ei, dict):
            msg_parts.append(f"extra_info.keys={list(ei.keys())[:8]}")
        raise KeyError(
            "sample_id not found in dataset item. "
            "Tried non_tensor_batch / extra_info / top-level. "
            + " | ".join(msg_parts)
        )

    # 规范化
    sid = str(sid)
    try:
        jd = int(jd) if jd is not None else None
    except Exception:
        jd = None
    try:
        stepnum = int(stepnum) if stepnum is not None else None
    except Exception:
        stepnum = None

    return sid, jd, stepnum

def _extract_sample_id(item: Any) -> str:
    sid, _, _ = _extract_meta(item)
    return sid

def _extract_judging_step(item: Any) -> Optional[int]:
    _, jd, _ = _extract_meta(item)
    return jd


class PackFullGroupsBatchSampler(Sampler[List[int]]):
    """
    按 sample_id 分组，打包多个完整组到一个 batch，且不拆分组。
    - 每个 batch 的大小 <= batch_size（若 allow_oversized_group=True 且遇到超大组，则该批可 > batch_size）
    - 组的顺序按 epoch 与 seed 打乱
    - 组内默认按 judging_step 升序（可通过 sort_group_by_jd 控制）
    - 支持 set_epoch / state_dict / load_state_dict，便于断点续训
    """
    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        allow_oversized_group: bool = True,   # 若某组本身 > batch_size，是否允许“单组超大批”
        sort_group_by_jd: bool = True,        # <—— 新增：按 judging_step 升序排组内索引
        # === 新增：让每个产出 batch 的长度对齐到 divisor（如 8）===
        divisor: Optional[int] = None,
        verbose: bool = True,
        # 迭代耗尽后是否自动进入下一 epoch 并重置顺序
        auto_reset_on_iter_end: bool = True,
    ):
        assert batch_size is not None and batch_size > 0
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.allow_oversized_group = bool(allow_oversized_group)
        self.sort_group_by_jd = bool(sort_group_by_jd)
        self.verbose = verbose
        self.divisor = int(divisor) if (divisor is not None) else None
        self.auto_reset_on_iter_end = bool(auto_reset_on_iter_end)
        # 若指定 divisor，要求 batch_size 也是该倍数，避免补齐时超过 batch_size
        if self.divisor is not None:
            assert self.divisor > 0, "divisor must be positive"
            assert (self.batch_size % self.divisor) == 0, \
                f"batch_size={self.batch_size} must be a multiple of divisor={self.divisor}"

        # 状态
        self._epoch = 0
        self._group_cursor = 0      # 下一组在 _order 里的下标
        self._pending: List[int] = []  # 尚未产出的“已累计但未发出”的索引

        # 构建分组与初始顺序
        self._groups: List[List[int]] = []   # 每个元素是一组索引（同一 sample_id）
        self._build_groups()
        self._reset_order()

    # 可选：手动复位（保持当前 epoch，不洗牌；如要洗牌请用 set_epoch）
    def reset(self):
        self._group_cursor = 0
        self._pending = []

    # === 新增：对齐函数，在 yield 前把长度补齐到 divisor 的倍数 ===
    def _pad_to_divisor(self, idxs: List[int], *, limit: Optional[int]) -> List[int]:
        """
        idxs: 当前准备产出的索引列表
        limit: 最大允许长度；None 表示不限制（用于单组超大批的场景）
        """
        if self.divisor is None or not idxs:
            return idxs
        rem = len(idxs) % self.divisor
        if rem == 0:
            return idxs
        need = self.divisor - rem
        if limit is not None:
            # 确保不超过上限（常规批用 batch_size 做上限）
            need = min(need, max(0, limit - len(idxs)))
        if need <= 0:
            return idxs
        pad_val = idxs[-1]
        return idxs + [pad_val] * need

    def _build_groups(self):
        from collections import defaultdict
        bucket: Dict[str, List[Tuple[int, Optional[int]]]] = defaultdict(list)
        n = len(self.dataset)

        miss_cnt = 0
        for idx in range(n):
            item = self.dataset[idx]
            try:
                sid = _extract_sample_id(item)
                jd  = _extract_judging_step(item)  # 可能为 None
            except KeyError as e:
                miss_cnt += 1
                # 提供最小干扰的退化：把这条样本单独放入自己的组，避免训练中断
                sid = f"__MISSING_SID__#{idx}"
                jd = None
            bucket[sid].append((idx, jd))

        groups: List[List[int]] = []
        for sid, pairs in bucket.items():
            if self.sort_group_by_jd:
                # None 放最后；有值按升序
                pairs.sort(key=lambda x: (x[1] is None, x[1] if isinstance(x[1], int) else 0))
            # 仅保留索引顺序
            groups.append([idx for idx, _ in pairs])

        self._groups = groups

        if self.verbose:
            sizes = [len(g) for g in self._groups]
            sizes_sorted = sorted(sizes)
            med = sizes_sorted[len(sizes)//2] if sizes_sorted else 0
            print(f"[PackFullGroupsBatchSampler] built {len(self._groups)} groups. "
                  f"min/median/max group size = {min(sizes) if sizes else 0}/{med}/{max(sizes) if sizes else 0}. "
                  f"missing_sid_rows={miss_cnt}")

    def _reset_order(self):
        rng = random.Random(self.seed + self._epoch)
        self._order = list(range(len(self._groups)))
        if self.shuffle:
            rng.shuffle(self._order)
        self._group_cursor = 0
        self._pending = []

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        self._reset_order()

    def __iter__(self):
        """
        严格“整组不可分”。用局部快照(cur, tmp_pending)模拟打包，
        只有在真正 yield 之前，才把快照回写到类状态，避免重入/预取引发的时序问题。
        """
        # 容错：确保存在该属性
        if not hasattr(self, "auto_reset_on_iter_end"):
            self.auto_reset_on_iter_end = True

        if self.auto_reset_on_iter_end and self._group_cursor >= len(getattr(self, "_order", [])) and not self._pending:
            self._epoch += 1
            self._reset_order()

        cur = self._group_cursor          # 局部快照，不直接动类属性
        tmp_pending = list(self._pending) # 局部缓冲

        while cur < len(self._order):
            gid = self._order[cur]
            g = self._groups[gid]
            gsz = len(g)

            # A) pending 为空且遇到“超大组”
            if not tmp_pending and gsz > self.batch_size:
                if not self.allow_oversized_group:
                    raise ValueError(
                        f"Found group (size={gsz}) larger than batch_size={self.batch_size}, "
                        f"and allow_oversized_group=False."
                    )
                # —— 修改：对齐到 divisor（不设上限，因为允许超大批）——
                out = self._pad_to_divisor(list(g), limit=None)
                self._group_cursor = cur + 1
                self._pending = []
                yield out
                cur = self._group_cursor
                tmp_pending = []
                continue

            remain = self.batch_size - len(tmp_pending)

            # B) 本组整组放不下 -> 先发当前 pending
            if tmp_pending and gsz > remain:
                out = list(tmp_pending)
                # —— 修改：对齐到 divisor（上限为 batch_size）——
                out = self._pad_to_divisor(out, limit=self.batch_size)
                self._group_cursor = cur
                self._pending = []
                yield out
                cur = self._group_cursor
                tmp_pending = []
                continue

            # C) 可以整组装入
            assert len(tmp_pending) + gsz <= self.batch_size, \
                f"internal error: group would overflow (pending={len(tmp_pending)}, gsz={gsz}, bs={self.batch_size})"
            tmp_pending.extend(g)
            cur += 1

            # 满了就发（batch_size 已保证是 divisor 的倍数）
            if len(tmp_pending) == self.batch_size:
                out = list(tmp_pending)
                self._group_cursor = cur
                self._pending = []
                yield out
                cur = self._group_cursor
                tmp_pending = []

        # 尾批
        if tmp_pending:
            out = list(tmp_pending)
            # —— 修改：对齐到 divisor（上限为 batch_size）——
            out = self._pad_to_divisor(out, limit=self.batch_size)
            self._group_cursor = cur
            self._pending = []
            assert len(out) <= self.batch_size or (self.allow_oversized_group and self.batch_size < len(out)), \
                f"Tail batch overflow: {len(out)} > {self.batch_size}"
            yield out

    def __len__(self) -> int:
        """
        用“与上面完全一致”的打包模拟来计数，基于当前状态快照，不修改类属性。
        """
        cnt = 0
        cur = getattr(self, "_group_cursor", 0)
        pending_sz = len(getattr(self, "_pending", []))

        while cur < len(self._order):
            gid = self._order[cur]
            gsz = len(self._groups[gid])

            # A) 超大组
            if pending_sz == 0 and gsz > self.batch_size:
                if not self.allow_oversized_group:
                    raise ValueError(
                        f"Found group (size={gsz}) larger than batch_size={self.batch_size}, "
                        f"and allow_oversized_group=False."
                    )
                cnt += 1
                cur += 1
                continue

            remain = self.batch_size - pending_sz

            # B) 放不下 -> 先结算当前 pending
            if pending_sz and gsz > remain:
                cnt += 1
                pending_sz = 0
                continue  # 不动 cur，这组下一轮处理

            # C) 整组装入
            assert pending_sz + gsz <= self.batch_size
            pending_sz += gsz
            cur += 1

            if pending_sz == self.batch_size:
                cnt += 1
                pending_sz = 0

        if pending_sz > 0:
            cnt += 1
        return cnt

    # —— DataLoader/StatefulDataLoader 可能会取/存 sampler 的状态 ——
    def state_dict(self) -> dict:
        return {
            "epoch": self._epoch,
            "group_cursor": self._group_cursor,
            "order": list(self._order),
            "pending": list(self._pending),
            "seed": self.seed,
            "shuffle": self.shuffle,
            "batch_size": self.batch_size,
            "allow_oversized_group": self.allow_oversized_group,
            "sort_group_by_jd": self.sort_group_by_jd,
            "auto_reset_on_iter_end": self.auto_reset_on_iter_end,
        }

    def load_state_dict(self, state: dict):
        self._epoch = int(state.get("epoch", 0))
        self._group_cursor = int(state.get("group_cursor", 0))
        self._order = list(state.get("order", list(range(len(self._groups)))))
        self._pending = list(state.get("pending", []))
        self.seed = int(state.get("seed", self.seed))
        self.shuffle = bool(state.get("shuffle", self.shuffle))
        self.batch_size = int(state.get("batch_size", self.batch_size))
        self.allow_oversized_group = bool(state.get("allow_oversized_group", self.allow_oversized_group))
        self.sort_group_by_jd = bool(state.get("sort_group_by_jd", self.sort_group_by_jd))
        self.auto_reset_on_iter_end = bool(state.get("auto_reset_on_iter_end", getattr(self, "auto_reset_on_iter_end", True)))
        
        # 关键修改：即使 state 里没有，也要保证定义
        if "auto_reset_on_iter_end" in state:
            self.auto_reset_on_iter_end = bool(state["auto_reset_on_iter_end"])
        else:
            # 强制兜底，避免 AttributeError
            self.auto_reset_on_iter_end = True