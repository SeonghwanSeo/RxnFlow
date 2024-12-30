import copy
import math
import torch
import torch_geometric.data as gd
from torch_scatter import scatter_sum

from gflownet.envs.graph_building_env import GraphActionCategorical, ActionIndex
from gflownet.utils.misc import get_worker_device
from rxnflow.envs.action import RxnActionType
from rxnflow.envs.env_context import SynthesisEnvContext
from rxnflow.policy.action_space_subsampling import SubsamplingPolicy
from rxnflow.utils.misc import get_worker_env

from torch import Tensor


class RxnActionCategorical(GraphActionCategorical):
    def __init__(
        self,
        graphs: gd.Batch,
        emb: Tensor,
        action_masks: list[Tensor],
        model: torch.nn.Module,
    ):
        self.model = model
        self.graphs = graphs
        self.num_graphs = graphs.num_graphs
        self.emb: Tensor = emb
        self.dev = dev = self.emb.device
        self.ctx: SynthesisEnvContext = get_worker_env("ctx")
        self.action_subsampler: SubsamplingPolicy = get_worker_env("action_subsampler")
        self._epsilon = 1e-38

        subsampler: SubsamplingPolicy = get_worker_env("action_subsampler")
        self.subsample_results: list[tuple[list[int], float]] = []
        for protocol in self.ctx.protocols:
            block_type = protocol.block_type
            if block_type is None:  # dummy
                block_idcs = []
                sr = 1.0
            else:
                block_idcs = subsampler.sampling(block_type)
                sr = subsampler.sampling_ratios[block_type]
            self.subsample_results.append((block_idcs, sr))

        self._action_masks: list[Tensor] = action_masks
        self._masked_logits: list[Tensor] = self._calculate_logits()
        self.batch = [torch.arange(self.num_graphs, device=dev)] * self.ctx.num_protocols
        self.slice = [torch.arange(self.num_graphs + 1, device=dev)] * self.ctx.num_protocols

    def to(self, device):
        self.dev = device
        self._masked_logits = [i.to(device) for i in self._masked_logits]
        self._action_masks = [i.to(device) for i in self._action_masks]
        # self.batch = [i.to(device) for i in self.batch]
        # self.slice = [i.to(device) for i in self.slice]
        # if self.logprobs is not None:
        #     self.logprobs = [i.to(device) for i in self.logprobs]
        # if self.log_n is not None:
        #     self.log_n = self.log_n.to(device)
        return self

    def detach(self):
        new = copy.copy(self)
        new._masked_logits = [i.detach() for i in new._masked_logits]
        return new

    def _calculate_logits(self) -> list[Tensor]:
        """
        Due to action space subsampling, we calculate logits for actions in ActionCategorical class
        instead of GFN model.
        """
        device = get_worker_device()
        placeholder = lambda x: torch.full(x, -torch.inf, dtype=torch.float32, device=device)  # noqa: E731

        logits: Tensor
        masked_logits: list[Tensor] = []
        for protocol_idx, protocol in enumerate(self.ctx.protocols):
            mask = self._action_masks[protocol_idx]
            if protocol.action in (RxnActionType.FirstBlock, RxnActionType.BiRxn):
                assert protocol.block_type is not None
                use_block_idcs, subsampling_ratio = self.subsample_results[protocol_idx]
                logits = placeholder((self.num_graphs, len(use_block_idcs)))  # [Nstate, Nblock]
                if mask.any():
                    emb = self.emb[mask]  # [Ninit, Fstate]
                    block_data = self.ctx.get_block_data(protocol.block_type, use_block_idcs, device)
                    _logits = self.model.hook_adding_block(emb, block_data, protocol.name)
                    logits[mask] = _logits - math.log(subsampling_ratio)  # Importance sampling
            elif protocol.action is RxnActionType.UniRxn:
                if mask.any():
                    logits = self.model.hook_unirxn(self.emb, protocol.name)
                    logits = logits.masked_fill(~mask.view(-1, 1), -torch.inf)
                else:
                    logits = placeholder((self.num_graphs, 1))  # [Nstate, 1]
            elif protocol.action is RxnActionType.Stop:
                logits = placeholder((self.num_graphs, 1))
            else:
                raise ValueError(protocol.action)
            masked_logits.append(logits)
        return masked_logits

    def _cal_action_logits(self, actions: list[ActionIndex]) -> Tensor:
        """Calculate the logit values for sampled actions"""
        # NOTE: placeholder of action_logits
        device = get_worker_device()
        action_logits = torch.full((len(actions),), -torch.inf, device=self.dev)
        for i, action in enumerate(actions):
            protocol_idx, _, block_idx = action
            assert self._action_masks[protocol_idx][i]  # it should be not masked.
            protocol = self.ctx.protocols[protocol_idx]
            if protocol.action in (RxnActionType.FirstBlock, RxnActionType.BiRxn):
                assert protocol.block_type is not None
                block_data = self.ctx.get_block_data(protocol.block_type, [block_idx], device).view(-1)
                logit = self.model.single_hook_adding_block(self.emb[i], block_data, protocol.name)
            elif protocol.action is RxnActionType.UniRxn:
                logit = self._masked_logits[protocol_idx][i, 0]
            else:
                raise ValueError(protocol.action)
            action_logits[i] = logit
        return action_logits

    # NOTE: Function override
    def sample(self) -> list[ActionIndex]:
        """override function of sample()
        Since we perform action space subsampling, the indices of block is from the partial space.
        Therefore, we reassign the block indices on the entire block library.
        """
        action_list = super().sample()
        global_index_actions: list[ActionIndex] = []
        for action in action_list:
            protocol_idx, row_idx, local_block_idx = action
            assert row_idx == 0
            use_block_idcs, subsampling_ratio = self.subsample_results[protocol_idx]
            if subsampling_ratio < 1.0:
                # NOTE: Index of a partial action space -> Index of an entire action space
                action = ActionIndex(protocol_idx, row_idx, use_block_idcs[local_block_idx])
            global_index_actions.append(action)
        return global_index_actions

    def log_prob(
        self,
        actions: list[ActionIndex],
        logprobs: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> Tensor:
        """The log-probability of a list of action tuples, effectively indexes `logprobs` using internal
        slice indices.

        Parameters
        ----------
        actions: List[ActionIndex]
            A list of n action tuples denoting indices
        logprobs: None (dummy)
        batch: None (dummy)

        Returns
        -------
        action_logprobs: Tensor
            The log probability of each action.
        """
        assert logprobs is None
        assert batch is None

        maxl: Tensor = self._compute_batchwise_max(self.logits).values  # [Ngraph,]
        if True:
            # when graph-wise prediction is only performed
            corr_logits: list[Tensor] = [(i - maxl.unsqueeze(1)) for i in self._masked_logits]
            exp_logits: list[Tensor] = [i.exp().clamp(self._epsilon) for i in corr_logits]
            logZ: Tensor = sum([i.sum(1) for i in exp_logits]).log()
        else:
            # when node-wise or edge-wise prediction is required
            corr_logits = [(i - maxl[b, None]) for i, b in zip(self.logits, self.batch, strict=True)]
            exp_logits = [i.exp().clamp(self._epsilon) for i in corr_logits]
            logZ: Tensor = sum(
                [
                    scatter_sum(i, b, dim=0, dim_size=self.num_graphs).sum(1)
                    for i, b in zip(exp_logits, self.batch, strict=True)
                ]
            ).log()
        action_logits = self._cal_action_logits(actions) - maxl
        action_logprobs = (action_logits - logZ).clamp(max=0.0)
        return action_logprobs

    @property
    def logits(self) -> list[Tensor]:
        return self._masked_logits

    @logits.setter
    def logits(self, new_raw_logits):
        self.raw_logits = new_raw_logits
        self._apply_action_masks()

    def _mask(self, x: Tensor, m: Tensor) -> Tensor:
        """mask the logit

        Parameters
        ----------
        x : FloatTensor
            logit of action (protocol) type, [Ngraph, Naction]
        m : BoolTensor
            mask of action (protocol) type, [Ngraph,]

        Returns
        -------
        masked_x: FloatTensor
            masked logit
        """
        assert m.dtype == torch.bool
        m = m.unsqueeze(-1)
        return x.masked_fill(~m, -torch.inf)

    # NOTE: same but faster (optimized for graph-wise predictions)
    def argmax(
        self,
        x: list[Tensor],
        batch: list[Tensor] | None = None,
        dim_size: int | None = None,
    ) -> list[ActionIndex]:
        """10x Faster argmax() under graph-wise batching (no node-wise or edge-wise)"""
        # NOTE: Find Protocol Type
        max_per_type = [torch.max(tensor, dim=1) for tensor in x]
        max_values_per_type = [pair[0] for pair in max_per_type]
        type_max: list[int] = torch.max(torch.stack(max_values_per_type), dim=0)[1].tolist()
        assert len(type_max) == self.num_graphs

        # NOTE: Find Action Index
        col_max_per_type = [pair[1] for pair in max_per_type]
        col_max: list[int] = [int(col_max_per_type[t][i]) for i, t in enumerate(type_max)]

        argmaxes = [ActionIndex(i, 0, j) for i, j in zip(type_max, col_max, strict=True)]
        return argmaxes

    def _compute_batchwise_max(
        self,
        x: list[Tensor],
        detach: bool = True,
        batch: list[Tensor] | None = None,
        reduce_columns: bool = True,
    ):
        """Compute the maximum value of each batch element in `x`

        Parameters
        ----------
        x: list[Tensor]
            A list of tensors of shape `(n, m)` (e.g. representing _masked_logits)
        detach: bool, default=True
            If true, detach the tensors before computing the max
        batch: List[Tensor], default=None
            The batch index of each element in `x`. If None, uses self.batch
        reduce_columns: bool, default=True
            If true computes the max over the columns, and returns a tensor of shape `(k,)`
            If false, only reduces over rows, returns a list of (values, indexes) tuples.

        Returns
        -------
        maxl: (values: Tensor, indices: Tensor)
            A named tuple of tensors of shape `(k,)` where `k` is the number of graphs in the batch, unless
            reduce_columns is False. In the latter case, returns a list of named tuples that don't have columns reduced.
        """
        if detach:
            x = [i.detach() for i in x]
        if reduce_columns:
            return torch.cat(x, dim=1).max(1)
        else:
            res = [
                (i, torch.arange(i.shape[0], dtype=torch.long, device=x[0].device).view(-1, 1).repeat(1, i.shape[1]))
                for i in x
            ]
            return res
