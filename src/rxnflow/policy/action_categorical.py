import torch
import torch_geometric.data as gd
from torch import Tensor

from gflownet.envs.graph_building_env import ActionIndex, GraphActionCategorical
from rxnflow.envs.action import Protocol, RxnActionType
from rxnflow.envs.env_context import SynthesisEnvContext
from rxnflow.policy.action_space_subsampling import SubsamplingPolicy
from rxnflow.utils.misc import get_worker_env


def placeholder(size: tuple[int, ...], device: torch.device, **kwargs) -> Tensor:
    return torch.empty(size, dtype=torch.float32, device=device, **kwargs)


def neginf(size: tuple[int, ...], device: torch.device, **kwargs) -> Tensor:
    return torch.full(size, -torch.inf, dtype=torch.float32, device=device, **kwargs)


class RxnActionCategorical(GraphActionCategorical):
    def __init__(
        self,
        graphs: gd.Batch,
        emb: Tensor,
        logit_scale: Tensor,
        protocol_masks: list[Tensor],
        model: torch.nn.Module,
    ):
        self._epsilon = 1e-38

        self.ctx: SynthesisEnvContext = get_worker_env("ctx")
        self.model = model
        self.graphs = graphs
        self.num_graphs = graphs.num_graphs
        self.emb: Tensor = emb
        self.logit_scale: Tensor = logit_scale
        self._protocol_masks: list[Tensor] = protocol_masks
        self.dev = self.emb.device

        # NOTE: action subsampling
        subsampler: SubsamplingPolicy = get_worker_env("action_subsampler")
        self.subsamples: list[Tensor] = subsampler.sampling()
        self._importance_weights: list[float] = subsampler.protocol_weights  # importance weight

        self._masked_logits: list[Tensor] = self._calculate_logits()
        self.raw_logits: list[Tensor] = self._masked_logits

        # self.batch = [torch.arange(self.num_graphs, device=dev)] * self.ctx.num_protocols
        # self.slice = [torch.arange(self.num_graphs + 1, device=dev)] * self.ctx.num_protocols

    def _calculate_logits(self) -> list[Tensor]:
        # TODO: add descriptors
        # PERF: optimized for performance but bad readability
        masked_logits: list[Tensor] = []
        # logits: [Nstate, Naction]
        for protocol_idx, protocol in enumerate(self.ctx.protocols):
            protocol_mask = self._protocol_masks[protocol_idx]
            subsample_idcs = self.subsamples[protocol_idx]
            num_actions = len(subsample_idcs)
            if num_actions == 0:
                logits = neginf((self.num_graphs, 1), device=self.dev)
            elif protocol_mask.all():
                # calculate logit then perform logit-scaling (Logit-GFN)
                logits = self.model_hook(protocol, self.emb, subsample_idcs)
                logits = logits * self.logit_scale.view(-1, 1)  # [Nstate, Naction]
            elif protocol_mask.any():
                # calculate logit
                emb_allowed = self.emb[protocol_mask]  # [Nstate', Fstate]
                allowed_logits = self.model_hook(protocol, emb_allowed, subsample_idcs)
                # logit-scaling (Logit-GFN)
                allowed_logit_scale = self.logit_scale[protocol_mask].view(-1, 1)  # [Nstate', 1]
                allowed_logits = allowed_logits * allowed_logit_scale
                # create placeholder first and then insert the calculated.
                logits = neginf((self.num_graphs, num_actions), device=self.dev)
                logits[protocol_mask] = allowed_logits
            else:
                # set 1 instead of `num_actions` to reduce overhead
                logits = neginf((self.num_graphs, 1), device=self.dev)
            masked_logits.append(logits)
        return masked_logits

    def _cal_action_logits(self, actions: list[ActionIndex]) -> Tensor:
        """Calculate the logit values for sampled actions"""
        action_logits = placeholder((len(actions),), device=self.dev)
        for i, action in enumerate(actions):
            protocol_idx, block_type_idx, block_idx = action
            protocol = self.ctx.protocols[protocol_idx]
            emb = self.emb[i].view(1, -1)
            logit = self.model_hook(protocol, emb, subsampled_idcs=block_idx)
            action_logits[i] = logit
        action_logits = action_logits * self.logit_scale.view(-1)
        return action_logits

    def model_hook(
        self,
        protocol: Protocol,
        emb: Tensor,
        subsampled_idcs: Tensor | int,
    ) -> Tensor:
        if protocol.action is RxnActionType.FirstBlock:
            # collect block data; [Nblock, F]
            block = self.ctx.get_block_data(subsampled_idcs, self.dev)
            # calculate the logit for each starting block - (state, action)
            return self.model.hook_firstblock(emb, block)
        elif protocol.action is RxnActionType.UniRxn:
            # calculate the logit for corresponding reaction - (state, 1)
            return self.model.hook_unirxn(emb, protocol.name)
        elif protocol.action is RxnActionType.BiRxn:
            # collect block data; [Nblock, F]
            block = self.ctx.get_block_data(subsampled_idcs, self.dev)
            # calculate the logit for each reactant - (state, action)
            return self.model.hook_birxn(emb, block, protocol.name)
        elif protocol.action is RxnActionType.Stop:
            # calculate the logit for stop action - (state, 1)
            return self.model.hook_stop(emb)
        else:
            raise ValueError(protocol)

    # NOTE: Function override
    def sample(self) -> list[ActionIndex]:
        """Sample the action
        Since we perform action space subsampling, the indices of block is from the partial space.
        Therefore, we reassign the block indices on the entire block library.
        """
        action_list = super().sample()
        reindexed_actions: list[ActionIndex] = []
        for action in action_list:
            protocol_idx, row_idx, col_idx = action
            assert row_idx == 0
            action_type = self.ctx.protocols[protocol_idx].action
            if action_type in (RxnActionType.FirstBlock, RxnActionType.BiRxn):
                _col_idx = int(self.subsamples[protocol_idx][col_idx])
                action = ActionIndex(protocol_idx, 0, _col_idx)
            elif action_type in (RxnActionType.Stop, RxnActionType.UniRxn):
                assert row_idx == 0 and col_idx == 0
            else:
                raise ValueError(action)
            reindexed_actions.append(action)
        return reindexed_actions

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

        logZ: Tensor = self.calc_logZ()
        action_logits = self._cal_action_logits(actions)
        action_logprobs = (action_logits - logZ).clamp(max=0.0)
        return action_logprobs

    def calc_logZ(self) -> Tensor:
        """calculate logZ"""
        # importance weighting
        logits = self.importance_weighting(alpha=1.0)
        maxl = self._compute_batchwise_max_opt(logits).values  # [Ngraph,]
        corr_logits: list[Tensor] = [(i - maxl.unsqueeze(1)) for i in logits]
        exp_logits: list[Tensor] = [i.exp() for i in corr_logits]
        logZ = sum([i.sum(1).clamp(self._epsilon) for i in exp_logits]).log()
        return logZ + maxl

    def importance_weighting(self, alpha: float = 1.0) -> list[Tensor]:
        """importance weighting; calibrate logits with the action subsampling ratio"""
        if alpha == 0.0:
            return self.logits
        else:
            return [logits + alpha * w for logits, w in zip(self.logits, self._importance_weights, strict=True)]

    def _apply_action_masks(self):
        self._masked_logits = [
            self._mask(logits, mask) for logits, mask in zip(self.raw_logits, self._protocol_masks, strict=True)
        ]

    def _mask(self, x: Tensor, m: Tensor) -> Tensor:
        assert m.dtype == torch.bool
        m = m.unsqueeze(-1)  # [Ngraph,] -> [Ngraph, 1]
        return x.masked_fill_(~m, -torch.inf)  # [Ngraph, Naction]

    # NOTE: same but 10x faster (optimized for graph-wise predictions)
    def argmax(
        self,
        x: list[Tensor],
        batch: list[Tensor] | None = None,
        dim_size: int | None = None,
    ) -> list[ActionIndex]:
        # Find protocol type
        max_per_type = [torch.max(tensor, dim=1) for tensor in x]
        max_values_per_type = [pair[0] for pair in max_per_type]
        type_max: list[int] = torch.max(torch.stack(max_values_per_type), dim=0)[1].tolist()
        assert len(type_max) == self.num_graphs

        # find action indexes
        col_max_per_type = [pair[1] for pair in max_per_type]
        col_max: list[int] = [int(col_max_per_type[t][i]) for i, t in enumerate(type_max)]

        # return argmaxes
        argmaxes = [ActionIndex(i, 0, j) for i, j in zip(type_max, col_max, strict=True)]
        return argmaxes

    # NOTE: same but faster (optimized for graph-wise predictions)
    def _compute_batchwise_max_opt(self, x: list[Tensor]):
        x = [i.detach() for i in x]
        return torch.cat(x, dim=1).max(1)

    # NOTE: unused (replaced to ***_opt())
    def _compute_batchwise_max(
        self,
        x: list[Tensor],
        detach: bool = True,
        batch: list[Tensor] | None = None,
        reduce_columns: bool = True,
    ):
        if detach:
            x = [i.detach() for i in x]
        if batch is None:
            batch = self.batch
        if reduce_columns:
            return torch.cat(x, dim=1).max(1)
        return [(i, b.view(-1, 1).repeat(1, i.shape[1])) for i, b in zip(x, batch, strict=True)]
