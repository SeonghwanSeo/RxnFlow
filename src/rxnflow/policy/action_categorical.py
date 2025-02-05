import torch
import torch_geometric.data as gd
from torch import Tensor

from gflownet.envs.graph_building_env import GraphActionCategorical, ActionIndex
from rxnflow.envs.action import RxnActionType
from rxnflow.envs.env_context import SynthesisEnvContext
from rxnflow.policy.action_space_subsampling import SubsamplingPolicy
from rxnflow.utils.misc import get_worker_env


def placeholder(size: tuple[int, ...], device: torch.device) -> Tensor:
    return torch.empty(size, dtype=torch.float32, device=device)


def neginf(size: tuple[int, ...], device: torch.device) -> Tensor:
    return torch.full(size, -torch.inf, dtype=torch.float32, device=device)


class RxnActionCategorical(GraphActionCategorical):
    def __init__(
        self,
        graphs: gd.Batch,
        emb: Tensor,
        action_masks: list[Tensor],
        model: torch.nn.Module,
    ):
        self.ctx: SynthesisEnvContext = get_worker_env("ctx")
        self.model = model
        self.graphs = graphs
        self.num_graphs = graphs.num_graphs
        self.emb: Tensor = emb
        self.dev = dev = self.emb.device
        self._epsilon = 1e-38
        self._action_masks: list[Tensor] = action_masks

        # NOTE: action subsampling
        sampler: SubsamplingPolicy = get_worker_env("action_subsampler")
        self.subsamples: list[Tensor] = []
        self.subsample_size: list[int] = []
        self._weights: list[float] = []  # importance weight
        for protocol in self.ctx.protocols:
            if protocol.action in (RxnActionType.FirstBlock, RxnActionType.BiRxn):
                # subsampling
                subsample, importance_weight = sampler.sampling(protocol.name)
            else:
                subsample, importance_weight = torch.tensor([torch.inf]), 0.0
            self.subsamples.append(subsample)
            self.subsample_size.append(subsample.shape[0])
            self._weights.append(importance_weight)

        self._masked_logits: list[Tensor] = self._calculate_logits()
        self.raw_logits: list[Tensor] = self._masked_logits
        self.weighted_logits: list[Tensor] = self.importance_weighting(1.0)

        self.batch = [torch.arange(self.num_graphs, device=dev)] * self.ctx.num_protocols
        self.slice = [torch.arange(self.num_graphs + 1, device=dev)] * self.ctx.num_protocols

    def _calculate_logits(self) -> list[Tensor]:
        # TODO: add descriptors
        # PERF: optimized for performance but bad readability
        # logits: [Nstate, Naction]
        masked_logits: list[Tensor] = []
        for protocol_idx, protocol in enumerate(self.ctx.protocols):
            subsample = self.subsamples[protocol_idx]
            num_actions = self.subsample_size[protocol_idx]
            protocol_mask = self._action_masks[protocol_idx]
            if num_actions == 0:
                logits = neginf((self.num_graphs, 1), device=self.dev)
            elif protocol_mask.any():
                emb_allowed = self.emb[protocol_mask]  # [Nstate', Fstate]
                if protocol.action is RxnActionType.FirstBlock:
                    # collect block data; [Nblock, F]
                    block_data = self.ctx.get_block_data(subsample).to(self.dev)
                    # calculate the logit for each action - (state, action)
                    allowed_logits = self.model.hook_firstblock(emb_allowed, block_data)
                elif protocol.action is RxnActionType.UniRxn:
                    allowed_logits = self.model.hook_unirxn(emb_allowed, protocol.name)
                elif protocol.action is RxnActionType.BiRxn:
                    # collect block data; [Nblock, F]
                    block_data = self.ctx.get_block_data(subsample).to(self.dev)
                    # calculate the logit for each action - (state, action)
                    allowed_logits = self.model.hook_birxn(emb_allowed, block_data, protocol.name)
                elif protocol.action is RxnActionType.Stop:
                    allowed_logits = self.model.hook_stop(emb_allowed)
                else:
                    raise ValueError(protocol)

                if protocol_mask.all():
                    # directly use the calculate logit
                    logits = allowed_logits
                else:
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
            if protocol.action is RxnActionType.Stop:
                logit = self.model.hook_stop(emb).view(-1)
            elif protocol.action is RxnActionType.FirstBlock:
                block_data = self.ctx.get_block_data(block_idx).to(self.dev)
                logit = self.model.hook_firstblock(emb, block_data).view(-1)
            elif protocol.action is RxnActionType.UniRxn:
                logit = self.model.hook_unirxn(emb, protocol.name).view(-1)
            elif protocol.action is RxnActionType.BiRxn:
                block_data = self.ctx.get_block_data(block_idx).to(self.dev)
                logit = self.model.hook_birxn(emb, block_data, protocol.name).view(-1)
            else:
                raise ValueError(protocol.action)
            action_logits[i] = logit
        return action_logits

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

        # when graph-wise prediction is only performed
        logits = self.weighted_logits  # use logit from importance weighting
        maxl: Tensor = self._compute_batchwise_max(logits).values  # [Ngraph,]
        corr_logits: list[Tensor] = [(i - maxl.unsqueeze(1)) for i in logits]
        exp_logits: list[Tensor] = [i.exp().clamp(self._epsilon) for i in corr_logits]
        logZ: Tensor = sum([i.sum(1) for i in exp_logits]).log()

        action_logits = self._cal_action_logits(actions) - maxl
        action_logprobs = (action_logits - logZ).clamp(max=0.0)
        return action_logprobs

    def importance_weighting(self, alpha: float = 1.0) -> list[Tensor]:
        if alpha == 0.0:
            return self.logits
        elif alpha == 1.0:
            return [logits + w for logits, w in zip(self.logits, self._weights, strict=True)]
        else:
            return [logits + alpha * w for logits, w in zip(self.logits, self._weights, strict=True)]

    def _mask(self, x: Tensor, m: Tensor) -> Tensor:
        assert m.dtype == torch.bool
        m = m.unsqueeze(-1)  # [Ngraph,] -> [Ngraph, 1]
        return x.masked_fill_(~m, -torch.inf)  # [Ngraph, Naction]

    @staticmethod
    def _get_pairwise_mask(level_state: Tensor, level_block: Tensor) -> Tensor:
        """Mask of the action (state, block)
        if level(state) + level(block) > 1, the action (state, block) is masked.

        Parameters
        ----------
        level_state : Tensor [Nstate, Nprop]
            level of state; here, only num atoms
        level_block : Tensor [Nblock, Nprop]
            level of state; here, only num atoms
        Returns
        -------
        action_mask: Tensor [Nstate, Nblock]
            mask of the action (state, block)
        """
        return ((level_state.unsqueeze(1) + level_block.unsqueeze(0)) < 1).all(-1)  # [Nstate, Nblock]

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
