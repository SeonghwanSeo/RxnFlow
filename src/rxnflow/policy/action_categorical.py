import math
import torch
import torch_geometric.data as gd

from gflownet.envs.graph_building_env import GraphActionCategorical
from rxnflow.envs.action import RxnActionIndex, RxnActionType
from rxnflow.envs.env_context import SynthesisEnvContext
from rxnflow.policy.action_space_subsampling import SubsamplingPolicy
from rxnflow.utils.misc import get_worker_env


class RxnActionCategorical(GraphActionCategorical):
    def __init__(
        self,
        graphs: gd.Batch,
        emb: torch.Tensor,
        model: torch.nn.Module,
        fwd: bool,
    ):
        self.model = model
        self.graphs = graphs
        self.num_graphs = graphs.num_graphs
        self.traj_indices = graphs.traj_idx
        self.emb: torch.Tensor = emb
        self.dev = dev = self.emb.device
        self.ctx: SynthesisEnvContext = get_worker_env("ctx")
        self.action_subsampler: SubsamplingPolicy = get_worker_env("action_subsampler")
        self.fwd = fwd
        self._epsilon = 1e-38

        self.types: list[RxnActionType]
        if fwd:
            self.types = self.ctx.action_type_order
        else:
            self.types = self.ctx.bck_action_type_order

        self._logits: list[torch.Tensor] = []
        self.masks: dict[RxnActionType, torch.Tensor] = {
            RxnActionType.UniRxn: graphs[RxnActionType.UniRxn.mask_name],
            RxnActionType.BiRxn: graphs[RxnActionType.BiRxn.mask_name],
        }
        self.batch = torch.arange(self.num_graphs, device=dev)
        self.random_action_mask: torch.Tensor | None = None

    def sample(
        self,
        sample_temp: float = 1.0,
        min_len: int = 2,
        onpolicy_temp: float = 0.0,
    ) -> list[RxnActionIndex]:
        """
        Samples from the categorical distribution
        sample_temp:
            Softmax temperature used when sampling
        onpolicy_temp:
            t<1.0: more UniRxn/Stop
            t=1.0: on-policy sampling
            t>1.0: more BiRxn
        """
        traj_idx = self.traj_indices[0]
        assert (self.traj_indices == traj_idx).all()  # For sampling, we use the same traj index
        if traj_idx == 0:
            return self.sample_initial_state(sample_temp)
        else:
            return self.sample_later_state(sample_temp, min_len, onpolicy_temp)

    def sample_initial_state(self, sample_temp: float = 1.0):
        # NOTE: The first action in a trajectory is always FirstBlock (select a building block)
        type_idx = self.types.index(RxnActionType.FirstBlock)

        action_subsampler: SubsamplingPolicy = get_worker_env("action_subsampler")
        block_space = action_subsampler.get_space(RxnActionType.FirstBlock)
        block_indices = block_space.sampling()
        block_emb = self.model.block_embedding(self.ctx.get_block_data(block_indices, self.dev))

        # NOTE: PlaceHolder
        logits = self.model.hook_firstblock(self.emb, block_emb)
        self._logits.append(logits)

        if self.random_action_mask is not None:
            self._logits[0][self.random_action_mask, :] = 0.0

        # NOTE: Softmax temperature used when sampling
        if sample_temp != 1:
            self._logits = [logit / sample_temp for logit in self._logits]

        # NOTE: Use the Gumbel trick to sample categoricals
        noise = torch.rand_like(self._logits[0])
        gumbel = self._logits[0] - (-noise.log()).log()
        argmax = self.argmax(x=[gumbel])
        return [RxnActionIndex.create(type_idx, block_idx=block_indices[block_idx]) for _, block_idx in argmax]

    def sample_later_state(
        self,
        sample_temp: float = 1.0,
        min_len: int = 2,
        onpolicy_temp: float = 0.0,
    ):
        action_subsampler: SubsamplingPolicy = get_worker_env("action_subsampler")
        self._logits.append(self.model.hook_stop(self.emb))
        self._logits.append(self.model.hook_uni_rxn(self.emb, self.masks[RxnActionType.UniRxn]))

        block_indices_bi_rxn = []
        for rxn_idx in range(self.ctx.num_bi_rxns):
            for block_is_first in (True, False):
                # TODO: Check the efficiency
                reactant_space = action_subsampler.get_space(RxnActionType.BiRxn, (rxn_idx, block_is_first))
                if reactant_space.num_blocks == 0:
                    block_indices_bi_rxn.append([int(1e9)])  # invald index, it should not be sampled
                    self._logits.append(torch.full((self.num_graphs, 1), -torch.inf, device=self.dev))
                    continue
                else:
                    block_indices = reactant_space.sampling()
                    block_indices_bi_rxn.append(block_indices)
                    block_emb = self.model.block_embedding(self.ctx.get_block_data(block_indices, self.dev))
                    mask = self.masks[RxnActionType.BiRxn][:, rxn_idx, int(block_is_first)]
                    logits = self.model.hook_bi_rxn(self.emb, rxn_idx, block_is_first, block_emb, mask)
                    if onpolicy_temp != 0.0 and reactant_space.sampling_ratio != 1.0:
                        logits = logits - onpolicy_temp * math.log(reactant_space.sampling_ratio)
                    self._logits.append(logits)

        if self.random_action_mask is not None:
            mask = self.random_action_mask
            random_action_idx = torch.where(mask)[0]
            for i in random_action_idx:
                uni_rxn_mask = self.masks[RxnActionType.UniRxn][i]
                bi_rxn_mask = self.masks[RxnActionType.BiRxn][i]
                num_uni_rxn = uni_rxn_mask.sum().item()
                num_bi_rxn = bi_rxn_mask.sum().item()

                self._logits[0][i] = 0
                if num_uni_rxn > 0:
                    self._logits[1][i, uni_rxn_mask] = -math.log(num_uni_rxn)
                if num_bi_rxn > 0:
                    offset = 2
                    for rxn_idx in range(self.ctx.num_bi_rxns):
                        for block_is_first in (True, False):
                            if bi_rxn_mask[rxn_idx, int(block_is_first)]:
                                num_blocks = self._logits[offset].shape[1]
                                self._logits[offset][i, None] = -(math.log(num_bi_rxn) + math.log(num_blocks))
                            offset += 1

        # NOTE: Softmax temperature used when sampling
        if sample_temp != 1:
            self._logits = [logit / sample_temp for logit in self._logits]

        # NOTE: Use the Gumbel trick to sample categoricals
        gumbel = []
        for i, logit in enumerate(self._logits):
            if (i == 0) and (self.traj_indices[0] < min_len):
                gumbel.append(torch.full_like(logit, -1e6))
            else:
                noise = torch.rand_like(logit)
                gumbel.append(logit - (-noise.log()).log())
        argmax = self.argmax(x=gumbel)  # tuple of action type, action idx

        actions: list[RxnActionIndex] = []
        for idx1, idx2 in argmax:
            rxn_idx = block_idx = block_is_first = None
            if idx1 == 0:
                t = RxnActionType.Stop
            elif idx1 == 1:
                t = RxnActionType.UniRxn
                rxn_idx = idx2
            else:
                t = RxnActionType.BiRxn
                idx1 = idx1 - 2
                block_idx = block_indices_bi_rxn[idx1][idx2]
                rxn_idx, block_is_first = idx1 // 2, bool(idx1 % 2 == 0)
            type_idx = self.types.index(t)
            actions.append(RxnActionIndex.create(type_idx, rxn_idx, block_idx, block_is_first))
        return actions

    def argmax(self, x: list[torch.Tensor]) -> list[tuple[int, int]]:
        # for each graph in batch and for each action type, get max value and index
        max_per_type = [torch.max(tensor, dim=1) for tensor in x]
        max_values_per_type = [pair[0] for pair in max_per_type]
        argmax_indices_per_type = [pair[1] for pair in max_per_type]
        _, type_indices = torch.max(torch.stack(max_values_per_type), dim=0)
        action_indices = torch.gather(torch.stack(argmax_indices_per_type), 0, type_indices.unsqueeze(0)).squeeze(0)
        argmax_pairs = list(zip(type_indices.tolist(), action_indices.tolist(), strict=True))  # action type, action idx
        return argmax_pairs

    def log_prob_after_sampling(self, actions: list[RxnActionIndex]) -> torch.Tensor:
        """Access the log-probability of actions after sampling for entropy estimation"""
        logits = torch.cat(self._logits, dim=-1)
        max_logits = logits.detach().max(dim=-1, keepdim=True).values
        logZ = torch.logsumexp(logits - max_logits, dim=-1) + max_logits.squeeze(-1)
        action_logits = self.cal_action_logits(actions)
        return action_logits - logZ

    def cal_action_logits(self, actions: list[RxnActionIndex]):
        # NOTE: placeholder of action_logits
        action_logits = torch.empty(len(actions), device=self.dev)
        for i, action in enumerate(actions):
            type_idx, rxn_idx, block_idx, block_is_first = action
            t = self.types[type_idx]
            if t is RxnActionType.FirstBlock:
                block_emb = self.model.block_embedding(self.ctx.get_block_data([int(block_idx)], self.dev).view(-1))
                logit = self.model.single_hook_firstblock(self.emb[i], block_emb)
            elif t is RxnActionType.Stop:
                logit = self.model.single_hook_stop(self.emb[i])
            elif t is RxnActionType.UniRxn:
                self.masks[t][i, rxn_idx] = True
                logit = self.model.single_hook_uni_rxn(self.emb[i], rxn_idx)
            elif t is RxnActionType.BiRxn:
                self.masks[t][i, rxn_idx, int(block_is_first)] = True
                block_emb = self.model.block_embedding(self.ctx.get_block_data([int(block_idx)], self.dev).view(-1))
                logit = self.model.single_hook_bi_rxn(self.emb[i], rxn_idx, block_is_first, block_emb)
            else:
                raise ValueError
            action_logits[i] = logit
        return action_logits

    def log_prob(self, actions: list[RxnActionIndex]) -> torch.Tensor:
        """Access the log-probability of actions"""
        assert len(actions) == self.num_graphs, f"num_graphs: {self.num_graphs}, num_actions: {len(actions)}"
        if self.fwd:
            logZ = self.cal_logZ()
            action_logits = self.cal_action_logits(actions)
            action_logits = action_logits - logZ
            return action_logits.clamp(math.log(self._epsilon))
        else:
            raise NotImplementedError

    def cal_logZ(self):
        initial_indices = torch.where(self.traj_indices == 0)[0]
        later_indices = torch.where(self.traj_indices != 0)[0]
        logZ = torch.empty(self.num_graphs, device=self.dev)
        logZ[initial_indices] = self.cal_logZ_initial(initial_indices)
        logZ[later_indices] = self.cal_logZ_later(later_indices)
        return logZ

    def cal_logZ_initial(self, state_indices: torch.Tensor) -> torch.Tensor:
        emb = self.emb[state_indices]
        block_space = self.action_subsampler.get_space(RxnActionType.FirstBlock)
        sampling_ratio = block_space.sampling_ratio
        if sampling_ratio < 1.0:
            block_indices = block_space.sampling()
            block_emb = self.model.block_embedding(self.ctx.get_block_data(block_indices, self.dev))
            logits = self.model.hook_firstblock(emb, block_emb)
            max_logits = logits.detach().max(dim=-1, keepdim=True).values
            logZ = torch.logsumexp(logits - max_logits, dim=-1) - math.log(sampling_ratio)
        else:
            block_emb = self.model.block_embedding(self.ctx.get_block_data(block_space.block_indices, self.dev))
            logits = self.model.hook_firstblock(emb, block_emb)
            max_logits = logits.detach().max(dim=-1, keepdim=True).values
            logZ = torch.logsumexp(logits - max_logits, dim=-1)
        return logZ + max_logits.squeeze(-1)

    def cal_logZ_later(self, state_indices: torch.Tensor) -> torch.Tensor:
        emb = self.emb[state_indices]
        logit_list = []
        logit_list.append(self.model.hook_stop(emb))
        logit_list.append(self.model.hook_uni_rxn(emb, self.masks[RxnActionType.UniRxn][state_indices]))

        for rxn_idx in range(self.ctx.num_bi_rxns):
            for block_is_first in (True, False):
                reactant_space = self.action_subsampler.get_space(RxnActionType.BiRxn, (rxn_idx, block_is_first))
                if reactant_space.num_blocks == 0:
                    continue
                mask = self.masks[RxnActionType.BiRxn][state_indices, rxn_idx, int(block_is_first)]
                sampling_ratio = reactant_space.sampling_ratio
                if sampling_ratio < 1:
                    block_indices = reactant_space.sampling()
                    block_emb = self.model.block_embedding(self.ctx.get_block_data(block_indices, self.dev))
                    logits = self.model.hook_bi_rxn(emb, rxn_idx, block_is_first, block_emb, mask)
                    logit_list.append(logits - math.log(sampling_ratio))
                else:
                    block_emb = self.model.block_embedding(
                        self.ctx.get_block_data(reactant_space.block_indices, self.dev)
                    )
                    logits = self.model.hook_bi_rxn(emb, rxn_idx, block_is_first, block_emb, mask)
                    logit_list.append(logits)
        logits = torch.cat(logit_list, dim=-1)
        max_logits = logits.detach().max(dim=-1, keepdim=True).values
        logZ = torch.logsumexp(logits - max_logits, dim=-1)
        return logZ + max_logits.squeeze(-1)
