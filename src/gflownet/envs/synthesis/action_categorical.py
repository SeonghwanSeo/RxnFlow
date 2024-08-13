import math
import torch
import torch_geometric.data as gd


from gflownet.envs.graph_building_env import GraphActionCategorical
from gflownet.envs.synthesis.action import (
    ReactionActionIdx,
    ReactionActionType,
    get_action_idx,
)
from gflownet.envs.synthesis.action_sampling import ActionSamplingPolicy
from gflownet.envs.synthesis.env_context import SynthesisEnvContext
from gflownet.utils.misc import get_worker_env


class ReactionActionCategorical(GraphActionCategorical):
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
        self.fwd = fwd
        self._epsilon = 1e-38

        if fwd:
            self.types: list[ReactionActionType] = self.ctx.action_type_order
        else:
            self.types: list[ReactionActionType] = self.ctx.bck_action_type_order

        self.logits: list[torch.Tensor] = []
        self.masks: dict[ReactionActionType, torch.Tensor] = {
            ReactionActionType.ReactUni: graphs[ReactionActionType.ReactUni.mask_name],
            ReactionActionType.ReactBi: graphs[ReactionActionType.ReactBi.mask_name],
        }
        self.batch = torch.arange(self.num_graphs, device=dev)
        self.random_action_mask: torch.Tensor | None = None

    def sample(
        self,
        action_sampler: ActionSamplingPolicy,
        onpolicy_temp: float = 0.0,
        sample_temp: float = 1.0,
        min_len: int = 2,
    ) -> list[ReactionActionIdx]:
        """
        Samples from the categorical distribution
        sample_temp:
            Softmax temperature used when sampling
        onpolicy_temp:
            t<1.0: more ReactUni/Stop
            t=1.0: on-policy sampling
            t>1.0: more ReactBi
        """
        traj_idx = self.traj_indices[0]
        assert (self.traj_indices == traj_idx).all()  # For sampling, we use the same traj index
        if traj_idx == 0:
            return self.sample_initial_state(action_sampler, sample_temp)
        else:
            return self.sample_later_state(action_sampler, onpolicy_temp, sample_temp, min_len)

    def sample_initial_state(
        self,
        action_sampler: ActionSamplingPolicy,
        sample_temp: float = 1.0,
    ):
        # NOTE: The first action in a trajectory is always AddFirstReactant (select a building block)
        type_idx = self.types.index(ReactionActionType.AddFirstReactant)

        block_space = action_sampler.get_space(ReactionActionType.AddFirstReactant)
        block_indices = block_space.sampling()
        block_emb = self.model.block_mlp(self.ctx.get_block_data(block_indices, self.dev))

        # NOTE: PlaceHolder
        logits = self.model.hook_add_first_reactant(self.emb, block_emb)
        self.logits.append(logits)

        if self.random_action_mask is not None:
            self.logits[0][self.random_action_mask, :] = 0.0

        # NOTE: Softmax temperature used when sampling
        if sample_temp != 1:
            self.logits = [logit / sample_temp for logit in self.logits]

        # NOTE: Use the Gumbel trick to sample categoricals
        noise = torch.rand_like(self.logits[0])
        gumbel = self.logits[0] - (-noise.log()).log()
        argmax = self.argmax(x=[gumbel])
        return [get_action_idx(type_idx, block_idx=block_indices[block_idx]) for _, block_idx in argmax]

    def sample_later_state(
        self,
        action_sampler: ActionSamplingPolicy,
        onpolicy_temp: float = 0.0,
        sample_temp: float = 1.0,
        min_len: int = 2,
    ):
        self.logits.append(self.model.hook_stop(self.emb))
        self.logits.append(self.model.hook_reactuni(self.emb, self.masks[ReactionActionType.ReactUni]))

        block_indices_reactbi = []
        for rxn_idx in range(self.ctx.num_bimolecular_rxns):
            for block_is_first in (True, False):
                # TODO: Check the efficiency
                reactant_space = action_sampler.get_space(ReactionActionType.ReactBi, (rxn_idx, block_is_first))
                if reactant_space.num_blocks == 0:
                    block_indices_reactbi.append([int(1e9)])  # invald index, it should not be sampled
                    self.logits.append(torch.full((self.num_graphs, 1), -torch.inf, device=self.dev))
                    continue
                else:
                    block_indices = reactant_space.sampling()
                    block_indices_reactbi.append(block_indices)
                    block_emb = self.model.block_mlp(self.ctx.get_block_data(block_indices, self.dev))
                    mask = self.masks[ReactionActionType.ReactBi][:, rxn_idx, int(block_is_first)]
                    logits = self.model.hook_reactbi(self.emb, rxn_idx, block_is_first, block_emb, mask)
                    if onpolicy_temp != 0.0 and reactant_space.sampling_ratio != 1.0:
                        logits = logits - onpolicy_temp * math.log(reactant_space.sampling_ratio)
                    self.logits.append(logits)

        if self.random_action_mask is not None:
            mask = self.random_action_mask
            random_action_idx = torch.where(mask)[0]
            for i in random_action_idx:
                reactuni_mask = self.masks[ReactionActionType.ReactUni][i]
                reactbi_mask = self.masks[ReactionActionType.ReactBi][i]
                num_reactuni = reactuni_mask.sum().item()
                num_reactbi = reactbi_mask.sum().item()

                self.logits[0][i] = 0
                if num_reactuni > 0:
                    self.logits[1][i, reactuni_mask] = -math.log(num_reactuni)
                if num_reactbi > 0:
                    offset = 2
                    for rxn_idx in range(self.ctx.num_bimolecular_rxns):
                        for block_is_first in (True, False):
                            if reactbi_mask[rxn_idx, int(block_is_first)]:
                                num_blocks = self.logits[offset].shape[1]
                                self.logits[offset][i, None] = -(math.log(num_reactbi) + math.log(num_blocks))
                            offset += 1

        # NOTE: Softmax temperature used when sampling
        if sample_temp != 1:
            self.logits = [logit / sample_temp for logit in self.logits]

        # NOTE: Use the Gumbel trick to sample categoricals
        gumbel = []
        for i, logit in enumerate(self.logits):
            if (i == 0) and (self.traj_indices[0] < min_len):
                gumbel.append(torch.full_like(logit, -1e6))
            else:
                noise = torch.rand_like(logit)
                gumbel.append(logit - (-noise.log()).log())
        argmax = self.argmax(x=gumbel)  # tuple of action type, action idx

        actions: list[ReactionActionIdx] = []
        for idx1, idx2 in argmax:
            rxn_idx = block_idx = block_is_first = None
            if idx1 == 0:
                t = ReactionActionType.Stop
            elif idx1 == 1:
                t = ReactionActionType.ReactUni
                rxn_idx = idx2
            else:
                t = ReactionActionType.ReactBi
                idx1 = idx1 - 2
                block_idx = block_indices_reactbi[idx1][idx2]
                rxn_idx, block_is_first = idx1 // 2, bool(idx1 % 2 == 0)
            type_idx = self.types.index(t)
            actions.append(get_action_idx(type_idx, rxn_idx, block_idx, block_is_first))
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

    def log_prob_after_sampling(self, actions: list[ReactionActionIdx]) -> torch.Tensor:
        """Access the log-probability of actions after sampling for entropy estimation"""
        logits = torch.cat(self.logits, dim=-1)
        max_logits = logits.detach().max(dim=-1, keepdim=True).values
        logZ = torch.logsumexp(logits - max_logits, dim=-1) + max_logits.squeeze(-1)
        action_logits = self.cal_action_logits(actions)
        return action_logits - logZ

    def cal_action_logits(self, actions: list[ReactionActionIdx]):
        # NOTE: placeholder of action_logits
        action_logits = torch.empty(len(actions), device=self.dev)
        for i, action in enumerate(actions):
            type_idx, rxn_idx, block_idx, block_is_first = action
            t = self.types[type_idx]
            if t is ReactionActionType.AddFirstReactant:
                block_emb = self.model.block_mlp(self.ctx.get_block_data([int(block_idx)], self.dev).view(-1))
                logit = self.model.single_hook_add_first_reactant(self.emb[i], block_emb)
            elif t is ReactionActionType.Stop:
                logit = self.model.single_hook_stop(self.emb[i])
            elif t is ReactionActionType.ReactUni:
                self.masks[t][i, rxn_idx] = True
                logit = self.model.single_hook_reactuni(self.emb[i], rxn_idx)
            elif t is ReactionActionType.ReactBi:
                self.masks[t][i, rxn_idx, int(block_is_first)] = True
                block_emb = self.model.block_mlp(self.ctx.get_block_data([int(block_idx)], self.dev).view(-1))
                logit = self.model.single_hook_reactbi(self.emb[i], rxn_idx, block_is_first, block_emb)
            else:
                raise ValueError
            action_logits[i] = logit
        return action_logits

    def log_prob(self, actions: list[ReactionActionIdx], action_sampler: ActionSamplingPolicy) -> torch.Tensor:
        """Access the log-probability of actions"""
        assert len(actions) == self.num_graphs, f"num_graphs: {self.num_graphs}, num_actions: {len(actions)}"
        if self.fwd:
            logZ = self.cal_logZ(action_sampler)
            action_logits = self.cal_action_logits(actions)
            action_logits = action_logits - logZ
            return action_logits.clamp(math.log(self._epsilon))
        else:
            raise NotImplementedError

    def cal_logZ(self, action_sampler: ActionSamplingPolicy):
        initial_indices = torch.where(self.traj_indices == 0)[0]
        later_indices = torch.where(self.traj_indices != 0)[0]
        logZ = torch.empty(self.num_graphs, device=self.dev)
        logZ[initial_indices] = self.cal_logZ_initial(action_sampler, initial_indices)
        logZ[later_indices] = self.cal_logZ_later(action_sampler, later_indices)
        return logZ

    def cal_logZ_initial(self, action_sampler: ActionSamplingPolicy, state_indices: torch.Tensor) -> torch.Tensor:
        num_mc_sampling = action_sampler.num_mc_sampling
        emb = self.emb[state_indices]
        block_space = action_sampler.get_space(ReactionActionType.AddFirstReactant)
        sampling_ratio = block_space.sampling_ratio
        if sampling_ratio < 1.0:
            # NOTE: MC Sampling
            logit_list = []
            for _ in range(num_mc_sampling):
                block_indices = block_space.sampling()
                block_emb = self.model.block_mlp(self.ctx.get_block_data(block_indices, self.dev))
                logit_list.append(self.model.hook_add_first_reactant(emb, block_emb))
            logits = torch.cat(logit_list, -1)
            max_logits = logits.detach().max(dim=-1, keepdim=True).values
            logZ = torch.logsumexp(logits - max_logits, dim=-1) - math.log(sampling_ratio * num_mc_sampling)
        else:
            block_emb = self.model.block_mlp(self.ctx.get_block_data(block_space.block_indices, self.dev))
            logits = self.model.hook_add_first_reactant(emb, block_emb)
            max_logits = logits.detach().max(dim=-1, keepdim=True).values
            logZ = torch.logsumexp(logits - max_logits, dim=-1)
        return logZ + max_logits.squeeze(-1)

    def cal_logZ_later(self, action_sampler: ActionSamplingPolicy, state_indices: torch.Tensor) -> torch.Tensor:
        emb = self.emb[state_indices]
        logit_list = []
        logit_list.append(self.model.hook_stop(emb))
        logit_list.append(self.model.hook_reactuni(emb, self.masks[ReactionActionType.ReactUni][state_indices]))

        num_mc_sampling = action_sampler.num_mc_sampling
        for rxn_idx in range(self.ctx.num_bimolecular_rxns):
            for block_is_first in (True, False):
                reactant_space = action_sampler.get_space(ReactionActionType.ReactBi, (rxn_idx, block_is_first))
                if reactant_space.num_blocks == 0:
                    continue
                mask = self.masks[ReactionActionType.ReactBi][state_indices, rxn_idx, int(block_is_first)]
                sampling_ratio = reactant_space.sampling_ratio
                if sampling_ratio < 1:
                    for _ in range(num_mc_sampling):  # NOTE: MC Sampling
                        block_indices = reactant_space.sampling()
                        block_emb = self.model.block_mlp(self.ctx.get_block_data(block_indices, self.dev))
                        logits = self.model.hook_reactbi(emb, rxn_idx, block_is_first, block_emb, mask)
                        logit_list.append(logits - math.log(sampling_ratio * num_mc_sampling))
                else:
                    block_emb = self.model.block_mlp(self.ctx.get_block_data(reactant_space.block_indices, self.dev))
                    logits = self.model.hook_reactbi(emb, rxn_idx, block_is_first, block_emb, mask)
                    logit_list.append(logits)
        logits = torch.cat(logit_list, dim=-1)
        max_logits = logits.detach().max(dim=-1, keepdim=True).values
        logZ = torch.logsumexp(logits - max_logits, dim=-1)
        return logZ + max_logits.squeeze(-1)
