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

""" For Comparison with RGFN """


class HierarchicalReactionActionCategorical(GraphActionCategorical):
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
        self.ctx: SynthesisEnvContext = model.env_ctx
        self.fwd = fwd
        self._epsilon = 1e-38

        if fwd:
            self.types: list[ReactionActionType] = self.ctx.action_type_order
        else:
            self.types: list[ReactionActionType] = self.ctx.bck_action_type_order

        self.logits: list[torch.Tensor] = []
        self.secondary_logits: list[torch.Tensor | None] = []
        self.action_logits: list[torch.Tensor] = []
        self.masks: dict[ReactionActionType, torch.Tensor] = {
            ReactionActionType.ReactUni: graphs[ReactionActionType.ReactUni.mask_name],
            ReactionActionType.ReactBi: graphs[ReactionActionType.ReactBi.mask_name],
        }
        self.batch = torch.arange(self.num_graphs, device=dev)
        self.random_action_mask: torch.Tensor | None = None

    def sample(
        self,
        action_sampler: ActionSamplingPolicy,
        onpolicy_temp: float = 1.0,
        sample_temp: float = 1.0,
        min_len: int = 2,
    ) -> list[ReactionActionIdx]:
        traj_idx = self.traj_indices[0]
        assert (self.traj_indices == traj_idx).all()  # For sampling, we use the same traj index
        if traj_idx == 0:
            return self.sample_initial_state(sample_temp)
        else:
            return self.sample_later_state(action_sampler, sample_temp, min_len)

    def sample_initial_state(self, sample_temp: float = 1.0):
        # NOTE: The first action in a trajectory is always AddFirstReactant (select a building block)
        type_idx = self.types.index(ReactionActionType.AddFirstReactant)
        block_emb = self.model.block_mlp(self.ctx.get_block_data(list(range(self.ctx.num_building_blocks)), self.dev))

        # NOTE: PlaceHolder
        logits = self.model.hook_add_first_reactant(self.emb, block_emb)
        self.logits.append(logits)

        # NOTE: Softmax temperature used when sampling
        if sample_temp != 1:
            self.logits = [logit / sample_temp for logit in self.logits]

        if self.random_action_mask is not None:
            self.logits[0][self.random_action_mask, :] = 0.0

        # NOTE: Use the Gumbel trick to sample categoricals
        noise = torch.rand_like(logits)
        gumbel = logits - (-noise.log()).log()
        argmax = self.argmax(x=[gumbel])
        return [get_action_idx(type_idx, block_idx=block_idx) for _, block_idx in argmax]

    def sample_later_state(
        self,
        action_sampler: ActionSamplingPolicy,
        sample_temp: float = 1.0,
        min_len: int = 2,
    ):
        self.logits.append(self.model.hook_stop(self.emb))
        self.logits.append(self.model.hook_reactbi_primary(self.emb, self.masks[ReactionActionType.ReactBi]))

        # NOTE: Softmax temperature used when sampling
        if sample_temp != 1:
            self.logits = [logit / sample_temp for logit in self.logits]

        if self.random_action_mask is not None:
            mask = self.random_action_mask
            random_action_idx = torch.where(mask)[0]
            for i in random_action_idx:
                self.logits[0][i] = 0
                self.logits[1][i, self.masks[ReactionActionType.ReactBi][i].view(-1)] = 0

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
        for i, (idx1, idx2) in enumerate(argmax):
            rxn_idx = block_idx = block_is_first = None
            if idx1 == 0:
                t = ReactionActionType.Stop
                secondary_logit = None
            else:
                t = ReactionActionType.ReactBi
                rxn_idx, block_is_first = idx2 // 2, bool(idx2 % 2)
                reactant_space = action_sampler.get_space_reactbi(rxn_idx, block_is_first)
                block_indices = reactant_space.block_indices
                assert len(block_indices) > 0

                if (self.random_action_mask is not None) and (self.random_action_mask[i] is True):
                    secondary_logit = torch.zeros((len(block_indices),), dtype=torch.float32)
                else:
                    block_emb = self.model.block_mlp(self.ctx.get_block_data(block_indices, self.dev))
                    secondary_logit = self.model.hook_reactbi_secondary(self.emb[i], rxn_idx, block_is_first, block_emb)
                    if sample_temp != 1:
                        secondary_logit = secondary_logit / sample_temp
                noise = torch.rand_like(secondary_logit)
                gumbel = secondary_logit - (-noise.log()).log()
                max_idx = int(gumbel.argmax())
                block_idx = block_indices[max_idx]
            self.secondary_logits.append(secondary_logit)
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
        """Access the log-probability of actions"""
        assert len(actions) == self.num_graphs, f"num_graphs: {self.num_graphs}, num_actions: {len(actions)}"
        if self.traj_indices[0] == 0:
            log_prob = self.log_prob_initial_after_sampling(actions)
        else:
            log_prob = self.log_prob_later_after_sampling(actions)
        return log_prob.clamp(math.log(self._epsilon))

    def log_prob_initial_after_sampling(self, actions: list[ReactionActionIdx]) -> torch.Tensor:
        logits = self.logits[0]
        max_logit = logits.detach().max(dim=-1, keepdim=True).values
        logZ = torch.logsumexp(logits - max_logit, dim=-1) + max_logit.squeeze(-1)

        action_logits = torch.empty((self.num_graphs,), device=self.dev)
        for i, aidx in enumerate(actions):
            type_idx, rxn_idx, block_idx, block_is_first = aidx
            action_logits[i] = logits[i, block_idx]
        return action_logits - logZ

    def log_prob_later_after_sampling(self, actions: list[ReactionActionIdx]) -> torch.Tensor:
        logits = torch.cat(self.logits, dim=-1)
        max_logits = logits.detach().max(dim=-1, keepdim=True).values
        logZ = torch.logsumexp(logits - max_logits, dim=-1) + max_logits.squeeze(-1)

        log_prob = torch.full((len(actions),), -torch.inf, device=self.dev)
        for i, aidx in enumerate(actions):
            type_idx, rxn_idx, block_idx, block_is_first = aidx
            if type_idx == 0:
                log_prob[i] = self.logits[0][i, 0]
                continue
            else:
                primary_logprob = self.logits[1][i, rxn_idx * 2 + int(block_is_first)] - logZ[i]

                secondary_logit = self.secondary_logits[i]
                assert secondary_logit is not None
                _max_logit = secondary_logit.detach().max(dim=-1, keepdim=True).values
                _logZ = torch.logsumexp(secondary_logit - _max_logit, dim=-1) + _max_logit.squeeze(-1)

                _single_block_emb = self.model.block_mlp(self.ctx.get_block_data([int(block_idx)], self.dev).view(-1))
                _action_logit = self.model.single_hook_reactbi_secondary(
                    self.emb[i], rxn_idx, block_is_first, _single_block_emb
                )
                secondary_logprob = _action_logit - _logZ
                log_prob[i] = primary_logprob + secondary_logprob
        return log_prob

    def log_prob(self, actions: list[ReactionActionIdx], action_sampler: ActionSamplingPolicy) -> torch.Tensor:
        """Access the log-probability of actions"""
        assert len(actions) == self.num_graphs, f"num_graphs: {self.num_graphs}, num_actions: {len(actions)}"
        if self.fwd:
            initial_indices = torch.where(self.traj_indices == 0)[0]
            later_indices = torch.where(self.traj_indices != 0)[0]
            log_prob = torch.empty(self.num_graphs, device=self.dev)
            log_prob[initial_indices] = self.log_prob_initial(actions, initial_indices)
            log_prob[later_indices] = self.log_prob_later(actions, action_sampler, later_indices)
            return log_prob.clamp(math.log(self._epsilon))
        else:
            raise NotImplementedError

    def log_prob_initial(self, actions: list[ReactionActionIdx], state_indices: torch.Tensor) -> torch.Tensor:
        emb = self.emb[state_indices]
        block_emb = self.model.block_mlp(self.ctx.get_block_data(list(range(self.ctx.num_building_blocks)), self.dev))
        log_prob = torch.full((emb.shape[0],), -torch.inf, device=self.dev)

        logits = self.model.hook_add_first_reactant(emb, block_emb)
        max_logits = logits.detach().max(dim=-1, keepdim=True).values
        logZ = torch.logsumexp(logits - max_logits, dim=-1)

        for i, j in enumerate(state_indices):
            type_idx, rxn_idx, block_idx, block_is_first = actions[j]
            log_prob[i] = logits[i, block_idx] - logZ[i]
        return log_prob

    def log_prob_later(
        self, actions: list[ReactionActionIdx], action_sampler: ActionSamplingPolicy, state_indices: torch.Tensor
    ) -> torch.Tensor:
        emb = self.emb[state_indices]
        stop_logit = self.model.hook_stop(emb)
        react_logit = self.model.hook_reactbi_primary(emb, self.masks[ReactionActionType.ReactBi][state_indices])

        logits = torch.cat([stop_logit, react_logit], dim=-1)
        max_logits = logits.detach().max(dim=-1, keepdim=True).values
        logZ = torch.logsumexp(logits - max_logits, dim=-1) + max_logits.squeeze(-1)

        log_prob = torch.full((emb.shape[0],), -torch.inf, device=self.dev)
        for i, j in enumerate(state_indices):
            type_idx, rxn_idx, block_idx, block_is_first = actions[j]
            if type_idx == 0:
                log_prob[i] = stop_logit[i, 0] - logZ[i]
            else:
                primary_action_logprob = react_logit[i, rxn_idx * 2 + int(block_is_first)] - logZ[i]
                secondary_action_logprob = self.cal_secondary_log_prob(
                    emb[i], rxn_idx, block_idx, block_is_first, action_sampler
                )
                log_prob[i] = primary_action_logprob + secondary_action_logprob

        return log_prob

    def cal_secondary_log_prob(self, emb, rxn_idx, block_idx, block_is_first, action_sampler):
        block_space = action_sampler.get_space_reactbi(int(rxn_idx), bool(block_is_first))
        block_emb = self.model.block_mlp(self.ctx.get_block_data(block_space.block_indices, self.dev))
        logits = self.model.hook_reactbi_secondary(emb, rxn_idx, bool(block_is_first), block_emb)
        max_logits = logits.detach().max(dim=-1, keepdim=True).values
        logZ = torch.logsumexp(logits - max_logits, dim=-1)

        single_block_emb = self.model.block_mlp(self.ctx.get_block_data([int(block_idx)], self.dev).view(-1))
        action_logit = self.model.single_hook_reactbi_secondary(emb, rxn_idx, bool(block_is_first), single_block_emb)
        return action_logit - logZ
