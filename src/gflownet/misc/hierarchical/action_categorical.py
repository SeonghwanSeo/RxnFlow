import math
import torch
import torch_geometric.data as gd

from typing import Dict, List

from gflownet.envs.graph_building_env import GraphActionCategorical
from gflownet.envs.synthesis.action import (
    ReactionActionIdx,
    ReactionActionType,
    get_action_idx,
)
from gflownet.envs.synthesis.action_sampling import ActionSamplingPolicy
from gflownet.envs.synthesis.env_context import SynthesisEnvContext


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
            self.types: List[ReactionActionType] = self.ctx.action_type_order
        else:
            self.types: List[ReactionActionType] = self.ctx.bck_action_type_order

        self.logits: List[torch.Tensor] = []
        self.secondary_logits: List[Optional[torch.Tensor]] = []
        self.masks: Dict[ReactionActionType, torch.Tensor] = {
            ReactionActionType.ReactUni: graphs[ReactionActionType.ReactUni.mask_name],
            ReactionActionType.ReactBi: graphs[ReactionActionType.ReactBi.mask_name],
        }
        self.batch = torch.arange(self.num_graphs, device=dev)

    def sample(
        self,
        action_sampler: ActionSamplingPolicy,
        onpolicy_temp: float = 0.0,
        sample_temp: float = 1.0,
    ) -> List[ReactionActionIdx]:
        """
        Samples from the categorical distribution
        sample_temp:
            Softmax temperature used when sampling
        onpolicy_temp:
            0.0: importance sampling
            1.0: on-policy sampling
        """
        # TODO: Reweighting for Online Policy?
        traj_idx = self.traj_indices[0]
        assert (self.traj_indices == traj_idx).all()  # For sampling, we use the same traj index
        if traj_idx == 0:
            return self.sample_initial_state(action_sampler, sample_temp)
        else:
            return self.sample_later_state(action_sampler, sample_temp)

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
        sample_temp: float = 1.0,
    ):
        self.logits.append(self.model.hook_stop(self.emb))
        self.logits.append(self.model.hook_reactuni(self.emb, self.masks[ReactionActionType.ReactUni]))
        self.logits.append(self.model.hook_reactbi_primary(self.emb, self.masks[ReactionActionType.ReactBi]))

        # NOTE: Softmax temperature used when sampling
        if sample_temp != 1:
            self.logits = [logit / sample_temp for logit in self.logits]

        # NOTE: Use the Gumbel trick to sample categoricals
        gumbel = []
        for i, logit in enumerate(self.logits):
            if i == 0 and self.traj_indices[0] == 1:
                # SynFlowNet, Mask that the generated molecule is building block.
                gumbel.append(torch.full_like(logit, -torch.inf))
            else:
                noise = torch.rand_like(logit)
                gumbel.append(logit - (-noise.log()).log())
        argmax = self.argmax(x=gumbel)  # tuple of action type, action idx

        actions: List[ReactionActionIdx] = []
        for i, (idx1, idx2) in enumerate(argmax):
            rxn_idx = block_idx = block_is_first = None
            if idx1 == 0:
                t = ReactionActionType.Stop
                secondary_logit = None
            elif idx1 == 1:
                t = ReactionActionType.ReactUni
                rxn_idx = idx2
                secondary_logit = None
            else:
                t = ReactionActionType.ReactBi
                rxn_idx, block_is_first = idx2 // 2, bool(idx2 % 2)
                reactant_space = action_sampler.get_space_reactbi(rxn_idx, block_is_first)
                block_indices = reactant_space.sampling()
                assert len(block_indices) > 0
                block_emb = self.model.block_mlp(self.ctx.get_block_data(block_indices, self.dev))
                secondary_logit = self.model.hook_reactbi_secondary(self.emb[i], rxn_idx, block_is_first, block_emb)

                noise = torch.rand_like(secondary_logit)
                gumbel = secondary_logit - (-noise.log()).log()
                max_idx = int(gumbel.argmax())
                block_idx = block_indices[max_idx]
            self.secondary_logits.append(secondary_logit)

            type_idx = self.types.index(t)
            actions.append(get_action_idx(type_idx, rxn_idx, block_idx, block_is_first))
        return actions

    def argmax(self, x: List[torch.Tensor]) -> List[tuple[int, int]]:
        # for each graph in batch and for each action type, get max value and index
        max_per_type = [torch.max(tensor, dim=1) for tensor in x]
        max_values_per_type = [pair[0] for pair in max_per_type]
        argmax_indices_per_type = [pair[1] for pair in max_per_type]
        _, type_indices = torch.max(torch.stack(max_values_per_type), dim=0)
        action_indices = torch.gather(torch.stack(argmax_indices_per_type), 0, type_indices.unsqueeze(0)).squeeze(0)
        argmax_pairs = list(zip(type_indices.tolist(), action_indices.tolist(), strict=True))  # action type, action idx
        return argmax_pairs

    def log_prob_after_sampling(self, actions: List[ReactionActionIdx]) -> torch.Tensor:
        """Access the log-probability of actions"""
        assert len(actions) == self.num_graphs, f"num_graphs: {self.num_graphs}, num_actions: {len(actions)}"
        if self.traj_indices[0] == 0:
            log_prob = self.log_prob_initial_after_sampling(actions)
        else:
            log_prob = self.log_prob_later_after_sampling(actions)
        return log_prob.clamp(math.log(self._epsilon))

    def log_prob_initial_after_sampling(self, actions: List[ReactionActionIdx]) -> torch.Tensor:
        logit = self.logits[0]
        max_logit = logit.detach().max(dim=-1, keepdim=True).values
        logZ = torch.logsumexp(logit - max_logit, dim=-1) + max_logit.squeeze(-1)

        action_logits = torch.empty((self.num_graphs,), device=self.dev)
        for i, aidx in enumerate(actions):
            type_idx, rxn_idx, block_idx, block_is_first = aidx
            single_block_emb = self.model.block_mlp(self.ctx.get_block_data([int(block_idx)], self.dev).view(-1))
            action_logits[i] = self.model.single_hook_add_first_reactant(self.emb[i], single_block_emb)

        return action_logits - logZ

    def log_prob_later_after_sampling(self, actions: List[ReactionActionIdx]) -> torch.Tensor:
        logits = torch.cat(self.logits, dim=-1)
        max_logits = logits.detach().max(dim=-1, keepdim=True).values
        logZ = torch.logsumexp(logits - max_logits, dim=-1) + max_logits.squeeze(-1)

        action_logits = torch.full((len(actions),), -torch.inf, device=self.dev)
        for i, action in enumerate(actions):
            type_idx, rxn_idx, block_idx, block_is_first = action
            t = self.types[type_idx]
            if t is ReactionActionType.Stop:
                action_logits[i] = self.logits[0][i, 0]
            elif t is ReactionActionType.ReactUni:
                action_logits[i] = self.logits[1][i, rxn_idx]
            elif t is ReactionActionType.ReactBi:
                action_logits[i] = self.logits[2][i, rxn_idx * 2 + int(block_is_first)]
            else:
                raise ValueError
        log_prob = action_logits - logZ
        del logits, max_logits, action_logits

        for i, aidx in enumerate(actions):
            type_idx, rxn_idx, block_idx, block_is_first = aidx
            if self.types[type_idx] is ReactionActionType.ReactBi:
                secondary_logit = self.secondary_logits[i]
                assert secondary_logit is not None
                max_logit = secondary_logit.detach().max(dim=-1, keepdim=True).values
                logZ = torch.logsumexp(secondary_logit - max_logit, dim=-1) + max_logit.squeeze(-1)
                single_block_emb = self.model.block_mlp(self.ctx.get_block_data([int(block_idx)], self.dev).view(-1))
                action_logit = self.model.single_hook_reactbi_secondary(
                    self.emb[i], rxn_idx, block_is_first, single_block_emb
                )
                log_prob[i] = log_prob[i] + (action_logit - logZ)
        return log_prob

    def log_prob(self, actions: List[ReactionActionIdx], action_sampler: ActionSamplingPolicy) -> torch.Tensor:
        """Access the log-probability of actions"""
        assert len(actions) == self.num_graphs, f"num_graphs: {self.num_graphs}, num_actions: {len(actions)}"
        if self.fwd:
            initial_indices = torch.where(self.traj_indices == 0)[0]
            later_indices = torch.where(self.traj_indices != 0)[0]
            log_prob = torch.empty(self.num_graphs, device=self.dev)
            log_prob[initial_indices] = self.log_prob_initial(actions, action_sampler, initial_indices)
            log_prob[later_indices] = self.log_prob_later(actions, action_sampler, later_indices)
            return log_prob.clamp(math.log(self._epsilon))
        else:
            raise NotImplementedError

    def log_prob_initial(
        self, actions: List[ReactionActionIdx], action_sampler: ActionSamplingPolicy, state_indices: torch.Tensor
    ) -> torch.Tensor:
        emb = self.emb[state_indices]
        log_prob = torch.full((emb.shape[0],), -torch.inf, device=self.dev)

        block_space = action_sampler.get_space_addfirstreactant()
        sampling_ratio = block_space.sampling_ratio
        if sampling_ratio < 1.0:
            num_mc_sampling = action_sampler.num_mc_sampling
            raise NotImplementedError(
                "This code block is implemented but unused. We do not guarantee that this code is work"
            )
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

        for i, j in enumerate(state_indices):
            type_idx, rxn_idx, block_idx, block_is_first = actions[j]
            _block_emb = self.model.block_mlp(self.ctx.get_block_data([int(block_idx)], self.dev).view(-1))
            action_logit = self.model.single_hook_add_first_reactant(emb[i], _block_emb)
            log_prob[i] = action_logit - logZ[i]
        return log_prob

    def log_prob_later(
        self, actions: List[ReactionActionIdx], action_sampler: ActionSamplingPolicy, state_indices: torch.Tensor
    ) -> torch.Tensor:
        emb = self.emb[state_indices]
        logit_list = []
        logit_list.append(self.model.hook_stop(emb))
        logit_list.append(self.model.hook_reactuni(emb, self.masks[ReactionActionType.ReactUni][state_indices]))
        logit_list.append(self.model.hook_reactbi_primary(emb, self.masks[ReactionActionType.ReactBi][state_indices]))
        logits = torch.cat(logit_list, dim=-1)
        max_logits = logits.detach().max(dim=-1, keepdim=True).values
        logZ = torch.logsumexp(logits - max_logits, dim=-1) + max_logits.squeeze(-1)

        log_prob = torch.full((emb.shape[0],), -torch.inf, device=self.dev)
        for i, j in enumerate(state_indices):
            type_idx, rxn_idx, block_idx, block_is_first = actions[j]
            t = self.types[type_idx]
            if t is ReactionActionType.Stop:
                action_logit = logit_list[0][i, 0]
            elif t is ReactionActionType.ReactUni:
                action_logit = logit_list[1][i, rxn_idx]
            elif t is ReactionActionType.ReactBi:
                action_logit = logit_list[2][i, rxn_idx * 2 + int(block_is_first)]
            else:
                raise ValueError
            log_prob[i] = action_logit - logZ[i]

            if t is ReactionActionType.ReactBi:
                log_prob[i] = log_prob[i] + self.cal_secondary_log_prob(
                    emb[i], rxn_idx, block_idx, block_is_first, action_sampler
                )

        return log_prob

    def cal_secondary_log_prob(self, emb, rxn_idx, block_idx, block_is_first, action_sampler):
        num_mc_sampling = action_sampler.num_mc_sampling
        block_space = action_sampler.get_space_reactbi(int(rxn_idx), bool(block_is_first))
        sampling_ratio = block_space.sampling_ratio
        if sampling_ratio < 1.0:
            raise NotImplementedError(
                "This code block is implemented but unused. We do not guarantee that this code is work"
            )
            logit_list = []
            for _ in range(num_mc_sampling):
                block_indices = block_space.sampling()
                block_emb = self.model.block_mlp(self.ctx.get_block_data(block_indices, self.dev))
                logit_list.append(self.model.hook_reactbi_secondary(emb, rxn_idx, bool(block_is_first), block_emb))
            logits = torch.cat(logit_list, -1)
            max_logits = logits.detach().max(dim=-1, keepdim=True).values
            logZ = torch.logsumexp(logits - max_logits, dim=-1) - math.log(sampling_ratio * num_mc_sampling)
        else:
            block_emb = self.model.block_mlp(self.ctx.get_block_data(block_space.block_indices, self.dev))
            logits = self.model.hook_reactbi_secondary(emb, rxn_idx, bool(block_is_first), block_emb)
            max_logits = logits.detach().max(dim=-1, keepdim=True).values
            logZ = torch.logsumexp(logits - max_logits, dim=-1)

        single_block_emb = self.model.block_mlp(self.ctx.get_block_data([int(block_idx)], self.dev).view(-1))
        action_logit = self.model.single_hook_reactbi_secondary(emb, rxn_idx, bool(block_is_first), single_block_emb)
        return action_logit - logZ
