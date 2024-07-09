import math
import random

from rdkit.Chem import Mol
import torch
import torch_geometric.data as gd

from typing import Dict, List, Optional, Union

from gflownet.envs.graph_building_env import Graph, GraphActionCategorical
from gflownet.envs.synthesis.action import (
    ReactionActionIdx,
    ReactionActionType,
    get_action_idx,
)
from gflownet.envs.synthesis.env import SynthesisEnv
from gflownet.envs.synthesis.env_context import SynthesisEnvContext


class ReactionActionCategorical(GraphActionCategorical):
    def __init__(
        self,
        graphs: gd.Batch,
        raw_logits: Dict[ReactionActionType, torch.Tensor],
        masks: Dict[ReactionActionType, Optional[torch.Tensor]],
        emb: Dict[str, torch.Tensor],
        model: torch.nn.Module,
        fwd: bool,
    ):
        self.model = model
        self.graphs = graphs
        self.num_graphs = graphs.num_graphs
        self.traj_indices = graphs.traj_idx
        self.graph_embedding: torch.Tensor = emb["graph"]
        self.dev = dev = self.graph_embedding.device
        self.ctx: SynthesisEnvContext = model.env_ctx
        self.env: SynthesisEnv = self.ctx.env
        self.fwd = fwd
        self._epsilon = 1e-38

        if fwd:
            self.types: List[ReactionActionType] = self.ctx.action_type_order
            self.primary_action_types: List[ReactionActionType] = self.ctx.primary_action_type_order
            self.secondary_action_types: List[ReactionActionType] = self.ctx.secondary_action_type_order
        else:
            self.types: List[ReactionActionType] = self.ctx.bck_action_type_order
            self.primary_action_types: List[ReactionActionType] = self.ctx.primary_bck_action_type_order
            self.secondary_action_types: List[ReactionActionType] = self.ctx.secondary_bck_action_type_order

        self.logits: Dict[ReactionActionType, torch.Tensor] = raw_logits
        self.masks: Dict[ReactionActionType, Optional[torch.Tensor]] = masks

        self.batch = torch.arange(self.num_graphs, device=dev)

        self.logprobs: Optional[Dict[ReactionActionType, torch.Tensor]] = None

        self.setuped: bool = False

    def _compute_batchwise_max(self) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute the argmax for each batch element in the batch of logits.

        Parameters
        ----------

        Returns
        -------
        overall_max_per_graph: Tensor
            A tensor of shape (n,m) where n is the number of graphs in the batch.
            Each element is the max value of the logits for the corresponding graph.
            m is 1 if there is one hierarchy of actions, and 2 if there are two hierarchies.
        """
        # Compute max for primary logits
        max_per_primary_type = [torch.max(self.logits[typ], dim=1)[0] for typ in self.primary_action_types]
        overall_max_per_graph_primary, _ = torch.max(torch.stack(max_per_primary_type), dim=0)

        # Compute max for secondary logits if they exist
        if len(self.secondary_action_types) > 0:
            max_per_secondary_type = [torch.max(self.logits[typ], dim=1)[0] for typ in self.secondary_action_types]
            overall_max_per_graph_secondary, _ = torch.max(torch.stack(max_per_secondary_type), dim=0)
            return overall_max_per_graph_primary, overall_max_per_graph_secondary
        else:
            return overall_max_per_graph_primary, None

    def logsoftmax(self) -> Dict[ReactionActionType, torch.Tensor]:
        """Compute log-probabilities given logits"""
        if self.logprobs is not None:
            return self.logprobs

        # we need to compute the log-probabilities (1) for the primary logits and (2) for the secondary logits
        max_logits_primary, max_logits_secondary = self._compute_batchwise_max()

        # correct primary logits by max and exponentiate
        corr_logits_primary = {
            typ: self.logits[typ] - max_logits_primary.view(-1, 1) for typ in self.primary_action_types
        }
        exp_logits_primary = [i.exp().clamp(self._epsilon) for i in corr_logits_primary.values()]

        # compute logZ for primary logits
        merged_exp_logits_primary = torch.cat(exp_logits_primary, dim=1)
        logZ_primary = merged_exp_logits_primary.sum(dim=1).log()

        # compute log-probabilities for primary logits
        logprobs = {typ: logit - logZ_primary.view(-1, 1) for typ, logit in corr_logits_primary.items()}

        # if there are secondary logits, compute log-probabilities for them
        if max_logits_secondary is not None:
            corr_logits_secondary = {
                typ: self.logits[typ] - max_logits_secondary.view(-1, 1) for typ in self.secondary_action_types
            }
            exp_logits_secondary = [i.exp().clamp(self._epsilon) for i in corr_logits_secondary.values()]
            merged_exp_logits_secondary = torch.cat(exp_logits_secondary, dim=1)
            logZ_secondary = merged_exp_logits_secondary.sum(dim=1).log()
            for key, value in corr_logits_secondary.items():
                logprobs[key] = value - logZ_secondary.view(-1, 1)
        self.logprobs = logprobs
        return self.logprobs

    def sample(
        self,
        block_indices: List[int],
        block_emb: Optional[torch.Tensor],
    ) -> List[ReactionActionIdx]:
        """Samples from the categorical distribution"""

        if block_emb is None:
            block_fp = self.ctx.get_block_data(block_indices, self.dev)
            block_emb = self.model.block_mlp(block_fp)
        precomputed_bb_mask = torch.from_numpy(self.ctx.precomputed_bb_masks[:, block_indices, :])
        precomputed_bb_mask = precomputed_bb_mask.to(self.dev)  # [R2, B, 2]

        # NOTE: Masking
        for t in self.primary_action_types:
            mask = self.masks.get(t)
            self.logits[t] = self._mask(self.logits[t], mask)

        # NOTE: Bimolecular Reactant mask
        mol_mask = self.graphs.react_bi_mask_detail  # [N, R2, 2]
        bireaction_mask_detail = mol_mask.unsqueeze(2) & precomputed_bb_mask.unsqueeze(0)  # [N, R2, B, 2]

        if not self.setuped:
            # NOTE: PlaceHolder
            if ReactionActionType.AddFirstReactant in self.secondary_action_types:
                self.logits[ReactionActionType.AddFirstReactant] = torch.full(
                    (self.num_graphs, len(block_indices)), -torch.inf, device=self.dev
                )
            if ReactionActionType.AddReactant in self.secondary_action_types:
                self.logits[ReactionActionType.AddReactant] = torch.full(
                    (self.num_graphs, len(block_indices)), -torch.inf, device=self.dev
                )

        traj_idx = self.traj_indices[0]
        assert (self.traj_indices == traj_idx).all()  # For sampling, we use the same traj index

        # NOTE: The first action in a trajectory is always AddFirstReactant (select a building block)
        if traj_idx == 0:
            type_idx = self.types.index(ReactionActionType.AddFirstReactant)
            logits = self.logits[ReactionActionType.AddFirstReactant]
            if not self.setuped:
                for i in range(self.num_graphs):
                    logits[i] = self.model.add_first_reactant_hook(self.graph_embedding[i], block_emb)
            noise = torch.rand_like(logits)
            gumbel = logits - (-noise.log()).log()
            argmax = self.argmax(x=[gumbel])
            self.setuped = True
            return [get_action_idx(type_idx, block_idx=block_idx) for _, block_idx in argmax]

        # NOTE: Use the Gumbel trick to sample categoricals
        gumbel = []
        for t in self.primary_action_types:
            logit = self.logits[t]

            # NOTE: Additional Masking
            if t == ReactionActionType.Stop and traj_idx == 1:
                # NOTE: Mask the Stop for the second action
                logit = torch.full_like(logit, -torch.inf)

            elif t == ReactionActionType.ReactBi and bireaction_mask_detail is not None:
                # NOTE: Mask the invalid bimolecular reaction
                masked_logit = torch.full_like(logit, -torch.inf)
                logit = torch.where(bireaction_mask_detail.flatten(2).any(-1), logit, masked_logit)

            noise = torch.rand_like(logit)
            gumbel.append(logit - (-noise.log()).log())
        argmax = self.argmax(x=gumbel)  # tuple of action type, action idx

        actions: List[ReactionActionIdx] = []
        for i, (type_idx, rxn_idx) in enumerate(argmax):
            t = self.types[type_idx]
            if t == ReactionActionType.Stop:
                actions.append(get_action_idx(type_idx, is_stop=True))
            elif t == ReactionActionType.ReactUni:
                actions.append(get_action_idx(type_idx, rxn_idx=rxn_idx))
            elif t == ReactionActionType.ReactBi:  # sample reactant
                reactant_mask_detail = bireaction_mask_detail[i, rxn_idx]  # [B, 2]
                reactant_mask = torch.any(reactant_mask_detail, dim=1)
                if not torch.any(reactant_mask):
                    # NOTE: This script is not necessary.
                    # Just include it to prevent error caused by bug.
                    stop_idx = self.types.index(ReactionActionType.Stop)
                    actions.append(get_action_idx(stop_idx, is_stop=True))
                    continue

                if not self.setuped:
                    # NOTE: Call the hook to get the logits for the AddReactant action
                    logit = self.model.add_reactant_hook(rxn_idx, self.graph_embedding[i], block_emb)
                    logit = self._mask(logit, reactant_mask)
                    self.logits[ReactionActionType.AddReactant][i] = logit
                logit = self.logits[ReactionActionType.AddReactant][i]

                noise = torch.rand_like(logit)
                gumbel = logit - (-noise.log()).log()
                max_idx = int(gumbel.argmax())

                # NOTE: Check what is block: (mol, block) | (block, mol)
                assert reactant_mask[max_idx], "This index should not be masked"
                block_is_first, block_is_second = reactant_mask_detail[max_idx]
                if block_is_first and block_is_second:
                    block_is_first = random.random() < 0.5
                actions.append(
                    get_action_idx(type_idx, rxn_idx=rxn_idx, block_idx=max_idx, block_is_first=bool(block_is_first))
                )
            else:
                raise ValueError
        self.setuped = True
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

    def _mask(self, x: torch.Tensor, m: Optional[torch.Tensor]) -> torch.Tensor:
        if m is None:
            return x
        assert m.dtype == torch.bool
        return x.masked_fill(torch.logical_not(m), -torch.inf)

    def log_prob(
        self,
        actions: List[ReactionActionIdx],
        block_indices: List[int],
        block_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Access the log-probability of actions"""
        assert len(actions) == self.num_graphs, f"num_graphs: {self.num_graphs}, num_actions: {len(actions)}"
        if self.fwd:
            return self.log_prob_fwd(actions, block_indices, block_emb)
        else:
            return self.log_prob_bck(actions)

    def log_prob_fwd(
        self,
        actions: List[ReactionActionIdx],
        block_indices: List[int],
        block_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Access the log-probability of forward actions"""
        if not self.setuped:
            self.setup_mask_fwd(actions, block_indices, block_emb)
        self.logprobs = self.logsoftmax()

        # NOTE: placeholder of log_action_probs and log_action_space_sizes
        log_action_probs = torch.empty(len(actions), device=self.dev)

        for i, (traj_idx, action) in enumerate(zip(self.traj_indices, actions, strict=True)):
            type_idx, is_stop, rxn_idx, block_idx, block_is_first = action
            t = self.types[type_idx]
            if t is ReactionActionType.Stop:
                assert is_stop
                log_prob = self.logprobs[t][i]
            elif t is ReactionActionType.AddFirstReactant:
                log_prob = self.logprobs[t][i, block_idx]
            elif t is ReactionActionType.ReactUni:
                log_prob = self.logprobs[t][i, rxn_idx]
            elif t is ReactionActionType.ReactBi:
                bireact_log_prob = self.logprobs[t][i, rxn_idx]
                addreactant_log_prob = self.logprobs[ReactionActionType.AddReactant][i, block_idx]
                log_prob = bireact_log_prob + addreactant_log_prob
            else:
                raise ValueError
            log_action_probs[i] = log_prob
        return log_action_probs.clamp(math.log(self._epsilon))

    def log_prob_bck(self, actions: List[ReactionActionIdx]) -> torch.Tensor:
        """Access the log-probability of backward actions"""
        if not self.setuped:
            self.setup_mask_bck(actions)
        self.logprobs = self.logsoftmax()

        # NOTE: placeholder of log_action_probs and log_action_space_sizes
        log_action_probs = torch.empty(len(actions), device=self.dev)

        for i, (traj_idx, action) in enumerate(zip(self.traj_indices, actions, strict=True)):
            type_idx, is_stop, rxn_idx, block_idx, block_is_first = action
            t = self.types[type_idx]
            if is_stop:
                log_prob = torch.tensor([0.0], device=self.dev, dtype=torch.float)
            elif t is ReactionActionType.BckRemoveFirstReactant:
                log_prob = self.logprobs[t][i, block_idx]
            elif t is ReactionActionType.BckReactUni:
                log_prob = self.logprobs[t][i, rxn_idx]
            elif t is ReactionActionType.BckReactBi:
                log_prob = self.logprobs[t][i, rxn_idx]
                if traj_idx == 1:  # NOTE: when traj_idx == 1, block + block
                    log_prob = log_prob - math.log(2)
            else:
                raise ValueError
            log_action_probs[i] = log_prob
        return log_action_probs.clamp(math.log(self._epsilon))

    def setup_mask_fwd(
        self,
        actions: List[ReactionActionIdx],
        block_indices: List[int],
        block_emb: Optional[torch.Tensor] = None,
    ):
        if block_emb is None:
            block_fp = self.ctx.get_block_data(block_indices, self.dev)
            block_emb = self.model.block_mlp(block_fp)
        precomputed_bb_mask = torch.from_numpy(self.ctx.precomputed_bb_masks[:, block_indices, :])
        precomputed_bb_mask = precomputed_bb_mask.to(self.dev)  # [R2, B, 2]

        # NOTE: PlaceHolder
        if ReactionActionType.AddFirstReactant in self.secondary_action_types:
            self.logits[ReactionActionType.AddFirstReactant] = torch.full(
                (self.num_graphs, len(block_indices)), -torch.inf, device=self.dev
            )
        if ReactionActionType.AddReactant in self.secondary_action_types:
            self.logits[ReactionActionType.AddReactant] = torch.full(
                (self.num_graphs, len(block_indices)), -torch.inf, device=self.dev
            )

        for i, (traj_idx, action) in enumerate(zip(self.traj_indices, actions)):
            type_idx, is_stop, rxn_idx, block_local_idx, block_is_first = action
            t = self.types[type_idx]

            if t is ReactionActionType.AddFirstReactant:
                logit = self.model.add_first_reactant_hook(self.graph_embedding[i], block_emb)
                self.logits[ReactionActionType.AddFirstReactant][i] = logit

            elif t is ReactionActionType.ReactUni:
                mask = self.masks.get(t)
                if mask is not None:
                    mask[i, rxn_idx] = True

            elif t is ReactionActionType.ReactBi:  # secondary logits were computed
                mask = self.masks.get(t)
                if mask is not None:
                    mask[i, rxn_idx] = True

                mol_mask = self.graphs.react_bi_mask_detail[i, rxn_idx, :].reshape(1, 2)
                reactant_mask_detail = mol_mask & precomputed_bb_mask[rxn_idx]  # [R2, B, 2]
                reactant_mask = reactant_mask_detail.any(dim=1)  # [R2, B]
                reactant_mask[block_local_idx] = True

                logit = self.model.add_reactant_hook(rxn_idx, self.graph_embedding[i], block_emb)
                logit = self._mask(logit, reactant_mask)
                self.logits[ReactionActionType.AddReactant][i] = logit
        for t in self.primary_action_types:  # NOTE: Masking
            mask = self.masks.get(t)
            self.logits[t] = self._mask(self.logits[t], mask)

    def setup_mask_bck(self, actions: List[ReactionActionIdx]):
        for i, action in enumerate(actions):
            type_idx, is_stop, rxn_idx, block_local_idx, block_is_first = action
            t = self.types[type_idx]
            if t in [ReactionActionType.BckReactUni, ReactionActionType.BckReactBi]:
                mask = self.masks.get(t)
                if mask is not None:
                    mask[i, rxn_idx] = True
        for t in self.primary_action_types:  # NOTE: Masking
            mask = self.masks.get(t)
            self.logits[t] = self._mask(self.logits[t], mask)

    def reweight_logp(
        self,
        log_p_F,
        actions,
        rev_traj_lens_list: List[List[int]],
        rev_traj_lens_partial_list: List[List[int]],
        block_indices: List[int],
    ):
        """Access the log-probability of forward actions"""
        assert self.logprobs is not None
        if len(block_indices) == self.ctx.num_building_blocks:
            return log_p_F

        logp_stop = self.logprobs[ReactionActionType.Stop]
        logp_react_uni = self.logprobs[ReactionActionType.ReactUni]
        p_stop = logp_stop.exp().squeeze(-1)
        p_react_uni = logp_react_uni.exp().sum(-1)
        p_react_bi = (1 - p_stop - p_react_uni).clamp(self._epsilon)

        r_block = len(block_indices) / self.ctx.num_building_blocks
        expansion_rate = self.ctx.env.num_average_possible_actions
        expansion_rate_partial = self.ctx.env.num_average_possible_actions * r_block

        converted_log_p_F = torch.empty_like(log_p_F)
        assert log_p_F.size(0) == len(actions)

        max_traj_len = len(rev_traj_lens_list[0]) - 1

        # NOTE: Re-weight
        for i, (traj_idx, action) in enumerate(zip(self.traj_indices, actions)):
            type_idx, is_stop, _, _, _ = action
            action_type = self.types[type_idx]
            if traj_idx == 0:
                assert action_type is ReactionActionType.AddFirstReactant
                converted_log_p_F[i] = log_p_F[i] + math.log(r_block)

            else:
                rev_traj_lens = rev_traj_lens_list[i]
                rev_traj_lens_partial = rev_traj_lens_partial_list[i]
                assert rev_traj_lens[0] == rev_traj_lens_partial[0] == 0  # non-initial-state

                num_traj_stop = sum(rev_traj_lens)
                num_traj_stop_partial = sum(rev_traj_lens_partial)

                num_traj_nonstop = sum(
                    rev_traj_lens[_t] * sum(expansion_rate**_i for _i in range(max_traj_len - _t))
                    for _t in range(max_traj_len)  # t=0~N-1, i=0~N-t-1
                )
                num_traj_nonstop_partial = sum(
                    rev_traj_lens_partial[_t] * sum(expansion_rate_partial**_i for _i in range(max_traj_len - _t))
                    for _t in range(max_traj_len)  # t=0~N-1, i=0~N-t-1
                )
                weight_nonstop = max(1, num_traj_nonstop / num_traj_nonstop_partial)
                weight_stop = max(1, num_traj_stop / num_traj_stop_partial)

                if is_stop:
                    log_numerator = log_p_F[i] + math.log(weight_stop)
                else:
                    log_numerator = log_p_F[i] + math.log(weight_nonstop)

                log_denominator = torch.log(
                    weight_stop * p_stop[i] + weight_nonstop * (p_react_uni[i] + p_react_bi[i] / r_block)
                )
                converted_log_p_F[i] = log_numerator - log_denominator
        return converted_log_p_F.clamp(max=0)
