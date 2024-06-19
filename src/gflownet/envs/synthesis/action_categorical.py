import math
import random

import torch
import torch_geometric.data as gd

from typing import Dict, List, Optional

from gflownet.envs.synthesis.action import ReactionActionIdx, ReactionActionType
from gflownet.envs.synthesis.env import Graph
from gflownet.envs.synthesis.env_context import SynthesisEnvContext


class ReactionActionCategorical:
    def __init__(
        self,
        ctx: SynthesisEnvContext,
        graphs: gd.Batch,
        emb: Dict[str, torch.Tensor],
        logits: Dict[ReactionActionType, torch.Tensor],
        fwd: bool,
    ):
        self.num_graphs = graphs.num_graphs
        self.graphs: gd.Batch = graphs
        self.graph_embedding: torch.Tensor = emb["graph"]
        self.dev = dev = graphs.x.device
        self.ctx: SynthesisEnvContext = ctx
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

        self.logits = logits

        if ReactionActionType.AddFirstReactant in self.secondary_action_types:
            # NOTE: PlaceHolder
            self.logits[ReactionActionType.AddFirstReactant] = torch.full(
                (self.num_graphs, self.ctx.num_block_sampling), -torch.inf, device=dev
            )
        if ReactionActionType.AddReactant in self.secondary_action_types:
            self.logits[ReactionActionType.AddReactant] = torch.empty(
                (self.num_graphs, self.ctx.num_block_sampling), device=dev
            )
        self.batch = torch.arange(graphs.num_graphs, device=dev)

        self.logprobs: Optional[Dict[ReactionActionType, torch.Tensor]] = None
        self.log_n: Dict[ReactionActionType, float] = self.compute_log_n()

    def compute_log_n(self):
        log_n = {}
        # NOTE:
        # AddFirstReactant, BckRemoveFirstReactant: Always at Step 0 -> math.log(1) + math.log(num_blocks)
        log_n[ReactionActionType.Stop] = math.log(3)
        log_n[ReactionActionType.AddFirstReactant] = math.log(1) + math.log(self.ctx.num_block_sampling)
        log_n[ReactionActionType.ReactUni] = math.log(3)
        log_n[ReactionActionType.ReactBi] = math.log(3) + math.log(self.ctx.num_block_sampling)
        log_n[ReactionActionType.BckRemoveFirstReactant] = math.log(1) + math.log(self.ctx.num_building_blocks)
        log_n[ReactionActionType.BckReactUni] = math.log(3)
        log_n[ReactionActionType.BckReactBi] = math.log(3) + math.log(self.ctx.num_building_blocks)
        return log_n

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

    def _mask(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        assert m.dtype == torch.bool
        return x.masked_fill(torch.logical_not(m), -torch.inf)

    def argmax(self, x: List[torch.Tensor]) -> List[tuple[int, int]]:
        # for each graph in batch and for each action type, get max value and index
        max_per_type = [torch.max(tensor, dim=1) for tensor in x]
        max_values_per_type = [pair[0] for pair in max_per_type]
        argmax_indices_per_type = [pair[1] for pair in max_per_type]
        _, type_indices = torch.max(torch.stack(max_values_per_type), dim=0)
        action_indices = torch.gather(torch.stack(argmax_indices_per_type), 0, type_indices.unsqueeze(0)).squeeze(0)
        argmax_pairs = list(zip(type_indices.tolist(), action_indices.tolist(), strict=True))  # action type, action idx
        return argmax_pairs

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
        log_Z_primary = merged_exp_logits_primary.sum(dim=1).log()

        # compute log-probabilities for primary logits
        logprobs = {typ: logit - log_Z_primary.view(-1, 1) for typ, logit in corr_logits_primary.items()}

        # if there are secondary logits, compute log-probabilities for them
        if max_logits_secondary is not None:
            corr_logits_secondary = {
                typ: self.logits[typ] - max_logits_secondary.view(-1, 1) for typ in self.secondary_action_types
            }
            exp_logits_secondary = [i.exp().clamp(self._epsilon) for i in corr_logits_secondary.values()]
            merged_exp_logits_secondary = torch.cat(exp_logits_secondary, dim=1)
            log_Z_secondary = merged_exp_logits_secondary.sum(dim=1).log()
            for key, value in corr_logits_secondary.items():
                logprobs[key] = value - log_Z_secondary.view(-1, 1)
        self.logprobs = logprobs
        return self.logprobs

    def sample(
        self,
        model: torch.nn.Module,
        traj_idx: int,
        graphs: List[Graph],
        block_emb: torch.Tensor,
        block_indices: torch.Tensor,
    ) -> List[ReactionActionIdx]:
        """Samples from the categorical distribution"""
        # NOTE: The first action in a trajectory is always AddFirstReactant (select a building block)

        if traj_idx == 0:
            type_idx = self.types.index(ReactionActionType.AddFirstReactant)
            logit = model.add_first_reactant_hook(self.graph_embedding, block_emb)
            self.logits[ReactionActionType.AddFirstReactant] = logit
            noise = torch.rand_like(logit)
            gumbel = logit - (-noise.log()).log()
            argmax = self.argmax(x=[gumbel])
            return [(type_idx, 0, -1, a[1], -1) for a in argmax]

        # NOTE: Use the Gumbel trick to sample categoricals
        gumbel = []
        for t in self.primary_action_types:
            logit = self.logits[t]
            noise = torch.rand_like(logit)
            gumbel.append(logit - (-noise.log()).log())
        argmax = self.argmax(x=gumbel)  # tuple of action type, action idx

        actions: List[ReactionActionIdx] = []
        for i, (act_idx, rxn_idx) in enumerate(argmax):
            t = self.types[act_idx]
            if t == ReactionActionType.Stop:
                actions.append((act_idx, 1, -1, -1, -1))
            elif t == ReactionActionType.ReactUni:
                actions.append((act_idx, 0, rxn_idx, -1, -1))
            elif t == ReactionActionType.ReactBi:  # sample reactant
                mask = self.ctx.create_masks_for_bb_from_precomputed(graphs[i], rxn_idx)
                mask = torch.from_numpy(mask[block_indices]).to(self.dev)
                reactant_mask = torch.any(mask, dim=1)

                if not torch.any(reactant_mask):
                    stop_idx = self.types.index(ReactionActionType.Stop)
                    actions.append((stop_idx, 1, -1, -1, -1))
                    continue

                # NOTE: Call the hook to get the logits for the AddReactant action
                logit = model.add_reactant_hook(rxn_idx, self.graph_embedding[i], block_emb)
                logit = self._mask(logit, reactant_mask)
                self.logits[ReactionActionType.AddReactant][i] = logit
                noise = torch.rand_like(logit)
                gumbel = logit - (-noise.log()).log()
                max_idx = int(gumbel.argmax())
                assert reactant_mask[max_idx], "This index should not be masked"

                # NOTE: Check what is block: (mol, block) | (block, mol)
                first_is_block = bool(mask[max_idx, 0])
                second_is_block = bool(mask[max_idx, 1])
                assert first_is_block or second_is_block
                if first_is_block and second_is_block:
                    first_is_block = random.random() < 0.5
                    actions.append((act_idx, 0, rxn_idx, max_idx, first_is_block))
                else:
                    actions.append((act_idx, 0, rxn_idx, max_idx, first_is_block))
            else:
                raise ValueError

        return actions

    def log_prob(
        self,
        actions: List[ReactionActionIdx],
        traj_indices: torch.Tensor,
        model: torch.nn.Module,
        graphs: List[Graph],
        block_indices_list: torch.Tensor,
    ) -> torch.Tensor:
        assert len(graphs) == len(traj_indices) == len(actions)
        """Access the log-probability of actions"""
        # Initialize a tensor to hold the log probabilities for each action
        block_emb_list: List[Optional[torch.Tensor]] = [None] * (int(traj_indices.max()) + 1)
        if self.fwd:
            assert traj_indices is not None
            for i, (traj_idx, action) in enumerate(zip(traj_indices, actions, strict=True)):
                type_idx, is_stop, rxn_idx, block_local_idx, block_is_first = action
                t = self.types[type_idx]

                # NOTE: if len(block_indices_list): max_traj_len, it is stop
                if traj_idx < len(block_indices_list):
                    block_indices = block_indices_list[traj_idx]
                    block_emb: Optional[torch.Tensor] = block_emb_list[traj_idx]
                    if block_emb_list[traj_idx] is None:
                        block_g = gd.Batch.from_data_list([self.ctx.building_block_datas[idx] for idx in block_indices])
                        block_emb = model.block_transf.forward(block_g.to(self.dev))
                        block_emb_list[traj_idx] = block_emb
                    else:
                        block_emb = block_emb_list[traj_idx]
                else:
                    assert t is ReactionActionType.Stop
                    continue

                if t is ReactionActionType.AddFirstReactant:
                    assert block_emb is not None
                    logit = model.add_first_reactant_hook(self.graph_embedding[i : i + 1], block_emb)
                    self.logits[ReactionActionType.AddFirstReactant][i] = logit

                elif t is not ReactionActionType.ReactBi:  # secondary logits were computed
                    assert block_emb is not None
                    mask = torch.from_numpy(self.ctx.create_masks_for_bb_from_precomputed(graphs[i], rxn_idx))
                    mask = mask.to(self.dev)
                    reactant_mask = torch.any(mask[block_indices], dim=1)
                    logit = model.add_reactant_hook(rxn_idx, self.graph_embedding[i], block_emb)
                    logit = self._mask(logit, reactant_mask)
                    self.logits[ReactionActionType.AddReactant][i] = logit

        self.logprobs = self.logsoftmax()

        # NOTE: placeholder of log_action_probs and log_action_space_sizes
        log_action_probs = torch.empty(len(actions), device=self.dev)

        for i, (traj_idx, action) in enumerate(zip(traj_indices, actions, strict=True)):
            type_idx, is_stop, rxn_idx, block_idx, block_is_first = action
            t = self.types[type_idx]
            if is_stop:
                if self.fwd:
                    log_prob = self.logprobs[t][i]
                else:
                    log_prob = torch.tensor([0.0], device=self.dev, dtype=torch.float64)
            elif t == ReactionActionType.ReactUni:
                log_prob = self.logprobs[t][i, rxn_idx]
            elif t == ReactionActionType.BckReactUni:
                log_prob = self.logprobs[ReactionActionType.BckReactUni][i, rxn_idx]
            elif t == ReactionActionType.ReactBi:
                bireact_log_prob = self.logprobs[ReactionActionType.ReactBi][i, rxn_idx]
                addreactant_log_prob = self.logprobs[ReactionActionType.AddReactant][i, block_idx]
                log_prob = bireact_log_prob + addreactant_log_prob
                if log_prob.isnan() or log_prob.isinf():
                    print(self.fwd, t, action, bireact_log_prob, addreactant_log_prob)
            elif t == ReactionActionType.BckReactBi:
                log_prob = self.logprobs[ReactionActionType.BckReactBi][i, rxn_idx]
                if traj_idx == 1:  # NOTE: when traj_idx == 1, block + block
                    log_prob = log_prob - math.log(2)
            elif t == ReactionActionType.AddFirstReactant:
                log_prob = self.logprobs[ReactionActionType.AddFirstReactant][i, block_idx]
                if log_prob.isnan() or log_prob.isinf():
                    print(self.fwd, t, action, log_prob)
            elif t == ReactionActionType.BckRemoveFirstReactant:
                log_prob = self.logprobs[ReactionActionType.BckRemoveFirstReactant][i, block_idx]
            else:
                raise ValueError
            log_action_probs[i] = log_prob
        return log_action_probs

    def log_n_actions(self, actions):
        return torch.tensor([self.log_n[self.types[action_type]] for action_type, *_ in actions], device=self.dev)
