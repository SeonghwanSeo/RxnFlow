import math
from typing import List, Optional, Tuple

from rdkit import Chem
import torch
import torch.nn as nn
from torch import Tensor

from gflownet.config import Config
from gflownet.envs.synthesis import (
    Graph,
    SynthesisEnv,
    SynthesisEnvContext,
    ReactionActionType,
    BackwardAction,
    ForwardAction,
    RetroSynthesisTree,
)


class SynthesisSampler:
    """A helper class to sample from ActionCategorical-producing models"""

    def __init__(
        self,
        ctx,
        env,
        max_len,
        rng,
        cfg: Config,
        sample_temp=1,
        correct_idempotent=False,
        pad_with_terminal_state=False,
    ):
        """
        Parameters
        ----------
        env: ReactionTemplateEnv
            A reaction template environment.
        ctx: ReactionTemplateEnvContext
            A context.
        max_len: int
            If not None, ends trajectories of more than max_len steps.
        pad_with_terminal_state: bool
        """
        self.ctx: SynthesisEnvContext = ctx
        self.env: SynthesisEnv = env
        self.max_len = max_len if max_len is not None else 4
        self.rng = rng
        # Experimental flags
        self.sample_temp = sample_temp
        self.sanitize_samples = True
        self.correct_idempotent = correct_idempotent
        self.pad_with_terminal_state = pad_with_terminal_state

        self.cfg = cfg.algo.action_sampling
        self.num_block_sampling = min(self.cfg.num_building_block_sampling, self.ctx.num_building_blocks)

    def sample_from_model(
        self,
        model: nn.Module,
        n: int,
        cond_info: Tensor,
        dev: torch.device,
        random_action_prob: float = 0.0,
    ):
        """Samples a model in a minibatch

        Parameters
        ----------
        model: nn.Module
            Model whose forward() method returns ActionCategorical instances
        n: int
            Number of graphs to sample
        action_sampling_size: int,
            Number of actions (building blocks) to sample
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        dev: torch.device
            Device on which data is manipulated

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]], the list of states and actions
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - bck_logratio: sum logratios P_Sb
           - is_valid: is the generated graph valid according to the env & ctx
        """

        # NOTE: Block Sampling
        block_indices = self.ctx.sample_blocks(self.num_block_sampling)
        if len(block_indices) == self.env.num_building_blocks:
            block_set = None
        else:
            block_set = set(self.env.building_blocks[i] for i in block_indices)
        block_emb = model.block_mlp(self.ctx.get_block_data(block_indices, dev))

        # This will be returned
        data = [{"traj": [], "retro_tree": [], "reward_pred": None, "is_valid": True, "is_sink": []} for _ in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: List[List[float]] = [[] for _ in range(n)]
        bck_logprob: List[List[float]] = [[] for _ in range(n)]

        retro_tree: List[RetroSynthesisTree] = [RetroSynthesisTree()] * n
        retro_tree_partial: List[RetroSynthesisTree] = [RetroSynthesisTree()] * n

        graphs = [self.env.new() for _ in range(n)]
        rdmols = [Chem.Mol() for _ in range(n)]
        done = [False] * n
        fwd_a: List[List[Optional[ForwardAction]]] = [[None] for _ in range(n)]
        bck_a: List[List[BackwardAction]] = [[BackwardAction(ReactionActionType.Stop)] for _ in range(n)]

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        for traj_idx in range(self.max_len):
            torch_graphs = [self.ctx.graph_to_Data(graphs[i], traj_idx) for i in not_done(range(n))]
            not_done_mask = [not v for v in done]

            fwd_cat, *_, log_reward_preds = model(self.ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask])
            if random_action_prob > 0:
                raise NotImplementedError()
            if self.sample_temp != 1:
                raise NotImplementedError()
            else:
                actions = fwd_cat.sample(block_indices, block_emb)
            reaction_actions: List[ForwardAction] = [
                self.ctx.aidx_to_ReactionAction(a, block_indices=block_indices) for a in actions
            ]
            log_probs = fwd_cat.log_prob(actions, block_indices, block_emb)

            # Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n)):
                i: int
                j: int
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]["traj"].append((graphs[i], reaction_actions[j]))
                data[i]["retro_tree"].append((retro_tree[i], retro_tree_partial[i]))
                fwd_a[i].append(reaction_actions[j])
                bck_a[i].append(self.env.reverse(rdmols[i], reaction_actions[j]))
                # Check if we're done
                if reaction_actions[j].action == ReactionActionType.Stop:  # 0 is ReactionActionType.Stop
                    done[i] = True
                    bck_logprob[i].append(0.0)
                    data[i]["is_sink"].append(1)

                else:  # If not done, step the self.environment
                    try:
                        next_rdmol = self.env.step(rdmols[i], reaction_actions[j])
                    except Exception as e:
                        done[i] = True
                        bck_logprob[i].append(0.0)
                        data[i]["is_valid"] = False
                        next_rdmol = Chem.Mol()
                        next_rt = next_rt_partial = RetroSynthesisTree()
                    else:
                        max_depth = self.max_len
                        expansion_rate = self.env.num_average_possible_actions
                        bck_action = bck_a[i][-1]
                        assert isinstance(bck_action, BackwardAction)

                        curr_rt = retro_tree[i]
                        curr_rt_partial = retro_tree_partial[i]

                        known_branches = [(bck_action, curr_rt)]
                        known_branches_partial = [(bck_action, curr_rt_partial)]

                        next_rt = self.env.retrosynthesis(next_rdmol, max_depth, known_branches=known_branches)
                        next_rt_partial = self.env.retrosynthesis(
                            next_rdmol, max_depth, block_set, known_branches_partial
                        )

                        curr_rt_lens = curr_rt.length_distribution(self.max_len)
                        next_rt_lens = next_rt.length_distribution(self.max_len)

                        numerator = sum(
                            curr_rt_lens[_t] * sum(expansion_rate**_i for _i in range(self.max_len - _t))
                            for _t in range(0, self.max_len)  # T(s->s'), t=0~N-1, i=0~N-t-1
                        )
                        denominator = sum(
                            next_rt_lens[_t] * sum(expansion_rate**_i for _i in range(self.max_len - _t + 1))
                            for _t in range(1, self.max_len + 1)  # T(s'), t=1~N, i=0~N-t
                        )
                        if False:
                            # NOTE: UNIFORM-PB
                            n_back = max(1, self.env.count_backward_transitions(next_rdmol))
                            bck_logprob[i].append(math.log(1 / n_back))
                        else:
                            # NOTE: RETROSYNTHESIS-PB
                            bck_logprob[i].append(math.log(numerator) - math.log(denominator))

                    if traj_idx == self.max_len - 1:
                        done[i] = True

                    data[i]["is_sink"].append(0)
                    rdmols[i] = next_rdmol
                    graphs[i] = self.ctx.mol_to_graph(next_rdmol)
                    retro_tree[i] = next_rt
                    retro_tree_partial[i] = next_rt_partial

                if done[i] and len(data[i]["traj"]) <= 2:
                    data[i]["is_valid"] = False
            if all(done):
                break

        # is_sink indicates to a GFN algorithm that P_B(s) must be 1
        for i in range(n):
            data[i]["result"] = graphs[i]
            data[i]["fwd_logprob"] = sum(fwd_logprob[i])
            data[i]["bck_logprob"] = sum(bck_logprob[i])
            data[i]["bck_logprobs"] = torch.tensor(bck_logprob[i], device=dev).reshape(-1)
            data[i]["bck_a"] = bck_a[i]
            data[i]["block_indices"] = block_indices
            if self.pad_with_terminal_state:
                data[i]["traj"].append((graphs[i], ForwardAction(ReactionActionType.Stop)))
                data[i]["retro_tree"].append((retro_tree[i], retro_tree_partial[i]))
                data[i]["is_sink"].append(1)

        return data

    def sample_backward_from_graphs(
        self,
        graphs: List[Graph],
        model: Optional[nn.Module],
        cond_info: Tensor,
        dev: torch.device,
    ):
        """Sample a model's P_B starting from a list of graphs.

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints
        model: nn.Module
            Model whose forward() method returns ActionCategorical instances
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        dev: torch.device
            Device on which data is manipulated

        """
        raise NotImplementedError()
