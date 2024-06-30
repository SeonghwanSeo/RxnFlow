import math
from typing import List, Optional

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
        block_emb = model.block_mlp(self.ctx.get_block_data(block_indices, dev))

        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for _ in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: List[List[float]] = [[] for _ in range(n)]
        bck_logprob: List[List[float]] = [[] for _ in range(n)]

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
                self.ctx.aidx_to_ReactionAction(g, a, block_indices=block_indices)
                for g, a in zip(torch_graphs, actions)
            ]
            log_probs = fwd_cat.log_prob(actions, block_indices, block_emb)

            # Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n)):
                i: int
                j: int
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]["traj"].append((graphs[i], reaction_actions[j]))
                fwd_a[i].append(reaction_actions[j])
                bck_a[i].append(self.env.reverse(rdmols[i], reaction_actions[j]))
                # Check if we're done
                if reaction_actions[j].action == ReactionActionType.Stop:  # 0 is ReactionActionType.Stop
                    done[i] = True
                    bck_logprob[i].append(0.0)
                    data[i]["is_sink"].append(1)

                else:  # If not done, step the self.environment
                    try:
                        rdmol = self.env.step(rdmols[i], reaction_actions[j])
                    except Exception as e:
                        rdmol = None
                    if rdmol is None:
                        done[i] = True
                        data[i]["is_valid"] = False
                        rdmol = Chem.Mol()

                    if traj_idx == self.max_len - 1:
                        done[i] = True

                    n_back = max(1, self.env.count_backward_transitions(rdmol))
                    bck_logprob[i].append(math.log(1 / n_back))
                    data[i]["is_sink"].append(0)
                    rdmols[i] = rdmol
                    graphs[i] = self.ctx.mol_to_graph(rdmol)

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
