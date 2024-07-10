import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from rdkit import Chem

from typing import List, Optional

from gflownet.envs.graph_building_env import Graph
from gflownet.envs.synthesis import (
    SynthesisEnv,
    SynthesisEnvContext,
    ReactionActionType,
    ReactionAction,
)
from gflownet.envs.synthesis.action_sampling import ActionSamplingPolicy
from gflownet.envs.synthesis.retrosynthesis import MultiRetroSyntheticAnalyzer, RetroSynthesisTree


class SynthesisSampler:
    """A helper class to sample from ActionCategorical-producing models"""

    def __init__(
        self,
        ctx: SynthesisEnvContext,
        env: SynthesisEnv,
        max_len: int,
        rng,
        action_sampler: ActionSamplingPolicy,
        sample_temp: float = 1,
        correct_idempotent: bool = False,
        pad_with_terminal_state: bool = False,
        num_workers: int = 4,
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

        self.action_sampler: ActionSamplingPolicy = action_sampler
        self.retro_analyzer = MultiRetroSyntheticAnalyzer(self.env.retrosynthetic_analyzer, num_workers)
        self.uniform_bck_logprob: bool = False

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
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        dev: torch.device
            Device on which data is manipulated

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, ReactionAction]], the list of states and actions
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """

        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for _ in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: List[List[float]] = [[] for _ in range(n)]
        bck_logprob: List[List[float]] = [[] for _ in range(n)]

        self.retro_analyzer.init()
        retro_tree: List[RetroSynthesisTree] = [RetroSynthesisTree(Chem.Mol())] * n

        graphs = [self.env.new() for _ in range(n)]
        rdmols = [Chem.Mol() for _ in range(n)]
        done = [False] * n
        fwd_a: List[List[Optional[ReactionAction]]] = [[None] for _ in range(n)]
        bck_a: List[List[ReactionAction]] = [[ReactionAction(ReactionActionType.Stop)] for _ in range(n)]

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        for traj_idx in range(self.max_len):
            torch_graphs = [self.ctx.graph_to_Data(graphs[i], traj_idx) for i in not_done(range(n))]
            not_done_mask = [not v for v in done]

            fwd_cat, *_, log_reward_preds = model(self.ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask])
            if np.random.rand() < random_action_prob:
                actions = fwd_cat.random_sample(self.action_sampler)
            else:
                actions = fwd_cat.sample(self.action_sampler, sample_temp=self.sample_temp)
            reaction_actions: List[ReactionAction] = [self.ctx.aidx_to_ReactionAction(a) for a in actions]
            log_probs = fwd_cat.log_prob_after_sampling(actions)
            for i, next_rt in self.retro_analyzer.result():
                bck_logprob[i].append(self.cal_bck_logprob(retro_tree[i], next_rt))
                retro_tree[i] = next_rt
            # Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n)):
                i: int
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]["traj"].append((graphs[i], reaction_actions[j]))
                fwd_a[i].append(reaction_actions[j])
                bck_a[i].append(self.env.reverse(rdmols[i], reaction_actions[j]))
                # Check if we're done
                if reaction_actions[j].action == ReactionActionType.Stop:  # 0 is ReactionActionType.Stop
                    done[i] = True
                    bck_logprob[i].append(0.0)
                    data[i]["is_sink"].append(1)
                    continue
                # If not done, step the self.environment
                try:
                    next_rdmol = self.env.step(rdmols[i], reaction_actions[j])
                except Exception:
                    done[i] = True
                    bck_logprob[i].append(0.0)
                    fwd_a[i][-1] = bck_a[i][-1] = ReactionAction(ReactionActionType.Stop)
                    data[i]["is_sink"].append(1)
                else:
                    self.retro_analyzer.submit(i, next_rdmol, traj_idx + 1, [(bck_a[i][-1], retro_tree[i])])
                    data[i]["is_sink"].append(0)
                    rdmols[i] = next_rdmol
                    graphs[i] = self.ctx.mol_to_graph(next_rdmol)
            if all(done):
                break
        for i, next_rt in self.retro_analyzer.result():
            bck_logprob[i].append(self.cal_bck_logprob(retro_tree[i], next_rt))
            retro_tree[i] = next_rt

        # is_sink indicates to a GFN algorithm that P_B(s) must be 1
        for i in range(n):
            data[i]["result_rdmol"] = rdmols[i]
            data[i]["result"] = graphs[i]
            data[i]["fwd_logprob"] = sum(fwd_logprob[i])
            data[i]["bck_logprob"] = sum(bck_logprob[i])
            data[i]["bck_logprobs"] = torch.tensor(bck_logprob[i]).reshape(-1)
            data[i]["bck_a"] = bck_a[i]

        return data

    def sample_inference(self, model: nn.Module, n: int, cond_info: Tensor, dev: torch.device):
        """Model Sampling (Inference)

        Parameters
        ----------
        model: nn.Module
            Model whose forward() method returns ActionCategorical instances
        n: int
            Number of samples
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        dev: torch.device
            Device on which data is manipulated

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Chem.Mol, ReactionAction]], the list of states and actions
           - fwd_logprob: P_F(tau)
           - is_valid: is the generated graph valid according to the env & ctx
        """

        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True} for _ in range(n)]
        fwd_logprob: List[List[float]] = [[] for _ in range(n)]

        graphs = [self.env.new() for _ in range(n)]
        rdmols = [Chem.Mol() for _ in range(n)]
        done = [False] * n
        fwd_a: List[List[Optional[ReactionAction]]] = [[None] for _ in range(n)]

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        for traj_idx in range(self.max_len):
            torch_graphs = [self.ctx.graph_to_Data(graphs[i], traj_idx) for i in not_done(range(n))]
            not_done_mask = [not v for v in done]

            fwd_cat, *_, log_reward_preds = model(self.ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask])
            actions = fwd_cat.sample(self.action_sampler, sample_temp=self.sample_temp)
            reaction_actions: List[ReactionAction] = [self.ctx.aidx_to_ReactionAction(a) for a in actions]
            log_probs = fwd_cat.log_prob_after_sampling(actions)

            for i, j in zip(not_done(range(n)), range(n)):
                i: int
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]["traj"].append((rdmols[i], reaction_actions[j]))
                fwd_a[i].append(reaction_actions[j])

                if reaction_actions[j].action == ReactionActionType.Stop:  # 0 is ReactionActionType.Stop
                    done[i] = True
                    # data[i]["is_valid"] = len(data[i]["traj"]) > 1
                    continue
                try:
                    next_rdmol = self.env.step(rdmols[i], reaction_actions[j])
                except Exception:
                    done[i] = True
                    data[i]["is_valid"] = False
                    rdmols[i] = Chem.Mol()
                else:
                    rdmols[i] = next_rdmol
                    if traj_idx == self.max_len - 1:
                        data[i]["traj"].append((rdmols[i], ReactionAction(ReactionActionType.Stop)))
                    else:
                        graphs[i] = self.ctx.mol_to_graph(next_rdmol)
            if all(done):
                break
        for i in range(n):
            data[i]["result"] = rdmols[i]
            data[i]["fwd_logprob"] = sum(fwd_logprob[i])
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

    def cal_bck_logprob(self, curr_rt: RetroSynthesisTree, next_rt: RetroSynthesisTree):
        if self.uniform_bck_logprob:
            # NOTE: PB is uniform
            return -math.log(len(next_rt))
        else:
            # NOTE: PB is proportional to the number of passing trajectories
            curr_rt_lens = curr_rt.length_distribution(self.max_len)
            next_rt_lens = next_rt.length_distribution(self.max_len)
            numerator = sum(
                curr_rt_lens[_t] * sum(self.env.num_total_actions**_i for _i in range(self.max_len - _t))
                for _t in range(0, self.max_len)  # T(s->s'), t=0~N-1, i=0~N-t-1
            )
            denominator = sum(
                next_rt_lens[_t] * sum(self.env.num_total_actions**_i for _i in range(self.max_len - _t + 1))
                for _t in range(1, self.max_len + 1)  # T(s'), t=1~N, i=0~N-t
            )
            return math.log(numerator) - math.log(denominator)

    def random_sample(self, n: int):
        """Random Samples in a minibatch

        Parameters
        ----------
        n: int
            Number of graphs to sample

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, ReactionAction]], the list of states and actions
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """

        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for _ in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        bck_logprob: List[List[float]] = [[] for _ in range(n)]

        self.retro_analyzer.init()
        retro_tree: List[RetroSynthesisTree] = [RetroSynthesisTree(Chem.Mol())] * n

        graphs = [self.env.new() for _ in range(n)]
        rdmols = [Chem.Mol() for _ in range(n)]
        done = [False] * n
        fwd_a: List[List[Optional[ReactionAction]]] = [[None] for _ in range(n)]
        bck_a: List[List[ReactionAction]] = [[ReactionAction(ReactionActionType.Stop)] for _ in range(n)]

        for traj_idx in range(self.max_len):
            for i, next_rt in self.retro_analyzer.result():
                bck_logprob[i].append(self.cal_bck_logprob(retro_tree[i], next_rt))
                retro_tree[i] = next_rt

            for i in range(n):
                if done[i]:
                    continue
                if traj_idx == 0:
                    block_idx = int(np.random.choice(self.ctx.num_building_blocks))
                    block = self.ctx.building_blocks[block_idx]
                    action = ReactionAction(ReactionActionType.AddFirstReactant, block=block, block_idx=block_idx)
                else:
                    v = np.random.rand()
                    if v < 0.1:
                        action = ReactionAction(ReactionActionType.Stop)
                    else:
                        rdmol = rdmols[i]
                        react_uni_mask = self.ctx.create_masks(rdmol, fwd=True, unimolecular=True).reshape(-1)
                        react_bi_mask = self.ctx.create_masks(rdmol, fwd=True, unimolecular=False).reshape(-1)
                        uni_is_allow = bool(react_uni_mask.any())
                        bi_is_allow = bool(react_bi_mask.any())
                        if uni_is_allow and bi_is_allow:
                            t = ReactionActionType.ReactUni if v < 0.2 else ReactionActionType.ReactBi
                        if uni_is_allow:
                            t = ReactionActionType.ReactUni
                        elif bi_is_allow:
                            t = ReactionActionType.ReactBi
                        else:
                            t = ReactionActionType.Stop

                        if t is ReactionActionType.ReactUni:
                            rxn_idx = int(np.random.choice(np.where(react_uni_mask)[0]))
                            rxn = self.ctx.unimolecular_reactions[rxn_idx]
                            action = ReactionAction(t, rxn)
                        elif t is ReactionActionType.ReactBi:
                            rxn_idx = int(np.random.choice(np.where(react_bi_mask)[0]))
                            rxn_idx, block_is_first = rxn_idx // 2, bool(rxn_idx % 2)
                            block_indices = self.action_sampler.get_space(t, (rxn_idx, block_is_first)).block_indices
                            block_idx = int(np.random.choice(block_indices))
                            rxn = self.ctx.bimolecular_reactions[rxn_idx]
                            block = self.ctx.building_blocks[block_idx]
                            action = ReactionAction(t, rxn, block, block_idx, block_is_first)
                        else:
                            action = ReactionAction(t)

                next_rdmol = self.env.step(rdmols[i], action)
                data[i]["traj"].append((graphs[i], action))
                fwd_a[i].append(action)
                bck_a[i].append(self.env.reverse(rdmols[i], action))
                if action.action == ReactionActionType.Stop:
                    done[i] = True
                    bck_logprob[i].append(0.0)
                    data[i]["is_sink"].append(1)
                else:
                    self.retro_analyzer.submit(i, next_rdmol, traj_idx + 1, [(bck_a[i][-1], retro_tree[i])])
                    rdmols[i] = next_rdmol
                    graphs[i] = self.ctx.mol_to_graph(rdmols[i])
                    data[i]["is_sink"].append(0)
            if all(done):
                break
        for i, next_rt in self.retro_analyzer.result():
            bck_logprob[i].append(self.cal_bck_logprob(retro_tree[i], next_rt))
            retro_tree[i] = next_rt

        for i in range(n):
            data[i]["result_rdmol"] = rdmols[i]
            data[i]["result"] = graphs[i]
            data[i]["bck_logprob"] = sum(bck_logprob[i])
            data[i]["bck_logprobs"] = torch.tensor(bck_logprob[i]).reshape(-1)
            data[i]["bck_a"] = bck_a[i]

        return data
