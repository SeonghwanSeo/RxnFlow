import copy
import math

import torch
from torch import Tensor
from torch_geometric import data as gd

from gflownet.algo.graph_sampling import GraphSampler
from gflownet.envs.graph_building_env import ActionIndex, Graph
from gflownet.utils.misc import get_worker_device, get_worker_rng
from rxnflow.envs import MolGraph, RxnAction, RxnActionType, SynthesisEnv, SynthesisEnvContext
from rxnflow.envs.retrosynthesis import MultiRetroSyntheticAnalyzer, RetroSynthesisTree
from rxnflow.models.gfn import RxnFlow
from rxnflow.policy.action_categorical import RxnActionCategorical
from rxnflow.policy.action_space_subsampling import SubsamplingPolicy


class SyntheticPathSampler(GraphSampler):
    """A helper class to sample from ActionCategorical-producing models"""

    def __init__(
        self,
        ctx: SynthesisEnvContext,
        env: SynthesisEnv,
        action_subsampler: SubsamplingPolicy,
        min_len: int = 2,
        max_len: int = 4,
        importance_temp: float = 1.0,
        sample_temp: float = 1.0,
        correct_idempotent: bool = False,
        pad_with_terminal_state: bool = False,
    ):
        """
        Parameters
        ----------
        env: SynthesisEnv
            A synthesis-oriented environment.
        ctx: SynthesisEnvContext
            A context.
        aciton_subsampler: SubsamplingPolicy
            Action subsampler.
        importance_temp: float
            [Experimental] Temperature when importance weighting
        sample_temp: float
            [Experimental] Softmax temperature used when sampling
        correct_idempotent: bool
            [Experimental] Correct for idempotent actions when counting
        pad_with_terminal_state: bool
            [Experimental] If true pads trajectories with a terminal
        """
        self.ctx: SynthesisEnvContext = ctx
        self.env: SynthesisEnv = env
        self.min_len = min_len
        self.max_len = max_len

        self.action_subsampler: SubsamplingPolicy = action_subsampler
        self.retro_analyzer: MultiRetroSyntheticAnalyzer = self.env.retro_analyzer

        # Experimental flags
        self.importance_temp = importance_temp
        self.sample_temp = sample_temp
        self.correct_idempotent = correct_idempotent
        self.pad_with_terminal_state = pad_with_terminal_state

    def _estimate_policy(
        self,
        model: RxnFlow,
        torch_graphs: list[gd.Data],
        cond_info: torch.Tensor,
        not_done_mask: list[bool],
    ) -> RxnActionCategorical:
        dev = get_worker_device()
        g = self.ctx.collate(torch_graphs).to(dev)
        cond_info = cond_info[not_done_mask]
        fwd_cat, *_ = model.forward(g, cond_info)
        return fwd_cat

    def _sample_action(
        self,
        torch_graphs: list[gd.Data],
        fwd_cat: RxnActionCategorical,
        random_action_prob: float = 0,
    ) -> list[ActionIndex]:
        # NOTE: sample from forward policy (on-policy & random policy)
        sample_cat = copy.copy(fwd_cat)
        if self.importance_temp > 0:
            sample_cat.raw_logits = sample_cat.importance_weighting(self.importance_temp)
        if random_action_prob > 0:
            dev = get_worker_device()
            rng = get_worker_rng()
            is_random = torch.tensor(rng.uniform(size=len(torch_graphs)) < random_action_prob, device=dev)
            new_logits = []
            for protocol_idx in range(len(self.env.protocols)):
                logit = sample_cat.raw_logits[protocol_idx]
                subsample = sample_cat.subsamples[protocol_idx]

                # calibrate so that the protocols are sampled equally.
                num_actions = len(subsample)
                if num_actions > 0:
                    logit[is_random] = 1000 - math.log(num_actions)
                new_logits.append(logit)
            sample_cat.raw_logits = new_logits
        if self.sample_temp != 1:
            sample_cat.raw_logits = [i / self.sample_temp for i in sample_cat.raw_logits]
        sample_cat._apply_action_masks()
        actions = sample_cat.sample()
        return actions

    def sample_from_model(
        self,
        model: RxnFlow,
        n: int,
        cond_info: Tensor,
        random_action_prob: float = 0.0,
    ) -> list[dict]:
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
        data: list[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: list[Tuple[Graph, RxnAction]], the list of states and actions
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for _ in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: list[list[float]] = [[] for _ in range(n)]
        bck_logprob: list[list[float]] = [[] for _ in range(n)]

        fwd_a: list[list[RxnAction]] = [[] for _ in range(n)]
        bck_a: list[list[RxnAction]] = [[RxnAction(RxnActionType.Stop)] for _ in range(n)]

        graphs: list[MolGraph] = [self.env.new() for _ in range(n)]
        retro_trees: list[RetroSynthesisTree] = [RetroSynthesisTree("")] * n
        done: list[bool] = [False] * n

        def not_done(lst) -> list[int]:
            return [e for i, e in enumerate(lst) if not done[i]]

        for traj_idx in range(self.max_len):
            # Label the state is last or not
            for i in not_done(range(n)):
                graphs[i].graph.update({"allow_stop": (traj_idx >= self.min_len), "sample_idx": i})

            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [self.ctx.graph_to_Data(graphs[i]) for i in not_done(range(n))]
            not_done_mask = [not v for v in done]

            # NOTE: forward transition probability (forward policy) estimation
            fwd_cat: RxnActionCategorical = self._estimate_policy(model, torch_graphs, cond_info, not_done_mask)
            actions: list[ActionIndex] = self._sample_action(torch_graphs, fwd_cat, random_action_prob)
            reaction_actions: list[RxnAction] = [
                self.ctx.ActionIndex_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions, strict=True)
            ]
            log_probs = fwd_cat.log_prob(actions)

            # NOTE: async retro analysis
            for i, analysis_res in self.retro_analyzer.result():
                retro_trees[i] = analysis_res
                bck_logprob[i].append(self.calc_bck_logprob(bck_a[i][-1], analysis_res))
                # error handling
                if analysis_res is None:
                    done[i] = True
                    data[i]["is_sink"][-1] = 1
                    data[i]["is_valid"] = False

            # NOTE: Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n), strict=False):
                data[i]["traj"].append((graphs[i], reaction_actions[j]))
                fwd_logprob[i].append(log_probs[j].item())
                fwd_a[i].append(reaction_actions[j])
                bck_a[i].append(self.env.reverse(graphs[i], reaction_actions[j]))
                if reaction_actions[j].action is RxnActionType.Stop:
                    done[i] = True
                    bck_logprob[i].append(0.0)
                    data[i]["is_sink"].append(1)
                    continue
                try:
                    gp = self.env.step(graphs[i], reaction_actions[j])
                    assert gp.mol is not None
                except AssertionError:
                    done[i] = True
                    data[i]["is_valid"] = False
                    bck_logprob[i].append(0.0)
                    data[i]["is_sink"].append(1)
                else:
                    graphs[i] = gp
                    data[i]["is_sink"].append(0)
                    self.retro_analyzer.submit(i, gp.smi, traj_idx, [(bck_a[i][-1], retro_trees[i])])
            if all(done):
                break

        for i, analysis_res in self.retro_analyzer.result():
            bck_logprob[i].append(self.calc_bck_logprob(bck_a[i][-1], analysis_res))
            if analysis_res is None:
                data[i]["is_sink"][-1] = 1
                data[i]["is_valid"] = False

        for i in range(n):
            data[i]["fwd_logprob"] = sum(fwd_logprob[i])
            data[i]["bck_logprob"] = sum(bck_logprob[i])
            data[i]["bck_logprobs"] = torch.tensor(bck_logprob[i]).reshape(-1)
            data[i]["result"] = graphs[i]
            data[i]["bck_a"] = bck_a[i]
        return data

    def sample_inference(
        self,
        model: RxnFlow,
        n: int,
        cond_info: Tensor,
    ):
        """Model Sampling (Inference - Non Retrosynthetic Analysis)

        Parameters
        ----------
        model: RxnFlow
            Model whose forward() method returns RxnActionCategorical instances
        n: int
            Number of samples
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        dev: torch.device
            Device on which data is manipulated

        Returns
        -------
        data: list[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: list[Tuple[Chem.Mol, RxnAction]], the list of states and actions
           - is_valid: is the generated graph valid according to the env & ctx
        """
        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for _ in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_a: list[list[RxnAction]] = [[] for _ in range(n)]

        graphs: list[MolGraph] = [self.env.new() for _ in range(n)]
        done: list[bool] = [False] * n

        def not_done(lst) -> list[int]:
            return [e for i, e in enumerate(lst) if not done[i]]

        for traj_idx in range(self.max_len):
            # Label the state is last or not
            for i in not_done(range(n)):
                graphs[i].graph.update({"allow_stop": (traj_idx >= self.min_len), "sample_idx": i})

            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [self.ctx.graph_to_Data(graphs[i]) for i in not_done(range(n))]
            not_done_mask = [not v for v in done]

            # NOTE: forward transition probability (forward policy) estimation
            fwd_cat: RxnActionCategorical = self._estimate_policy(model, torch_graphs, cond_info, not_done_mask)
            actions: list[ActionIndex] = self._sample_action(torch_graphs, fwd_cat, 0)
            reaction_actions: list[RxnAction] = [
                self.ctx.ActionIndex_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions, strict=True)
            ]

            # NOTE: Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n), strict=False):
                data[i]["traj"].append((graphs[i], reaction_actions[j]))
                fwd_a[i].append(reaction_actions[j])
                if reaction_actions[j].action is RxnActionType.Stop:
                    done[i] = True
                    data[i]["is_sink"].append(1)
                    continue
                try:
                    gp = self.env.step(graphs[i], reaction_actions[j])
                    assert gp.mol is not None
                except AssertionError:
                    done[i] = True
                    data[i]["is_valid"] = False
                    data[i]["is_sink"].append(1)
                else:
                    data[i]["is_sink"].append(0)
                    graphs[i] = gp
            if all(done):
                break
        for i in range(n):
            data[i]["result"] = graphs[i]
        return data

    def sample_backward_from_graphs(self, graphs: list[Graph], model: RxnFlow | None, cond_info: Tensor):
        raise NotImplementedError()

    def calc_bck_logprob(self, action: RxnAction, retro_tree: RetroSynthesisTree) -> float:
        if retro_tree is None:
            return 0.0
        # retro_tree.print()
        COEFF = self.ctx.num_total_actions
        numerator = 0
        denomiator = 0
        for _action, child in retro_tree.branches:
            weight = sum(COEFF ** (-d) for d in child.iteration_depth())
            denomiator += weight
            if action == _action:
                numerator += weight
        if numerator > 0:
            return math.log(numerator) - math.log(denomiator)
        else:
            return 0.0
