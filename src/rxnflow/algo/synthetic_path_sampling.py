import copy
import math
import torch
from torch import Tensor
from torch_geometric import data as gd
from rdkit import Chem

from gflownet.algo.graph_sampling import GraphSampler
from gflownet.envs.graph_building_env import ActionIndex, Graph
from gflownet.utils.misc import get_worker_device, get_worker_rng
from rxnflow.envs import SynthesisEnv, SynthesisEnvContext, RxnActionType, RxnAction
from rxnflow.models.gfn import RxnFlow
from rxnflow.policy.action_categorical import RxnActionCategorical
from rxnflow.policy.action_space_subsampling import SubsamplingPolicy
from rxnflow.envs.retrosynthesis import MultiRetroSyntheticAnalyzer, RetroSynthesisTree


class SyntheticPathSampler(GraphSampler):
    """A helper class to sample from ActionCategorical-producing models"""

    def __init__(
        self,
        ctx: SynthesisEnvContext,
        env: SynthesisEnv,
        action_subsampler: SubsamplingPolicy,
        min_len: int = 2,
        max_len: int = 3,
        sample_temp: float = 1.0,
        correct_idempotent: bool = False,
        pad_with_terminal_state: bool = False,
        num_workers: int = 4,
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

        # Experimental flags
        self.sample_temp = sample_temp
        self.sanitize_samples = True
        self.correct_idempotent = correct_idempotent
        self.pad_with_terminal_state = pad_with_terminal_state

        self.action_subsampler: SubsamplingPolicy = action_subsampler
        self.retro_analyzer = MultiRetroSyntheticAnalyzer(self.env.retro_analyzer, num_workers)

    def _estimate_policy(
        self,
        model: RxnFlow,
        torch_graphs: list[gd.Data],
        cond_info: torch.Tensor,
        not_done_mask: list[bool],
    ) -> RxnActionCategorical:
        dev = get_worker_device()
        ci = cond_info[not_done_mask] if cond_info is not None else None
        fwd_cat, *_, log_reward_preds = model(self.ctx.collate(torch_graphs).to(dev), ci)
        return fwd_cat

    def _sample_action(
        self,
        torch_graphs: list[gd.Data],
        fwd_cat: RxnActionCategorical,
        random_action_prob: float = 0,
    ) -> list[ActionIndex]:
        # NOTE: sample from forward policy (online policy & random policy)
        if random_action_prob > 0:
            dev = get_worker_device()
            rng = get_worker_rng()
            # Device which graphs in the minibatch will get their action randomized
            is_random_action = torch.tensor(rng.uniform(size=len(torch_graphs)) < random_action_prob, device=dev)
            # Set the logits to some large value to have a uniform distribution
            new_logits = []
            for logit, batch in zip(fwd_cat.logits, fwd_cat.batch, strict=True):
                is_random = is_random_action[batch]
                logit[is_random, :] = 1000 - math.log(logit.shape[-1])
                new_logits.append(logit)
            fwd_cat.logits = new_logits
        if self.sample_temp != 1:
            sample_cat = copy.copy(fwd_cat)
            sample_cat.logits = [i / self.sample_temp for i in fwd_cat.logits]
            actions = sample_cat.sample()
        else:
            actions = fwd_cat.sample()
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

        fwd_a: list[list[RxnAction | None]] = [[None] for _ in range(n)]
        bck_a: list[list[RxnAction]] = [[RxnAction(RxnActionType.Stop, self.ctx.stop_list[0])] for _ in range(n)]

        graphs: list[Graph] = [self.env.new() for _ in range(n)]
        rdmols: list[Chem.Mol] = [self.ctx.graph_to_obj(g) for g in graphs]
        rts: list[RetroSynthesisTree] = [RetroSynthesisTree("", None)] * n
        done: list[bool] = [False] * n

        def not_done(lst) -> list[int]:
            return [e for i, e in enumerate(lst) if not done[i]]

        # instead `for` statement, we use `while` iteration to ensure the synple workflows.
        for traj_idx in range(self.max_len):
            # Label the trajectory length is longer than min length
            allow_stop = traj_idx >= self.min_len
            for i in not_done(range(n)):
                graphs[i].graph["allow_stop"] = allow_stop

            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [self.ctx.graph_to_Data(graphs[i]) for i in not_done(range(n))]
            not_done_mask = [not v for v in done]

            # NOTE: estimate forward transition probability (forward policy)
            fwd_cat: RxnActionCategorical = self._estimate_policy(model, torch_graphs, cond_info, not_done_mask)
            actions: list[ActionIndex] = self._sample_action(torch_graphs, fwd_cat, random_action_prob)
            reaction_actions: list[RxnAction] = [
                self.ctx.ActionIndex_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions, strict=True)
            ]
            log_probs = fwd_cat.log_prob(actions)

            # NOTE: calculate backward transition probability (backward policy)
            for i, next_rt in self.retro_analyzer.result():
                bck_logprob[i].append(self.calc_bck_logprob(rts[i], next_rt))
                rts[i] = next_rt

            # NOTE: Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n), strict=False):
                fwd_logprob[i].append(log_probs[j].item())
                data[i]["traj"].append((graphs[i], reaction_actions[j]))
                fwd_a[i].append(reaction_actions[j])
                bck_a[i].append(self.env.reverse(graphs[i], reaction_actions[j]))
                # Check if we're done
                if reaction_actions[j].action == RxnActionType.Stop:
                    done[i] = True
                    bck_logprob[i].append(0.0)
                    data[i]["is_sink"].append(1)
                else:
                    try:
                        rdmol = self.env.step(rdmols[i], reaction_actions[j])
                        assert rdmol is not None
                    except Exception:
                        done[i] = True
                        data[i]["is_valid"] = False
                        bck_logprob[i].append(0.0)
                        data[i]["is_sink"].append(1)
                        continue
                    if traj_idx == self.max_len - 1:
                        done[i] = True
                    self.retro_analyzer.submit(i, rdmol, traj_idx + 1, [(bck_a[i][-1], rts[i])])
                    n_back = self.env.count_backward_transitions(rdmol)
                    bck_logprob[i].append(-math.log(n_back))
                    data[i]["is_sink"].append(0)
                    graphs[i] = self.ctx.obj_to_graph(rdmol)
                    rdmols[i] = rdmol
            if all(done):
                break

        for i, next_rt in self.retro_analyzer.result():
            bck_logprob[i].append(self.calc_bck_logprob(rts[i], next_rt))
            rts[i] = next_rt

        # is_sink indicates to a GFN algorithm that P_B(s) must be 1
        for i in range(n):
            data[i]["fwd_logprob"] = sum(fwd_logprob[i])
            data[i]["bck_logprob"] = sum(bck_logprob[i])
            data[i]["bck_logprobs"] = torch.tensor(bck_logprob[i]).reshape(-1)
            data[i]["result"] = graphs[i]
            data[i]["bck_a"] = bck_a[i]

        return data

    def sample_inference(self, model: RxnFlow, n: int, cond_info: Tensor):
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
           - fwd_logprob: P_F(tau)
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = get_worker_device()
        raise NotImplementedError

        # This will be returned
        data = [{"traj": [], "is_valid": True} for _ in range(n)]
        graphs: list[Graph] = [self.env.new() for _ in range(n)]
        rdmols: list[Chem.Mol] = [Chem.Mol() for _ in range(n)]
        done: list[bool] = [False] * n
        cond_info = cond_info.to(dev)

        def not_done(lst) -> list[int]:
            return [e for i, e in enumerate(lst) if not done[i]]

        traj_idx: int = 0
        while True:
            assert traj_idx < self.max_len
            traj_idx += 1
            torch_graphs = [self.ctx.graph_to_Data(graphs[i]) for i in not_done(range(n))]
            not_done_mask = [not v for v in done]

            fwd_cat: RxnActionCategorical = self._estimate_policy(model, torch_graphs, cond_info, not_done_mask)
            actions: list[ActionIndex] = self._sample_action(torch_graphs, fwd_cat)
            reaction_actions: list[RxnAction] = [
                self.ctx.ActionIndex_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions, strict=True)
            ]
            for i, j in zip(not_done(range(n)), range(n), strict=False):
                data[i]["traj"].append((graphs[i], reaction_actions[j]))
                rdmols[i] = rdmol = self.env.step(rdmols[i], reaction_actions[j])
                graphs[i] = self.ctx.obj_to_graph(rdmol)
                if self.is_terminate(rdmol):
                    done[i] = True
            if all(done):
                break
        for i in range(n):
            data[i]["result"] = graphs[i]
        return data

    def sample_backward_from_graphs(self, graphs: list[Graph], model: RxnFlow | None, cond_info: Tensor):
        """Sample a model's P_B starting from a list of graphs.

        Parameters
        ----------
        graphs: list[Graph]
            list of Graph endpoints
        model: nn.Module
            Model whose forward() method returns ActionCategorical instances
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        """
        raise NotImplementedError()

    def cal_bck_logprob(self, prev_rt: RetroSynthesisTree, next_rt: RetroSynthesisTree):
        """Estimate P_B of rdmol

        Parameters
        ----------
        prev_rt :
            backward trajectories of s_t
        next_rt :
            backward trajectories of s_{t+1}

        Returns
        -------
        P_B: float
        """

        # NOTE: PB is proportional to the number of passing trajectories
        prev_rt_lens = list(prev_rt._iteration_length(0))
        next_rt_lens = list(next_rt._iteration_length(0))

        prev_len_dist = [sum(length == _t for length in prev_rt_lens) for _t in range(0, self.max_len + 1)]
        next_len_dist = [sum(length == _t for length in next_rt_lens) for _t in range(0, self.max_len + 1)]

        numerator = sum(
            prev_len_dist[_t] * sum(self.env.avg_num_actions**_i for _i in range(self.max_len - _t))
            for _t in range(0, self.max_len)  # T(s->s'), t=0~N-1, i=0~N-t-1
        )
        denominator = sum(
            next_len_dist[_t] * sum(self.env.avg_num_actions**_i for _i in range(self.max_len - _t + 1))
            for _t in range(1, self.max_len + 1)  # T(s'), t=1~N, i=0~N-t
        )
        return math.log(numerator) - math.log(denominator)
