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
from rxnflow.envs.action import Protocol
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
        sample_temp: float
            [Experimental] Softmax temperature used when sampling
        correct_idempotent: bool
            [Experimental] Correct for idempotent actions when counting
        pad_with_terminal_state: bool
            [Experimental] If true pads trajectories with a terminal
        """
        self.ctx: SynthesisEnvContext = ctx
        self.env: SynthesisEnv = env
        self.max_len = 3

        # Experimental flags
        self.sample_temp = sample_temp
        self.sanitize_samples = True
        self.correct_idempotent = correct_idempotent
        self.pad_with_terminal_state = pad_with_terminal_state

        self.action_subsampler: SubsamplingPolicy = action_subsampler

        self.protocol_dict: dict[str, Protocol] = env.workflow.protocol_dict

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
        bck_a: list[list[RxnAction]] = [[RxnAction(RxnActionType.Stop, self.protocol_dict["stop"])] for _ in range(n)]

        graphs: list[Graph] = [self.env.new() for _ in range(n)]
        rdmols: list[Chem.Mol] = [Chem.Mol() for _ in range(n)]
        done: list[bool] = [False] * n

        def not_done(lst) -> list[int]:
            return [e for i, e in enumerate(lst) if not done[i]]

        # NOTE: instead `for` statement, we use `while` iteration to ensure the synple workflows.
        traj_idx: int = 0
        while True:
            assert traj_idx < self.max_len
            traj_idx += 1
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

            # NOTE: Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n), strict=False):
                fwd_logprob[i].append(log_probs[j].item())
                data[i]["traj"].append((graphs[i], reaction_actions[j]))
                fwd_a[i].append(reaction_actions[j])
                bck_a[i].append(self.env.reverse(rdmols[i], reaction_actions[j]))

                # Step the self.environment
                rdmol = rdmols[i] = self.env.step(rdmols[i], reaction_actions[j])
                graphs[i] = self.ctx.obj_to_graph(rdmols[i])
                if self.is_terminate(rdmol):
                    done[i] = True
                    data[i]["is_sink"].append(1)
                else:
                    data[i]["is_sink"].append(0)
                n_incoming_edges = self.env.count_backward_transitions(rdmol)
                bck_logprob[i].append(-math.log(n_incoming_edges))
            if all(done):
                break

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

    @staticmethod
    def is_terminate(rdmol: Chem.Mol) -> bool:
        """check whether the connecting parts remain."""
        return not any(atom.GetSymbol() == "At" for atom in rdmol.GetAtoms())
