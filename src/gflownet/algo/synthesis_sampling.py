import math
import torch
import torch.nn as nn
from torch import Tensor
from rdkit import Chem

from gflownet.envs.graph_building_env import Graph
from gflownet.envs.synthesis import SynthesisEnv, SynthesisEnvContext, ReactionActionType, ReactionAction
from gflownet.envs.synthesis.action_sampling import ActionSamplingPolicy
from gflownet.envs.synthesis.retrosynthesis import MultiRetroSyntheticAnalyzer, RetroSynthesisTree
from gflownet.models.synthesis_gfn import SynthesisGFN


class SynthesisSampler:
    """A helper class to sample from ActionCategorical-producing models"""

    def __init__(
        self,
        ctx: SynthesisEnvContext,
        env: SynthesisEnv,
        min_len: int,
        max_len: int,
        rng,
        action_sampler: ActionSamplingPolicy,
        onpolicy_temp: float = 0.0,
        sample_temp: float = 1.0,
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
        self.min_len = min_len if min_len is not None else 2
        self.max_len = max_len if max_len is not None else 4
        self.rng = rng
        # Experimental flags
        self.onpolicy_temp = onpolicy_temp
        self.sample_temp = sample_temp
        self.sanitize_samples = True
        self.correct_idempotent = correct_idempotent
        self.pad_with_terminal_state = pad_with_terminal_state

        self.action_sampler: ActionSamplingPolicy = action_sampler
        self.retro_analyzer = MultiRetroSyntheticAnalyzer(self.env.retrosynthetic_analyzer, num_workers)
        self.uniform_bck_logprob: bool = False

    def sample_from_model(
        self,
        model: SynthesisGFN,
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
        data: list[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: list[Tuple[Graph, ReactionAction]], the list of states and actions
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """

        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for _ in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: list[list[float]] = [[] for _ in range(n)]
        bck_logprob: list[list[float]] = [[] for _ in range(n)]

        self.retro_analyzer.init()
        retro_tree: list[RetroSynthesisTree] = [RetroSynthesisTree(Chem.Mol())] * n

        graphs: list[Graph] = [self.env.new() for _ in range(n)]
        rdmols: list[Chem.Mol] = [Chem.Mol() for _ in range(n)]
        done: list[bool] = [False] * n
        fwd_a: list[list[ReactionAction | None]] = [[None] for _ in range(n)]
        bck_a: list[list[ReactionAction]] = [[ReactionAction(ReactionActionType.Stop)] for _ in range(n)]

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        for traj_idx in range(self.max_len):
            torch_graphs = [self.ctx.graph_to_Data(graphs[i], traj_idx) for i in not_done(range(n))]
            not_done_mask = [not v for v in done]

            fwd_cat, *_, log_reward_preds = model(self.ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask])
            fwd_cat.random_action_mask = torch.tensor(
                self.rng.uniform(size=len(torch_graphs)) < random_action_prob, device=dev
            ).bool()

            actions = fwd_cat.sample(self.action_sampler, self.onpolicy_temp, self.sample_temp, self.min_len)
            reaction_actions: list[ReactionAction] = [self.ctx.aidx_to_GraphAction(a) for a in actions]
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

    def sample_inference(
        self,
        model: SynthesisGFN,
        n: int,
        cond_info: Tensor,
        dev: torch.device,
    ):
        """Model Sampling (Inference - Non Retrosynthetic Analysis)

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
        data: list[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: list[Tuple[Chem.Mol, ReactionAction]], the list of states and actions
           - fwd_logprob: P_F(tau)
           - is_valid: is the generated graph valid according to the env & ctx
        """

        # This will be returned
        data = [{"traj": [], "is_valid": True} for _ in range(n)]
        graphs: list[Graph] = [self.env.new() for _ in range(n)]
        rdmols: list[Chem.Mol] = [Chem.Mol() for _ in range(n)]
        done: list[bool] = [False] * n
        cond_info = cond_info.to(dev)

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        for traj_idx in range(self.max_len):
            torch_graphs = [self.ctx.graph_to_Data(graphs[i], traj_idx) for i in not_done(range(n))]
            not_done_mask = [not v for v in done]

            fwd_cat, *_ = model(self.ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask])
            actions = fwd_cat.sample(self.action_sampler, sample_temp=self.sample_temp)
            reaction_actions: list[ReactionAction] = [self.ctx.aidx_to_GraphAction(a) for a in actions]
            for i, j in zip(not_done(range(n)), range(n)):
                i: int
                data[i]["traj"].append((rdmols[i], reaction_actions[j]))
                if reaction_actions[j].action == ReactionActionType.Stop:  # 0 is ReactionActionType.Stop
                    done[i] = True
                    continue
                try:
                    next_rdmol = self.env.step(rdmols[i], reaction_actions[j])
                except Exception:
                    done[i] = True
                else:
                    rdmols[i] = next_rdmol
                    graphs[i] = self.ctx.mol_to_graph(next_rdmol)
            if all(done):
                break
        for i in range(n):
            data[i]["result"] = graphs[i]
            data[i]["result_rdmol"] = rdmols[i]
        return data

    def sample_backward_from_graphs(
        self,
        graphs: list[Graph],
        model: nn.Module | None,
        cond_info: Tensor,
        dev: torch.device,
    ):
        """Sample a model's P_B starting from a list of graphs.

        Parameters
        ----------
        graphs: list[Graph]
            list of Graph endpoints
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
