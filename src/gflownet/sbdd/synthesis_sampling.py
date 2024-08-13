import torch
from torch import Tensor
from rdkit import Chem

from gflownet.algo.synthesis_sampling import SynthesisSampler
from gflownet.envs.graph_building_env import Graph
from gflownet.envs.synthesis import ReactionActionType, ReactionAction
from gflownet.envs.synthesis.retrosynthesis import RetroSynthesisTree
from gflownet.sbdd.gfn import SynthesisGFN_SBDD


class SynthesisSampler_SBDD(SynthesisSampler):
    """A helper class to sample from ActionCategorical-producing models"""

    def sample_from_model(
        self,
        model: SynthesisGFN_SBDD,
        n: int,
        cond_info: Tensor,
        dev: torch.device,
        random_action_prob: float = 0.0,
    ):
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for _ in range(n)]
        fwd_logprob: list[list[float]] = [[] for _ in range(n)]
        bck_logprob: list[list[float]] = [[] for _ in range(n)]

        self.retro_analyzer.init()
        retro_tree: list[RetroSynthesisTree] = [RetroSynthesisTree(Chem.Mol())] * n

        graphs = [self.env.new() for _ in range(n)]
        rdmols = [Chem.Mol() for _ in range(n)]
        done = [False] * n
        fwd_a: list[list[ReactionAction | None]] = [[None] for _ in range(n)]
        bck_a: list[list[ReactionAction]] = [[ReactionAction(ReactionActionType.Stop)] for _ in range(n)]

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        for traj_idx in range(self.max_len):
            torch_graphs = [self.ctx.graph_to_Data(graphs[i], traj_idx) for i in not_done(range(n))]
            not_done_mask = [not v for v in done]

            # NOTE: Different
            fwd_cat, *_, log_reward_preds = model(
                self.ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask], not_done(range(n))
            )
            fwd_cat.random_action_mask = torch.tensor(
                self.rng.uniform(size=len(torch_graphs)) < random_action_prob, device=dev
            ).bool()

            actions = fwd_cat.sample(self.action_sampler, self.onpolicy_temp, self.sample_temp, self.min_len)
            reaction_actions: list[ReactionAction] = [self.ctx.aidx_to_ReactionAction(a) for a in actions]
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
                if reaction_actions[j].action == ReactionActionType.Stop:
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
        model: SynthesisGFN_SBDD,
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
        data = [{"traj": [], "reward_pred": None, "is_valid": True} for _ in range(n)]
        graphs: list[Graph] = [self.env.new() for _ in range(n)]
        rdmols: list[Chem.Mol] = [Chem.Mol() for _ in range(n)]
        done: list[bool] = [False] * n

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        for traj_idx in range(self.max_len):
            torch_graphs = [self.ctx.graph_to_Data(graphs[i], traj_idx) for i in not_done(range(n))]
            not_done_mask = [not v for v in done]

            # NOTE: Different
            fwd_cat, *_ = model(self.ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask], not_done(range(n)))
            actions = fwd_cat.sample(self.action_sampler, sample_temp=self.sample_temp)
            reaction_actions: list[ReactionAction] = [self.ctx.aidx_to_ReactionAction(a) for a in actions]
            for i, j in zip(not_done(range(n)), range(n)):
                i: int
                data[i]["traj"].append((rdmols[i], reaction_actions[j]))
                if reaction_actions[j].action == ReactionActionType.Stop:
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
            data[i]["result"] = rdmols[i]
        return data
