import torch
from torch import Tensor

from gflownet.envs.graph_building_env import ActionIndex
from rxnflow.envs import MolGraph, RxnActionType, RxnAction
from rxnflow.envs.retrosynthesis import RetroSynthesisTree
from rxnflow.models.gfn import RxnFlow
from rxnflow.policy.action_categorical import RxnActionCategorical
from rxnflow.algo.trajectory_balance import SynthesisTB
from rxnflow.algo.synthetic_path_sampling import SyntheticPathSampler

from rxnflow_3d.envs.env import SynthesisEnv3D


class SynthesisTB3D(SynthesisTB):
    def setup_graph_sampler(self):
        self.graph_sampler = SyntheticPathSampler3D(
            self.ctx,
            self.env,
            self.action_subsampler,
            max_len=self.max_len,
            importance_temp=self.importance_temp,
            sample_temp=self.sample_temp,
            correct_idempotent=self.cfg.do_correct_idempotent,
            pad_with_terminal_state=self.cfg.do_parameterize_p_b,
            num_workers=self.global_cfg.num_workers_retrosynthesis,
        )


class SyntheticPathSampler3D(SyntheticPathSampler):
    """A helper class to sample from ActionCategorical-producing models"""

    env: SynthesisEnv3D

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
            is_last_step = traj_idx == (self.max_len - 1)
            for i in not_done(range(n)):
                graphs[i].graph.update({"is_last_step": is_last_step, "sample_idx": i})

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

            for i, analysis_res in self.retro_analyzer.result():
                bck_logprob[i].append(self.calc_bck_logprob(bck_a[i][-1], analysis_res))
                retro_trees[i] = analysis_res
                if analysis_res is None:
                    done[i] = True
                    data[i]["is_sink"][-1] = 1
                    data[i]["is_valid"] = False

            # NOTE: Step each trajectory, and accumulate statistics
            need_to_binding = []
            for i, j in zip(not_done(range(n)), range(n), strict=False):
                data[i]["traj"].append((graphs[i], reaction_actions[j]))
                fwd_logprob[i].append(log_probs[j].item())
                fwd_a[i].append(reaction_actions[j])
                bck_a[i].append(self.env.reverse(graphs[i], reaction_actions[j]))
                try:
                    graphs[i] = g = self.env.step(graphs[i], reaction_actions[j])
                    assert g.mol is not None
                    need_to_binding.append(i)
                except AssertionError:
                    done[i] = True
                    data[i]["is_valid"] = False
                    data[i]["is_sink"].append(1)
                    bck_logprob[i].append(0.0)
                    continue
                self.retro_analyzer.submit(i, g.smi, traj_idx, [(bck_a[i][-1], retro_trees[i])])
                if self.is_terminate(g.mol):
                    done[i] = True
                    data[i]["is_sink"].append(1)
                else:
                    data[i]["is_sink"].append(0)

            # NOTE: run docking
            objs = [graphs[i].mol for i in need_to_binding]
            self.env.set_binding_pose_batch(objs)
            for i, obj in zip(need_to_binding, objs, strict=True):
                if obj.GetNumConformers() == 0:
                    done[i] = True
                    data[i]["is_valid"] = False
                    data[i]["is_sink"][-1] = 1

            if all(done):
                break
        assert all(done)

        for i, analysis_res in self.retro_analyzer.result():
            bck_logprob[i].append(self.calc_bck_logprob(bck_a[i][-1], analysis_res))
            if analysis_res is None:
                data[i]["is_valid"] = False

        for i in range(n):
            data[i]["fwd_logprob"] = sum(fwd_logprob[i])
            data[i]["bck_logprob"] = sum(bck_logprob[i])
            data[i]["bck_logprobs"] = torch.tensor(bck_logprob[i]).reshape(-1)
            data[i]["result"] = graphs[i]
            data[i]["bck_a"] = bck_a[i]
        return data
