from typing import Optional

import numpy as np
import torch
import torch_geometric.data as gd
from torch import Tensor
from torch_scatter import scatter

from gflownet.algo.config import TBVariant
from gflownet.config import Config
from gflownet.algo.trajectory_balance import TrajectoryBalanceModel
from gflownet.trainer import GFNAlgorithm

from gflownet.algo.synthesis_sampling import SynthesisSampler
from gflownet.envs.synthesis import (
    Graph,
    SynthesisEnv,
    SynthesisEnvContext,
    ReactionAction,
)


class ActionSamplingTrajectoryBalance(GFNAlgorithm):
    """Trajectory-based GFN loss implementations. Implements
    - TB: Trajectory Balance: Improved Credit Assignment in GFlowNets Nikolay Malkin, Moksh Jain,
    Emmanuel Bengio, Chen Sun, Yoshua Bengio
    https://arxiv.org/abs/2201.13259

    - ASTB: #T ODO

    # TODO:
        - SubTB(1): Learning GFlowNets from partial episodes for improved convergence and stability, Kanika Madan, Jarrid
        Rector-Brooks, Maksym Korablyov, Emmanuel Bengio, Moksh Jain, Andrei Cristian Nica, Tom Bosc, Yoshua Bengio,
        Nikolay Malkin
        https://arxiv.org/abs/2209.12782
        Note: We implement the lambda=1 version of SubTB here (this choice is based on empirical results from the paper)

        - DB: GFlowNet Foundations, Yoshua Bengio, Salem Lahlou, Tristan Deleu, Edward J. Hu, Mo Tiwari, Emmanuel Bengio
        https://arxiv.org/abs/2111.09266
        Note: This is the trajectory version of Detailed Balance (i.e. transitions are not iid, but trajectories are).
        Empirical results in subsequent papers suggest that DB may be improved by training on iid transitions (sampled from
        a replay buffer) instead of trajectories.
    """

    def __init__(
        self,
        env: SynthesisEnv,
        ctx: SynthesisEnvContext,
        rng: np.random.RandomState,
        cfg: Config,
    ):
        """Instanciate a TB algorithm.

        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        rng: np.random.RandomState
            rng used to take random actions
        cfg: Config
            Hyperparameters
        """
        self.ctx = ctx
        self.env = env
        self.rng = rng
        self.global_cfg = cfg
        self.cfg = cfg.algo.tb
        self.max_len = cfg.algo.max_len
        self.max_nodes = cfg.algo.max_nodes
        self.length_normalize_losses = cfg.algo.tb.do_length_normalize
        # Experimental flags
        self.reward_loss_is_mae = True
        self.tb_loss_is_mae = False
        self.tb_loss_is_huber = False
        self.mask_invalid_rewards = False
        self.reward_normalize_losses = False
        self.sample_temp = 1
        self.bootstrap_own_reward = self.cfg.bootstrap_own_reward
        # When the model is autoregressive, we can avoid giving it ["A", "AB", "ABC", ...] as a sequence of inputs, and
        # instead give "ABC...Z" as a single input, but grab the logits at every timestep. Only works if using something
        # like a transformer with causal self-attention.
        self.model_is_autoregressive = False

        self.graph_sampler = SynthesisSampler(
            ctx,
            env,
            cfg.algo.max_len,
            rng,
            self.global_cfg,
            self.sample_temp,
            correct_idempotent=self.cfg.do_correct_idempotent,
            pad_with_terminal_state=self.cfg.do_parameterize_p_b,
        )
        if self.cfg.variant == TBVariant.SubTB1:
            raise NotImplementedError

    def create_training_data_from_own_samples(
        self,
        model: TrajectoryBalanceModel,
        n: int,
        cond_info: Tensor,
        random_action_prob: float,
    ):
        """Generate trajectories by sampling a model

        Parameters
        ----------
        model: TrajectoryBalanceModel
           The model being sampled
        n: int
            Number of trajectories to sample
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        random_action_prob: float
            Probability of taking a random action
        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]]
           - reward_pred: float, -100 if an illegal action is taken, predicted R(x) if bootstrapping, None otherwise
           - fwd_logprob: log Z + sum logprobs P_F -> NO USE
           - bck_logprob: sum logprobs P_B
           - logZ: predicted log Z
           - loss: predicted loss (if bootstrapping)
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = self.ctx.device
        cond_info = cond_info.to(dev)
        data = None
        while data is None:
            try:
                data = self.graph_sampler.sample_from_model(model, n, cond_info, dev, random_action_prob)
            except Exception as e:
                print(f"ERROR - create_training_data_from_own_samples - {e}")
                # raise e
                data = None
        logZ_pred = model.logZ(cond_info)
        for i in range(n):
            data[i]["logZ"] = logZ_pred[i].item()
        return data

    def create_training_data_from_graphs(
        self,
        graphs,
        model: Optional[TrajectoryBalanceModel],
        cond_info: Optional[Tensor] = None,
        random_action_prob: Optional[float] = None,
    ):
        # raise NotImplementedError()
        return []

    def get_idempotent_actions(
        self, g: Graph, gd: gd.Data, gp: Graph, action: ReactionAction, return_aidx: bool = True
    ):
        raise NotImplementedError()

    def construct_batch(self, trajs, cond_info, log_rewards):
        """Construct a batch from a list of trajectories and their information

        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        log_rewards: Tensor
            The transformed log-reward (e.g. torch.log(R(x) ** beta) ) for each trajectory. Shape (N,)
        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """

        if self.model_is_autoregressive:
            raise NotImplementedError
        else:
            torch_graphs = [
                self.ctx.graph_to_Data(traj[0], tj_idx) for tj in trajs for tj_idx, traj in enumerate(tj["traj"])
            ]
            actions = [self.ctx.ReactionAction_to_aidx(traj[1]) for tj in trajs for traj in tj["traj"]]
            retro_tree = [tree1 for tj in trajs for tree1, tree2 in tj["retro_tree"]]
            retro_tree_partial = [tree2 for tj in trajs for tree1, tree2 in tj["retro_tree"]]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.log_p_B = torch.cat([i["bck_logprobs"] for i in trajs], 0)
        batch.actions = torch.tensor(actions)
        if self.cfg.do_parameterize_p_b:
            batch.bck_actions = torch.tensor(
                [self.ctx.ReactionAction_to_aidx(traj[1]) for tj in trajs for traj in tj["bck_a"]]
            )
            batch.is_sink = torch.tensor(sum([i["is_sink"] for i in trajs], []))
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()

        # NOTE: ASGFN
        # NOTE: NON-TENSOR-DATAS
        batch.rev_traj_lens = [tree.length_distribution(self.max_len) for tree in retro_tree]
        batch.rev_traj_lens_partial = [tree.length_distribution(self.max_len) for tree in retro_tree_partial]
        batch.block_indices = trajs[0]["block_indices"]

        if self.cfg.do_correct_idempotent:
            raise NotImplementedError()

        return batch

    def compute_batch_losses(
        self,
        model: TrajectoryBalanceModel,
        batch: gd.Batch,
        num_bootstrap: int = 0,  # type: ignore[override]
    ):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: TrajectoryBalanceModel
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: gd.Batch
          batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap: int
          the number of trajectories for which the reward loss is computed. Ignored if 0."""
        dev = batch.x.device
        # A single trajectory is comprised of many graphs
        num_trajs = int(batch.traj_lens.shape[0])
        log_rewards = batch.log_rewards
        # Clip rewards
        assert log_rewards.ndim == 1
        clip_log_R = torch.maximum(
            log_rewards, torch.tensor(self.global_cfg.algo.illegal_action_logreward, device=dev)
        ).float()
        cond_info = batch.cond_info
        invalid_mask = 1 - batch.is_valid

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # The position of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # Forward pass of the model, returns a GraphActionCategorical representing the forward
        # policy P_F, optionally a backward policy P_B, and per-graph outputs (e.g. F(s) in SubTB).
        if self.cfg.do_parameterize_p_b:
            fwd_cat, bck_cat, per_graph_out = model(batch, cond_info[batch_idx])
        else:
            if self.model_is_autoregressive:
                fwd_cat, per_graph_out = model(batch, cond_info, batched=True)
            else:
                fwd_cat, per_graph_out = model(batch, cond_info[batch_idx])
        # Retreive the reward predictions for the full graphs,
        # i.e. the final graph of each trajectory
        log_reward_preds = per_graph_out[final_graph_idx, 0]
        # Compute trajectory balance objective
        log_Z = model.logZ(cond_info)[:, 0]
        # Compute the log prob of each action in the trajectory
        if self.cfg.do_correct_idempotent:
            raise NotImplementedError()
        else:
            # Else just naively take the logprob of the actions we took
            _log_p_F = fwd_cat.log_prob_fwd(batch.actions, batch.block_indices)
            log_p_F = fwd_cat.reweight_logp(
                _log_p_F, batch.actions, batch.rev_traj_lens, batch.rev_traj_lens_partial, batch.block_indices
            )
            if self.cfg.do_parameterize_p_b:
                log_p_B = bck_cat.log_prob_bck(batch.bck_actions)

        if self.cfg.do_parameterize_p_b:
            raise NotImplementedError
            # If we're modeling P_B then trajectories are padded with a virtual terminal state sF,
            # zero-out the logP_F of those states
            log_p_F[final_graph_idx] = 0
            if self.cfg.variant == TBVariant.SubTB1 or self.cfg.variant == TBVariant.DB:
                raise NotImplementedError

            # To get the correct P_B we need to shift all predictions by 1 state, and ignore the
            # first P_B prediction of every trajectory.
            # Our batch looks like this:
            # [(s1, a1), (s2, a2), ..., (st, at), (sF, None),   (s1, a1), ...]
            #                                                   ^ new trajectory begins
            # For the P_B of s1, we need the output of the model at s2.

            # We also have access to the is_sink attribute, which tells us when P_B must = 1, which
            # we'll use to ignore the last padding state(s) of each trajectory. This by the same
            # occasion masks out the first P_B of the "next" trajectory that we've shifted.
            log_p_B = torch.roll(log_p_B, -1, 0) * (1 - batch.is_sink)
        else:
            log_p_B = batch.log_p_B
        assert log_p_F.shape == log_p_B.shape

        # This is the log probability of each trajectory
        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")

        if self.cfg.variant == TBVariant.SubTB1:
            raise NotImplementedError()
        elif self.cfg.variant == TBVariant.DB:
            raise NotImplementedError()
        else:
            # Compute log numerator and denominator of the ASTB objective
            numerator = log_Z + traj_log_p_F
            denominator = clip_log_R + traj_log_p_B

            if self.mask_invalid_rewards:
                # Instead of being rude to the model and giving a
                # logreward of -100 what if we say, whatever you think the
                # logprobablity of this trajetcory is it should be smaller
                # (thus the `numerator - 1`). Why 1? Intuition?
                denominator = denominator * (1 - invalid_mask) + invalid_mask * (numerator.detach() - 1)

            if self.cfg.epsilon is not None:
                # Numerical stability epsilon
                epsilon = torch.tensor([self.cfg.epsilon], device=dev).float()
                numerator = torch.logaddexp(numerator, epsilon)
                denominator = torch.logaddexp(denominator, epsilon)
            if self.tb_loss_is_mae:
                traj_losses = abs(numerator - denominator)
            elif self.tb_loss_is_huber:
                pass  # TODO
            else:
                traj_losses = (numerator - denominator).pow(2)

        # Normalize losses by trajectory length
        if self.length_normalize_losses:
            traj_losses = traj_losses / batch.traj_lens
        if self.reward_normalize_losses:
            # multiply each loss by how important it is, using R as the importance factor
            # factor = Rp.exp() / Rp.exp().sum()
            factor = -clip_log_R.min() + clip_log_R + 1
            factor = factor / factor.sum()
            assert factor.shape == traj_losses.shape
            # * num_trajs because we're doing a convex combination, and a .mean() later, which would
            # undercount (by 2N) the contribution of each loss
            traj_losses = factor * traj_losses * num_trajs

        if self.cfg.bootstrap_own_reward:
            num_bootstrap = num_bootstrap or len(log_rewards)
            if self.reward_loss_is_mae:
                reward_losses = abs(log_rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap])
            else:
                reward_losses = (log_rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap]).pow(2)
            reward_loss = reward_losses.mean() * self.cfg.reward_loss_multiplier
        else:
            reward_loss = 0

        loss = traj_losses.mean() + reward_loss
        info = {
            "offline_loss": traj_losses[: batch.num_offline].mean() if batch.num_offline > 0 else 0,
            "online_loss": traj_losses[batch.num_offline :].mean() if batch.num_online > 0 else 0,
            "reward_loss": reward_loss,
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            "invalid_logprob": (invalid_mask * traj_log_p_F).sum() / (invalid_mask.sum() + 1e-4),
            "invalid_losses": (invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4),
            "logZ": log_Z.mean(),
            "loss": loss.item(),
        }
        return loss, info
