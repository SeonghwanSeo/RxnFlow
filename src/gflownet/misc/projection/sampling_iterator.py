import os
import numpy as np
import torch

from rdkit import Chem

from gflownet.data.sampling_iterator import SamplingIterator
from gflownet.algo.trajectory_balance_synthesis import SynthesisTrajectoryBalance


def tanimoto_similarity(array1, array2):
    intersection = np.sum(np.logical_and(array1, array2))
    union = np.sum(np.logical_or(array1, array2))
    similarity = intersection / union
    return similarity


class ProjectionSamplingIterator(SamplingIterator):
    algo: SynthesisTrajectoryBalance

    def __iter__(self):
        self.iter += 1
        if self.det_after is not None and self.iter > self.det_after:
            self.random_action_prob = 0
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        # Now that we know we are in a worker instance, we can initialize per-worker things
        self.rng = self.algo.rng = self.task.rng = np.random.default_rng(142857 + self._wid)
        self.ctx.device = self.device
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = f"{self.log_dir}/generated_mols_{self._wid}.db"
            self.log.connect(self.log_path)

        for idcs in self._idx_iterator():
            num_offline = idcs.shape[0]  # This is in [0, self.offline_batch_size]
            # Sample conditional info such as temperature, trade-off weights, etc.

            if self.sample_cond_info:
                num_online = self.online_batch_size
                cond_info = self.task.sample_conditional_information(
                    num_offline + self.online_batch_size, self.train_it
                )

                # NOTE: Diff-Part1 - num_offline: num sample with random_policy
                # Sample some dataset data
                trajs = self.algo.create_training_data_from_random_samples(num_offline)
                rewards = []
                for i in range(num_offline):
                    prop, prop_t = self.task.get_property(trajs[i]["result_rdmol"])
                    rewards.append([1.0] * 6)
                    cond_info["encoding"][i, -(2048 + 5) :] = prop_t
                flat_rewards = list(self.task.flat_reward_transform(rewards)) if num_offline > 0 else []

            else:  # If we're not sampling the conditionals, then the idcs refer to listed preferences
                raise NotImplementedError

            # Sample some on-policy data
            is_valid = torch.ones(num_offline + num_online).bool()
            if num_online > 0:
                with torch.no_grad():
                    trajs += self.algo.create_training_data_from_own_samples(
                        self.model,
                        num_online,
                        cond_info["encoding"][num_offline:],
                        random_action_prob=self.random_action_prob,
                    )
                if self.algo.bootstrap_own_reward:
                    # The model can be trained to predict its own reward,
                    # i.e. predict the output of cond_info_to_logreward
                    pred_reward = [i["reward_pred"].cpu().item() for i in trajs[num_offline:]]
                    flat_rewards += pred_reward
                else:
                    # Otherwise, query the task for flat rewards
                    valid_idcs = torch.tensor(
                        [i + num_offline for i in range(num_online) if trajs[i + num_offline]["is_valid"]]
                    ).long()
                    # fetch the valid trajectories endpoints
                    mols = [self.ctx.graph_to_mol(trajs[i]["result"]) for i in valid_idcs]
                    # ask the task to compute their reward
                    # NOTE: Diff-Part2
                    online_flat_rew, m_is_valid = self.task.compute_flat_rewards(mols, valid_idcs)
                    assert (
                        online_flat_rew.ndim == 2
                    ), "FlatRewards should be (mbsize, n_objectives), even if n_objectives is 1"
                    # The task may decide some of the mols are invalid, we have to again filter those
                    valid_idcs = valid_idcs[m_is_valid]
                    pred_reward = torch.zeros((num_online, online_flat_rew.shape[1]))
                    pred_reward[valid_idcs - num_offline] = online_flat_rew
                    is_valid[num_offline:] = False
                    is_valid[valid_idcs] = True
                    flat_rewards += list(pred_reward)
                    # Override the is_valid key in case the task made some mols invalid
                    for i in range(num_online):
                        trajs[num_offline + i]["is_valid"] = is_valid[num_offline + i].item()

            # Compute scalar rewards from conditional information & flat rewards
            flat_rewards = torch.stack(flat_rewards)
            log_rewards = self.task.cond_info_to_logreward(cond_info, flat_rewards)
            log_rewards[torch.logical_not(is_valid)] = self.illegal_action_logreward

            assert len(trajs) == num_online + num_offline
            # Computes some metrics
            extra_info = {"random_action_prob": self.random_action_prob}
            if num_online > 0:
                H = sum(i["fwd_logprob"] for i in trajs[num_offline:])
                extra_info["entropy"] = -H / num_online
                extra_info["length"] = np.mean([len(i["traj"]) for i in trajs[num_offline:]])
            if not self.sample_cond_info:
                # If we're using a dataset of preferences, the user may want to know the id of the preference
                for i, j in zip(trajs, idcs):
                    i["data_idx"] = j
            #  note: we convert back into natural rewards for logging purposes
            #  (allows to take averages and plot in objective space)
            #  TODO: implement that per-task (in case they don't apply the same beta and log transformations)
            rewards = torch.exp(log_rewards / cond_info["beta"])
            if num_online > 0 and self.log_dir is not None:
                self.log_generated(
                    trajs[num_offline:],
                    rewards[num_offline:],
                    flat_rewards[num_offline:],
                    {k: v[num_offline:] for k, v in cond_info.items()},
                )
            if num_online > 0:
                extra_info["sampled_reward_avg"] = rewards[num_offline:].mean().item()
                for hook in self.log_hooks:
                    extra_info.update(
                        hook(
                            trajs[num_offline:],
                            rewards[num_offline:],
                            flat_rewards[num_offline:],
                            {k: v[num_offline:] for k, v in cond_info.items()},
                        )
                    )

            if self.replay_buffer is not None:
                # If we have a replay buffer, we push the online trajectories in it
                # and resample immediately such that the "online" data in the batch
                # comes from a more stable distribution (try to avoid forgetting)

                # cond_info is a dict, so we need to convert it to a list of dicts
                cond_info = [{k: v[i] for k, v in cond_info.items()} for i in range(num_offline + num_online)]

                # push the online trajectories in the replay buffer and sample a new 'online' batch
                for i in range(num_offline, len(trajs)):
                    self.replay_buffer.push(
                        trajs[i],
                        log_rewards[i],
                        flat_rewards[i],
                        cond_info[i],
                        is_valid[i],
                    )
                replay_trajs, replay_logr, replay_fr, replay_condinfo, replay_valid = self.replay_buffer.sample(
                    num_online
                )

                # append the online trajectories to the offline ones
                trajs = trajs[:num_offline] + replay_trajs
                log_rewards = torch.cat([log_rewards[:num_offline], replay_logr], dim=0)
                flat_rewards = torch.cat([flat_rewards[:num_offline], replay_fr], dim=0)
                cond_info = cond_info[:num_offline] + replay_condinfo  # list of dicts
                is_valid = torch.cat([is_valid[:num_offline], replay_valid], dim=0)

                # convert cond_info back to a dict
                cond_info = {k: torch.stack([d[k] for d in cond_info]) for k in cond_info[0]}

            if self.hindsight_ratio > 0.0:
                # Relabels some of the online trajectories with hindsight
                assert hasattr(
                    self.task, "relabel_condinfo_and_logrewards"
                ), "Hindsight requires the task to implement relabel_condinfo_and_logrewards"
                # samples indexes of trajectories without repeats
                hindsight_idxs = torch.randperm(num_online)[: int(num_online * self.hindsight_ratio)] + num_offline
                cond_info, log_rewards = self.task.relabel_condinfo_and_logrewards(
                    cond_info, log_rewards, flat_rewards, hindsight_idxs
                )
                log_rewards[torch.logical_not(is_valid)] = self.illegal_action_logreward

            # Construct batch
            batch = self.algo.construct_batch(trajs, cond_info["encoding"], log_rewards)
            batch.num_offline = num_offline
            batch.num_online = num_online
            batch.flat_rewards = flat_rewards
            batch.preferences = cond_info.get("preferences", None)
            batch.focus_dir = cond_info.get("focus_dir", None)
            batch.extra_info = extra_info
            if self.ctx.has_n():
                log_ns = [self.ctx.traj_log_n(i["traj"]) for i in trajs]
                batch.log_n = torch.tensor([i[-1] for i in log_ns], dtype=torch.float32)
                batch.log_ns = torch.tensor(sum(log_ns, start=[]), dtype=torch.float32)
            # TODO: we could very well just pass the cond_info dict to construct_batch above,
            # and the algo can decide what it wants to put in the batch object

            # Only activate for debugging your environment or dataset (e.g. the dataset could be
            # generating trajectories with illegal actions)
            if self.do_validate_batch:
                self.validate_batch(batch, trajs)

            self.train_it += worker_info.num_workers if worker_info is not None else 1
            yield batch
