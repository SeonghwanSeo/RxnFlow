import os
import sqlite3

import torch

from gflownet.utils.sqlite_log import SQLiteLog


class CustomSQLiteLogHook:
    def __init__(self, log_dir, ctx) -> None:
        self.log = None  # Only initialized in __call__, which will occur inside the worker
        self.log_dir = log_dir
        self.ctx = ctx
        self.data_labels = None

    def __call__(self, trajs, rewards, obj_props, cond_info):
        if self.log is None:
            worker_info = torch.utils.data.get_worker_info()
            self._wid = worker_info.id if worker_info is not None else 0
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = f"{self.log_dir}/generated_objs_{self._wid}.db"
            self.log = SQLiteLog()
            self.log.connect(self.log_path)

        if hasattr(self.ctx, "object_to_log_repr"):
            objs = [self.ctx.object_to_log_repr(t["result"]) if t["is_valid"] else "" for t in trajs]
        else:
            objs = [""] * len(trajs)

        """ This is New """
        if hasattr(self.ctx, "traj_to_log_repr"):
            traj_str = [self.ctx.traj_to_log_repr(t["traj"]) if t["is_valid"] else "" for t in trajs]
        else:
            traj_str = [""] * len(trajs)
        """ END """

        obj_props = obj_props.reshape((len(obj_props), -1)).data.numpy().tolist()
        rewards = rewards.data.numpy().tolist()
        preferences = cond_info.get("preferences", torch.zeros((len(objs), 0))).data.numpy().tolist()
        focus_dir = cond_info.get("focus_dir", torch.zeros((len(objs), 0))).data.numpy().tolist()
        logged_keys = [k for k in sorted(cond_info.keys()) if k not in ["encoding", "preferences", "focus_dir"]]

        data = [
            [objs[i], rewards[i], traj_str[i]]
            + obj_props[i]
            + preferences[i]
            + focus_dir[i]
            + [cond_info[k][i].item() for k in logged_keys]
            for i in range(len(trajs))
        ]
        if self.data_labels is None:
            self.data_labels = (
                ["smi", "r", "traj"]
                + [f"fr_{i}" for i in range(len(obj_props[0]))]
                + [f"pref_{i}" for i in range(len(preferences[0]))]
                + [f"focus_{i}" for i in range(len(focus_dir[0]))]
                + [f"ci_{k}" for k in logged_keys]
            )

        self.log.insert_many(data, self.data_labels)
        return {}


def read_all_results(path):
    # E402: module level import not at top of file, but pandas is an optional dependency
    import pandas as pd  # noqa: E402

    num_workers = len([f for f in os.listdir(path) if f.startswith("generated_objs")])
    dfs = [
        pd.read_sql_query("SELECT * FROM results", sqlite3.connect(f"file:{path}/generated_objs_{i}.db?mode=ro"))
        for i in range(num_workers)
    ]
    return pd.concat(dfs).sort_index().reset_index(drop=True)
