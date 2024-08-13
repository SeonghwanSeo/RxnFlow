# https://github.com/mirunacrt/synflownet
from itertools import chain

import torch
import torch.nn as nn
import torch_geometric.data as gd

from gflownet.config import Config
from gflownet.models.graph_transformer import GraphTransformer

from .env import ActionCategorical, ActionType


def mlp(n_in, n_hid, n_out, n_layer, act=nn.LeakyReLU):
    """Creates a fully-connected network with no activation after the last layer.
    If `n_layer` is 0 then this corresponds to `nn.Linear(n_in, n_out)`.
    """
    n = [n_in] + [n_hid] * n_layer + [n_out]
    return nn.Sequential(*sum([[nn.Linear(n[i], n[i + 1]), act()] for i in range(n_layer + 1)], [])[:-1])


torch.autograd.set_detect_anomaly(True)


class SynFlowNet(nn.Module):
    """GraphTransfomer class for a GFlowNet which outputs an ActionCategorical.

    Outputs logits corresponding to each action (template).
    """

    # The GraphTransformer outputs per-graph embeddings

    def __init__(
        self,
        env_ctx,
        cfg: Config,
        num_graph_out=1,
        do_bck=False,
    ) -> None:
        super().__init__()
        self.transf = GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=cfg.model.num_emb,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.graph_transformer.num_heads,
            ln_type=cfg.model.graph_transformer.ln_type,
        )
        self.env_ctx = env_ctx
        num_emb = cfg.model.num_emb
        num_glob_final = num_emb * 2  # *2 for concatenating global mean pooling & node embeddings
        self.action_type_order = env_ctx.action_type_order
        self.bck_action_type_order = env_ctx.bck_action_type_order

        # 3 Action types: ADD_REACTANT, REACT_UNI, REACT_BI
        # Every action type gets its own MLP that is fed the output of the GraphTransformer.
        # Here we define the number of inputs and outputs of each of those (potential) MLPs.
        self._action_type_to_num_inputs_outputs = {
            ActionType.Stop: (num_glob_final, 1),
            ActionType.AddFirstReactant: (num_glob_final, env_ctx.num_building_blocks),
            ActionType.AddReactant: (num_glob_final + env_ctx.num_bimolecular_rxns, env_ctx.num_building_blocks),
            ActionType.ReactUni: (num_glob_final, env_ctx.num_unimolecular_rxns),
            ActionType.ReactBi: (num_glob_final, env_ctx.num_bimolecular_rxns),
            ActionType.BckReactUni: (num_glob_final, env_ctx.num_unimolecular_rxns),
            ActionType.BckReactBi: (num_glob_final, env_ctx.num_bimolecular_rxns),
            ActionType.BckRemoveFirstReactant: (num_glob_final, 1),
        }

        self.add_reactant_hook = None

        self.do_bck = do_bck
        mlps = {}
        for atype in chain(self.action_type_order, self.bck_action_type_order if self.do_bck else []):
            num_in, num_out = self._action_type_to_num_inputs_outputs[atype]
            mlps[atype.cname] = mlp(num_in, num_emb, num_out, cfg.model.graph_transformer.num_mlp_layers)
        self.mlps = nn.ModuleDict(mlps)

        self.emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, cfg.model.graph_transformer.num_mlp_layers)
        self._logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2)

    def logZ(self, cond: torch.Tensor) -> torch.Tensor:
        return self._logZ(cond)

    def register_add_reactant_hook(self, hook):
        """
        Registers a custom hook for the AddReactant action.
        hook : callable
            The hook function to call with arguments (self, rxn_id, emb, g).
        """
        self.add_reactant_hook = hook

    def call_add_reactant_hook(self, rxn_id, emb, g):
        """
        Calls the registered hook for the AddReactant action, if any.
        rxn_id : int
            The ID of the reaction selected by the sampler.
        emb : torch.Tensor
            The embedding tensor for the current state.
        g : Graph
            The current graph.
        """
        if self.add_reactant_hook is not None:
            return self.add_reactant_hook(self, rxn_id, emb, g)
        else:
            raise RuntimeError("AddReactant hook not registered.")

    # ActionType.AddReactant gets masked in ActionCategorical class, not here
    def _action_type_to_mask(self, t, g):
        # if it is the first action, all logits get masked except those for AddFirstReactant
        # print(t.cname, getattr(g, "traj_len")[0])
        if hasattr(g, t.mask_name):
            masks = getattr(g, t.mask_name)
        att = []
        device = g.x.device
        for i in range(g.num_graphs):
            if getattr(g, "traj_len")[i] == 0 and t != ActionType.AddFirstReactant:
                att.append(torch.zeros(self._action_type_to_num_inputs_outputs[t][1]).to(device))
            elif getattr(g, "traj_len")[i] > 0 and t == ActionType.AddFirstReactant:
                att.append(torch.zeros(self._action_type_to_num_inputs_outputs[t][1]).to(device))
            else:
                att.append(
                    masks[
                        i * self._action_type_to_num_inputs_outputs[t][1] : (i + 1)
                        * self._action_type_to_num_inputs_outputs[t][1]
                    ]
                    if hasattr(g, t.mask_name)
                    else torch.ones((self._action_type_to_num_inputs_outputs[t][1]), device=device)
                )
        att = torch.stack(att)
        return att.view(g.num_graphs, self._action_type_to_num_inputs_outputs[t][1]).to(device)

    def _action_type_to_logit(self, t, emb, g):
        logits = self.mlps[t.cname](emb)
        return self._mask(logits, self._action_type_to_mask(t, g))

    def _mask(self, x, m):
        # mask logit vector x with binary mask m, -1000 is a tiny log-value
        # Note to self: we can't use torch.inf here, because inf * 0 is nan (but also see issue #99)
        return x * m + -1000 * (1 - m)

    def _make_cat(self, g, emb, action_types, fwd):
        return ActionCategorical(
            self.env_ctx,
            g,
            emb,
            logits=[self._action_type_to_logit(t, emb, g) for t in action_types],
            masks=[self._action_type_to_mask(t, g) for t in action_types],
            types=action_types,
            fwd=fwd,
        )

    def forward(self, g: gd.Batch, cond: torch.Tensor, is_first_action: bool = False):
        """
        Forward pass of the GraphTransformerReactionsGFN.

        Parameters
        ----------
        g : gd.Batch
            A standard torch_geometric Batch object. Expects `edge_attr` to be set.
        cond : torch.Tensor
            The per-graph conditioning information. Shape: (g.num_graphs, self.g_dim).

        Returns
        -------
        ActionCategorical
        """
        _, graph_embeddings = self.transf(g, cond)
        graph_out = self.emb2graph_out(graph_embeddings)
        action_type_order = [a for a in self.action_type_order if a not in [ActionType.AddReactant]]
        # Map graph embeddings to action logits
        fwd_cat = self._make_cat(g, graph_embeddings, action_type_order, fwd=True)
        if self.do_bck:
            bck_cat = self._make_cat(g, graph_embeddings, self.bck_action_type_order, fwd=False)
            return fwd_cat, bck_cat, graph_out
        return fwd_cat, graph_out
