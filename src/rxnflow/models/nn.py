"""Shared neural-network building blocks; weights are initialized by the model."""

from torch import nn


def mlp(
    d_in: int,
    d_hid: int,
    d_out: int,
    n_layer: int = 2,
    *,
    activation: type[nn.Module] = nn.SiLU,
    layernorm: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build n_layer Linear layers, including the output layer.

    Hidden blocks use Linear → optional LayerNorm → activation → optional
    dropout. With n_layer=1, this is a single Linear(d_in, d_out).
    """
    assert n_layer >= 1
    modules = []
    for _ in range(n_layer - 1):
        modules.append(nn.Linear(d_in, d_hid))
        if layernorm:
            modules.append(nn.LayerNorm(d_hid))
        modules.append(activation())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        d_in = d_hid
    return nn.Sequential(*modules, nn.Linear(d_in, d_out))
