from torch import nn


def init_weight_linear(m: nn.Module, act: type[nn.Module]):
    if act is nn.LeakyReLU:
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.01)
    elif act is nn.ReLU:
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
    elif act is nn.SiLU:
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
    else:
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", a=1e-5)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def mlp(
    n_in: int,
    n_hid: int,
    n_out: int,
    n_layer: int,
    act: type[nn.Module] = nn.LeakyReLU,
    norm: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Creates a fully-connected network with no activation after the last layer.
    If `n_layer` is 0 then this corresponds to `nn.Linear(n_in, n_out)`.
    """
    layers = []
    for i in range(n_layer):
        d_in = n_in if i == 0 else n_hid
        d_out = n_hid
        layers.append(nn.Linear(d_in, d_out))
        if norm:
            layers.append(nn.LayerNorm(d_out))
        layers.append(act())
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(n_hid, n_out))
    return nn.Sequential(*layers)
