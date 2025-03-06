from torch import nn


def init_weight_linear(m: nn.Module, act: type[nn.Module]):
    if act is nn.LeakyReLU:
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.01)
    elif act in [nn.ReLU or nn.SiLU]:
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
    else:
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", a=1e-5)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
