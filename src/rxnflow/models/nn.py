from torch import nn


class Linear(nn.Linear):
    """A linear layer with various initialization methods."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        init: str = "xavier",
    ):
        self.init: str = init
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        init = self.init
        if init == "xavier":
            nn.init.xavier_uniform_(self.weight)
        elif init == "xavier_small":
            nn.init.xavier_uniform_(self.weight, gain=0.25)
        elif init == "zero":
            nn.init.zeros_(self.weight)
        else:
            raise ValueError(f"Unknown initialization method: {init}")

        if self.bias is not None:
            nn.init.zeros_(self.bias)
