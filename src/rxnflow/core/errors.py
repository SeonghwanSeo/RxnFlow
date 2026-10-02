"""Exceptions shared by the synthesis environment and GFlowNet policy."""


class InvalidTransition(ValueError):
    """A selected reaction has no structurally valid product within capacity."""


class NoValidActions(ValueError):
    """The sampled space has no budget-feasible continuation."""
