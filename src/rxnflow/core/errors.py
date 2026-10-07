"""Exceptions shared by the synthesis environment and GFlowNet policy."""


class InvalidTransition(ValueError):
    """A selected action violates synthesis budgets or has no valid product."""
