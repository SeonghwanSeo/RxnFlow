"""Exceptions shared by the synthesis environment and GFlowNet policy."""


class InvalidTransition(ValueError):
    """A selected action is unavailable or has no valid product."""
