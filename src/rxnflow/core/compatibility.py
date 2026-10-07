"""Check artifact release versions against the last incompatible changes."""

from rxnflow.__version__ import __version__

# Update each minimum only when its format or semantics become incompatible.
LIBRARY_MIN_VERSION = "0.9.2"
MODEL_MIN_VERSION = "0.9.2"


def _version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = map(int, version.split("."))
    return major, minor, patch


def check_library_compatibility(version: str) -> None:
    if not (
        _version_tuple(LIBRARY_MIN_VERSION)
        <= _version_tuple(version)
        <= _version_tuple(__version__)
    ):
        raise ValueError(
            f"unsupported library version {version!r}; "
            f"supported range is {LIBRARY_MIN_VERSION} through {__version__}; "
            "regenerate the prepared environment"
        )


def check_model_compatibility(version: str) -> None:
    if not (
        _version_tuple(MODEL_MIN_VERSION)
        <= _version_tuple(version)
        <= _version_tuple(__version__)
    ):
        raise ValueError(
            f"unsupported model version {version!r}; "
            f"supported range is {MODEL_MIN_VERSION} through {__version__}; "
            "use a compatible checkpoint or retrain the model"
        )
