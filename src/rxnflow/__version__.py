"""Release version used for models and prepared synthon libraries."""

__version__ = "2.2.0"


def check_library_compatibility(version: str) -> None:
    if version != __version__:
        raise ValueError(
            f"unsupported library version {version!r}, expected {__version__!r}. "
            "Regenerate the prepared environment with the installed version."
        )


def check_model_compatibility(version: str) -> None:
    if version != __version__:
        raise ValueError(
            f"unsupported model version {version!r}, expected {__version__!r}. "
            "Load the model with the version that produced it or retrain it."
        )
