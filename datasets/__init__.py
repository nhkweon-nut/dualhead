from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .argoverse_v1_dataset import ArgoverseV1Dataset as ArgoverseV1Dataset


def __getattr__(name: str):
    if name == "ArgoverseV1Dataset":
        from .argoverse_v1_dataset import ArgoverseV1Dataset

        return ArgoverseV1Dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ArgoverseV1Dataset"]
