from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """
    Ensure a directory exists.
    Args:
        path: Path or string pointing to the directory to create.
    Returns:
        The Path object of the created (or existing) directory.
    """

    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p


def project_path(*parts: str) -> Path:
    """
    Return a Path relative to the project root.
    The function assumes this file is located at repo/src/utils.py and that
    "project root" is two levels above this file (i.e. repo/).
    Args:
        *parts: Path components to join under the project root.
    Returns:
        pathlib.Path object representing the requested path.
    """

    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.joinpath(*parts)
