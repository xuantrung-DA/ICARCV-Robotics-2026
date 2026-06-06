from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_output_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def resolve_input_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = project_root() / p
    if repo_candidate.exists():
        return repo_candidate
    return repo_candidate