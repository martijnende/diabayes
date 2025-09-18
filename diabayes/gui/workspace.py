import os
from pathlib import Path

def create_workspace(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "uploads").mkdir(exist_ok=True)
    (path / "exports").mkdir(exist_ok=True)

    # Create an empty SQLite DB if not exists
    db_path = path / "app.db"
    if not db_path.exists():
        db_path.touch()

    # Maybe add a config file for user-level options
    cfg = path / "myapp.cfg"
    if not cfg.exists():
        cfg.write_text("[myapp]\nworkspace = {}\n".format(path))


def find_workspace(start: Path) -> Path | None:
    """Look for a workspace directory by checking cwd and parents."""
    path = start
    while path != path.parent:
        if (path / "app.db").exists() and (path / "myapp.cfg").exists():
            return path
        path = path.parent
    return None

