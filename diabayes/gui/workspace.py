import os
from pathlib import Path


def create_workspace(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "uploads").mkdir(exist_ok=True)
    (path / "exports").mkdir(exist_ok=True)

    db_path = path / "app.db"
    if not db_path.exists():
        db_path.touch()

    cfg = path / "workspace.toml"
    if not cfg.exists():
        with open(cfg, "w") as f:
            f.write(f"WORKSPACE = '{path}'\n")
            f.write(f"SQLALCHEMY_DATABASE_URI = 'sqlite:///{db_path}'\n")
            f.write(f"DEBUG = '{True}'\n")
            f.write(f"TESTING = '{True}'\n")
            f.write(f"UPLOAD_FOLDER = '{path / 'uploads'}'\n")
            f.write(f"EXPORT_FOLDER = '{path / 'exports'}'\n")


def find_workspace(start: Path) -> Path | None:
    """Look for a workspace directory by checking cwd and parents."""
    path = start
    while path != path.parent:
        if (path / "app.db").exists() and (path / "workspace.toml").exists():
            return path
        path = path.parent
    return None
