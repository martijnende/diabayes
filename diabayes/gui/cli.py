import os
import sys
import click
from pathlib import Path

from .workspace import create_workspace, find_workspace
from . import create_app

@click.group()
def cli():
    """MyApp CLI tool."""

@cli.command()
@click.argument("dirname", required=False, default="workspace")
def init(dirname):
    """Initialize a new workspace in the current directory."""
    path = Path.cwd() / dirname
    create_workspace(path)
    click.echo(f"Workspace created at {path}")

@cli.command()
def run():
    """Run the web app inside the current workspace."""
    ws = find_workspace(Path.cwd())
    if ws is None:
        click.echo("Error: no workspace found in current or parent directories.")
        sys.exit(1)

    app = create_app(workspace=ws)
    app.run(host="127.0.0.1", port=5000)

