import sys
from pathlib import Path

import click
from flask_migrate import downgrade as mig_downgrade
from flask_migrate import init as mig_init
from flask_migrate import migrate as mig_migrate
from flask_migrate import upgrade as mig_upgrade

from . import create_app, socketio
from .workspace import create_workspace, find_workspace


def _create_app():
    """Run the web app inside the current workspace."""
    ws = find_workspace(Path.cwd())
    if ws is None:
        click.echo("Error: no workspace found in current or parent directories.")
        sys.exit(1)

    app = create_app(workspace=ws)
    return app


"""
Basic CLI
---------

DiaBayes comes with a set of CLI tools to launch/manage the user interface.
First create a default workspace with `diabayes init` (creates "workspace")
or specify a workspace name with `diabayes init myworkspace`. Then enter
into the created workspace and run `diabayes run`.

For existing workspaces, skip the initialisation.
"""


@click.group
def cli():
    """MyApp CLI tool."""


@cli.command()
@click.argument("dirname", required=False, default="workspace")
def init(dirname):
    """Initialize a new workspace in the current directory."""
    path = Path.cwd() / dirname
    # Create workspace dir including default config file
    create_workspace(path)
    click.echo(f"Workspace created at {path}")


@cli.command()
def run():
    app = _create_app()
    socketio.run(app, host="127.0.0.1", port=5000, debug=bool(app.config["DEBUG"]))


"""
Database migrations
-------------------

When the structure of the database changes, the existing database entries
need to be mapped to this new structure (called "migration"). This is only
relevant to developers; users should never have to migrate.

Usage:

Initialise migration dir:       diabayes migrate init
Create a commit:                diabayes migrate revision -m "Add trace column"
Attempt to upgrade database:    diabayes migrate upgrade
Rollback in case it fails:      diabayes migrate downgrade --revision -1
"""


@click.group()
def migrate():
    """Database migration commands."""
    pass


@migrate.command()
def init():
    """Initialize migrations directory."""
    app = _create_app()
    with app.app_context():
        mig_init()


@migrate.command()
@click.option("-m", "--message", help="Revision message")
def revision(message):
    """Create new migration revision."""
    app = _create_app()
    with app.app_context():
        mig_migrate(message=message)


@migrate.command()
def upgrade():
    """Apply migrations."""
    app = _create_app()
    with app.app_context():
        mig_upgrade()


@migrate.command()
@click.option(
    "--revision", default="-1", help="Which revision to downgrade to (default: -1)"
)
def downgrade(revision):
    """Revert migrations."""
    app = _create_app()
    with app.app_context():
        mig_downgrade(revision)


cli.add_command(migrate)
