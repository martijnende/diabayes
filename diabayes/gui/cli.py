import signal
import sys
from pathlib import Path
from threading import Event, Thread

import click
from bokeh.server.server import Server
from flask_migrate import downgrade as mig_downgrade
from flask_migrate import init as mig_init
from flask_migrate import migrate as mig_migrate
from flask_migrate import upgrade as mig_upgrade
from tornado.ioloop import IOLoop

from . import create_app, socketio
from .workspace import create_workspace, find_workspace

stop_event = Event()


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
def init(dirname):  # type:ignore
    """Initialize a new workspace in the current directory."""
    path = Path.cwd() / dirname
    # Create workspace dir including default config file
    create_workspace(path)
    click.echo(f"Workspace created at {path}")


@cli.command()
def run():
    app = _create_app()

    def _bk_worker():
        io_loop = IOLoop()
        IOLoop.make_current(io_loop)
        # Create a Bokeh rendering server bound to port 5006
        bokeh_server = Server(
            app.extensions["plot_handler"].make_bokeh_doc,
            allow_websocket_origin=["localhost:5000"],
            port=5006,
            io_loop=io_loop,
            session_token_expiration=3600,
        )
        app.extensions["plot_handler"].server = bokeh_server
        bokeh_server.start()

        try:
            io_loop.start()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                print("Stopping Bokeh server...")
                bokeh_server.unlisten()
                bokeh_server.stop()
            except Exception:
                pass

    bk_thread = Thread(target=_bk_worker, daemon=True)
    bk_thread.start()

    def handle_signal(signum, frame):
        print("Received shutdown signal. Wrapping up...")
        stop_event.set()
        try:
            bokeh_server = app.extensions["plot_handler"].server
            if bokeh_server is not None and hasattr(bokeh_server, "io_loop"):
                bokeh_server.io_loop.add_callback(bokeh_server.io_loop.stop)
        except Exception as e:
            print("Failed to schedule Bokeh stop:")
            print(e)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        # Run the main application on port 5000
        socketio.run(
            app,
            host="127.0.0.1",
            port=5000,
            debug=bool(app.config["DEBUG"]),
            use_reloader=False,
        )
    except KeyboardInterrupt:
        print("Stopping Flask server...")
    finally:
        IOLoop.current().add_callback(IOLoop.current().stop)
        bk_thread.join(timeout=0)
        print("Application stopped cleanly")


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
