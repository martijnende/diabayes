import logging
import tomllib

from flask import Flask
from flask_migrate import Migrate
from flask_socketio import SocketIO

from .logger import SQLiteHandler
from .models import db

socketio = SocketIO()


def create_app(workspace: str | None = None):
    app = Flask(__name__)

    if workspace is None:
        raise RuntimeError("Workspace path is required")

    app.config.from_file(workspace / "workspace.toml", load=tomllib.load, text=False)
    app.config["SECRET_KEY"] = "123"

    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")

    from . import routes

    app.register_blueprint(routes.bp)

    with app.app_context():
        db.create_all()

    log_handler = SQLiteHandler(socketio=socketio)
    log_handler.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(message)s")
    log_handler.setFormatter(log_formatter)

    app.logger.addHandler(log_handler)
    app.logger.setLevel(logging.INFO)

    # Create file and plot handlers
    # Need to check for data files?

    return app


@socketio.on("connect", namespace="/logs")
def test_connect(auth):
    socketio.emit("connection_response", {"data": "Connected"}, namespace="/logs")


@socketio.on("disconnect")
def test_disconnect(reason):
    print("Client disconnected, reason:", reason)
