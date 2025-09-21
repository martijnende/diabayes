import logging
import os
import tomllib

from flask import Flask
from flask_migrate import Migrate
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy

from .logger import SQLiteHandler
from .models import db

socketio = SocketIO()


def create_app(workspace: str | None = None):
    app = Flask(__name__)

    if workspace is None:
        raise RuntimeError("Workspace path is required")

    app.config.from_file(workspace / "workspace.toml", load=tomllib.load, text=False)

    db.init_app(app)
    migrate = Migrate(app, db)
    socketio.init_app(app)

    from . import routes

    app.register_blueprint(routes.bp)

    with app.app_context():
        db.create_all()

    handler = SQLiteHandler(socketio=socketio)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

    return app
