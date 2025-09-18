from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

db = SQLAlchemy()

def create_app(workspace: str | None = None):
    app = Flask(__name__)

    if workspace is None:
        raise RuntimeError("Workspace path is required")

    app.config["WORKSPACE"] = workspace
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(workspace, 'app.db')}"
    app.config["UPLOAD_FOLDER"] = os.path.join(workspace, "uploads")
    app.config["EXPORT_FOLDER"] = os.path.join(workspace, "exports")

    db.init_app(app)

    from . import routes
    app.register_blueprint(routes.bp)

    return app

