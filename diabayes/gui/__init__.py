import logging
import tomllib
from pathlib import Path

from bokeh.server.server import Server
from flask import Flask
from flask_migrate import Migrate
from flask_socketio import SocketIO
from tornado.ioloop import IOLoop
from tornado.web import Application, FallbackHandler
from tornado.wsgi import WSGIContainer

from .file_handler import FileHandler
from .logger import SQLiteHandler
from .models import db
from .plot_handler import PlotHandler

socketio = SocketIO()
migrate = Migrate()
fhandler = FileHandler()
phandler = PlotHandler()


def create_app(workspace: Path | None = None):
    app = Flask(__name__)

    if workspace is None:
        raise RuntimeError("Workspace path is required")

    app.config.from_file(workspace / "workspace.toml", load=tomllib.load, text=False)
    app.config["SECRET_KEY"] = "123"

    db.init_app(app)
    migrate.init_app(app, db)
    socketio.init_app(app, cors_allowed_origins="*")

    from . import routes

    app.register_blueprint(routes.bp)

    with app.app_context():
        db.create_all()

    log_handler = SQLiteHandler(socketio=socketio)
    log_level = logging.DEBUG if app.config.get("DEBUG") else logging.INFO
    log_handler.setLevel(log_level)
    log_formatter = logging.Formatter("%(message)s")
    log_handler.setFormatter(log_formatter)
    app.logger.addHandler(log_handler)

    fhandler.init_app(app)
    # TODO: add current data file to config toml?
    phandler.init_app(app)

    bokeh_apps = {"/bkapp": phandler.make_bokeh_doc}
    bokeh_server = Server(
        bokeh_apps, io_loop=IOLoop.current(), allow_websocket_origin=["*"], port=5006
    )
    bokeh_server.start()

    container = WSGIContainer(app)
    tornado_app = Application(
        [
            (r"/bkapp.*", FallbackHandler, {"fallback": bokeh_server._tornado}),
            (r".*", FallbackHandler, {"fallback": container}),
        ]
    )
    tornado_app.listen(5000)

    try:
        bokeh_server.io_loop.start()
    except KeyboardInterrupt:
        print("Stopping...")
        bokeh_server.io_loop.stop()

    pass


@socketio.on("connect", namespace="/logs")
def test_connect(auth):
    socketio.server.emit(
        "connection_response", {"data": "Connected"}, namespace="/logs"
    )


@socketio.on("disconnect")
def test_disconnect(reason):
    print("Client disconnected, reason:", reason)
