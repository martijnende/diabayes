import logging
import os

import numpy as np
import pandas as pd
import plotly.express as px
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO
from flask_socketio import emit as sock_emit

from .database import db

# Init Flask application with database
app = Flask(__name__)
# Specify in-memory SQLite database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///"
app.config["SECRET_KEY"] = "42"
socketio = SocketIO(app, cors_allowed_origins="*")
# Init database
db.init_app(app)
# Create all tables
with app.app_context():
    db.create_all()
# Get the log handler (imports app)
from .logger import SQlAlchemyHandler, get_logs

# Init/register log handler
db_handler = SQlAlchemyHandler()
db_handler.setLevel(logging.INFO)
app.logger.addHandler(db_handler)

with app.app_context():
    app.logger.info("Initialisation complete")


def create_plot():
    N = 10_000
    x = np.linspace(0, 1, N)
    y = np.random.randn(N)

    fig = px.line(x=x, y=y)
    graph_json = fig.to_json()

    return graph_json


@socketio.on("request_logs")
def send_logs():
    logs = get_logs()
    sock_emit("update_logs", logs)


@app.route("/")
@app.route("/load_data")
def load_data():
    return render_template("load_data.html", log=get_logs())


@app.route("/read_file", methods=["POST"])
def read_data():
    data = request.get_json()
    filename = data.get("filename")

    app.logger.info(f"Attempting to read file `{filename}`")

    if not filename or not os.path.isfile(filename):
        app.logger.error("File not found")
        return jsonify({"success": False, "error": "File not found"}), 400

    try:
        df = pd.read_csv(filename)
        return jsonify({"success": True, "content": list(df.columns)})
    except Exception as e:
        app.logger.error(str(e))
        app.logger.error("An error was encountered while reading the CSV file:")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/processing")
def processing():
    return render_template("processing.html", plot=create_plot())


@app.route("/inversion")
def inversion():
    return render_template("inversion.html")


if __name__ == "__main__":
    socketio.run(app, debug=True)
