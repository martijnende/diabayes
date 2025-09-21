import os

from flask import Blueprint, current_app, jsonify, render_template, request

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/upload", methods=["POST"])
def upload_file():
    try:
        file = request.files["data_file"]
        upload_folder = current_app.config["UPLOAD_FOLDER"]
        path = os.path.join(upload_folder, file.filename)
        file.save(path)
        current_app.logger.info("File upload successful")
    except Exception:
        current_app.logger.error("File upload failed")
    return jsonify({"status": "ok", "message": "pong"})


@bp.route("/api/echo", methods=["POST"])
def echo():
    data = request.get_json(silent=True) or {}
    return jsonify({"you_sent": data})


@bp.route("/hello/<name>")
def hello(name):
    return f"Hello, {name}!"
