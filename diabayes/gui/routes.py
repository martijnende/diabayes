import os

from flask import Blueprint, current_app, jsonify, render_template, request

from .models import LogEntry

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
        return jsonify({"status": "ok", "message": ""}), 200
    except Exception:
        current_app.logger.error("File upload failed")
        return jsonify({"status": "error", "message": f"{Exception}"}), 500


@bp.route("/logs/all")
def check_db():
    entries = LogEntry.query.order_by(LogEntry.timestamp.desc()).all()
    return jsonify(
        [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "level": e.level.lower(),
                "msg": e.msg,
            }
            for e in entries
        ]
    )
